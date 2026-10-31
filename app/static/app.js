'use strict';

const $ = (id) => document.getElementById(id);
const state = { mode: 'retrieve', busy: false, uploading: false, pipelineRunning: false, modelInitialized: false, result: null, history: [] };
const HISTORY_KEY = 'noria-history-v1';
let pipelineTimer;
let elapsedTimer;

try {
  const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? localStorage.getItem('rag-studio-history-v1') ?? '[]');
  if (Array.isArray(saved)) state.history = saved.filter((entry) => entry && typeof entry.question === 'string' && entry.result && Array.isArray(entry.result.sources)).slice(0, 20);
} catch { /* Un stockage local indisponible ne bloque pas les tests. */ }

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function message(text, error = false) {
  $('global-message').textContent = text;
  $('global-message').classList.toggle('error', error);
  $('global-message').hidden = !text;
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error('Le serveur est injoignable. Vérifiez que FastAPI est démarré.');
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error('Le serveur a renvoyé une réponse illisible. Consultez ses journaux.'); }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'Paramètres invalides. Vérifiez votre saisie.';
    throw new Error(detail);
  }
  return data;
}

function switchView(view) {
  $('view-corpus').hidden = view !== 'corpus';
  $('view-lab').hidden = view !== 'lab';
  $('page-label').textContent = view === 'corpus' ? 'Corpus & pipeline' : 'Laboratoire';
  document.querySelectorAll('.nav-item').forEach((button) => {
    button.classList.toggle('active', button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
}

document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => switchView(button.dataset.view)));

function updateControls() {
  const locked = state.pipelineRunning || state.uploading || state.busy;
  $('run-pipeline').disabled = locked;
  $('file-input').disabled = locked;
  $('submit-query').disabled = locked;
  ['chunk-size', 'chunk-overlap', 'corpus-lang', 'embeddings-model'].forEach((id) => { $(id).disabled = locked; });
  document.querySelectorAll('[data-mode], [data-question]').forEach((button) => { button.disabled = state.busy; });
  $('export-test').disabled = !state.result || state.busy;
}

async function refreshStatus() {
  $('refresh-status').disabled = true;
  try {
    const data = await api('/status');
    for (const name of ['qdrant', 'ollama']) {
      const service = data.services[name];
      $(`${name}-status`).textContent = service.detail;
      $(`${name}-dot`).className = `status-dot ${service.ok ? 'ok' : 'off'}`;
    }
    if (!state.modelInitialized) {
      $('embeddings-model').replaceChildren();
      for (const model of data.config.embedding_models) {
        const option = element('option', '', model.label);
        option.value = model.id;
        $('embeddings-model').append(option);
      }
      $('embeddings-model').value = data.config.embeddings || data.config.embeddings_default;
      state.modelInitialized = true;
    }
    $('active-embeddings').textContent = data.config.embeddings
      ? `Modèle de l’index : ${data.config.embeddings}${data.config.embeddings_legacy ? ' · Ancien encodage : relancez la pipeline pour appliquer le profil multilingue.' : ''}`
      : 'Modèle de l’index indisponible.';
    $('generation-caption').textContent = data.config.model;
  } catch (error) {
    for (const name of ['qdrant', 'ollama']) {
      $(`${name}-status`).textContent = 'Diagnostic indisponible';
      $(`${name}-dot`).className = 'status-dot off';
    }
    message(error.message, true);
  } finally { $('refresh-status').disabled = false; }
}

$('refresh-status').addEventListener('click', refreshStatus);

function formatBytes(bytes) { return bytes < 1024 ? `${bytes} o` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} Ko` : `${(bytes / (1024 * 1024)).toFixed(1)} Mo`; }

async function loadDocuments() {
  const data = await api('/documents');
  $('document-count').textContent = data.documents.length;
  $('document-list').replaceChildren();
  if (!data.documents.length) $('document-list').append(element('li', 'muted', 'Aucun document. Importez votre premier fichier.'));
  for (const file of data.documents) {
    const item = element('li');
    const info = element('div', 'file-info');
    info.append(element('span', 'file-name', file.name.split('/').pop()), element('small', 'file-path', file.name));
    item.append(element('span', 'file-icon', file.name.split('.').pop().toUpperCase()), info, element('span', 'file-size', formatBytes(file.size)));
    $('document-list').append(item);
  }
}

async function uploadFiles(files) {
  if (!files.length || state.uploading || state.pipelineRunning || state.busy) return;
  state.uploading = true;
  updateControls();
  message('');
  let uploaded = 0;
  const errors = [];
  for (const file of files) {
    $('upload-feedback').textContent = `Import de ${file.name}…`;
    try {
      if (!/\.(txt|md|pdf)$/i.test(file.name)) throw new Error(`${file.name} : seuls .pdf, .txt et .md sont acceptés.`);
      if (file.size > 5 * 1024 * 1024) throw new Error(`${file.name} : limite de 5 Mo dépassée.`);
      await api(`/documents?filename=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
      uploaded++;
    } catch (error) { errors.push(error.message); }
  }
  state.uploading = false;
  $('file-input').value = '';
  $('upload-feedback').textContent = `${uploaded} fichier(s) importé(s). Lancez la pipeline pour les rendre recherchables.`;
  if (errors.length) message(errors.join(' '), true);
  else if (uploaded) message('Import terminé. Réglez le découpage et lancez la pipeline pour indexer votre corpus.');
  try { await loadDocuments(); } catch (error) { message(error.message, true); }
  updateControls();
}

$('file-input').addEventListener('change', (event) => uploadFiles(Array.from(event.target.files)));
for (const type of ['dragenter', 'dragover']) $('drop-zone').addEventListener(type, (event) => { event.preventDefault(); $('drop-zone').classList.add('dragover'); });
for (const type of ['dragleave', 'drop']) $('drop-zone').addEventListener(type, (event) => { event.preventDefault(); $('drop-zone').classList.remove('dragover'); });
$('drop-zone').addEventListener('drop', (event) => uploadFiles(Array.from(event.dataTransfer.files)));

function renderPipeline(data) {
  state.pipelineRunning = data.status === 'running';
  $('pipeline-stage').textContent = data.stage;
  $('pipeline-percent').textContent = `${data.progress} %`;
  $('pipeline-progress').value = data.progress;
  if (data.status === 'done') {
    $('pipeline-detail').textContent = `${data.documents} document(s) · ${data.chunks} extraits indexés. Le corpus est prêt.`;
  } else if (data.status === 'error') {
    $('pipeline-detail').textContent = data.error;
  } else if (state.pipelineRunning) {
    $('pipeline-detail').textContent = data.chunks ? `${data.documents} document(s) · ${data.chunks} extraits. Laissez le serveur ouvert.` : 'Lecture et découpage de tous les documents locaux…';
  }
  updateControls();
}

async function pollPipeline() {
  clearTimeout(pipelineTimer);
  try {
    const wasRunning = state.pipelineRunning;
    const data = await api('/pipeline');
    renderPipeline(data);
    if (data.status === 'running') pipelineTimer = setTimeout(pollPipeline, 1500);
    else if (wasRunning && data.status === 'done') {
      message('Indexation terminée. Ouvrez le laboratoire pour tester votre corpus.');
      await refreshStatus();
    } else if (wasRunning && data.status === 'error') message(data.error, true);
  } catch (error) {
    message(`${error.message} Le suivi reprendra automatiquement.`, true);
    pipelineTimer = setTimeout(pollPipeline, 4000);
  }
}

$('run-pipeline').addEventListener('click', async () => {
  if (state.pipelineRunning || state.uploading || state.busy) return;
  const size = Number($('chunk-size').value);
  const overlap = Number($('chunk-overlap').value);
  if (!$('chunk-size').reportValidity() || !$('chunk-overlap').reportValidity()) return;
  if (!Number.isInteger(size) || !Number.isInteger(overlap) || overlap >= size) {
    message('Le chevauchement doit être un entier inférieur à la taille des extraits.', true);
    return;
  }
  state.pipelineRunning = true;
  updateControls();
  message('');
  try {
    const lang = $('corpus-lang').value;
    const data = await api('/pipeline', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ chunk_size: size, overlap, lang, embeddings_model: $('embeddings-model').value }) });
    $('search-lang').value = '';
    renderPipeline(data);
    pipelineTimer = setTimeout(pollPipeline, 1000);
  } catch (error) {
    state.pipelineRunning = false;
    message(error.message, true);
    updateControls();
    pollPipeline();
  }
});

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('[data-mode]').forEach((button) => {
    button.classList.toggle('active', button.dataset.mode === mode);
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  });
  $('submit-query').replaceChildren(document.createTextNode(mode === 'ask' ? 'Générer la réponse ' : 'Rechercher '), element('span', '', '↗'));
}

document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => { $('question').value = button.dataset.question; $('question').focus(); }));
$('question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $('query-form').requestSubmit(); } });

function duration(ms) { return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`; }
function percent(value) { return `${Number(value).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} %`; }

function renderResult(entry) {
  const data = entry.result;
  state.result = entry;
  $('results').hidden = false;
  $('empty-state').hidden = true;
  $('query-error').hidden = true;
  $('best-score').textContent = data.sources.length ? percent(Math.max(...data.sources.map((source) => source.correspondence_pct))) : '—';
  $('retrieval-time').textContent = duration(data.retrieval_latency_ms);
  $('generation-time').textContent = entry.mode === 'ask' && data.sources.length ? duration(data.llm_latency_ms) : '—';
  $('generation-caption').textContent = entry.mode === 'retrieve' ? 'Mode recherche seule' : !data.sources.length ? 'Aucun appel au modèle' : 'Ollama';
  $('total-time').textContent = duration(data.total_latency_ms);
  $('result-count').textContent = `${data.sources.length} extrait(s) retrouvé(s)`;
  $('answer-panel').hidden = !data.answer;
  $('answer-text').textContent = data.answer || '';
  $('sources-list').replaceChildren();
  if (!data.sources.length) $('sources-list').append(element('div', 'notice', 'Aucun extrait trouvé. Vérifiez la langue sélectionnée et relancez l’indexation si nécessaire.'));
  data.sources.forEach((source, index) => {
    const card = element('article', 'source-card');
    const top = element('div', 'source-top');
    top.append(element('span', 'source-rank', String(index + 1).padStart(2, '0')), element('strong', 'source-title', source.doc_id || 'Document'), element('span', 'source-score', percent(source.correspondence_pct)));
    const track = element('div', 'score-track');
    const fill = element('div', 'score-fill');
    fill.style.width = `${Math.max(0, Math.min(100, Number(source.correspondence_pct) || 0))}%`;
    track.append(fill);
    card.append(top, track, element('p', 'source-text', source.text || 'Extrait non disponible.'), element('div', 'source-meta', `${source.source || ''}${source.page ? ` · page ${source.page}` : ''} · cosinus : ${Number(source.score).toFixed(4)}`));
    $('sources-list').append(card);
  });
  $('raw-result').textContent = JSON.stringify(data, null, 2);
  updateControls();
}

function saveHistory() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history)); }
  catch { message('Le stockage du navigateur est plein ou indisponible. Exportez votre test pour le conserver.'); }
  renderHistory();
}

function renderHistory() {
  $('history-list').replaceChildren();
  if (!state.history.length) $('history-list').append(element('p', 'sidebar-muted', 'Vos recherches apparaîtront ici.'));
  state.history.forEach((entry) => {
    const button = element('button', 'history-item', entry.question);
    button.title = entry.question;
    button.append(element('small', '', `${entry.mode === 'ask' ? 'Réponse' : 'Recherche'} · ${new Date(entry.date).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`));
    button.addEventListener('click', () => {
      if (state.busy) return;
      switchView('lab');
      $('question').value = entry.question;
      $('search-lang').value = entry.lang;
      $('top-k').value = String(entry.k);
      setMode(entry.mode);
      renderResult(entry);
    });
    $('history-list').append(button);
  });
}

$('clear-history').addEventListener('click', () => { state.history = []; saveHistory(); });
$('export-test').addEventListener('click', () => {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = element('a');
  anchor.href = url;
  anchor.download = `noria-test-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$('query-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.busy || state.pipelineRunning || state.uploading) return;
  const question = $('question').value.trim();
  if (!question) { $('question').focus(); return; }
  const mode = state.mode;
  const lang = $('search-lang').value;
  const k = Number($('top-k').value);
  state.busy = true;
  state.result = null;
  updateControls();
  $('query-error').hidden = true;
  $('results').hidden = true;
  $('empty-state').hidden = true;
  $('query-loading').hidden = false;
  $('loading-title').textContent = mode === 'ask' ? 'Recherche des extraits et génération de la réponse…' : 'Recherche dans vos documents…';
  const started = Date.now();
  $('elapsed').textContent = '0 s';
  elapsedTimer = setInterval(() => { $('elapsed').textContent = `${Math.floor((Date.now() - started) / 1000)} s`; }, 1000);
  try {
    const result = await api(`/${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, lang, k }) });
    const entry = { question, lang, k, mode, date: new Date().toISOString(), result };
    state.history.unshift(entry);
    state.history = state.history.slice(0, 20);
    saveHistory();
    renderResult(entry);
  } catch (error) {
    $('query-error').textContent = error.message;
    $('query-error').hidden = false;
  } finally {
    clearInterval(elapsedTimer);
    $('query-loading').hidden = true;
    state.busy = false;
    updateControls();
  }
});

renderHistory();
refreshStatus();
loadDocuments().catch((error) => message(error.message, true));
pollPipeline();
