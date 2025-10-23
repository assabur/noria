.PHONY: setup ingest index serve eval fmt lint test


setup:
poetry install


ingest:
poetry run python rag/ingest.py --path data/raw --chunk-size 300 --overlap 60


index:
poetry run python rag/index.py


serve:
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


fmt:
poetry run black . && poetry run ruff check --fix .


lint:
poetry run ruff check . && poetry run mypy rag app


test:
poetry run pytest -q


eval:
poetry run pytest -q tests/test_eval.py::test_retrieval_quality