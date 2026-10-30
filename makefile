.PHONY: setup ingest index serve eval fmt lint test

PYTHON ?= python3.12

setup:
	POETRY_VIRTUALENVS_USE_POETRY_PYTHON=true poetry env use "$(PYTHON)"
	poetry install --no-root

ingest:
	poetry run python -m rag.ingest --path data/raw --chunk-size 300 --overlap 60

index:
	poetry run python -m rag.index

serve:
	poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

fmt:
	poetry run black . && poetry run ruff check --fix .

lint:
	poetry run ruff check . && poetry run mypy rag app

test:
	poetry run python -m pytest -q test/test_ui_api.py

eval:
	poetry run python -m pytest -q test/test_eval.py::test_retrieval_quality
