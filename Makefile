.PHONY: install test lint format api dashboard demo

install:
	uv pip install -e ".[dev,api,dashboard]"

test:
	python -m pytest

lint:
	ruff check .

format:
	ruff format .

api:
	uvicorn fingraph.api.app:app --reload

dashboard:
	streamlit run src/fingraph/dashboard/app.py

demo:
	python -m fingraph.scoring.run_scoring
	python -m fingraph.explain.run_investigation
