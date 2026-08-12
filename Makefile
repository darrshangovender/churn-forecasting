# Makefile for churn-forecasting reference implementation.
# Windows-friendly: all real work goes through `uv run`.

.PHONY: install data train eval forecast test notebooks benchmark clean

install:
	uv sync

# Data is downloaded lazily on first pipeline run.
data:
	@echo "Data is fetched lazily by churn.data.loader on first run."

train:
	uv run python -m churn.pipeline

eval:
	uv run python -m churn.pipeline

forecast:
	uv run python -m mrr_forecast.forecaster

test:
	uv run pytest

notebooks:
	uv run jupyter notebook notebooks/

benchmark:
	uv run python benchmarks/run.py

clean:
	@echo "Removing build artifacts (not data/)."
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ *.egg-info
