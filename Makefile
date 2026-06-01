.PHONY: install data train evaluate experiments test lint api ui

## Install full development dependencies
install:
	pip install -r requirements-dev.txt

## Download M5 dataset
data:
	python scripts/download_data.py

## Train LightGBM model
train:
	python -m src.train_lgbm

## Run Chronos zero-shot (requires chronos-forecasting)
train-chronos:
	python -m src.train_chronos

## Run full model comparison experiment
experiments:
	python -m src.experiments

## Generate evaluation plots
evaluate:
	python -m src.evaluate

## Run all in sequence
all: data train experiments evaluate

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Lint
lint:
	ruff check src/ tests/ api/ app/

## Start FastAPI server
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Start Gradio UI
ui:
	python app/gradio_app.py
