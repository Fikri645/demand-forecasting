.PHONY: install data train train-chronos finetune-chronos tune-chronos \
        tune-lgbm tune-ensemble experiments evaluate drift \
        test lint api ui docker-up all

PYTHON := C:/Users/fikri/AppData/Local/Programs/Python/Python311/python.exe

## Install full development dependencies
install:
	$(PYTHON) -m pip install -r requirements-dev.txt

## Download and extract Store Sales dataset (put zip in data/raw/ first)
data:
	$(PYTHON) scripts/download_data.py

## Train LightGBM model (default hyperparameters)
train:
	$(PYTHON) -m src.train_lgbm

## Tune LightGBM with Optuna (50 trials) — saves models/lgbm_tuned.pkl
tune-lgbm:
	$(PYTHON) -m src.tune_lgbm

## Run Chronos-2 zero-shot (requires GPU + chronos-forecasting)
train-chronos:
	$(PYTHON) -m src.train_chronos

## Fine-tune Chronos-2 on Favorita (1000 steps) — saves models/chronos_finetuned/
finetune-chronos:
	$(PYTHON) -m src.finetune_chronos

## Extended Chronos fine-tuning (2000 more steps from checkpoint)
tune-chronos:
	$(PYTHON) -m src.tune_chronos

## Grid-search ensemble weights — saves models/ensemble_weights.json
tune-ensemble:
	$(PYTHON) -m src.tune_ensemble

## Run full 8-model comparison experiment + MLflow logging
experiments:
	$(PYTHON) -m src.experiments

## Generate evaluation plots (reports/figures/)
evaluate:
	$(PYTHON) -m src.evaluate

## Run data drift report — compares last 30 days vs historical baseline
drift:
	$(PYTHON) -m monitoring.drift_report

## Run drift with custom window (e.g. make drift-window WINDOW=60)
drift-window:
	$(PYTHON) -m monitoring.drift_report --window $(WINDOW)

## Run all in sequence (requires Kaggle zip already in data/raw/)
all: data train tune-lgbm experiments evaluate

## Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

## Lint
lint:
	$(PYTHON) -m ruff check src/ tests/ api/ app/ monitoring/ --ignore E501

## Start FastAPI server
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Start Gradio UI locally
ui:
	$(PYTHON) app/gradio_app.py

## Start full stack (API + MLflow) via Docker Compose
docker-up:
	docker-compose up --build
