.PHONY: install data test lint train evaluate baseline ablations figures reproduce demo

install:
	pip install -e ".[dev]"

data:
	airdet download

test:
	pytest

lint:
	ruff check . && ruff format --check .

train:
	airdet train --config configs/default.yaml --output-dir runs/main

evaluate:
	airdet evaluate --model models/airplane_fcn.keras --split test --out results/test_metrics.json

baseline:
	airdet baseline --out results/baseline_v1.json --save-model runs/baseline_v1.keras

ablations:
	python scripts/run_ablations.py

figures:
	python scripts/make_figures.py

# Everything behind the README, from an empty checkout (about 2 h on 2 CPU cores).
reproduce: data train
	cp runs/main/model.keras models/airplane_fcn.keras
	cp runs/main/model.yaml models/airplane_fcn.yaml
	cp runs/main/history.json results/training_history.json
	$(MAKE) evaluate baseline ablations
	airdet train --config configs/no_mining.yaml
	airdet evaluate --model runs/no_mining/model.keras --split test --out results/no_mining_test_metrics.json
	$(MAKE) figures

demo:
	python app.py
