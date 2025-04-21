# Makefile for full pipeline

STATIONS := 20380357 21609641
PYTHON := python

.PHONY: all preprocess train tune train_tuned evaluate evaluate_tuned compare clean run run_optimized

# Define 'run' as default target that runs the full pipeline with default hyperparameters
run: setup preprocess train tune evaluate

# Define 'run_optimized' target that includes training with tuned hyperparameters
run_optimized: setup preprocess train tune train_tuned evaluate_tuned

all: run

# Setup required directories
setup:
	@echo "==> Creating required directories"
	@mkdir -p data/processed/train data/processed/val data/processed/test
	@mkdir -p models
	@mkdir -p results/plots results/metrics results/tuning

preprocess:
	@echo "==> Preprocessing data for all stations"
	@for s in $(STATIONS); do \
		$(PYTHON) -m src.preprocess --station $$s --data-dir . --out-dir data/processed; \
	done

train:
	@echo "==> Training models with default hyperparameters"
	$(PYTHON) -m src.train --data-dir data/processed --models-dir models

tune:
	@echo "==> Hyperparameter tuning per station"
	@for s in $(STATIONS); do \
		$(PYTHON) -m src.tune --station $$s; \
	done

train_tuned:
	@echo "==> Training models with tuned hyperparameters"
	$(PYTHON) -m src.train_tuned --data-dir data/processed --models-dir models --tuning-dir results/tuning

evaluate:
	@echo "==> Evaluating default models"
	$(PYTHON) -m src.evaluate --models-dir models --data-dir data/processed --plots-dir results/plots --metrics-dir results/metrics

evaluate_tuned:
	@echo "==> Evaluating tuned models"
	$(PYTHON) -m src.evaluate --models-dir models --data-dir data/processed --plots-dir results/plots --metrics-dir results/metrics --use-tuned

compare:
	@echo "==> Comparing default and tuned models"
	$(PYTHON) -m src.evaluate --models-dir models --data-dir data/processed --plots-dir results/plots/comparison --metrics-dir results/metrics --compare

clean:
	rm -rf data/processed/* models/* results/* results/tuning/*
