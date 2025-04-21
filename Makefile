# Makefile for full pipeline

STATIONS := 20380357 21609641
PYTHON := python

.PHONY: all preprocess train tune evaluate clean run

# Define 'run' as default target that runs the full pipeline
run: setup preprocess train tune evaluate

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
	@echo "==> Training models"
	$(PYTHON) -m src.train --data-dir data/processed --models-dir models

tune:
	@echo "==> Hyperparameter tuning per station"
	@for s in $(STATIONS); do \
		$(PYTHON) -m src.tune --station $$s; \
	done

evaluate:
	@echo "==> Evaluating models"
	$(PYTHON) -m src.evaluate --models-dir models --data-dir data/processed --plots-dir results/plots --metrics-dir results/metrics

clean:
	rm -rf data/processed/* models/* results/* results/tuning/*
