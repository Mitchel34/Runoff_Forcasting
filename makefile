# Makefile for Runoff Forecasting workflow

# Define stations and model types
STATIONS = 20380357 21609641
MODEL_TYPES = lstm transformer

# Default target that runs the complete workflow
all: preprocess tune train evaluate

# Verify raw data exists
verify-data:
	@echo "Verifying raw data exists..."
	@test -d data/raw/20380357 || (echo "Error: Missing data for station 20380357" && exit 1)
	@test -d data/raw/21609641 || (echo "Error: Missing data for station 21609641" && exit 1)
	@echo "Raw data verified."

# Preprocessing step
preprocess: verify-data
	@echo "====== Running Data Preprocessing ======"
	python3 src/preprocess.py
	@echo "====== Preprocessing Complete ======"

# Hyperparameter tuning (both stages)
tune: preprocess tune_hyperband tune_bayesian

# Stage 1: Hyperband tuning
tune_hyperband: preprocess
	@echo "====== Running Hyperband Tuning ======"
	python3 src/tune.py --station_id 21609641 --model_type lstm --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30
	python3 src/tune.py --station_id 21609641 --model_type transformer --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30
	python3 src/tune.py --station_id 20380357 --model_type lstm --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30
	python3 src/tune.py --station_id 20380357 --model_type transformer --tuner hyperband --epochs_per_trial 50 --batch_size 64 --max_trials 30
	@echo "====== Hyperband Tuning Complete ======"

# Stage 2: Bayesian optimization
tune_bayesian: tune_hyperband
	@echo "====== Running Bayesian Optimization ======"
	python3 src/tune.py --station_id 21609641 --model_type lstm --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20
	python3 src/tune.py --station_id 21609641 --model_type transformer --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20
	python3 src/tune.py --station_id 20380357 --model_type lstm --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20
	python3 src/tune.py --station_id 20380357 --model_type transformer --tuner bayesian --epochs_per_trial 80 --batch_size 64 --max_trials 20
	@echo "====== Bayesian Optimization Complete ======"

# Model training with the best hyperparameters from tuning
train: preprocess
	@echo "====== Training Models with Best Hyperparameters ======"
	python3 src/train.py --station_id 21609641 --model_type lstm --hyperparams_file results/hyperparameters/21609641_lstm_best_hps.json --epochs 100 --batch_size 64
	python3 src/train.py --station_id 21609641 --model_type transformer --hyperparams_file results/hyperparameters/21609641_transformer_best_hps.json --epochs 100 --batch_size 64
	python3 src/train.py --station_id 20380357 --model_type lstm --hyperparams_file results/hyperparameters/20380357_lstm_best_hps.json --epochs 100 --batch_size 64
	python3 src/train.py --station_id 20380357 --model_type transformer --hyperparams_file results/hyperparameters/20380357_transformer_best_hps.json --epochs 100 --batch_size 64
	@echo "====== Training Complete ======"

# Model evaluation
evaluate: train
	@echo "====== Evaluating Models ======"
	python3 src/evaluate.py --station_id 21609641 --model_type lstm
	python3 src/evaluate.py --station_id 21609641 --model_type transformer
	python3 src/evaluate.py --station_id 20380357 --model_type lstm
	python3 src/evaluate.py --station_id 20380357 --model_type transformer
	@echo "====== Evaluation Complete ======"

# Create directory structure
setup:
	@echo "====== Creating Directory Structure ======"
	mkdir -p data/raw/20380357
	mkdir -p data/raw/21609641
	mkdir -p data/processed/train
	mkdir -p data/processed/test
	mkdir -p data/processed/scalers
	mkdir -p models
	mkdir -p results/hyperparameters
	mkdir -p results/metrics
	mkdir -p results/plots
	mkdir -p results/tuning
	@echo "====== Directory Structure Created ======"

# Clean up all generated files
clean:
	@echo "====== Cleaning Up Generated Files ======"
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf results/hyperparameters/*
	rm -rf results/metrics/*
	rm -rf results/plots/*
	rm -rf results/tuning/*
	rm -rf tuner_logs/*
	rm -rf keras_tuner/*
	rm -rf logs/*
	@echo "====== Cleanup Complete ======"

# Clean only specific parts of the workflow
clean-processed:
	@echo "====== Cleaning Processed Data ======"
	rm -rf data/processed/*

clean-models:
	@echo "====== Cleaning Trained Models ======"
	rm -rf models/*

clean-results:
	@echo "====== Cleaning Results ======"
	rm -rf results/hyperparameters/*
	rm -rf results/metrics/*
	rm -rf results/plots/*
	rm -rf results/tuning/*

clean-logs:
	@echo "====== Cleaning Logs ======"
	rm -rf tuner_logs/*
	rm -rf keras_tuner/*
	rm -rf logs/*

# Mark targets that don't create files with their names
.PHONY: all preprocess tune tune_hyperband tune_bayesian train evaluate setup clean clean-processed clean-models clean-results clean-logs verify-data