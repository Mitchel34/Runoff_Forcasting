#!/bin/bash

# Script to train NWM post-processing models for multiple stream gauges
# Author: Mitchell Carson
# Date: April 13, 2025

# Set base directory
BASE_DIR=$(dirname "$0")
cd $BASE_DIR

# Create necessary directories
mkdir -p data/processed
mkdir -p models
mkdir -p reports/figures

# Display script header
echo "========================================================="
echo "  NWM Runoff Forecast Correction - Model Training Script"
echo "========================================================="

# Default parameters
RAW_DATA_PATH="data/raw"
PROCESSED_DATA_PATH="data/processed"
MODEL_SAVE_PATH="models"
SEQUENCE_LENGTH=24
FORECAST_HORIZON=18
LSTM_UNITS=64
DROPOUT_RATE=0.2
LEARNING_RATE=0.001
NUM_LAYERS=2
BATCH_SIZE=32
EPOCHS=100
PATIENCE=10

# Process command-line arguments
DO_TUNE=false
STREAM_IDS=("20380357" "21609641")
TRAIN_MODE="default"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --tune)
      DO_TUNE=true
      shift
      ;;
    --stream)
      if [[ -n "$2" && "$2" != --* ]]; then
        STREAM_IDS=($2)
        shift 2
      else
        echo "Error: Stream ID not provided"
        exit 1
      fi
      ;;
    --streams)
      if [[ -n "$2" && "$2" != --* ]]; then
        IFS=',' read -ra STREAM_IDS <<< "$2"
        shift 2
      else
        echo "Error: Stream IDs not provided"
        exit 1
      fi
      ;;
    --mode)
      if [[ -n "$2" ]]; then
        TRAIN_MODE="$2"
        shift 2
      else
        echo "Error: Training mode not provided"
        exit 1
      fi
      ;;
    --epochs)
      if [[ -n "$2" ]]; then
        EPOCHS="$2"
        shift 2
      else
        echo "Error: Epochs not provided"
        exit 1
      fi
      ;;
    --help)
      echo "Usage: ./train_models.sh [options]"
      echo ""
      echo "Options:"
      echo "  --tune                  Perform hyperparameter tuning"
      echo "  --stream <id>           Train model for specific stream ID"
      echo "  --streams <id1,id2,...> Train models for multiple stream IDs (comma-separated)"
      echo "  --mode <mode>           Training mode (default, quick, full)"
      echo "  --epochs <n>            Number of training epochs"
      echo "  --help                  Show this help message"
      echo ""
      echo "Training modes:"
      echo "  default: 100 epochs, no tuning"
      echo "  quick: 20 epochs, no tuning (for testing)"
      echo "  full: 100 epochs with hyperparameter tuning"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set training mode parameters
case $TRAIN_MODE in
  quick)
    EPOCHS=20
    PATIENCE=5
    DO_TUNE=false
    echo "Using quick training mode: $EPOCHS epochs, no tuning"
    ;;
  full)
    EPOCHS=100
    DO_TUNE=true
    echo "Using full training mode: $EPOCHS epochs with tuning"
    ;;
  default)
    echo "Using default training mode: $EPOCHS epochs"
    ;;
  *)
    echo "Unknown mode: $TRAIN_MODE. Using default settings."
    ;;
esac

# Print configuration
echo ""
echo "Training Configuration:"
echo "----------------------"
echo "Stream IDs: ${STREAM_IDS[*]}"
echo "Sequence length: $SEQUENCE_LENGTH"
echo "Forecast horizon: $FORECAST_HORIZON"
echo "Hyperparameter tuning: $DO_TUNE"
echo "Epochs: $EPOCHS"
echo ""

# Build command
CMD="python -m src.train --stream_ids ${STREAM_IDS[@]} --sequence_length $SEQUENCE_LENGTH"
CMD+=" --forecast_horizon $FORECAST_HORIZON --lstm_units $LSTM_UNITS --dropout_rate $DROPOUT_RATE"
CMD+=" --learning_rate $LEARNING_RATE --num_layers $NUM_LAYERS --batch_size $BATCH_SIZE"
CMD+=" --epochs $EPOCHS --patience $PATIENCE"

# Add tuning if enabled
if [ "$DO_TUNE" = true ]; then
  CMD+=" --tune --max_trials 30 --tuner_type hyperband"
fi

# Add paths
CMD+=" --raw_data_path $RAW_DATA_PATH --processed_data_path $PROCESSED_DATA_PATH --model_save_path $MODEL_SAVE_PATH"

# Run training
echo "Running command: $CMD"
echo "----------------------"
eval $CMD

# Check if training was successful
if [ $? -eq 0 ]; then
  echo ""
  echo "Training completed successfully!"
  echo "Models saved to: $MODEL_SAVE_PATH"
  echo ""
  echo "Next steps:"
  echo "1. Run the evaluation script to assess model performance"
  echo "2. Generate visualizations using notebooks/results_visualization.ipynb"
else
  echo ""
  echo "Error: Training failed. Check the error messages above."
fi