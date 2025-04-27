"""
Hyperparameter tuning script for LSTM and Transformer models using Keras Tuner.

Uses Hyperband or BayesianOptimization to find optimal hyperparameters.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split # For creating validation set

# Import model building functions and data loading
from models.lstm import build_lstm_model
from models.transformer import build_transformer_model
from train import load_data # Reuse data loading function from train.py

# Define paths
PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')
TUNER_LOG_DIR = os.path.join('..', 'tuner_logs') # Directory to store tuning results

class LSTMHyperModel(kt.HyperModel):
    """Keras Tuner HyperModel for LSTM."""
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):
        # Define hyperparameters to tune
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
        # num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3, step=1) # TODO: Implement multi-layer LSTM if needed
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Build the model (currently assumes single layer)
        # If implementing multi-layer, modify build_lstm_model or build here
        model = build_lstm_model(
            input_shape=self.input_shape,
            lstm_units=lstm_units,
            output_units=self.output_units
            # Add dropout to build_lstm_model if tuning dropout_rate
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

class TransformerHyperModel(kt.HyperModel):
    """Keras Tuner HyperModel for Transformer."""
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):
        # Define hyperparameters to tune
        num_encoder_blocks = hp.Int('num_encoder_blocks', min_value=2, max_value=6, step=1)
        num_heads = hp.Choice('num_heads', values=[2, 4, 8])
        ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        # mlp_units_choice = hp.Choice('mlp_units', values=['64', '128', '128,64']) # Example for tuning MLP head
        # mlp_units = [int(u) for u in mlp_units_choice.split(',')] if mlp_units_choice else []

        # Build the model
        model = build_transformer_model(
            input_shape=self.input_shape,
            head_size=256, # Keep fixed or tune: hp.Int('head_size', ...)
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_encoder_blocks=num_encoder_blocks,
            output_units=self.output_units,
            dropout=dropout_rate,
            mlp_units=[128], # Keep fixed or tune using hp, e.g., mlp_units variable above
            mlp_dropout=dropout_rate # Reuse dropout or define separate hp.Float
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

def run_tuning(station_id, model_type, max_trials, epochs_per_trial, batch_size, tuner_type='hyperband'):
    """Runs the hyperparameter tuning process."""
    print(f"\n--- Starting Hyperparameter Tuning for Station {station_id} ({model_type.upper()}) --- ")
    print(f"Tuner: {tuner_type}, Max Trials: {max_trials}, Epochs per Trial: {epochs_per_trial}, Batch Size: {batch_size}")

    # 1. Load Data
    X, y = load_data(station_id, 'train')
    input_shape = (X.shape[1], X.shape[2])
    output_units = y.shape[1]

    # 2. Create Validation Split (Temporal Order Matters!)
    # Split data without shuffling to maintain time series order
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Training data shape: X-{X_train.shape}, y-{y_train.shape}")
    print(f"Validation data shape: X-{X_val.shape}, y-{y_val.shape}")

    # 3. Instantiate HyperModel
    if model_type.lower() == 'lstm':
        hypermodel = LSTMHyperModel(input_shape, output_units)
    elif model_type.lower() == 'transformer':
        hypermodel = TransformerHyperModel(input_shape, output_units)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 4. Instantiate Tuner
    tuner_directory = os.path.join(TUNER_LOG_DIR, f"{station_id}_{model_type.lower()}_{tuner_type}")
    project_name = f"runoff_tuning_{station_id}_{model_type.lower()}"

    if tuner_type.lower() == 'hyperband':
        tuner = kt.Hyperband(
            hypermodel,
            objective='val_loss',
            max_epochs=epochs_per_trial, # Max epochs for the best models
            factor=3, # Reduction factor for rounds
            directory=tuner_directory,
            project_name=project_name
        )
    elif tuner_type.lower() == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            directory=tuner_directory,
            project_name=project_name
        )
    else:
        raise ValueError(f"Unsupported tuner type: {tuner_type}")

    # 5. Define Callbacks for Search
    # Early stopping within each trial
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # 6. Run Search
    print(f"\nStarting tuner search... Logs will be saved in: {tuner_directory}/{project_name}")
    tuner.search(
        X_train, y_train,
        epochs=epochs_per_trial, # For BayesianOptimization, this is epochs per trial
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1 # Set to 2 for more detailed trial logs, 0 for silent
    )

    # 7. Display Results
    print("\n--- Tuning Complete --- ")
    tuner.results_summary()

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters Found:")
    for param, value in best_hps.values.items():
        print(f"  {param}: {value}")

    # Optional: Build the best model and train it on full data (or do this in train.py)
    # best_model = tuner.hypermodel.build(best_hps)
    # history = best_model.fit(X, y, epochs=..., batch_size=..., validation_split=0.2)
    # best_model.save(...) 

    print(f"\nTo retrain the best model, use the hyperparameters above with train.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for runoff error correction models.")
    parser.add_argument("--station_id", type=str, required=True, help="USGS Station ID")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'transformer'], help="Model type to tune")
    parser.add_argument("--tuner", type=str, default='hyperband', choices=['hyperband', 'bayesian'], help="Keras Tuner algorithm")
    parser.add_argument("--max_trials", type=int, default=20, help="Maximum number of hyperparameter trials (used by BayesianOptimization)")
    parser.add_argument("--epochs_per_trial", type=int, default=50, help="Maximum epochs to train each model configuration (used by Hyperband max_epochs and Bayesian epochs)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training during tuning")

    args = parser.parse_args()

    run_tuning(
        station_id=args.station_id,
        model_type=args.model_type,
        max_trials=args.max_trials,
        epochs_per_trial=args.epochs_per_trial,
        batch_size=args.batch_size,
        tuner_type=args.tuner
    )

    print("\nTuning script finished.")
