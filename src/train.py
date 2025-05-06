"""
Train the LSTM or Transformer model for NWM error correction.

Handles loading data, building the model, training, and saving.
Uses command-line arguments to specify station, model type, and hyperparameters.
"""
import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from joblib import load as joblib_load

# Import model building functions
from models.lstm import build_lstm_model
from models.transformer import build_transformer_model

# Define paths using absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')
SCALERS_DIR = os.path.join(PROCESSED_DATA_DIR, 'scalers')
TUNER_LOG_DIR = os.path.join(PROJECT_ROOT, 'tuner_logs') # For potentially loading HPs

def load_data(station_id, data_type='train'):
    """Loads preprocessed data for a given station and type (train/test)."""
    file_path = os.path.join(PROCESSED_DATA_DIR, data_type, f"{station_id}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    data = np.load(file_path)
    
    # Check keys in the data file
    available_keys = list(data.keys())
    print(f"Available keys in the data file: {available_keys}")
    
    # Handle different key naming conventions
    if 'X_train' in data and 'y_train_scaled' in data:
        X = data['X_train']
        y = data['y_train_scaled']
    elif 'X_test' in data and 'y_test_scaled' in data:
        X = data['X_test']
        y = data['y_test_scaled']
    elif 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        raise KeyError(f"Expected data keys not found in {file_path}. Available keys: {available_keys}")
    
    print(f"Loaded {data_type} data for station {station_id} from {file_path}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def load_scaler(station_id, scaler_type='X'):
    """Loads the scaler object for a given station and type (X/y)."""
    scaler_filename = f"{station_id}_{scaler_type.lower()}_scaler.joblib"
    scaler_path = os.path.join(SCALERS_DIR, scaler_filename)
    if not os.path.exists(scaler_path):
        # Allow training without scaler for now, but evaluation will need it
        print(f"Warning: Scaler file not found: {scaler_path}. Proceeding without it.")
        return None
    scaler = joblib_load(scaler_path)
    print(f"Loaded {scaler_type} scaler for station {station_id} from {scaler_path}")
    return scaler

def train_model(station_id, model_type, epochs, batch_size, hyperparameters):
    """Trains the specified model for the given station using provided hyperparameters."""
    print(f"Starting training for Station {station_id} using {model_type.upper()} model.")
    print("Using hyperparameters:", hyperparameters)

    # Load training data
    X_train, y_train = load_data(station_id, 'train')

    # Determine input shape: (timesteps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1] # Should be 18 (for 18 lead times)
    print(f"Input shape: {input_shape}, Output units: {output_units}")

    # Build the model using hyperparameters from the dictionary
    if model_type.lower() == 'lstm':
        # Extract LSTM specific HPs, provide defaults if not found
        lstm_units = hyperparameters.get('lstm_units', 64)
        model = build_lstm_model(
            input_shape,
            lstm_units=lstm_units,
            output_units=output_units
        )
    elif model_type.lower() == 'transformer':
        # Extract Transformer specific HPs, provide defaults if not found
        transformer_params = {
            'head_size': hyperparameters.get('head_size', 256),
            'num_heads': hyperparameters.get('num_heads', 4),
            'ff_dim': hyperparameters.get('ff_dim', 128),
            'num_encoder_blocks': hyperparameters.get('num_encoder_blocks', 4),
            'dropout': hyperparameters.get('dropout_rate', 0.1),
            'mlp_units': hyperparameters.get('mlp_units', [128]),
            'mlp_dropout': hyperparameters.get('mlp_dropout', hyperparameters.get('dropout_rate', 0.1))
        }
        model = build_transformer_model(
            input_shape=input_shape,
            output_units=output_units,
            **transformer_params
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.summary()

    # Compile the model - Use learning rate from HPs
    learning_rate = hyperparameters.get('learning_rate', 1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error for regression

    # Callbacks
    model_save_path = os.path.join(MODELS_SAVE_DIR, f"{station_id}_{model_type.lower()}_best.keras")
    os.makedirs(MODELS_SAVE_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    print(f"\nTraining complete. Best model saved to {model_save_path}")

    return history, model_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM or Transformer model for runoff error correction.")
    parser.add_argument("--station_id", type=str, required=True, help="USGS Station ID (e.g., 21609641)")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'transformer'], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--hp_json_path", type=str, default=None, help="Optional path to a JSON file containing hyperparameters from tuning.")

    # Add arguments for hyperparameters (used if JSON is not provided or as overrides/defaults)
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (used if not in JSON)")
    parser.add_argument("--lstm_units", type=int, default=64, help="LSTM units (used if not in JSON)")
    parser.add_argument("--num_heads", type=int, default=4, help="Transformer heads (used if not in JSON)")
    parser.add_argument("--ff_dim", type=int, default=128, help="Transformer feed-forward dim (used if not in JSON)")
    parser.add_argument("--num_encoder_blocks", type=int, default=4, help="Transformer encoder blocks (used if not in JSON)")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Transformer dropout rate (used if not in JSON)")

    args = parser.parse_args()

    # Load hyperparameters
    hyperparameters = {}
    if args.hp_json_path:
        try:
            with open(args.hp_json_path, 'r') as f:
                hyperparameters = json.load(f)
            print(f"Loaded hyperparameters from {args.hp_json_path}")
        except FileNotFoundError:
            print(f"Warning: Hyperparameter JSON file not found at {args.hp_json_path}. Using command-line defaults/args.")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {args.hp_json_path}. Using command-line defaults/args.")

    # Populate hyperparameters dictionary, prioritizing JSON values, then command-line args
    hyperparameters.setdefault('learning_rate', args.learning_rate)

    # LSTM specific
    if args.model_type.lower() == 'lstm':
        hyperparameters.setdefault('lstm_units', args.lstm_units)

    # Transformer specific
    elif args.model_type.lower() == 'transformer':
        hyperparameters.setdefault('num_heads', args.num_heads)
        hyperparameters.setdefault('ff_dim', args.ff_dim)
        hyperparameters.setdefault('num_encoder_blocks', args.num_encoder_blocks)
        hyperparameters.setdefault('dropout_rate', args.dropout_rate)

    # Pass the consolidated hyperparameters dictionary
    train_model(
        station_id=args.station_id,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hyperparameters=hyperparameters
    )

    print("\nScript finished.")
