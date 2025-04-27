"""
Train the LSTM or Transformer model for NWM error correction.

Handles loading data, building the model, training, and saving.
Uses command-line arguments to specify station, model type, and hyperparameters.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from joblib import load as joblib_load

# Import model building functions
from models.lstm import build_lstm_model
from models.transformer import build_transformer_model

# Define paths (adjust if your structure differs)
PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')
MODELS_SAVE_DIR = os.path.join('..', 'models')
SCALERS_DIR = os.path.join(PROCESSED_DATA_DIR, 'scalers')

def load_data(station_id, data_type='train'):
    """Loads preprocessed data for a given station and type (train/test)."""
    file_path = os.path.join(PROCESSED_DATA_DIR, data_type, f"{station_id}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    data = np.load(file_path)
    print(f"Loaded {data_type} data for station {station_id} from {file_path}")
    print(f"  X shape: {data['X'].shape}, y shape: {data['y'].shape}")
    return data['X'], data['y']

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

def train_model(station_id, model_type, epochs, batch_size, learning_rate, lstm_units=64,
                transformer_params=None):
    """Trains the specified model for the given station."""
    print(f"Starting training for Station {station_id} using {model_type.upper()} model.")

    # Load training data
    X_train, y_train = load_data(station_id, 'train')

    # Determine input shape: (timesteps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1] # Should be 18 (for 18 lead times)
    print(f"Input shape: {input_shape}, Output units: {output_units}")

    # Build the model
    if model_type.lower() == 'lstm':
        model = build_lstm_model(input_shape, lstm_units=lstm_units, output_units=output_units)
    elif model_type.lower() == 'transformer':
        if transformer_params is None:
            # Default parameters if none provided
            transformer_params = {
                'head_size': 256, 'num_heads': 4, 'ff_dim': 128,
                'num_encoder_blocks': 4, 'dropout': 0.1, 'mlp_units': [128], 'mlp_dropout': 0.1
            }
            print("Using default Transformer parameters.")
        model = build_transformer_model(
            input_shape=input_shape,
            output_units=output_units,
            **transformer_params
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.summary()

    # Compile the model
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

    # Train the model (using a portion of training data as validation)
    # Note: A separate validation set is ideal, but for simplicity, we use validation_split.
    # Ensure shuffle=True for representative validation split.
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2, # Use 20% of training data for validation
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    print(f"\nTraining complete. Best model saved to {model_save_path}")

    # Optional: Load test data to evaluate final model on it (or do this in evaluate.py)
    # X_test, y_test = load_data(station_id, 'test')
    # test_loss = model.evaluate(X_test, y_test, verbose=0)
    # print(f"Final model evaluation on test set - Loss: {test_loss:.4f}")

    return history, model_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM or Transformer model for runoff error correction.")
    parser.add_argument("--station_id", type=str, required=True, help="USGS Station ID (e.g., 21609641)")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'transformer'], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    # Add model-specific args if needed, e.g., --lstm_units
    parser.add_argument("--lstm_units", type=int, default=64, help="Number of units in LSTM layer (if model_type is lstm)")
    # For Transformer, consider passing params as a JSON string or individual args
    # Example (simplified):
    parser.add_argument("--tf_heads", type=int, default=4, help="Transformer heads")
    parser.add_argument("--tf_ff_dim", type=int, default=128, help="Transformer feed-forward dim")
    parser.add_argument("--tf_blocks", type=int, default=4, help="Transformer encoder blocks")
    parser.add_argument("--tf_dropout", type=float, default=0.1, help="Transformer dropout")


    args = parser.parse_args()

    # Prepare transformer params if needed
    transformer_params = None
    if args.model_type.lower() == 'transformer':
         transformer_params = {
            'head_size': 256, # Keep head_size somewhat fixed or add arg
            'num_heads': args.tf_heads,
            'ff_dim': args.tf_ff_dim,
            'num_encoder_blocks': args.tf_blocks,
            'dropout': args.tf_dropout,
            'mlp_units': [128], # Keep fixed or add arg
            'mlp_dropout': args.tf_dropout # Reuse dropout or add specific arg
        }

    train_model(
        station_id=args.station_id,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lstm_units=args.lstm_units,
        transformer_params=transformer_params
    )

    print("\nScript finished.")
