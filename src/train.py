"""
Train models for NWM forecast error correction.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.lstm import build_lstm_model
from src.models.transformer import build_transformer_model


def train_model(station_id, data_dir, models_dir, batch_size=32, epochs=100, patience=20):
    """
    Train a model for a specific station.
    
    Args:
        station_id: Station identifier
        data_dir: Directory with processed data
        models_dir: Directory to save trained models
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Patience for early stopping
    
    Returns:
        Training history
    """
    # Load processed data
    data_file = os.path.join(data_dir, 'train', f"{station_id}.npz")
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return None
    
    data = np.load(data_file)
    X, y = data['X'], data['y']
    
    # Split training data into train/validation (80/20)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # Determine model type based on station ID and build model
    if station_id == '21609641':
        print(f"Building LSTM model for station {station_id}")
        model = build_lstm_model(
            input_shape=X_train.shape[1:],
            output_shape=y_train.shape[1],
            lstm_units=64,
            dropout_rate=0.2
        )
    elif station_id == '20380357':
        print(f"Building Transformer model for station {station_id}")
        model = build_transformer_model(
            input_shape=X_train.shape[1:],
            output_shape=y_train.shape[1],
            head_size=256,
            num_heads=4,
            ff_dim=512
        )
    else:
        raise ValueError(f"Unknown station ID: {station_id}")
    
    # Create model checkpoint directory
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{station_id}.h5")
    
    # Define callbacks
    callbacks = [
        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping if validation loss doesn't improve
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(models_dir, f"{station_id}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Station {station_id} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'Station {station_id} - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{station_id}_training_history.png"))
    plt.close()
    
    print(f"Model trained and saved to {model_path}")
    print(f"Training history saved to {history_path}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train model for NWM error correction")
    parser.add_argument("--station", type=str, required=False,
                        help="Station ID to train model for (if not specified, trains for both stations)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with processed data")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience for early stopping")
    args = parser.parse_args()
    
    if args.station:
        # Train model for specified station
        train_model(
            args.station,
            args.data_dir,
            args.models_dir,
            args.batch_size,
            args.epochs,
            args.patience
        )
    else:
        # Train models for both stations
        for station_id in ["21609641", "20380357"]:
            train_model(
                station_id,
                args.data_dir,
                args.models_dir,
                args.batch_size,
                args.epochs,
                args.patience
            )


if __name__ == "__main__":
    main()
