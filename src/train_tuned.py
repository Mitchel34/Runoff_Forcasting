import os
import sys
import glob
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import json

# Add the project root to the Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import build_model


def load_dataset(npz_path):
    data = np.load(npz_path)
    X, y = data['X'], data['y']
    # ensure y is at least 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # reshape for model input
    return X[..., np.newaxis], y


def load_best_hyperparameters(tuning_dir, station):
    """Load best hyperparameters from the tuning results."""
    hp_file = os.path.join(tuning_dir, f"{station}_best_hp.txt")
    if not os.path.exists(hp_file):
        print(f"No hyperparameter file found for station {station} at {hp_file}")
        return None
    
    hyperparams = {}
    with open(hp_file, 'r') as f:
        for line in f:
            if ':' in line:
                param, value = line.strip().split(':', 1)
                param = param.strip()
                value = value.strip()
                # Convert value to appropriate type
                if param in ['lstm_units', 'dense_units']:
                    hyperparams[param] = int(value)
                elif param in ['dropout', 'learning_rate']:
                    hyperparams[param] = float(value)
    
    return hyperparams


def main():
    tf.random.set_seed(42)  # Set the seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=os.path.join('data', 'processed'), help='Processed data directory')
    parser.add_argument('--tuning-dir', default=os.path.join('results', 'tuning'), help='Directory with tuning results')
    parser.add_argument('--models-dir', default='models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--suffix', default='_tuned', help='Suffix to add to tuned model filenames')
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    train_files = glob.glob(os.path.join(args.data_dir, 'train', '*.npz'))
    
    for tr_path in train_files:
        station = os.path.splitext(os.path.basename(tr_path))[0]
        print(f"Processing station {station}...")
        
        # Load best hyperparameters
        hyperparams = load_best_hyperparameters(args.tuning_dir, station)
        if not hyperparams:
            print(f"Skipping station {station}: no hyperparameter tuning results found")
            continue
            
        print(f"Using hyperparameters: {hyperparams}")
        
        # Load train and validation data
        X_train, y_train = load_dataset(tr_path)
        # Skip stations with no data or malformed shapes
        if X_train.size == 0 or y_train.size == 0 or y_train.ndim != 2 or y_train.shape[1] == 0:
            print(f"Skipping station {station}: insufficient or malformed training data")
            continue
            
        val_path = os.path.join(args.data_dir, 'val', f'{station}.npz')
        X_val, y_val = load_dataset(val_path)
        
        # Build model with tuned hyperparameters
        seq_len = X_train.shape[1]
        window = seq_len - y_train.shape[1]
        horizon = y_train.shape[1]
        
        model = build_model(
            window=window,
            horizon=horizon,
            lstm_units=hyperparams.get('lstm_units', 64),  # Default if not found
            dense_units=hyperparams.get('dense_units', 32),
            dropout=hyperparams.get('dropout', 0.2)
        )
        
        # Compile model with tuned learning rate
        model.compile(
            optimizer=Adam(learning_rate=hyperparams.get('learning_rate', 1e-3)),
            loss='mse'
        )
        
        # Define callbacks
        model_filename = f"{station}{args.suffix}.h5"
        ckpt_path = os.path.join(args.models_dir, model_filename)
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3, factor=0.5),
            ModelCheckpoint(ckpt_path, save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f'Trained model with tuned hyperparameters for station {station}')
        print(f'Model saved to {ckpt_path}')
        
        # Save training history
        history_path = os.path.join(args.models_dir, f"{station}{args.suffix}_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        print(f'Training history saved to {history_path}')


if __name__ == '__main__':
    main()