import os
import sys
import glob
import numpy as np
import argparse
import tensorflow as tf  # Add tensorflow import
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Add the project root to the Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import build_model


def load_dataset(npz_path):
    data = np.load(npz_path)
    X, y = data['X'], data['y']
    return X[..., np.newaxis], y


def model_builder(hp, window, horizon, features=1):
    # Hyperparameters to tune
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    # Build model
    model = build_model(window=window, horizon=horizon, features=features,
                        lstm_units=lstm_units, dense_units=dense_units, dropout=dropout)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def main():
    tf.random.set_seed(42)  # Set the seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', required=True, help='Station ID for tuning')
    parser.add_argument('--data-dir', default=os.path.join('data', 'processed'), help='Processed data directory')
    parser.add_argument('--max-trials', type=int, default=10, help='Number of hyperparameter trials')
    parser.add_argument('--executions', type=int, default=1, help='Executions per trial')
    parser.add_argument('--results-dir', default=os.path.join('results', 'tuning'), help='Directory to save tuning results')
    args = parser.parse_args()

    # Prepare data
    train_path = os.path.join(args.data_dir, 'train', f'{args.station}.npz')
    val_path = os.path.join(args.data_dir, 'val', f'{args.station}.npz')
    X_train, y_train = load_dataset(train_path)
    X_val, y_val = load_dataset(val_path)
    # skip if dataset dims are incorrect (empty or malformed)
    if X_train.ndim != 3 or y_train.ndim != 2 or X_train.shape[1] <= y_train.shape[1]:
        print(f"Skipping station {args.station}: insufficient or malformed data for tuning")
        return
    # skip stations with no data or malformed shapes
    if X_train.size == 0 or y_train.size == 0 or y_train.ndim != 2 or y_train.shape[1] == 0:
        print(f"Skipping station {args.station}: insufficient or malformed training data for tuning")
        return
    # derive window and horizon
    window = X_train.shape[1] - y_train.shape[1]
    horizon = y_train.shape[1]
    features = 1

    # Initialize tuner
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: model_builder(hp, window, horizon, features),
        objective='val_loss',
        max_trials=args.max_trials,
        executions_per_trial=args.executions,
        directory=args.results_dir,
        project_name=args.station,
        seed=42  # Add seed to Keras Tuner for reproducible search space sampling
    )
    # Search
    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=20,
                 batch_size=32,
                 callbacks=[EarlyStopping(patience=5)])
    # Summary and best hyperparameters
    tuner.results_summary()
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f'{args.station}_best_hp.txt'), 'w') as f:
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")
    print(f"Best hyperparameters for station {args.station} saved to {args.results_dir}")


if __name__ == '__main__':
    main()
