import os
import sys
import glob
import numpy as np
import argparse
import tensorflow as tf  # Add tensorflow import
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

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


def main():
    tf.random.set_seed(42)  # Set the seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=os.path.join('data', 'processed'), help='Processed data directory')
    parser.add_argument('--models-dir', default='models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    train_files = glob.glob(os.path.join(args.data_dir, 'train', '*.npz'))
    for tr_path in train_files:
        station = os.path.splitext(os.path.basename(tr_path))[0]
        # load train and val
        X_train, y_train = load_dataset(tr_path)
        # skip stations with no data or malformed shapes
        if X_train.size == 0 or y_train.size == 0 or y_train.ndim != 2 or y_train.shape[1] == 0:
            print(f"Skipping station {station}: insufficient or malformed training data")
            continue
        val_path = os.path.join(args.data_dir, 'val', f'{station}.npz')
        X_val, y_val = load_dataset(val_path)
        # build model
        seq_len = X_train.shape[1]
        model = build_model(window=seq_len - y_train.shape[1], horizon=y_train.shape[1])
        model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse')
        # callbacks
        ckpt_path = os.path.join(args.models_dir, f'{station}.h5')
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3, factor=0.5),
            ModelCheckpoint(ckpt_path, save_best_only=True)
        ]
        # training
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )
        print(f'Trained and saved model for station {station} at {ckpt_path}')


if __name__ == '__main__':
    main()
