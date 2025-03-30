import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories if they don't exist
os.makedirs('../results/models', exist_ok=True)

def load_processed_data():
    """
    Load preprocessed data from CSV files
    """
    print("Loading preprocessed data...")
    
    try:
        train_val_data = pd.read_csv('../data/processed/train_val_data.csv')
        test_data = pd.read_csv('../data/processed/test_data.csv')
        
        # Convert date strings back to datetime objects
        train_val_data['date'] = pd.to_datetime(train_val_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])
        
        # Load the scaler
        with open('../data/processed/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return train_val_data, test_data, scaler
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run preprocess.py first to generate preprocessed data files")
        return None, None, None

def create_sequences(data, seq_length, lead_times):
    """
    Create sequences for LSTM model using sliding window approach
    """
    print(f"Creating sequences with length {seq_length}...")
    
    X, y = [], []
    max_lead = max(lead_times)
    
    features = ['runoff_nwm', 'runoff_usgs']
    if 'precipitation' in data.columns:
        features.append('precipitation')
    
    # Calculate residuals (difference between NWM forecasts and USGS observations)
    data['residual'] = data['runoff_usgs'] - data['runoff_nwm']
    
    # Get data arrays
    feature_data = data[features].values
    residual_data = data['residual'].values
    
    for i in range(len(data) - seq_length - max_lead):
        # Input features: sequence of past observations
        X.append(feature_data[i:i+seq_length])
        
        # Target: residuals for each lead time
        target = []
        for lead in lead_times:
            target.append(residual_data[i+seq_length+lead-1])
        y.append(target)
    
    return np.array(X), np.array(y)

def build_model(seq_length, num_features, num_outputs):
    """
    Build LSTM model architecture
    """
    print("Building model...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, num_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_outputs)  # Output residuals for lead times
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the LSTM model
    """
    print("Training model...")
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath='../results/models/best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def save_model(model):
    """
    Save the trained model
    """
    print("Saving model...")
    model.save('../results/models/final_model.h5')

def main():
    # Load processed data
    train_val_data, test_data, scaler = load_processed_data()
    if train_val_data is None:
        return
    
    # Define parameters
    seq_length = 24
    lead_times = range(1, 19)  # 1-18 hours
    
    # Create sequences
    X_train, y_train = create_sequences(train_val_data, seq_length, lead_times)
    
    # Get number of features
    num_features = X_train.shape[2]
    num_outputs = y_train.shape[1]
    
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # Build model
    model = build_model(seq_length, num_features, num_outputs)
    
    # Train model
    model, history = train_model(model, X_train, y_train)
    
    # Save model
    save_model(model)
    
    print("Model training completed successfully.")

if __name__ == "__main__":
    main()
