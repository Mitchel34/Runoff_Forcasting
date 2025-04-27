"""
LSTM model architecture for station 21609641.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization


def build_lstm_model(input_shape, output_shape, lstm_units=64, dropout_rate=0.2):
    """
    Build an LSTM model for NWM forecast error correction.
    
    Args:
        input_shape: Shape of input sequences
        output_shape: Shape of output (number of lead times)
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled LSTM model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with return sequences for stacking
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second LSTM layer
    x = LSTM(lstm_units)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(output_shape)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_lstm_model_for_tuning(hp):
    """
    Build a tunable LSTM model with hyperparameters.
    
    Args:
        hp: HyperParameters instance
        
    Returns:
        Compiled LSTM model
    """
    # Define hyperparameter search space
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # Input shape is fixed for the application
    input_shape = (42, 1)  # 24 past observations + 18 NWM forecasts, 1 feature
    output_shape = 18  # 18 lead times
    
    # Build model with hyperparameters
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second LSTM layer
    x = LSTM(lstm_units)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(output_shape)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
