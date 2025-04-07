"""
Model definition and training for NWM runoff correction.
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_lstm_model(input_shape, output_shape=1):
    """
    Create an LSTM-based model for runoff prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    output_shape : int
        Number of output units
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_gru_model(input_shape, output_shape=1):
    """
    Create a GRU-based model for runoff prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    output_shape : int
        Number of output units
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled GRU model
    """
    model = Sequential([
        GRU(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Create a Transformer encoder block.
    
    Parameters:
    -----------
    inputs : tensorflow.Tensor
        Input tensor
    head_size : int
        Size of each attention head
    num_heads : int
        Number of attention heads
    ff_dim : int
        Hidden layer size in feed forward network
    dropout : float
        Dropout rate
        
    Returns:
    --------
    x : tensorflow.Tensor
        Output tensor
    """
    # Multi-head attention layer
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    # Skip connection and layer normalization
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ff_output = Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(inputs.shape[-1]),
        Dropout(dropout)
    ])(x)
    
    # Skip connection and layer normalization
    x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    return x

def create_transformer_model(input_shape, output_shape=1, head_size=256, num_heads=4, ff_dim=512, num_transformer_blocks=4, mlp_units=[128], dropout=0.2):
    """
    Create a Transformer-based model for runoff prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    output_shape : int
        Number of output units
    head_size : int
        Size of each attention head
    num_heads : int
        Number of attention heads
    ff_dim : int
        Hidden layer size in feed forward network
    num_transformer_blocks : int
        Number of transformer blocks
    mlp_units : list
        Units in final MLP layers
    dropout : float
        Dropout rate
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled Transformer model
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Final MLP layers
    for dim in mlp_units:
        x = Dense(dim, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    outputs = Dense(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_hybrid_model(input_shape, output_shape=1):
    """
    Create a hybrid CNN-LSTM model for runoff prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    output_shape : int
        Number of output units
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled hybrid model
    """
    inputs = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    
    # LSTM layers for sequential processing
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Dense layers for output
    x = Dense(16, activation='relu')(x)
    outputs = Dense(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, model_path=None):
    """
    Train the model with early stopping and learning rate reduction.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Model to train
    X_train, y_train : numpy.ndarray
        Training data
    X_val, y_val : numpy.ndarray
        Validation data
    batch_size : int
        Batch size for training
    epochs : int
        Maximum number of epochs
    model_path : str
        Path to save the best model
        
    Returns:
    --------
    history : tensorflow.keras.callbacks.History
        Training history
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    if model_path:
        callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'training_history.png'))
    
    plt.close()

def main(train=False, model_type='lstm'):
    """
    Main function for model creation and training.
    
    Parameters:
    -----------
    train : bool
        Whether to train the model
    model_type : str
        Type of model to create ('lstm', 'gru', 'transformer', 'hybrid')
    """
    # Define paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(root_dir, 'data', 'processed')
    models_dir = os.path.join(root_dir, 'models')
    
    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'nwm_dl_model.keras')
    
    if train:
        # Load processed data
        print("Loading processed data...")
        train_val_df = pd.read_csv(os.path.join(processed_dir, 'train_validation_data.csv'))
        
        # Prepare data for model (simplified for this example)
        # In a real scenario, you would load the actual preprocessed data
        # and transform it appropriately for the model
        
        # For demonstration, create dummy sequential data
        X_train = np.random.random((1000, 24, 10))  # 1000 samples, 24 time steps, 10 features
        y_train = np.random.random((1000, 1))
        X_val = np.random.random((200, 24, 10))
        y_val = np.random.random((200, 1))
        
        # Create model based on specified type
        input_shape = (24, 10)  # sequence_length, num_features
        
        if model_type == 'lstm':
            model = create_lstm_model(input_shape)
            print("Created LSTM model")
        elif model_type == 'gru':
            model = create_gru_model(input_shape)
            print("Created GRU model")
        elif model_type == 'transformer':
            model = create_transformer_model(input_shape)
            print("Created Transformer model")
        elif model_type == 'hybrid':
            model = create_hybrid_model(input_shape)
            print("Created Hybrid CNN-LSTM model")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        print("Training model...")
        history = train_model(model, X_train, y_train, X_val, y_val, model_path=model_path)
        
        # Plot training history
        plot_training_history(history)
        
        print(f"Model trained and saved to {model_path}")
    else:
        print(f"Use --train flag to train the model. Available models: lstm, gru, transformer, hybrid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a deep learning model for NWM runoff correction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'transformer', 'hybrid'], 
                        help='Type of model to create')
    args = parser.parse_args()
    
    main(train=args.train, model_type=args.model)
