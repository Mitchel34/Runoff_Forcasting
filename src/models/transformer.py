"""
Transformer model architecture for station 20380357.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    GlobalAveragePooling1D, MultiHeadAttention, Conv1D
)

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import positional_encoding

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """
    Transformer encoder block with multi-head attention and feed-forward network.
    
    Args:
        inputs: Input tensor
        head_size: Size of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden dimension of feed-forward network
        dropout: Dropout rate
        
    Returns:
        Output tensor
    """
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    return LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)


def build_transformer_model(
    input_shape, 
    output_shape, 
    head_size=256, 
    num_heads=4,
    ff_dim=512, 
    num_transformer_blocks=4, 
    mlp_units=[128], 
    dropout=0.2, 
    mlp_dropout=0.4
):
    """
    Build a Transformer model for NWM forecast error correction.
    
    Args:
        input_shape: Shape of input sequences
        output_shape: Shape of output (number of lead times)
        head_size: Size of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden dimension of feed-forward network
        num_transformer_blocks: Number of transformer blocks
        mlp_units: Units in the MLP layers after transformer
        dropout: Dropout rate in transformer
        mlp_dropout: Dropout rate in final MLP
        
    Returns:
        Compiled Transformer model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Extract features with 1D convolution
    x = Conv1D(filters=head_size, kernel_size=3, padding="same")(inputs)
    
    # Add positional encoding - implemented as a non-trainable addition
    pos_encoding = tf.convert_to_tensor(positional_encoding(input_shape[0], head_size))
    pos_encoding = tf.reshape(pos_encoding, (1, input_shape[0], head_size))
    x = x + pos_encoding
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x, head_size, num_heads, ff_dim, dropout
        )
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # MLP layers
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    
    # Output layer
    outputs = Dense(output_shape)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"]
    )
    
    return model


def build_transformer_model_for_tuning(hp):
    """
    Build a tunable Transformer model with hyperparameters.
    
    Args:
        hp: HyperParameters instance
        
    Returns:
        Compiled Transformer model
    """
    # Define hyperparameter search space
    head_size = hp.Int('head_size', min_value=128, max_value=512, step=128)
    num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    ff_dim = hp.Int('ff_dim', min_value=256, max_value=1024, step=256)
    num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=2, max_value=6, step=1)
    mlp_units = hp.Int('mlp_units', min_value=64, max_value=256, step=64)
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.1)
    mlp_dropout = hp.Float('mlp_dropout', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    
    # Input shape is fixed for the application
    input_shape = (42, 1)  # 24 past observations + 18 NWM forecasts, 1 feature
    output_shape = 18  # 18 lead times
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Extract features with 1D convolution
    x = Conv1D(filters=head_size, kernel_size=3, padding="same")(inputs)
    
    # Add positional encoding
    pos_encoding = tf.convert_to_tensor(positional_encoding(input_shape[0], head_size))
    pos_encoding = tf.reshape(pos_encoding, (1, input_shape[0], head_size))
    x = x + pos_encoding
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x, head_size, num_heads, ff_dim, dropout
        )
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # MLP layers
    x = Dense(mlp_units, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)
    
    # Output layer
    outputs = Dense(output_shape)(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    
    return model
