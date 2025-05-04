"""
Transformer model architecture for station 20380357.
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

# Based on the "Attention Is All You Need" paper (Vaswani et al., 2017)

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Creates a single transformer encoder block."""
    # Multi-Head Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x) # Self-attention
    x = Dropout(dropout)(x)
    res = x + inputs # Add & Norm (Residual connection)

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res # Add & Norm (Residual connection)

def build_transformer_model(
    input_shape, # (timesteps, features) e.g., (24, 36)
    head_size,   # Dimensionality of the attention key/query/value
    num_heads,   # Number of attention heads
    ff_dim,      # Hidden layer size in feed forward network
    num_encoder_blocks, # Number of stacked encoder blocks
    output_units=18, # Number of output units (lead times 1-18)
    dropout=0,   # Dropout rate
    mlp_units=[], # Units in the final MLP head
    mlp_dropout=0 # Dropout rate for the MLP head
):
    """Builds the Transformer model architecture."""
    inputs = Input(shape=input_shape)
    x = inputs

    # Create multiple Transformer Encoder blocks
    for _ in range(num_encoder_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)

    # Pooling: Global Average Pooling reduces the sequence dimension
    # Output shape: (batch_size, features)
    x = GlobalAveragePooling1D(data_format="channels_last")(x)

    # Final MLP Head for regression
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(output_units)(x) # Final output layer

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    # Example Usage
    example_input_shape = (24, 36) # (timesteps, num_features)
    model = build_transformer_model(
        input_shape=example_input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_encoder_blocks=4,
        output_units=18,
        dropout=0.1,
        mlp_units=[128],
        mlp_dropout=0.1
    )

    model.summary()

    # Test with dummy data
    dummy_input = tf.random.normal([32, example_input_shape[0], example_input_shape[1]])
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}") # Should be (batch_size, output_units) -> (32, 18)