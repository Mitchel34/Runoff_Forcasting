"""
LSTM model architecture for station 21609641.
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

def build_lstm_model(input_shape, lstm_units=64, output_units=18):
    """
    Builds the LSTM model architecture.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
                             Example: (24, 36) for 24 timesteps and 36 features.
        lstm_units (int): Number of units in the LSTM layer.
        output_units (int): Number of output units (corresponding to lead times).
                            Default is 18 for lead times 1-18.

    Returns:
        tf.keras.Model: The compiled LSTM model.
    """
    inputs = Input(shape=input_shape)

    # LSTM layer
    # return_sequences=False because we only need the output of the last timestep
    # to predict the next step's errors across all lead times.
    lstm_out = LSTM(lstm_units, return_sequences=False)(inputs)

    # Dense layer to produce the final output for each lead time
    outputs = Dense(output_units)(lstm_out)

    model = Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == '__main__':
    # Example usage: Define input shape based on preprocessing
    # (timesteps, num_features) -> (24, 36)
    example_input_shape = (24, 36)
    model = build_lstm_model(example_input_shape)
    model.summary() # Print model summary

    # You can add more test code here, e.g., creating dummy data
    # and checking the output shape.
    # Example dummy input: (batch_size, timesteps, features)
    dummy_input = tf.random.normal([32, example_input_shape[0], example_input_shape[1]])
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}") # Should be (batch_size, output_units) -> (32, 18)