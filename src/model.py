from tensorflow.keras import layers, Model, Input


def build_model(window=24, horizon=18, features=1, lstm_units=64, dense_units=32, dropout=0.2):
    """
    Build an LSTM model that takes a sequence of length window+horizon and predicts error vector of length horizon.
    """
    seq_len = window + horizon
    # Input shape: (seq_len, features)
    inp = Input(shape=(seq_len, features), name='input_sequence')
    x = layers.LSTM(lstm_units, return_sequences=False, name='lstm_layer')(inp)
    x = layers.Dropout(dropout, name='dropout_layer')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense_layer')(x)
    out = layers.Dense(horizon, activation='linear', name='error_output')(x)
    model = Model(inputs=inp, outputs=out, name='ErrorCorrector')
    return model
