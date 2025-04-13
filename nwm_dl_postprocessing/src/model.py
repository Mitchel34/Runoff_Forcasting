import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, ReLU, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

class Seq2SeqLSTMModel:
    """
    Sequence-to-Sequence LSTM model for NWM runoff forecast error correction.
    Predicts forecast errors for 1-18 hour lead times simultaneously.
    Uses ReLU activation and Adam optimizer as per project requirements.
    """
    
    def __init__(self, 
                 encoder_timesteps=24,
                 encoder_features=3,
                 decoder_timesteps=18,
                 lstm_units=64,
                 dropout_rate=0.2,
                 learning_rate=0.001,
                 num_layers=2):
        """
        Initialize the Seq2Seq model with customizable hyperparameters.
        
        Parameters:
        -----------
        encoder_timesteps : int
            Number of timesteps for encoder input sequence (default: 24 hours)
        encoder_features : int
            Number of features per timestep in encoder input (default: 3 features)
        decoder_timesteps : int
            Number of timesteps for decoder input/output (default: 18 hours for 1-18 lead times)
        lstm_units : int
            Number of LSTM units in each layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        num_layers : int
            Number of LSTM layers in encoder and decoder
        """
        self.encoder_timesteps = encoder_timesteps
        self.encoder_features = encoder_features
        self.decoder_timesteps = decoder_timesteps
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.model = None
        
    def build_model(self):
        """
        Build and compile the Seq2Seq LSTM model using functional API.
        
        Returns:
        --------
        model : tf.keras.Model
            Compiled Seq2Seq LSTM model
        """
        # Encoder
        encoder_inputs = Input(shape=(self.encoder_timesteps, self.encoder_features), name='encoder_inputs')
        
        # Encoder LSTM layers
        x = encoder_inputs
        for i in range(self.num_layers - 1):
            x = LSTM(self.lstm_units, return_sequences=True, name=f'encoder_lstm_{i+1}')(x)
            x = ReLU()(x)
            x = Dropout(self.dropout_rate)(x)
            
        # Final encoder LSTM layer
        encoder_outputs = LSTM(self.lstm_units, name=f'encoder_lstm_{self.num_layers}')(x)
        encoder_outputs = ReLU()(encoder_outputs)
        encoder_outputs = Dropout(self.dropout_rate)(encoder_outputs)
        
        # Decoder
        decoder_inputs = Input(shape=(self.decoder_timesteps,), name='decoder_inputs')
        
        # Repeat encoder output for each decoder timestep
        repeated_encoder = RepeatVector(self.decoder_timesteps)(encoder_outputs)
        
        # Combine repeated encoder output with decoder inputs
        decoder_dense = Dense(self.lstm_units, activation='relu')(decoder_inputs)
        decoder_dense = Dropout(self.dropout_rate)(decoder_dense)
        decoder_dense = RepeatVector(1)(decoder_dense)
        decoder_dense = tf.squeeze(decoder_dense, axis=1)
        
        combined_input = tf.concat([repeated_encoder, 
                                   tf.expand_dims(decoder_dense, axis=1)], 
                                  axis=1)
        
        # Decoder LSTM layers
        x = combined_input
        for i in range(self.num_layers - 1):
            x = LSTM(self.lstm_units, return_sequences=True, name=f'decoder_lstm_{i+1}')(x)
            x = ReLU()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Final decoder LSTM layer
        decoder_outputs = LSTM(self.lstm_units, return_sequences=True, name=f'decoder_lstm_{self.num_layers}')(x)
        decoder_outputs = ReLU()(decoder_outputs)
        decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
        
        # Output layer to predict forecast errors for each lead time
        outputs = TimeDistributed(Dense(1, activation='linear'))(decoder_outputs)
        outputs = tf.reshape(outputs, [-1, self.decoder_timesteps])
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], outputs)
        
        # Compile model with Adam optimizer and MSE loss
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_encoder, X_decoder, y, validation_data=None, 
              batch_size=32, epochs=50, callbacks=None, verbose=1):
        """
        Train the Seq2Seq model.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
        y : numpy.ndarray
            Target forecast error sequences
        validation_data : tuple, optional
            Validation data as (X_encoder_val, X_decoder_val, y_val)
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        callbacks : list, optional
            List of Keras callbacks
        verbose : int
            Verbosity mode (0, 1, or 2)
            
        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            [X_encoder, X_decoder], y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X_encoder, X_decoder):
        """
        Generate predictions with the trained model.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted forecast error sequences
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        return self.model.predict([X_encoder, X_decoder])
    
    def save(self, filepath):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load the model from disk"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model

    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        
        return self.model.summary()