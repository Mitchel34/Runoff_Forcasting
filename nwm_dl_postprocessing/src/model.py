import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import numpy as np

class Seq2SeqLSTMModel:
    """
    Sequence-to-Sequence LSTM model for NWM forecast error correction.
    
    The model consists of:
    1. Encoder: LSTM layers to encode historical errors, forecasts, and observations
    2. Decoder: LSTM layers to decode future forecast errors
    3. Output: Dense layer with linear activation to predict error corrections
    """
    
    def __init__(self, encoder_timesteps, encoder_features, decoder_timesteps,
                lstm_units=64, dropout_rate=0.2, learning_rate=0.001, num_layers=2):
        """
        Initialize the Seq2Seq LSTM model.
        
        Parameters:
        -----------
        encoder_timesteps : int
            Number of timesteps in encoder input sequence (past observations)
        encoder_features : int
            Number of features per timestep in encoder input
        decoder_timesteps : int
            Number of timesteps in decoder output (future predictions)
        lstm_units : int, optional
            Number of units in LSTM layers
        dropout_rate : float, optional
            Dropout rate for regularization
        learning_rate : float, optional
            Learning rate for Adam optimizer
        num_layers : int, optional
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
        Build the Seq2Seq LSTM model architecture.
        
        Returns:
        --------
        model : tensorflow.keras.models.Model
            Compiled Seq2Seq LSTM model
        """
        # Define encoder input
        encoder_inputs = Input(shape=(self.encoder_timesteps, self.encoder_features), name='encoder_input')
        
        # Define decoder input (NWM forecasts for future timesteps)
        decoder_inputs = Input(shape=(self.decoder_timesteps,), name='decoder_input')
        
        # Encoder
        # Multiple stacked LSTM layers with dropout
        x = encoder_inputs
        for i in range(self.num_layers - 1):
            x = LSTM(self.lstm_units, return_sequences=True, activation='relu', name=f'encoder_lstm_{i+1}')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Final encoder layer
        encoder_outputs = LSTM(self.lstm_units, return_state=True, activation='relu', name='encoder_lstm_final')
        _, state_h, state_c = encoder_outputs(x)
        encoder_states = [state_h, state_c]
        
        # Decoder
        # Reshape decoder inputs to match decoder expected format
        decoder_dense = Dense(self.lstm_units, activation='relu', name='decoder_dense')
        decoder_reshaped = decoder_dense(RepeatVector(self.decoder_timesteps)(decoder_inputs))
        
        # Multiple stacked LSTM layers for decoder with encoder states as initial states
        decoder_lstm = LSTM(self.lstm_units, return_sequences=True, activation='relu', name='decoder_lstm')
        decoder_outputs = decoder_lstm(decoder_reshaped, initial_state=encoder_states)
        
        # Add dropout
        decoder_outputs = Dropout(self.dropout_rate)(decoder_outputs)
        
        # Final dense layer to generate predictions
        outputs = TimeDistributed(Dense(1, activation='linear'), name='output_dense')(decoder_outputs)
        
        # Reshape to match target shape [batch_size, decoder_timesteps]
        outputs = tf.reshape(outputs, [-1, self.decoder_timesteps])
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_encoder, X_decoder, y, validation_data=None, 
             batch_size=32, epochs=100, callbacks=None, verbose=1):
        """
        Train the Seq2Seq LSTM model.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
        y : numpy.ndarray
            Target output sequences
        validation_data : tuple, optional
            Tuple of (X_encoder_val, X_decoder_val), y_val for validation
        batch_size : int, optional
            Batch size for training
        epochs : int, optional
            Number of epochs for training
        callbacks : list, optional
            List of Keras callbacks for training
        verbose : int, optional
            Verbosity mode
            
        Returns:
        --------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        
        # Train model
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
        Generate predictions using the trained model.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted forecast errors
        """
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train() first.")
        
        return self.model.predict([X_encoder, X_decoder])
    
    def evaluate(self, X_encoder, X_decoder, y):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
        y : numpy.ndarray
            Target output sequences
            
        Returns:
        --------
        evaluation : list
            List containing loss and metric values
        """
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train() first.")
        
        return self.model.evaluate([X_encoder, X_decoder], y, verbose=1)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from preprocess import DataPreprocessor
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed",
        sequence_length=24
    )
    
    # Process data for a single stream (for testing)
    data = preprocessor.process_data(stream_ids=["20380357"])
    
    # Extract training data
    stream_id = "20380357"
    X_encoder_train = data[stream_id]['train_val']['X_encoder']
    X_decoder_train = data[stream_id]['train_val']['X_decoder']
    y_train = data[stream_id]['train_val']['y']
    
    # Create and build model
    model = Seq2SeqLSTMModel(
        encoder_timesteps=X_encoder_train.shape[1],
        encoder_features=X_encoder_train.shape[2],
        decoder_timesteps=18
    )
    model.build_model()
    
    # Print model summary
    model.model.summary()
    
    # Train model (small number of epochs for testing)
    history = model.train(
        X_encoder_train, X_decoder_train, y_train,
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    model.save("../models/test_model.keras")