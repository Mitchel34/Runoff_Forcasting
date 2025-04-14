import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
import kerastuner as kt
from kerastuner.tuners import Hyperband, RandomSearch, BayesianOptimization
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Input
from tensorflow.keras.models import Model

class Seq2SeqTuner:
    """
    A class for hyperparameter tuning of the Seq2Seq LSTM model using KerasTuner.
    Supports Hyperband, RandomSearch, and BayesianOptimization tuning algorithms.
    """
    
    def __init__(self, encoder_timesteps, encoder_features, decoder_timesteps,
                 project_name='nwm_seq2seq_tuning', directory='../models'):
        """
        Initialize the Seq2SeqTuner.
        
        Parameters:
        -----------
        encoder_timesteps : int
            Number of timesteps in encoder input sequence
        encoder_features : int
            Number of features per timestep in encoder input
        decoder_timesteps : int
            Number of timesteps in decoder output
        project_name : str, optional
            Name of the tuning project
        directory : str, optional
            Directory to store tuning results
        """
        self.encoder_timesteps = encoder_timesteps
        self.encoder_features = encoder_features
        self.decoder_timesteps = decoder_timesteps
        self.project_name = project_name
        self.directory = directory
        self.tuner = None
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
    
    def build_model(self, hp):
        """
        Build a tunable Seq2Seq LSTM model with hyperparameters.
        
        Parameters:
        -----------
        hp : kerastuner.HyperParameters
            Hyperparameters to tune
            
        Returns:
        --------
        model : tensorflow.keras.models.Model
            Compiled Seq2Seq LSTM model with tunable hyperparameters
        """
        # Define hyperparameters to tune
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3, 3e-3])
        num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        
        # Define model inputs
        encoder_inputs = Input(shape=(self.encoder_timesteps, self.encoder_features), name='encoder_input')
        decoder_inputs = Input(shape=(self.decoder_timesteps,), name='decoder_input')
        
        # Encoder
        x = encoder_inputs
        for i in range(num_layers - 1):
            x = LSTM(lstm_units, return_sequences=True, activation='relu', name=f'encoder_lstm_{i+1}')(x)
            x = Dropout(dropout_rate)(x)
        
        # Final encoder layer
        encoder_outputs = LSTM(lstm_units, return_state=True, activation='relu', name='encoder_lstm_final')
        _, state_h, state_c = encoder_outputs(x)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_dense = Dense(lstm_units, activation='relu', name='decoder_dense')
        decoder_reshaped = decoder_dense(RepeatVector(self.decoder_timesteps)(decoder_inputs))
        
        decoder_lstm = LSTM(lstm_units, return_sequences=True, activation='relu', name='decoder_lstm')
        decoder_outputs = decoder_lstm(decoder_reshaped, initial_state=encoder_states)
        
        decoder_outputs = Dropout(dropout_rate)(decoder_outputs)
        outputs = TimeDistributed(Dense(1, activation='linear'), name='output_dense')(decoder_outputs)
        outputs = tf.reshape(outputs, [-1, self.decoder_timesteps])
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def setup_tuner(self, tuner_type='hyperband', max_trials=50, executions_per_trial=1):
        """
        Set up the hyperparameter tuner.
        
        Parameters:
        -----------
        tuner_type : str, optional
            Type of tuner to use ('hyperband', 'random', or 'bayesian')
        max_trials : int, optional
            Maximum number of trials to run
        executions_per_trial : int, optional
            Number of times to train each trial
        
        Returns:
        --------
        tuner : kerastuner.Tuner
            Configured hyperparameter tuner
        """
        if tuner_type == 'hyperband':
            self.tuner = Hyperband(
                self.build_model,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                directory=self.directory,
                project_name=self.project_name,
                max_trials=max_trials,
                executions_per_trial=executions_per_trial
            )
        elif tuner_type == 'random':
            self.tuner = RandomSearch(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=self.directory,
                project_name=self.project_name
            )
        elif tuner_type == 'bayesian':
            self.tuner = BayesianOptimization(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=self.directory,
                project_name=self.project_name
            )
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")
        
        return self.tuner
    
    def search_with_time_series_cv(self, X_encoder, X_decoder, y, n_splits=3, 
                                  batch_size=32, epochs=30, verbose=1):
        """
        Perform hyperparameter search with TimeSeriesSplit cross-validation.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
        y : numpy.ndarray
            Target output sequences
        n_splits : int, optional
            Number of splits for TimeSeriesSplit
        batch_size : int, optional
            Batch size for training
        epochs : int, optional
            Maximum number of epochs per trial
        verbose : int, optional
            Verbosity mode
        
        Returns:
        --------
        tuner : kerastuner.Tuner
            Trained hyperparameter tuner
        best_hps : kerastuner.HyperParameters
            Best hyperparameters
        """
        if self.tuner is None:
            raise ValueError("Tuner not set up. Call setup_tuner() first.")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Create a custom train/validation split for each fold
        for train_index, val_index in tscv.split(X_encoder):
            X_enc_train, X_enc_val = X_encoder[train_index], X_encoder[val_index]
            X_dec_train, X_dec_val = X_decoder[train_index], X_decoder[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Search hyperparameters using this fold
            self.tuner.search(
                [X_enc_train, X_dec_train],
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_enc_val, X_dec_val], y_val),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                ],
                verbose=verbose
            )
        
        # Get best hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(1)[0]
        
        print("\nBest hyperparameters found:")
        print(f"LSTM units: {best_hps.get('lstm_units')}")
        print(f"Dropout rate: {best_hps.get('dropout_rate')}")
        print(f"Learning rate: {best_hps.get('learning_rate')}")
        print(f"Number of layers: {best_hps.get('num_layers')}")
        
        return self.tuner, best_hps


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
    
    # Create tuner
    tuner = Seq2SeqTuner(
        encoder_timesteps=X_encoder_train.shape[1],
        encoder_features=X_encoder_train.shape[2],
        decoder_timesteps=18
    )
    
    # Setup tuner with reduced trials for testing
    tuner.setup_tuner(max_trials=5, tuner_type='hyperband')
    
    # Run hyperparameter search with TimeSeriesSplit
    tuner_results, best_hps = tuner.search_with_time_series_cv(
        X_encoder_train, X_decoder_train, y_train,
        n_splits=2,
        batch_size=32,
        epochs=50, 
        verbose=1
    )