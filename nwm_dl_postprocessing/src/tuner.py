import os
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, ReLU, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class Seq2SeqTuner:
    """
    Hyperparameter tuner for Seq2Seq LSTM models using keras-tuner.
    Performs hyperparameter optimization with TimeSeriesSplit cross-validation.
    """
    
    def __init__(self, 
                 encoder_timesteps=24, 
                 encoder_features=3,
                 decoder_timesteps=18,
                 project_name='nwm_seq2seq_tuning',
                 directory='../models'):
        """
        Initialize the Seq2SeqTuner.
        
        Parameters:
        -----------
        encoder_timesteps : int
            Number of timesteps for encoder input sequence
        encoder_features : int
            Number of features per timestep in encoder input
        decoder_timesteps : int
            Number of timesteps for decoder output (18 for 1-18h lead times)
        project_name : str
            Name for the tuning project
        directory : str
            Directory to save tuning results
        """
        self.encoder_timesteps = encoder_timesteps
        self.encoder_features = encoder_features
        self.decoder_timesteps = decoder_timesteps
        self.project_name = project_name
        self.directory = directory
        self.tuner = None
        
        # Make sure directory exists
        os.makedirs(directory, exist_ok=True)
    
    def build_model(self, hp):
        """
        Build model with hyperparameters from the search space.
        
        Parameters:
        -----------
        hp : keras_tuner.HyperParameters
            Hyperparameters to build model with
            
        Returns:
        --------
        model : tf.keras.Model
            Compiled Seq2Seq LSTM model with hyperparameters
        """
        # Hyperparameters to tune
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3])
        num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        
        # Encoder
        encoder_inputs = Input(shape=(self.encoder_timesteps, self.encoder_features), name='encoder_inputs')
        
        # Encoder LSTM layers
        x = encoder_inputs
        for i in range(num_layers - 1):
            x = LSTM(lstm_units, return_sequences=True, name=f'encoder_lstm_{i+1}')(x)
            x = ReLU()(x)
            x = Dropout(dropout_rate)(x)
            
        # Final encoder LSTM layer
        encoder_outputs = LSTM(lstm_units, name=f'encoder_lstm_{num_layers}')(x)
        encoder_outputs = ReLU()(encoder_outputs)
        encoder_outputs = Dropout(dropout_rate)(encoder_outputs)
        
        # Decoder
        decoder_inputs = Input(shape=(self.decoder_timesteps,), name='decoder_inputs')
        
        # Repeat encoder output for each decoder timestep
        repeated_encoder = RepeatVector(self.decoder_timesteps)(encoder_outputs)
        
        # Combine repeated encoder output with decoder inputs
        decoder_dense = Dense(lstm_units, activation='relu')(decoder_inputs)
        decoder_dense = Dropout(dropout_rate)(decoder_dense)
        decoder_dense = RepeatVector(1)(decoder_dense)
        decoder_dense = tf.squeeze(decoder_dense, axis=1)
        
        combined_input = tf.concat([repeated_encoder, 
                                   tf.expand_dims(decoder_dense, axis=1)], 
                                  axis=1)
        
        # Decoder LSTM layers
        x = combined_input
        for i in range(num_layers - 1):
            x = LSTM(lstm_units, return_sequences=True, name=f'decoder_lstm_{i+1}')(x)
            x = ReLU()(x)
            x = Dropout(dropout_rate)(x)
        
        # Final decoder LSTM layer
        decoder_outputs = LSTM(lstm_units, return_sequences=True, name=f'decoder_lstm_{num_layers}')(x)
        decoder_outputs = ReLU()(decoder_outputs)
        decoder_outputs = Dropout(dropout_rate)(decoder_outputs)
        
        # Output layer
        outputs = TimeDistributed(Dense(1, activation='linear'))(decoder_outputs)
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
    
    def setup_tuner(self, tuner_type='hyperband', max_trials=50, executions_per_trial=2):
        """
        Setup the keras-tuner for hyperparameter optimization.
        
        Parameters:
        -----------
        tuner_type : str
            Type of tuner to use ('hyperband' or 'bayesian')
        max_trials : int
            Maximum number of trials for hyperparameter search
        executions_per_trial : int
            Number of models to train per trial
            
        Returns:
        --------
        tuner : keras_tuner.Tuner
            Initialized tuner object
        """
        if tuner_type.lower() == 'hyperband':
            tuner = kt.Hyperband(
                self.build_model,
                objective='val_loss',
                max_epochs=50,
                factor=3,
                directory=self.directory,
                project_name=self.project_name,
                max_trials=max_trials,
                executions_per_trial=executions_per_trial
            )
        elif tuner_type.lower() == 'bayesian':
            tuner = kt.BayesianOptimization(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=self.directory,
                project_name=self.project_name
            )
        else:
            raise ValueError(f"Unsupported tuner type: {tuner_type}. Use 'hyperband' or 'bayesian'.")
            
        self.tuner = tuner
        return tuner
    
    def search_with_time_series_cv(self, X_encoder, X_decoder, y, 
                                  n_splits=3, batch_size=32, epochs=50, verbose=1):
        """
        Perform hyperparameter search with TimeSeriesSplit cross-validation.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences
        y : numpy.ndarray
            Target forecast error sequences
        n_splits : int
            Number of splits for TimeSeriesSplit
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of training epochs
        verbose : int
            Verbosity mode
            
        Returns:
        --------
        tuner : keras_tuner.Tuner
            Trained tuner object with results
        best_hps : keras_tuner.HyperParameters
            Best hyperparameters found
        """
        if self.tuner is None:
            self.setup_tuner()
            
        # Setup TimeSeriesSplit for temporal cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Setup callbacks for each fold
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        fold_idx = 0
        
        # Iterate through each fold
        for train_idx, val_idx in tscv.split(X_encoder):
            fold_idx += 1
            print(f"\nTraining on fold {fold_idx}/{n_splits}...")
            
            # Split data for this fold
            X_encoder_train, X_encoder_val = X_encoder[train_idx], X_encoder[val_idx]
            X_decoder_train, X_decoder_val = X_decoder[train_idx], X_decoder[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Search on this fold
            self.tuner.search(
                [X_encoder_train, X_decoder_train], y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_encoder_val, X_decoder_val], y_val),
                callbacks=callbacks,
                verbose=verbose
            )
            
        # Get best hyperparameters
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print("\nBest hyperparameters:")
        print(f"LSTM Units: {best_hps.get('lstm_units')}")
        print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
        print(f"Learning Rate: {best_hps.get('learning_rate')}")
        print(f"Number of Layers: {best_hps.get('num_layers')}")
        
        return self.tuner, best_hps
    
    def get_best_model(self):
        """
        Build the best model from the hyperparameter search.
        
        Returns:
        --------
        model : tf.keras.Model
            Best model from hyperparameter search
        """
        if self.tuner is None:
            raise ValueError("Tuner not initialized or search not performed yet.")
            
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = self.tuner.hypermodel.build(best_hps)
        
        return best_model

if __name__ == "__main__":
    # Example usage
    from preprocess import DataPreprocessor
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed",
        sequence_length=24
    )
    
    data = preprocessor.process_data(stream_ids=["20380357", "21609641"])
    
    # Get training data
    stream_id = "20380357"  # Example: use first stream for tuning
    X_encoder_train = data[stream_id]['train_val']['X_encoder']
    X_decoder_train = data[stream_id]['train_val']['X_decoder']
    y_train = data[stream_id]['train_val']['y']
    
    # Setup tuner
    tuner = Seq2SeqTuner(
        encoder_timesteps=X_encoder_train.shape[1],
        encoder_features=X_encoder_train.shape[2],
        decoder_timesteps=18
    )
    
    # Perform hyperparameter search
    tuner.setup_tuner(tuner_type='hyperband', max_trials=10)  # Reduced trials for example
    tuner, best_hps = tuner.search_with_time_series_cv(X_encoder_train, X_decoder_train, y_train, n_splits=3)
    
    # Build best model
    best_model = tuner.get_best_model()
    best_model.summary()
    
    # Save best model
    best_model.save('../models/nwm_lstm_model.keras')