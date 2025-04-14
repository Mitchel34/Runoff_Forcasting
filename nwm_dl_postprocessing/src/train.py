import os
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

# Import local modules
from preprocess import DataPreprocessor
from model import Seq2SeqLSTMModel
from tuner import Seq2SeqTuner

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train runoff forecasting models')
    
    # Data paths
    parser.add_argument('--raw_data_path', type=str, default='../data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--processed_data_path', type=str, default='../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--model_save_path', type=str, default='../models',
                        help='Path to save trained models')
    
    # Stream IDs
    parser.add_argument('--stream_ids', type=str, nargs='+', default=['20380357', '21609641'],
                        help='List of stream gauge IDs to process')
    
    # Model parameters
    parser.add_argument('--sequence_length', type=int, default=24,
                        help='Number of past timesteps to use as input (hours)')
    parser.add_argument('--forecast_horizon', type=int, default=18,
                        help='Number of future timesteps to predict (hours)')
    parser.add_argument('--lstm_units', type=int, default=64,
                        help='Number of LSTM units per layer')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Hyperparameter tuning
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--tuner_type', type=str, default='hyperband',
                        choices=['hyperband', 'random', 'bayesian'],
                        help='Type of hyperparameter tuner')
    parser.add_argument('--max_trials', type=int, default=30,
                        help='Maximum number of hyperparameter tuning trials')
    
    return parser.parse_args()

def train_stream_model(stream_id, data, args, use_best_params=None):
    """
    Train a model for a specific stream gauge.
    
    Parameters:
    -----------
    stream_id : str
        Stream gauge ID
    data : dict
        Preprocessed data dictionary
    args : argparse.Namespace
        Command line arguments
    use_best_params : dict, optional
        Best hyperparameters from tuning
        
    Returns:
    --------
    model : Seq2SeqLSTMModel
        Trained model
    history : tensorflow.keras.callbacks.History
        Training history
    metrics : dict
        Evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"Training model for stream gauge {stream_id}")
    print(f"{'='*50}")
    
    # Extract training and test data
    X_encoder_train = data[stream_id]['train_val']['X_encoder']
    X_decoder_train = data[stream_id]['train_val']['X_decoder']
    y_train = data[stream_id]['train_val']['y']
    
    X_encoder_test = data[stream_id]['test']['X_encoder']
    X_decoder_test = data[stream_id]['test']['X_decoder']
    y_test = data[stream_id]['test']['y']
    
    print(f"Training data shape: {X_encoder_train.shape}, {X_decoder_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_encoder_test.shape}, {X_decoder_test.shape}, {y_test.shape}")
    
    # Set model parameters from args or tuning results
    if use_best_params:
        print("Using hyperparameter tuning results:")
        lstm_units = use_best_params.get('lstm_units', args.lstm_units)
        dropout_rate = use_best_params.get('dropout_rate', args.dropout_rate)
        learning_rate = use_best_params.get('learning_rate', args.learning_rate)
        num_layers = use_best_params.get('num_layers', args.num_layers)
        
        print(f"LSTM units: {lstm_units}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Learning rate: {learning_rate}")
        print(f"Number of layers: {num_layers}")
    else:
        lstm_units = args.lstm_units
        dropout_rate = args.dropout_rate
        learning_rate = args.learning_rate
        num_layers = args.num_layers
    
    # Create model
    model = Seq2SeqLSTMModel(
        encoder_timesteps=X_encoder_train.shape[1],
        encoder_features=X_encoder_train.shape[2],
        decoder_timesteps=args.forecast_horizon,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        num_layers=num_layers
    )
    
    # Build model
    model.build_model()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'../logs/{stream_id}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
    ]
    
    # Train model
    history = model.train(
        X_encoder_train, X_decoder_train, y_train,
        validation_data=([X_encoder_test, X_decoder_test], y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_encoder_test, X_decoder_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    print("\nTest metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    
    # Save model
    model_save_dir = os.path.join(args.model_save_path, stream_id)
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = f"{stream_id}_seq2seq_lstm.keras"
    model.save(os.path.join(model_save_dir, model_filename))
    
    # Save metrics and parameters
    metadata = {
        'stream_id': stream_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'sequence_length': args.sequence_length,
            'forecast_horizon': args.forecast_horizon,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'num_layers': num_layers,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        },
        'metrics': metrics,
    }
    
    with open(os.path.join(model_save_dir, f"{stream_id}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{stream_id} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{stream_id} - Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f"{stream_id}_training_history.png"))
    plt.close()
    
    return model, history, metrics

def tune_stream_model(stream_id, data, args):
    """
    Perform hyperparameter tuning for a specific stream gauge.
    
    Parameters:
    -----------
    stream_id : str
        Stream gauge ID
    data : dict
        Preprocessed data dictionary
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    best_hp : kerastuner.HyperParameters
        Best hyperparameters
    """
    print(f"\n{'='*50}")
    print(f"Tuning hyperparameters for stream gauge {stream_id}")
    print(f"{'='*50}")
    
    # Extract training data
    X_encoder_train = data[stream_id]['train_val']['X_encoder']
    X_decoder_train = data[stream_id]['train_val']['X_decoder']
    y_train = data[stream_id]['train_val']['y']
    
    # Create tuner
    tuner = Seq2SeqTuner(
        encoder_timesteps=X_encoder_train.shape[1],
        encoder_features=X_encoder_train.shape[2],
        decoder_timesteps=args.forecast_horizon,
        project_name=f'nwm_seq2seq_{stream_id}',
        directory=os.path.join(args.model_save_path, 'tuning')
    )
    
    # Setup tuner
    tuner.setup_tuner(
        tuner_type=args.tuner_type,
        max_trials=args.max_trials,
        executions_per_trial=1
    )
    
    # Run hyperparameter search
    _, best_hp = tuner.search_with_time_series_cv(
        X_encoder_train, X_decoder_train, y_train,
        n_splits=3,
        batch_size=args.batch_size,
        epochs=args.epochs // 2,  # Reduce epochs for tuning
        verbose=1
    )
    
    return best_hp

def main():
    """Main function to run the training process."""
    args = parse_args()
    
    print(f"\nRunoff Forecasting - NWM Post-processing Training")
    print(f"{'='*50}")
    print(f"Stream IDs: {args.stream_ids}")
    print(f"Sequence length: {args.sequence_length} hours")
    print(f"Forecast horizon: {args.forecast_horizon} hours")
    print(f"Hyperparameter tuning: {'Yes' if args.tune else 'No'}")
    
    # Create data preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon
    )
    
    # Process data for all stream IDs
    print("\nProcessing data for all stream gauges...")
    data = preprocessor.process_data(stream_ids=args.stream_ids)
    
    # Train models for each stream
    results = {}
    for stream_id in args.stream_ids:
        if args.tune:
            # Tune hyperparameters
            print(f"\nTuning hyperparameters for stream {stream_id}...")
            best_hp = tune_stream_model(stream_id, data, args)
            
            # Train with best hyperparameters
            print(f"\nTraining model for stream {stream_id} with best hyperparameters...")
            model, history, metrics = train_stream_model(stream_id, data, args, use_best_params=best_hp)
        else:
            # Train with default hyperparameters
            print(f"\nTraining model for stream {stream_id} with default hyperparameters...")
            model, history, metrics = train_stream_model(stream_id, data, args)
        
        results[stream_id] = metrics
    
    # Print summary of results
    print("\nSummary of Results:")
    print(f"{'='*50}")
    for stream_id, metrics in results.items():
        print(f"Stream {stream_id}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.4f}")
        print()
    
    # Save combined results summary
    summary_path = os.path.join(args.model_save_path, "training_summary.json")
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'sequence_length': args.sequence_length,
            'forecast_horizon': args.forecast_horizon,
            'lstm_units': args.lstm_units,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'num_layers': args.num_layers,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        },
        'results': {stream_id: metrics for stream_id, metrics in results.items()}
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to {summary_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()