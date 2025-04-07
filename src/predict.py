"""
Prediction module for making runoff forecasts with trained model.
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """
    Load a trained TensorFlow model.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Loaded model
    """
    print(f"Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)

def prepare_input_data(df, feature_columns, sequence_length, scaler=None):
    """
    Prepare input data for prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with features
    feature_columns : list
        List of feature column names
    sequence_length : int
        Length of input sequences
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler for feature normalization
        
    Returns:
    --------
    X : numpy.ndarray
        Prepared input sequences
    scaler : sklearn.preprocessing.StandardScaler
        Fitted or provided scaler
    """
    # Extract features
    X = df[feature_columns].values
    
    # Scale features if scaler provided
    if scaler is not None:
        X = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Create sequences
    X_seq = []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
    
    return np.array(X_seq), scaler

def make_predictions(model, X):
    """
    Make predictions with the model.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    X : numpy.ndarray
        Input data
        
    Returns:
    --------
    predictions : numpy.ndarray
        Model predictions
    """
    print(f"Making predictions for {len(X)} samples")
    return model.predict(X)

def inverse_transform_predictions(predictions, scaler_y):
    """
    Inverse transform scaled predictions.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Scaled predictions
    scaler_y : sklearn.preprocessing.StandardScaler
        Scaler used for target variable
        
    Returns:
    --------
    predictions : numpy.ndarray
        Predictions in original scale
    """
    # Reshape if needed
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    return scaler_y.inverse_transform(predictions)

def save_predictions(df, predictions, sequence_length, output_file):
    """
    Save predictions to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame
    predictions : numpy.ndarray
        Model predictions
    sequence_length : int
        Length of input sequences
    output_file : str
        Path to output CSV file
        
    Returns:
    --------
    result_df : pandas.DataFrame
        DataFrame with predictions
    """
    # Create result DataFrame
    # The first sequence_length-1 rows won't have predictions
    result_df = df.copy()
    result_df['runoff_predicted'] = np.nan
    
    # Add predictions to the DataFrame
    result_df.loc[sequence_length-1:sequence_length-1+len(predictions), 'runoff_predicted'] = predictions.flatten()
    
    # Save to CSV
    print(f"Saving predictions to {output_file}")
    result_df.to_csv(output_file, index=False)
    
    return result_df

def main(input_file, output_file=None, model_path=None):
    """
    Main prediction function.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to output CSV file
    model_path : str, optional
        Path to model file
    """
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'nwm_dl_model.keras'
        )
    
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(input_file),
            'predictions.csv'
        )
    
    # Load data
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Load model
    model = load_model(model_path)
    
    # Define feature columns (this would be the same as used during training)
    # This is a placeholder - actual feature columns would depend on your specific dataset
    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    
    # Prepare data
    sequence_length = 24  # This should match the sequence length used during training
    X, scaler_X = prepare_input_data(df, feature_columns, sequence_length)
    
    # Make predictions
    predictions = make_predictions(model, X)
    
    # Create a dummy scaler for demonstration
    # In a real scenario, you would load the same scaler used during training
    scaler_y = StandardScaler()
    scaler_y.fit(df[['runoff_nwm']])
    
    # Inverse transform predictions if needed
    predictions = inverse_transform_predictions(predictions, scaler_y)
    
    # Save predictions
    result_df = save_predictions(df, predictions, sequence_length, output_file)
    
    print("Prediction completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make runoff predictions with trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--model', type=str, help='Path to model file')
    args = parser.parse_args()
    
    main(args.input, args.output, args.model)
