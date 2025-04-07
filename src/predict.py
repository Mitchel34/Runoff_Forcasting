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
    
    # Create predictions directory if it doesn't exist
    predictions_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'predictions'
    )
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure datetime is in correct format
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Load model
    model = load_model(model_path)
    
    # Get the expected input shape from the model
    input_shape = model.layers[0].input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]  # Get first input shape if multiple inputs
    expected_features = input_shape[-1]  # Last dimension is the number of features
    sequence_length = input_shape[1]  # Middle dimension is sequence length
    
    print(f"Model expects input shape with {expected_features} features and sequence length {sequence_length}")
    
    # Create the same features used during training
    print("Creating features for prediction...")
    df = create_features(df)
    
    # Hard-code exactly 10 features in the exact order needed
    feature_columns = [
        'hour', 'day', 'month', 'dayofweek', 'is_weekend',
        'season_winter', 'season_spring', 'season_summer', 'season_fall',
        'log_runoff_nwm'
    ]
    
    # Check if all required features are available
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"ERROR: Missing required features: {missing_features}")
        return
    
    print(f"Using features: {feature_columns}")
    
    # Prepare input data
    X, scaler_X = prepare_input_data(df, feature_columns, sequence_length)
    
    print(f"Input shape for prediction: {X.shape}")
    
    # Verify we have the correct number of features
    if X.shape[2] != expected_features:
        print(f"ERROR: Input has {X.shape[2]} features but model expects {expected_features}")
        return
    
    # Make predictions
    predictions = make_predictions(model, X)
    
    # Create a scaler for the target variable
    scaler_y = StandardScaler()
    if 'runoff_usgs' in df.columns:
        # If we have ground truth data, use it for the scaler
        scaler_y.fit(df[['runoff_usgs']])
    else:
        # Otherwise use NWM data as a proxy
        scaler_y.fit(df[['runoff_nwm']])
    
    # Inverse transform predictions if needed
    predictions = inverse_transform_predictions(predictions, scaler_y)
    
    # Save predictions
    if output_file is None:
        output_file = os.path.join(predictions_dir, 'predictions.csv')
    
    result_df = save_predictions(df, predictions, sequence_length, output_file)
    
    print(f"Prediction completed successfully! Results saved to {output_file}")

def create_features(df):
    """
    Create additional features for prediction.
    This should match the features created during training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime and runoff data
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional features
    """
    # Add date-related features if datetime is present
    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Add seasonal indicators, but only if they don't already exist
        season_columns = ['season_winter', 'season_spring', 'season_summer', 'season_fall']
        if not all(col in df.columns for col in season_columns):
            # Create season categorical variable
            df['season'] = pd.cut(
                df['datetime'].dt.month,
                bins=[0, 3, 6, 9, 12],
                labels=['winter', 'spring', 'summer', 'fall'],
                include_lowest=True
            )
            
            # Convert to one-hot encoding
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            
            # Drop any existing season columns to avoid duplicates
            df = df.drop(columns=[col for col in df.columns if col in season_columns], errors='ignore')
            
            # Add the new season columns
            df = pd.concat([df, season_dummies], axis=1)
    
    # Add logarithmic features for runoff data
    if 'runoff_nwm' in df.columns and 'log_runoff_nwm' not in df.columns:
        df['log_runoff_nwm'] = np.log1p(df['runoff_nwm'])
        
        # Add lagged features if data is time ordered
        if len(df) > 1 and 'datetime' in df.columns and 'runoff_nwm_lag1' not in df.columns:
            df = df.sort_values('datetime')
            df['runoff_nwm_lag1'] = df['runoff_nwm'].shift(1)
            
            # Replace NaN values from shift with the mean
            df['runoff_nwm_lag1'] = df['runoff_nwm_lag1'].fillna(df['runoff_nwm'].mean())
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make runoff predictions with trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--model', type=str, help='Path to model file')
    args = parser.parse_args()
    
    main(args.input, args.output, args.model)
