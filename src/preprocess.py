"""
Data preprocessing module for NWM runoff forecasting.
Handles loading, cleaning, and preparing data for model training.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(nwm_file, usgs_file):
    """
    Load NWM forecast and USGS observation data.
    
    Parameters:
    -----------
    nwm_file : str
        Path to NWM forecast CSV file
    usgs_file : str
        Path to USGS observations CSV file
        
    Returns:
    --------
    nwm_df : pandas.DataFrame
        DataFrame containing NWM forecasts
    usgs_df : pandas.DataFrame
        DataFrame containing USGS observations
    """
    print(f"Loading NWM data from {nwm_file}")
    nwm_df = pd.read_csv(nwm_file, parse_dates=['datetime'])
    
    print(f"Loading USGS data from {usgs_file}")
    usgs_df = pd.read_csv(usgs_file, parse_dates=['datetime'])
    
    return nwm_df, usgs_df

def align_data(nwm_df, usgs_df):
    """
    Align NWM forecasts with USGS observations based on datetime and station ID.
    
    Parameters:
    -----------
    nwm_df : pandas.DataFrame
        DataFrame containing NWM forecasts
    usgs_df : pandas.DataFrame
        DataFrame containing USGS observations
        
    Returns:
    --------
    aligned_df : pandas.DataFrame
        DataFrame with aligned NWM forecasts and USGS observations
    """
    print("Aligning NWM forecasts with USGS observations")
    # Merge on datetime and station/gauge ID
    aligned_df = pd.merge(
        nwm_df, 
        usgs_df,
        on=['datetime', 'station_id'],
        how='inner',
        suffixes=('_nwm', '_usgs')
    )
    
    print(f"Aligned data shape: {aligned_df.shape}")
    return aligned_df

def create_features(df):
    """
    Create additional features for the model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with aligned data
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional features
    """
    print("Creating additional features")
    
    # Extract time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Calculate error metrics
    df['nwm_error'] = df['runoff_nwm'] - df['runoff_usgs']
    df['nwm_error_pct'] = (df['nwm_error'] / df['runoff_usgs']) * 100
    
    return df

def split_data(df, test_start_date='2022-10-01'):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed DataFrame
    test_start_date : str
        Start date for test data (default: '2022-10-01')
        
    Returns:
    --------
    train_df : pandas.DataFrame
        Training data
    val_df : pandas.DataFrame
        Validation data
    test_df : pandas.DataFrame
        Test data
    """
    print(f"Splitting data with test set starting from {test_start_date}")
    
    # Split based on date
    train_val_df = df[df['datetime'] < test_start_date].copy()
    test_df = df[df['datetime'] >= test_start_date].copy()
    
    # Further split training data into train/validation
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"Validation set: {val_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    
    return train_df, val_df, test_df

def scale_data(train_df, val_df, test_df, feature_columns, target_column):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    train_df, val_df, test_df : pandas.DataFrame
        Training, validation, and test DataFrames
    feature_columns : list
        List of feature column names
    target_column : str
        Target column name
        
    Returns:
    --------
    X_train, X_val, X_test : numpy.ndarray
        Scaled feature arrays
    y_train, y_val, y_test : numpy.ndarray
        Target arrays
    scaler_X, scaler_y : sklearn.preprocessing.StandardScaler
        Fitted scalers for features and target
    """
    print("Scaling data")
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers on training data
    X_train = scaler_X.fit_transform(train_df[feature_columns])
    y_train = scaler_y.fit_transform(train_df[[target_column]])
    
    # Transform validation and test data
    X_val = scaler_X.transform(val_df[feature_columns])
    y_val = scaler_y.transform(val_df[[target_column]])
    
    X_test = scaler_X.transform(test_df[feature_columns])
    y_test = scaler_y.transform(test_df[[target_column]])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

def prepare_sequence_data(X, y, sequence_length):
    """
    Prepare sequential data for recurrent neural networks.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature array
    y : numpy.ndarray
        Target array
    sequence_length : int
        Length of sequences to create
        
    Returns:
    --------
    X_seq : numpy.ndarray
        Sequence feature array of shape (samples, sequence_length, features)
    y_seq : numpy.ndarray
        Target array corresponding to sequences
    """
    print(f"Creating sequences with length {sequence_length}")
    
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def main():
    """
    Main preprocessing pipeline.
    """
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Define file paths
    nwm_file = os.path.join(raw_dir, 'nwm_forecasts.csv')
    usgs_file = os.path.join(raw_dir, 'usgs_observations.csv')
    
    # Load data
    nwm_df, usgs_df = load_data(nwm_file, usgs_file)
    
    # Align data
    aligned_df = align_data(nwm_df, usgs_df)
    
    # Create features
    df = create_features(aligned_df)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Save processed data
    train_val_df = pd.concat([train_df, val_df])
    train_val_df.to_csv(os.path.join(processed_dir, 'train_validation_data.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)
    
    print("Preprocessing completed. Data saved to processed directory.")

if __name__ == "__main__":
    main()
