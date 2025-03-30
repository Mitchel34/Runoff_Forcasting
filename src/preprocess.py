import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Create results directory if it doesn't exist
os.makedirs('../data/processed', exist_ok=True)

def load_data():
    """
    Load NWM forecasts and USGS observed runoff data
    """
    print("Loading data...")
    try:
        # Load NWM forecasts data
        nwm_data = pd.read_csv('../data/nwm_forecasts.csv')
        # Load USGS observed runoff data
        usgs_data = pd.read_csv('../data/usgs_observations.csv')
        
        return nwm_data, usgs_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that the data files exist in the data directory")
        return None, None

def clean_data(nwm_data, usgs_data):
    """
    Clean and preprocess the data
    - Handle missing values
    - Align timestamps
    - Convert units if necessary
    """
    print("Cleaning data...")
    
    # Convert date strings to datetime objects
    nwm_data['date'] = pd.to_datetime(nwm_data['date'])
    usgs_data['date'] = pd.to_datetime(usgs_data['date'])
    
    # Align timestamps between datasets
    merged_data = pd.merge(nwm_data, usgs_data, on='date', suffixes=('_nwm', '_usgs'))
    
    # Handle missing values (interpolate)
    merged_data = merged_data.interpolate(method='linear')
    
    # Drop any remaining rows with missing values
    merged_data = merged_data.dropna()
    
    return merged_data

def split_data(data):
    """
    Split data into training/validation and testing sets
    - Training/Validation: April 2021 - September 2022
    - Testing: October 2022 - April 2023
    """
    print("Splitting data...")
    
    # Split based on date
    train_val_data = data[data['date'] < '2022-10-01'].copy()
    test_data = data[data['date'] >= '2022-10-01'].copy()
    
    return train_val_data, test_data

def normalize_data(train_val_data, test_data):
    """
    Normalize features using StandardScaler
    """
    print("Normalizing data...")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Select features for normalization (runoff, precipitation if available)
    features = ['runoff_nwm', 'runoff_usgs']
    if 'precipitation' in train_val_data.columns:
        features.append('precipitation')
    
    # Fit scaler on training data and transform both sets
    train_val_scaled = train_val_data.copy()
    test_scaled = test_data.copy()
    
    train_val_scaled[features] = scaler.fit_transform(train_val_data[features])
    test_scaled[features] = scaler.transform(test_data[features])
    
    # Save the scaler for later use
    with open('../data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return train_val_scaled, test_scaled, features, scaler

def save_processed_data(train_val_data, test_data):
    """
    Save preprocessed data to CSV files
    """
    print("Saving preprocessed data...")
    
    train_val_data.to_csv('../data/processed/train_val_data.csv', index=False)
    test_data.to_csv('../data/processed/test_data.csv', index=False)

def main():
    # Load data
    nwm_data, usgs_data = load_data()
    if nwm_data is None or usgs_data is None:
        return
    
    # Clean data
    merged_data = clean_data(nwm_data, usgs_data)
    
    # Split data
    train_val_data, test_data = split_data(merged_data)
    
    # Normalize data
    train_val_scaled, test_scaled, features, scaler = normalize_data(train_val_data, test_data)
    
    # Save processed data
    save_processed_data(train_val_scaled, test_scaled)
    
    print("Data preprocessing completed successfully.")
    print(f"Training/Validation samples: {len(train_val_scaled)}")
    print(f"Testing samples: {len(test_scaled)}")
    print(f"Features used: {features}")

if __name__ == "__main__":
    main()
