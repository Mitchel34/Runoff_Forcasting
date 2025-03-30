import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

# Create results directory if it doesn't exist
os.makedirs('../data/processed', exist_ok=True)

def load_data():
    """
    Load NWM forecasts and USGS observed runoff data from multiple files
    """
    print("Loading data...")
    try:
        # Load and combine NWM forecast data from folder1 (stream ID: 20380357)
        nwm_data_list1 = []
        folder1_path = '../data/folder1'
        for file in os.listdir(folder1_path):
            if file.startswith('streamflow_20380357_') and file.endswith('.csv'):
                file_path = os.path.join(folder1_path, file)
                df = pd.read_csv(file_path)
                df['station_id'] = '20380357'
                nwm_data_list1.append(df)

        # Load and combine NWM forecast data from folder2 (stream ID: 21609641)
        nwm_data_list2 = []
        folder2_path = '../data/folder2'
        for file in os.listdir(folder2_path):
            if file.startswith('streamflow_21609641_') and file.endswith('.csv'):
                file_path = os.path.join(folder2_path, file)
                df = pd.read_csv(file_path)
                df['station_id'] = '21609641'
                nwm_data_list2.append(df)
        
        # Combine all NWM data
        nwm_data = pd.concat(nwm_data_list1 + nwm_data_list2, ignore_index=True)
        
        # Rename columns to match expected format in the rest of the script
        nwm_data = nwm_data.rename(columns={
            'model_output_valid_time': 'date',
            'streamflow_value': 'runoff_nwm',
            'streamID': 'stream_id'
        })
        
        # Load USGS data
        usgs_data1 = pd.read_csv('../data/folder1/09520500_Strt_2021-04-20_EndAt_2023-04-21.csv')
        usgs_data1['station_id'] = '20380357'  # Associate with corresponding NWM station
        
        usgs_data2 = pd.read_csv('../data/folder2/11266500_Strt_2021-04-20_EndAt_2023-04-21.csv')
        usgs_data2['station_id'] = '21609641'  # Associate with corresponding NWM station
        
        # Combine USGS data
        usgs_data = pd.concat([usgs_data1, usgs_data2], ignore_index=True)
        
        # Rename columns to match expected format
        usgs_data = usgs_data.rename(columns={
            'DateTime': 'date',
            'USGSFlowValue': 'runoff_usgs'
        })
        
        return nwm_data, usgs_data
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure that the data files exist in the data directory with the expected structure.")
        return None, None

def clean_data(nwm_data, usgs_data):
    """
    Clean and preprocess the data
    """
    print("Cleaning data...")
    
    # Process NWM data
    nwm_data['date'] = pd.to_datetime(nwm_data['date'].str.replace('_', ' '))
    
    # Process USGS data
    usgs_data['date'] = pd.to_datetime(usgs_data['date'])
    
    # Convert both to naive datetime (no timezone) to fix the merge issue
    if nwm_data['date'].dt.tz is not None:
        nwm_data['date'] = nwm_data['date'].dt.tz_localize(None)
    
    if usgs_data['date'].dt.tz is not None:
        usgs_data['date'] = usgs_data['date'].dt.tz_localize(None)
    
    print("NWM date type:", nwm_data['date'].dtype)
    print("USGS date type:", usgs_data['date'].dtype)
    
    # Merge data by date and station ID
    merged_data = pd.merge(nwm_data, usgs_data, on=['date', 'station_id'])
    
    # Handle missing values (interpolate)
    merged_data = merged_data.interpolate(method='linear')
    
    # Drop any remaining rows with missing values
    merged_data = merged_data.dropna(subset=['runoff_nwm', 'runoff_usgs'])
    
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
