"""
Preprocess NWM forecast and USGS observation data.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import create_sequences, temporal_split


def align_and_merge(nwm_data, usgs_data):
    """
    Align NWM forecasts with USGS observations based on timestamp.
    
    Args:
        nwm_data: DataFrame with NWM forecasts (must include 'valid_time', 'nwm_flow', 'lead_time')
        usgs_data: DataFrame with USGS observations (must include 'datetime', 'usgs_flow')
        
    Returns:
        DataFrame with aligned data
    """
    # Rename USGS columns for clarity and consistency
    # NWM columns are assumed to be renamed before calling this function
    usgs_data = usgs_data.rename(columns={"value": "usgs_flow", "DateTime": "datetime"})

    # Ensure datetime columns are in the correct format
    # NWM valid_time is assumed to be datetime already
    # Assuming USGS 'datetime' is in a standard format pandas can infer or already parsed
    usgs_data["datetime"] = pd.to_datetime(usgs_data["datetime"])
    # Make USGS datetime naive to match NWM datetime before merging
    usgs_data["datetime"] = usgs_data["datetime"].dt.tz_localize(None)
    
    # Merge data based on the time the forecast is valid for
    merged = pd.merge(
        nwm_data[["valid_time", "nwm_flow", "lead_time"]], # Select required columns from NWM
        usgs_data[["datetime", "usgs_flow"]],
        left_on="valid_time",
        right_on="datetime",
        how="inner"
    )
    
    # Calculate error between NWM forecast and observed USGS flow
    merged["error"] = merged["usgs_flow"] - merged["nwm_flow"]
    
    # Keep necessary columns and set datetime as index
    merged = merged[["datetime", "nwm_flow", "usgs_flow", "error", "lead_time"]]
    merged = merged.set_index("datetime")
    
    return merged


def process_station(station_id, data_dir, output_dir, window_size=24, horizon=18):
    """
    Process data for a specific station.
    
    Args:
        station_id: Station identifier
        data_dir: Directory with raw data
        output_dir: Directory to save processed data
        window_size: Number of past time steps to use as input
        horizon: Number of lead times to predict (1-18h)
    """
    print(f"Processing data for station {station_id}...")
    
    # Find NWM forecast files for this station
    nwm_files = glob.glob(os.path.join(data_dir, f"streamflow_{station_id}_*.csv"))
    if not nwm_files:
        print(f"No NWM forecast files found for station {station_id}")
        return False

    # Find USGS observation file
    usgs_files = glob.glob(os.path.join(data_dir, f"*_Strt_*_EndAt_*.csv"))
    if not usgs_files:
        print(f"No USGS observation file found")
        return False
    
    usgs_file = None
    for file in usgs_files:
        if os.path.exists(file):
            usgs_file = file
            break
    
    if not usgs_file:
        print(f"USGS file not found")
        return False
    
    print(f"Found {len(nwm_files)} NWM files and USGS file: {os.path.basename(usgs_file)}")
    
    # Load NWM forecasts (monthly files)
    nwm_dfs = []
    for file in sorted(nwm_files):
        try:
            df = pd.read_csv(file, dtype={'streamflow_value': float})
            nwm_dfs.append(df)
            print(f"Loaded {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not nwm_dfs:
        print("No valid NWM data found")
        return False
        
    nwm_data = pd.concat(nwm_dfs, ignore_index=True)

    # --- Calculate Lead Time and Rename Columns ---
    # Rename NWM columns first
    nwm_data = nwm_data.rename(columns={
        "streamflow_value": "nwm_flow", 
        "model_output_valid_time": "valid_time",
        "model_initialization_time": "init_time"
    })

    # Convert time columns to datetime
    nwm_data["valid_time"] = pd.to_datetime(nwm_data["valid_time"], format='%Y-%m-%d_%H:%M:%S')
    nwm_data["init_time"] = pd.to_datetime(nwm_data["init_time"], format='%Y-%m-%d_%H:%M:%S')

    # Calculate lead time in hours
    nwm_data["lead_time"] = (nwm_data["valid_time"] - nwm_data["init_time"]).dt.total_seconds() / 3600
    nwm_data["lead_time"] = nwm_data["lead_time"].astype(int) # Convert to integer hours

    # Filter for required lead times (1-18 hours)
    nwm_data = nwm_data[nwm_data["lead_time"].between(1, horizon)]
    print(f"Filtered NWM data to {len(nwm_data)} records for lead times 1-{horizon}h")
    
    # Load USGS observations
    try:
        usgs_data = pd.read_csv(usgs_file, parse_dates=['DateTime'], index_col='DateTime')
        print(f"Loaded USGS columns: {usgs_data.columns.tolist()}") # Print columns to find the flow value
        
        # Identify the flow column 
        flow_col_name = 'USGSFlowValue' # Updated based on debug output

        if flow_col_name in usgs_data.columns:
            # Perform unit conversion if needed (assuming cfs to cms)
            usgs_data[flow_col_name] = usgs_data[flow_col_name] * 0.0283168
            # Rename the identified flow column to 'usgs_flow'
            usgs_data = usgs_data.rename(columns={flow_col_name: 'usgs_flow'})
        else:
            print(f"Error: Expected flow column '{flow_col_name}' not found in USGS data {usgs_file}. Cannot calculate error.")
            return False # Stop processing this station if flow data is missing
            
        usgs_data = usgs_data.reset_index() # Reset index after potential renaming
        print(f"Loaded and processed USGS data with {len(usgs_data)} records")
    except Exception as e:
        print(f"Error loading or processing USGS data: {e}")
        return False
    
    # Align and merge data
    # Pass the prepared nwm_data and usgs_data (now guaranteed to have 'usgs_flow')
    merged_data = align_and_merge(nwm_data, usgs_data)
    if merged_data.empty:
        print("Merged data is empty. Check alignment logic and data content.")
        return False
    print(f"Merged data has {len(merged_data)} records after alignment.")

    # --- Pivot data to have lead times as columns --- 
    # This structure is often better for sequence creation when predicting multiple steps ahead
    pivot_data = merged_data.pivot_table(
        index=merged_data.index, 
        columns='lead_time', 
        values=['nwm_flow', 'usgs_flow', 'error']
    )
    pivot_data.columns = [f'{val}_{lead}' for val, lead in pivot_data.columns]
    pivot_data = pivot_data.dropna() # Drop rows with any missing lead times after pivot
    print(f"Pivoted data shape: {pivot_data.shape}")

    if pivot_data.empty:
        print("Pivoted data is empty. Check for gaps in lead times.")
        return False

    # --- Feature Selection (using pivoted data) ---
    # Example: Use past errors and NWM forecasts for all lead times as features
    # Adjust feature selection based on modeling strategy
    feature_cols = [col for col in pivot_data.columns if col.startswith('error_') or col.startswith('nwm_flow_')]
    target_cols = [f'error_{i}' for i in range(1, horizon + 1)]
    
    features = pivot_data[feature_cols]
    target = pivot_data[target_cols]

    # Create sequences for time series modeling using only features
    # The function returns feature sequences (X) and the corresponding end-of-sequence indices
    X, _, sequence_end_indices = create_sequences(features.values, window_size, 1) # Use horizon=1 as we select targets later
    if X is None:
        print("Sequence creation failed. Check data and parameters.")
        return False
    
    # Select the corresponding targets using the indices returned by create_sequences
    # The indices correspond to the *end* of each sequence in the original pivot_data
    y = target.iloc[sequence_end_indices].values
    sequence_start_indices = pivot_data.index[sequence_end_indices] # Get the actual datetime indices

    # y shape should be (samples, num_target_features) where num_target_features = horizon
    print(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
    
    # Split data: Train (Apr 2021-Sep 2022), Test (Oct 2022-Apr 2023)
    split_date = pd.to_datetime('2022-10-01')
    # Use the datetime indices for splitting
    X_train, y_train, X_test, y_test, train_indices, test_indices = temporal_split(
        X, y, sequence_start_indices, split_date
    )
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Train or test split resulted in zero samples. Check split date and data range.")
        return False
        
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- Scaling ---
    # Reshape X for scaler (samples * timesteps, features)
    n_samples_train, n_timesteps_train, n_features_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features_train)
    
    n_samples_test, n_timesteps_test, n_features_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, n_features_test)

    # Reshape y for scaler (samples, horizon_outputs)
    # y is already (samples, horizon_outputs), no need to reshape like before
    y_train_reshaped = y_train 
    y_test_reshaped = y_test

    # Initialize scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler() # Scale targets together

    # Fit scalers ONLY on training data
    x_scaler.fit(X_train_reshaped)
    y_scaler.fit(y_train_reshaped) # Fit on (samples, horizon_outputs)

    # Transform train and test data
    X_train_scaled_reshaped = x_scaler.transform(X_train_reshaped)
    X_test_scaled_reshaped = x_scaler.transform(X_test_reshaped)
    y_train_scaled = y_scaler.transform(y_train_reshaped)
    y_test_scaled = y_scaler.transform(y_test_reshaped)

    # Reshape X back to original sequence format (samples, timesteps, features)
    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps_train, n_features_train)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps_test, n_features_test)
    # y_train_scaled and y_test_scaled are already in the correct shape (samples, horizon_outputs)

    print(f"Scaled Train X: {X_train_scaled.shape}, Scaled Test X: {X_test_scaled.shape}")
    print(f"Scaled Train y: {y_train_scaled.shape}, Scaled Test y: {y_test_scaled.shape}")

    # Get original NWM and USGS data corresponding to the test sequences
    # Use the test_indices which map back to the original pivot_data
    nwm_test_original = pivot_data[[f'nwm_flow_{i}' for i in range(1, horizon + 1)]].iloc[test_indices].values
    usgs_test_original = pivot_data[[f'usgs_flow_{i}' for i in range(1, horizon + 1)]].iloc[test_indices].values
    # --- ADDITION: Get timestamps for the test set ---
    # These timestamps correspond to the end of the input sequence window.
    # The forecast target is for the hour *after* this timestamp.
    test_timestamps = pivot_data.index[test_indices].values # Get as numpy array for saving

    # Save processed data and scalers
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    scaler_dir = os.path.join(output_dir, 'scalers')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    
    train_file = os.path.join(train_dir, f"{station_id}.npz")
    test_file = os.path.join(test_dir, f"{station_id}.npz")
    x_scaler_file = os.path.join(scaler_dir, f"{station_id}_x_scaler.joblib")
    y_scaler_file = os.path.join(scaler_dir, f"{station_id}_y_scaler.joblib")
    
    # Save with keys expected by evaluate.py, including timestamps
    np.savez(train_file, X_train=X_train_scaled, y_train_scaled=y_train_scaled)
    np.savez(test_file, 
             X_test=X_test_scaled, 
             y_test_scaled=y_test_scaled, 
             nwm_test_original=nwm_test_original, 
             usgs_test_original=usgs_test_original,
             test_timestamps=test_timestamps) # <-- Added timestamps
    
    joblib.dump(x_scaler, x_scaler_file)
    joblib.dump(y_scaler, y_scaler_file)
    
    print(f"Data saved to {train_file} and {test_file}")
    print(f"Scalers saved to {x_scaler_file} and {y_scaler_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess NWM and USGS data")
    parser.add_argument("--data-dir", type=str, default="data/raw", 
                        help="Directory with raw data files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--window-size", type=int, default=24,
                        help="Number of past time steps to use as input")
    parser.add_argument("--horizon", type=int, default=18,
                        help="Number of lead times to predict (1-18h)")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'test')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'scalers')).mkdir(parents=True, exist_ok=True)

    station_ids = ["20380357", "21609641"]
    success_count = 0
    for station_id in station_ids:
        station_data_dir = os.path.join(args.data_dir, station_id)
        if process_station(
            station_id, 
            station_data_dir,
            args.output_dir,
            args.window_size,
            args.horizon
        ):
            success_count += 1
            
    print(f"\nPreprocessing finished. Successfully processed {success_count}/{len(station_ids)} stations.")


if __name__ == "__main__":
    main()
