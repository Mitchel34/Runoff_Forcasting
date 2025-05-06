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
    Process data for a specific station, including resampling and interpolation.
    
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
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not nwm_dfs:
        print("No valid NWM data found")
        return False
        
    nwm_data = pd.concat(nwm_dfs, ignore_index=True)

    # --- Calculate Lead Time and Rename Columns ---
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
    nwm_data["lead_time"] = nwm_data["lead_time"].astype(int)

    # Filter for required lead times (1-18 hours)
    nwm_data = nwm_data[nwm_data["lead_time"].between(1, horizon)]
    print(f"Filtered NWM data to {len(nwm_data)} records for lead times 1-{horizon}h")
    
    # --- Load, Resample, and Interpolate USGS observations ---
    try:
        # Load with DateTime as index
        usgs_data = pd.read_csv(usgs_file, parse_dates=['DateTime'], index_col='DateTime')
        print(f"Loaded USGS columns: {usgs_data.columns.tolist()}") 
        
        # Identify the flow column 
        flow_col_name = 'USGSFlowValue' 

        if flow_col_name not in usgs_data.columns:
            print(f"Error: Expected flow column '{flow_col_name}' not found in USGS data {usgs_file}.")
            return False 

        # Select only the flow column for resampling/interpolation
        usgs_flow = usgs_data[[flow_col_name]].copy()

        # Resample to hourly frequency, taking the mean if multiple points fall in one hour
        usgs_flow_hourly = usgs_flow.resample('H').mean() 
        print(f"Resampled USGS to hourly: {len(usgs_flow_hourly)} records")
        
        # Interpolate missing values (e.g., linear)
        usgs_flow_hourly = usgs_flow_hourly.interpolate(method='linear') 
        print(f"Interpolated missing values using 'linear' method.")
        
        # Check for remaining NaNs after interpolation
        remaining_nans = usgs_flow_hourly[flow_col_name].isna().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain after interpolation. Consider backfill/forward fill.")
            usgs_flow_hourly = usgs_flow_hourly.fillna(method='bfill').fillna(method='ffill')
        
        # Rename the identified flow column to 'usgs_flow'
        usgs_flow_hourly = usgs_flow_hourly.rename(columns={flow_col_name: 'usgs_flow'})
        
        # Reset index to make 'DateTime' a column named 'datetime' for merging
        usgs_data_processed = usgs_flow_hourly.reset_index().rename(columns={'DateTime': 'datetime'})
        
        print(f"Processed USGS data with {len(usgs_data_processed)} hourly records after resampling/interpolation.")
        
    except Exception as e:
        print(f"Error loading or processing USGS data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # --- Align and merge data ---
    merged_data = align_and_merge(nwm_data, usgs_data_processed)
    if merged_data.empty:
        print("Merged data is empty. Check alignment logic and data content.")
        return False
    print(f"Merged data has {len(merged_data)} records after alignment.")

    # --- Pivot data to have lead times as columns --- 
    pivot_data = merged_data.pivot_table(
        index=merged_data.index, 
        columns='lead_time', 
        values=['nwm_flow', 'usgs_flow', 'error']
    )
    pivot_data.columns = [f'{val}_{lead}' for val, lead in pivot_data.columns]
    
    # Drop rows with ANY missing values AFTER pivoting
    initial_pivot_rows = len(pivot_data)
    pivot_data = pivot_data.dropna() 
    rows_dropped = initial_pivot_rows - len(pivot_data)
    print(f"Pivoted data shape before dropna: ({initial_pivot_rows}, {pivot_data.shape[1]})")
    print(f"Dropped {rows_dropped} rows due to NaNs after pivoting.")
    print(f"Final pivoted data shape: {pivot_data.shape}")

    if pivot_data.empty:
        print("Pivoted data is empty after dropna. Check for persistent gaps in NWM or USGS data.")
        return False

    # --- Feature Selection ---
    feature_cols = [col for col in pivot_data.columns if col.startswith('error_') or col.startswith('nwm_flow_')]
    target_cols = [f'error_{i}' for i in range(1, horizon + 1)]
    
    features = pivot_data[feature_cols]
    target = pivot_data[target_cols]

    # Create sequences for time series modeling using only features
    X, _, sequence_end_indices = create_sequences(features.values, window_size, 1) 
    if X is None:
        print("Sequence creation failed. Check data and parameters.")
        return False
    
    # Select the corresponding targets using the indices returned by create_sequences
    y = target.iloc[sequence_end_indices].values
    sequence_start_indices = pivot_data.index[sequence_end_indices] 

    print(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
    
    # Split data: Train (Apr 2021-Sep 2022), Test (Oct 2022-Apr 2023)
    split_date = pd.to_datetime('2022-10-01')
    X_train, y_train, X_test, y_test, train_indices, test_indices = temporal_split(
        X, y, sequence_start_indices, split_date
    )
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Train or test split resulted in zero samples. Check split date and data range.")
        return False
        
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- Scaling ---
    n_samples_train, n_timesteps_train, n_features_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features_train)
    
    n_samples_test, n_timesteps_test, n_features_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, n_features_test)

    y_train_reshaped = y_train 
    y_test_reshaped = y_test

    x_scaler = StandardScaler()
    y_scaler = StandardScaler() 

    x_scaler.fit(X_train_reshaped)
    y_scaler.fit(y_train_reshaped) 

    X_train_scaled_reshaped = x_scaler.transform(X_train_reshaped)
    X_test_scaled_reshaped = x_scaler.transform(X_test_reshaped)
    y_train_scaled = y_scaler.transform(y_train_reshaped)
    y_test_scaled = y_scaler.transform(y_test_reshaped)

    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps_train, n_features_train)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps_test, n_features_test)

    print(f"Scaled Train X: {X_train_scaled.shape}, Scaled Test X: {X_test_scaled.shape}")
    print(f"Scaled Train y: {y_train_scaled.shape}, Scaled Test y: {y_test_scaled.shape}")

    # Get original NWM and USGS data corresponding to the test sequences
    nwm_test_original = pivot_data[[f'nwm_flow_{i}' for i in range(1, horizon + 1)]].iloc[test_indices].values
    usgs_test_original = pivot_data[[f'usgs_flow_{i}' for i in range(1, horizon + 1)]].iloc[test_indices].values
    test_timestamps = pivot_data.index[test_indices].values 

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
    
    np.savez(train_file, X_train=X_train_scaled, y_train_scaled=y_train_scaled)
    np.savez(test_file, 
             X_test=X_test_scaled, 
             y_test_scaled=y_test_scaled, 
             nwm_test_original=nwm_test_original, 
             usgs_test_original=usgs_test_original,
             test_timestamps=test_timestamps) 
    
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
