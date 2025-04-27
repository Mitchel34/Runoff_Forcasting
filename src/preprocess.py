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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import create_sequences, temporal_split


def align_and_merge(nwm_data, usgs_data):
    """
    Align NWM forecasts with USGS observations based on timestamp.
    
    Args:
        nwm_data: DataFrame with NWM forecasts
        usgs_data: DataFrame with USGS observations
        
    Returns:
        DataFrame with aligned data
    """
    # Ensure datetime columns are in the correct format
    nwm_data["model_output_valid_time"] = pd.to_datetime(nwm_data["model_output_valid_time"])
    usgs_data["DateTime"] = pd.to_datetime(usgs_data["DateTime"])
    
    # Rename for clarity and consistency
    nwm_data = nwm_data.rename(columns={"streamflow": "nwm_flow", "model_output_valid_time": "valid_time"})
    usgs_data = usgs_data.rename(columns={"value": "usgs_flow", "DateTime": "datetime"})
    
    # Merge data based on the time the forecast is valid for
    merged = pd.merge(
        nwm_data[["valid_time", "nwm_flow", "lead_time"]], 
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
            df = pd.read_csv(file)
            nwm_dfs.append(df)
            print(f"Loaded {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not nwm_dfs:
        print("No valid NWM data found")
        return False
        
    nwm_data = pd.concat(nwm_dfs, ignore_index=True)
    
    # Load USGS observations
    try:
        usgs_data = pd.read_csv(usgs_file)
        print(f"Loaded USGS data with {len(usgs_data)} records")
    except Exception as e:
        print(f"Error loading USGS data: {e}")
        return False
    
    # Align and merge data
    merged_data = align_and_merge(nwm_data, usgs_data)
    print(f"Merged data has {len(merged_data)} records")
    
    # Create sequences for time series modeling
    X, y = create_sequences(merged_data, window_size, horizon)
    print(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
    
    # Add a channel dimension for the ML models
    X = X[..., np.newaxis]
    
    # Split data: Train/Val (Apr 2021-Sep 2022), Test (Oct 2022-Apr 2023)
    split_date = pd.to_datetime('2022-10-01')
    X_train, y_train, X_test, y_test = temporal_split(X, y, merged_data, split_date)
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Save processed data
    train_file = os.path.join(output_dir, 'train', f"{station_id}.npz")
    test_file = os.path.join(output_dir, 'test', f"{station_id}.npz")
    
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    np.savez(train_file, X=X_train, y=y_train)
    np.savez(test_file, X=X_test, y=y_test)
    
    print(f"Data saved to {train_file} and {test_file}")
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
    
    # Process both stations
    station_ids = ["20380357", "21609641"]
    for station_id in station_ids:
        process_station(
            station_id, 
            args.data_dir, 
            args.output_dir,
            args.window_size,
            args.horizon
        )


if __name__ == "__main__":
    main()
