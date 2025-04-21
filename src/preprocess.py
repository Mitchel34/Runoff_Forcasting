import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse

# Add the project root to the Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def load_data(station_dir):
    # Load USGS observations
    obs_file = glob.glob(os.path.join(station_dir, "*Strt*.csv"))[0]
    obs_df = pd.read_csv(obs_file, parse_dates=['DateTime'])
    obs_df = obs_df[['DateTime', 'USGSFlowValue']].dropna().set_index('DateTime')
    # Load NWM forecasts
    station = os.path.basename(station_dir)
    frc_files = glob.glob(os.path.join(station_dir, f"streamflow_{station}_*.csv"))
    df_list = []
    for f in frc_files:
        df = pd.read_csv(f, parse_dates=['model_output_valid_time'])
        df = df[['model_output_valid_time', 'streamflow_value']]
        df = df.rename(columns={'model_output_valid_time': 'DateTime', 'streamflow_value': 'NWM'}).dropna()
        df_list.append(df)
    frc_df = pd.concat(df_list).set_index('DateTime').sort_index()
    return obs_df, frc_df


def prepare_samples(obs_df, frc_df, window=24, horizon=18):
    df = obs_df.join(frc_df, how='inner')
    data = df[['USGSFlowValue', 'NWM']].values
    X, y, times = [], [], []
    for i in range(window, len(data) - horizon):
        obs_seq = data[i - window:i, 0]
        frc_seq = data[i:i + horizon, 1]
        err_seq = data[i:i + horizon, 0] - frc_seq
        sample = np.concatenate([obs_seq, frc_seq])
        X.append(sample)
        y.append(err_seq)
        times.append(df.index[i])
    return np.array(X), np.array(y), np.array(times)


def split_save(X, y, times, data_dir, station, train_end, val_ratio=0.8):
    # Directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # ensure times array is datetime64 for comparison
    times = pd.DatetimeIndex(times)
    # Train/val split by date
    idx_trval = times < train_end
    X_trval, y_trval = X[idx_trval], y[idx_trval]
    n_tr = int(len(X_trval) * val_ratio)
    # Save
    np.savez(os.path.join(train_dir, f'{station}.npz'), X=X_trval[:n_tr], y=y_trval[:n_tr])
    np.savez(os.path.join(val_dir, f'{station}.npz'), X=X_trval[n_tr:], y=y_trval[n_tr:])
    # Test
    idx_test = times >= train_end
    np.savez(os.path.join(test_dir, f'{station}.npz'), X=X[idx_test], y=y[idx_test])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', required=True, help='Station ID directory name')
    parser.add_argument('--data-dir', default='.', help='Root directory containing station folders')
    parser.add_argument('--out-dir', default=os.path.join('data', 'processed'), help='Output base directory for processed data')
    args = parser.parse_args()
    station_dir = os.path.join(args.data_dir, args.station)
    obs_df, frc_df = load_data(station_dir)
    X, y, times = prepare_samples(obs_df, frc_df)
    split_save(X, y, times, args.out_dir, args.station, pd.to_datetime('2022-10-01'))


if __name__ == '__main__':
    main()
