"""
Visualization module for runoff forecasting results.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_runoff_time_series(df, station_id=None, start_date=None, end_date=None):
    """
    Plot time series of observed vs. predicted runoff.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with runoff data
    station_id : str, optional
        Station ID to plot
    start_date : str, optional
        Start date for plotting
    end_date : str, optional
        End date for plotting
        
    Returns:
    --------
    None, saves figure to disk
    """
    # Filter data if needed
    plot_df = df.copy()
    
    if 'datetime' in plot_df.columns and not pd.api.types.is_datetime64_any_dtype(plot_df['datetime']):
        plot_df['datetime'] = pd.to_datetime(plot_df['datetime'])
    
    if station_id is not None:
        plot_df = plot_df[plot_df['station_id'] == station_id]
    
    if start_date is not None:
        plot_df = plot_df[plot_df['datetime'] >= start_date]
    
    if end_date is not None:
        plot_df = plot_df[plot_df['datetime'] <= end_date]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot observed, NWM, and corrected runoff
    ax.plot(plot_df['datetime'], plot_df['runoff_observed'], 'k-', label='Observed')
    ax.plot(plot_df['datetime'], plot_df['runoff_nwm'], 'b-', label='NWM')
    ax.plot(plot_df['datetime'], plot_df['runoff_predicted'], 'r-', label='ML Corrected')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Runoff (cms)')
    title = 'Runoff Time Series'
    if station_id is not None:
        title += f' for Station {station_id}'
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    figures_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'figures'
    )
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = 'runoff_time_series'
    if station_id is not None:
        filename += f'_{station_id}'
    plt.savefig(os.path.join(figures_dir, f'{filename}.png'))
    plt.close()

def plot_runoff_boxplots(df):
    """
    Create box plots comparing observed, NWM, and ML-corrected runoff.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with runoff data
        
    Returns:
    --------
    None, saves figure to disk
    """
    # Prepare data for seaborn
    plot_data = []
    
    for _, row in df.iterrows():
        plot_data.append({
            'Station': row['station_id'],
            'Type': 'Observed',
            'Runoff': row['runoff_observed']
        })
        plot_data.append({
            'Station': row['station_id'],
            'Type': 'NWM',
            'Runoff': row['runoff_nwm']
        })
        plot_data.append({
            'Station': row['station_id'],
            'Type': 'ML Corrected',
            'Runoff': row['runoff_predicted']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Station', y='Runoff', hue='Type', data=plot_df)
    
    # Set labels and title
    ax.set_xlabel('Station ID')
    ax.set_ylabel('Runoff (cms)')
    ax.set_title('Comparison of Runoff Distributions by Station')
    
    # Rotate x-axis labels if many stations
    if len(df['station_id'].unique()) > 5:
        plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    figures_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'figures'
    )
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'runoff_boxplots.png'))
    plt.close()

def plot_scatter_comparison(df):
    """
    Create scatter plots comparing observed vs. predicted runoff.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with runoff data
        
    Returns:
    --------
    None, saves figure to disk
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot NWM vs. Observed
    ax1.scatter(df['runoff_observed'], df['runoff_nwm'], alpha=0.5)
    max_val = max(df['runoff_observed'].max(), df['runoff_nwm'].max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'k--')
    ax1.set_xlabel('Observed Runoff (cms)')
    ax1.set_ylabel('NWM Runoff (cms)')
    ax1.set_title('NWM vs. Observed Runoff')
    ax1.set_aspect('equal')
    
    # Plot ML Corrected vs. Observed
    ax2.scatter(df['runoff_observed'], df['runoff_predicted'], alpha=0.5)
    max_val = max(df['runoff_observed'].max(), df['runoff_predicted'].max()) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--')
    ax2.set_xlabel('Observed Runoff (cms)')
    ax2.set_ylabel('ML Corrected Runoff (cms)')
    ax2.set_title('ML Corrected vs. Observed Runoff')
    ax2.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    figures_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'figures'
    )
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'runoff_scatter_comparison.png'))
    plt.close()

def plot_error_histograms(df):
    """
    Create histograms of prediction errors.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with runoff data
        
    Returns:
    --------
    None, saves figure to disk
    """
    # Calculate errors
    df['nwm_error'] = df['runoff_nwm'] - df['runoff_observed']
    df['ml_error'] = df['runoff_predicted'] - df['runoff_observed']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot NWM error histogram
    sns.histplot(df['nwm_error'], kde=True, ax=ax1)
    ax1.axvline(x=0, color='k', linestyle='--')
    ax1.set_xlabel('Error (cms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('NWM Error Distribution')
    
    # Plot ML error histogram
    sns.histplot(df['ml_error'], kde=True, ax=ax2)
    ax2.axvline(x=0, color='k', linestyle='--')
    ax2.set_xlabel('Error (cms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('ML Corrected Error Distribution')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    figures_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'figures'
    )
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'error_histograms.png'))
    plt.close()

def merge_data(predictions_file, observations_file):
    """
    Merge predictions and observations data.
    
    Parameters:
    -----------
    predictions_file : str
        Path to predictions CSV file
    observations_file : str
        Path to observations CSV file
        
    Returns:
    --------
    merged_df : pandas.DataFrame
        Merged DataFrame with predictions and observations
    """
    print(f"Loading predictions from {predictions_file}")
    pred_df = pd.read_csv(predictions_file)
    
    print(f"Loading observations from {observations_file}")
    obs_df = pd.read_csv(observations_file)
    
    print(f"Prediction columns: {list(pred_df.columns)}")
    print(f"Observation columns: {list(obs_df.columns)}")
    
    # Identify the column with observed values
    observed_runoff_column = None
    for possible_col in ['runoff_observed', 'runoff_usgs', 'observed_runoff']:
        if possible_col in obs_df.columns:
            observed_runoff_column = possible_col
            print(f"Found observed runoff data in column: '{observed_runoff_column}'")
            break
    
    if observed_runoff_column is None:
        raise ValueError("Could not find a column with observed runoff data. Expected one of: 'runoff_observed', 'runoff_usgs', 'observed_runoff'")
    
    # Ensure datetime is in correct format
    for df in [pred_df, obs_df]:
        if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Merge the data - use different approach to avoid column name conflicts
    try:
        # Rename columns before merge to avoid conflicts
        obs_subset = obs_df[['datetime', 'station_id', observed_runoff_column]].copy()
        obs_subset.rename(columns={observed_runoff_column: 'runoff_observed'}, inplace=True)
        
        # Check if we need to bring in runoff_nwm from observations
        if 'runoff_nwm' in obs_df.columns and 'runoff_nwm' not in pred_df.columns:
            obs_subset['runoff_nwm'] = obs_df['runoff_nwm']
        
        # Merge with renamed columns
        merged_df = pd.merge(
            pred_df,
            obs_subset,
            on=['datetime', 'station_id'],
            how='inner'
        )
        
        # If runoff_nwm exists in both datasets, prefer the one from predictions
        if 'runoff_nwm_x' in merged_df.columns:
            merged_df['runoff_nwm'] = merged_df['runoff_nwm_x']
            merged_df.drop(['runoff_nwm_x', 'runoff_nwm_y'], axis=1, inplace=True, errors='ignore')
        elif 'runoff_nwm' not in merged_df.columns and 'runoff_nwm_y' in merged_df.columns:
            merged_df['runoff_nwm'] = merged_df['runoff_nwm_y']
            merged_df.drop('runoff_nwm_y', axis=1, inplace=True, errors='ignore')
        
        print(f"Merged data contains {len(merged_df)} rows")
        print(f"Merged columns: {list(merged_df.columns)}")
        
        return merged_df
    except KeyError as e:
        raise KeyError(f"Error during merge: {e}. Make sure both files have 'datetime' and 'station_id' columns")

def main(predictions_file, observations_file, station_id=None, start_date=None, end_date=None):
    """
    Main visualization function.
    
    Parameters:
    -----------
    predictions_file : str
        Path to predictions CSV file
    observations_file : str
        Path to observations CSV file
    station_id : str, optional
        Station ID to plot
    start_date : str, optional
        Start date for plotting
    end_date : str, optional
        End date for plotting
    """
    # Merge prediction and observation data
    try:
        df = merge_data(predictions_file, observations_file)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return
    
    # Create reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Identify which columns are actually available
    available_columns = []
    for col in ['runoff_observed', 'runoff_predicted', 'runoff_nwm']:
        if col in df.columns:
            available_columns.append(col)
        else:
            print(f"Warning: Column '{col}' not found in the merged data")
    
    if not available_columns:
        print("ERROR: No required runoff columns available in the data")
        return
    
    # Filter out rows with NaN values
    print(f"Data has {df.isna().sum().sum()} NaN values across all columns")
    print(f"Checking for NaN values in columns: {available_columns}")
    df_clean = df.dropna(subset=available_columns)
    print(f"Using {len(df_clean)} rows after removing NaN values (from {len(df)} total)")
    
    # Create a smaller sample for testing and development
    if len(df_clean) > 10000:
        print(f"NOTE: Dataset is very large ({len(df_clean)} rows). Creating a sample of 10,000 rows for visualization.")
        df_small_sample = df_clean.sample(n=10000, random_state=42)
        
        # Make a small time series plot with this sample
        sample_station = df_small_sample['station_id'].iloc[0]
        station_sample = df_small_sample[df_small_sample['station_id'] == sample_station].head(100)
        print(f"Creating sample time series plot for station {sample_station} (first 100 rows)")
        
        # Simple time series plot
        plt.figure(figsize=(15, 6))
        plt.plot(station_sample['datetime'], station_sample['runoff_predicted'], 'r-', label='ML Predicted')
        
        if 'runoff_observed' in station_sample.columns:
            plt.plot(station_sample['datetime'], station_sample['runoff_observed'], 'k-', label='Observed')
        
        if 'runoff_nwm' in station_sample.columns:
            plt.plot(station_sample['datetime'], station_sample['runoff_nwm'], 'b-', label='NWM')
        
        plt.xlabel('Datetime')
        plt.ylabel('Runoff')
        plt.title(f'Sample Time Series for Station {sample_station}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(figures_dir, 'sample_time_series.png'))
        plt.close()
        print(f"Created sample time series plot: {os.path.join(figures_dir, 'sample_time_series.png')}")
    
    # Skip visualizations that require missing columns
    if all(col in available_columns for col in ['runoff_observed', 'runoff_nwm', 'runoff_predicted']):
        # Generate visualizations
        print("Generating time series plot...")
        plot_runoff_time_series(df_clean, station_id, start_date, end_date)
        
        # Sample data for box plots and scatter plots to avoid memory issues
        if len(df_clean) > 100000:
            print(f"Sampling {100000} rows for box plots and scatter plots (from {len(df_clean)} total)")
            df_sample = df_clean.sample(n=100000, random_state=42)
        else:
            df_sample = df_clean
        
        print("Generating runoff box plots...")
        plot_runoff_boxplots(df_sample)
        
        print("Generating scatter comparison plot...")
        plot_scatter_comparison(df_sample)
        
        print("Generating error histograms...")
        plot_error_histograms(df_sample)
    else:
        print("WARNING: Cannot generate all visualizations because some required columns are missing")
        print(f"Available columns: {list(df.columns)}")
        
        # Create scatterplot of predicted vs observed if those columns exist
        if 'runoff_observed' in df.columns and 'runoff_predicted' in df.columns:
            print("Generating scatter plot of predicted vs observed...")
            
            # Sample data to avoid memory issues
            if len(df_clean) > 100000:
                df_sample = df_clean.sample(n=100000, random_state=42)
            else:
                df_sample = df_clean
            
            plt.figure(figsize=(10, 10))
            plt.scatter(df_sample['runoff_observed'], df_sample['runoff_predicted'], alpha=0.5)
            
            # Determine max value for plot limits
            max_val = max(
                df_sample['runoff_observed'].max(),
                df_sample['runoff_predicted'].max()
            ) * 1.1
            
            # Add perfect prediction line
            plt.plot([0, max_val], [0, max_val], 'k--')
            plt.xlim([0, max_val])
            plt.ylim([0, max_val])
            
            plt.xlabel('Observed Runoff (cms)')
            plt.ylabel('ML Corrected Runoff (cms)')
            plt.title('ML Corrected vs. Observed Runoff')
            plt.tight_layout()
            
            plt.savefig(os.path.join(figures_dir, 'observed_vs_predicted.png'))
            plt.close()
            print(f"Saved scatter plot to {os.path.join(figures_dir, 'observed_vs_predicted.png')}")
            
            # Create error histogram
            df_sample['ml_error'] = df_sample['runoff_predicted'] - df_sample['runoff_observed']
            
            plt.figure(figsize=(10, 6))
            sns.histplot(df_sample['ml_error'], kde=True)
            plt.axvline(x=0, color='k', linestyle='--')
            plt.xlabel('Error (cms)')
            plt.ylabel('Frequency')
            plt.title('ML Corrected Error Distribution')
            plt.tight_layout()
            
            plt.savefig(os.path.join(figures_dir, 'ml_error_histogram.png'))
            plt.close()
            print(f"Saved error histogram to {os.path.join(figures_dir, 'ml_error_histogram.png')}")
    
    print("Visualization completed!")
    print(f"Figures saved to {figures_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize runoff prediction results')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--observations', type=str, required=True, help='Path to observations CSV file')
    parser.add_argument('--station', type=str, help='Station ID to plot')
    parser.add_argument('--start', type=str, help='Start date for plotting (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for plotting (YYYY-MM-DD)')
    args = parser.parse_args()
    
    main(args.predictions, args.observations, args.station, args.start, args.end)
