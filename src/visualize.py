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

def main(results_file, station_id=None, start_date=None, end_date=None):
    """
    Main visualization function.
    
    Parameters:
    -----------
    results_file : str
        Path to results CSV file
    station_id : str, optional
        Station ID to plot
    start_date : str, optional
        Start date for plotting
    end_date : str, optional
        End date for plotting
    """
    # Load data
    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)
    
    # Generate visualizations
    print("Generating time series plot...")
    plot_runoff_time_series(df, station_id, start_date, end_date)
    
    print("Generating runoff box plots...")
    plot_runoff_boxplots(df)
    
    print("Generating scatter comparison plot...")
    plot_scatter_comparison(df)
    
    print("Generating error histograms...")
    plot_error_histograms(df)
    
    print("Visualization completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize runoff prediction results')
    parser.add_argument('--results', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--station', type=str, help='Station ID to plot')
    parser.add_argument('--start', type=str, help='Start date for plotting (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for plotting (YYYY-MM-DD)')
    args = parser.parse_args()
    
    main(args.results, args.station, args.start, args.end)
