"""
Model evaluation module with hydrological metrics.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

def correlation_coefficient(obs, pred):
    """
    Calculate Pearson correlation coefficient.
    
    Parameters:
    -----------
    obs : numpy.ndarray
        Observed values
    pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    cc : float
        Correlation coefficient
    """
    return np.corrcoef(obs, pred)[0, 1]

def rmse(obs, pred):
    """
    Calculate Root Mean Square Error.
    
    Parameters:
    -----------
    obs : numpy.ndarray
        Observed values
    pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    rmse : float
        Root Mean Square Error
    """
    return np.sqrt(mean_squared_error(obs, pred))

def pbias(obs, pred):
    """
    Calculate Percent Bias.
    
    Parameters:
    -----------
    obs : numpy.ndarray
        Observed values
    pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    pbias : float
        Percent Bias
    """
    return 100 * np.sum(pred - obs) / np.sum(obs)

def nse(obs, pred):
    """
    Calculate Nash-Sutcliffe Efficiency.
    
    Parameters:
    -----------
    obs : numpy.ndarray
        Observed values
    pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    nse : float
        Nash-Sutcliffe Efficiency
    """
    return 1 - (np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

def evaluate_model(obs, nwm_pred, ml_pred):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    obs : numpy.ndarray
        Observed values
    nwm_pred : numpy.ndarray
        NWM predictions
    ml_pred : numpy.ndarray
        Machine learning corrected predictions
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Filter out NaN values from all arrays
    valid_indices = ~(np.isnan(obs) | np.isnan(nwm_pred) | np.isnan(ml_pred))
    
    if np.sum(valid_indices) == 0:
        print("WARNING: No valid data points (all NaN) for evaluation")
        return {
            'NWM': {'CC': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'NSE': np.nan, 'MAE': np.nan},
            'ML_Corrected': {'CC': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'NSE': np.nan, 'MAE': np.nan}
        }
    
    # Use only valid data for evaluation
    obs_valid = obs[valid_indices]
    nwm_pred_valid = nwm_pred[valid_indices]
    ml_pred_valid = ml_pred[valid_indices]
    
    print(f"Using {len(obs_valid)} valid data points out of {len(obs)} total")
    
    metrics = {
        'NWM': {
            'CC': correlation_coefficient(obs_valid, nwm_pred_valid),
            'RMSE': rmse(obs_valid, nwm_pred_valid),
            'PBIAS': pbias(obs_valid, nwm_pred_valid),
            'NSE': nse(obs_valid, nwm_pred_valid),
            'MAE': mean_absolute_error(obs_valid, nwm_pred_valid)
        },
        'ML_Corrected': {
            'CC': correlation_coefficient(obs_valid, ml_pred_valid),
            'RMSE': rmse(obs_valid, ml_pred_valid),
            'PBIAS': pbias(obs_valid, ml_pred_valid),
            'NSE': nse(obs_valid, ml_pred_valid),
            'MAE': mean_absolute_error(obs_valid, ml_pred_valid)
        }
    }
    
    return metrics

def print_evaluation_metrics(metrics):
    """
    Print evaluation metrics in a formatted table.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of evaluation metrics
    """
    print("Evaluation Metrics:")
    print("-" * 50)
    print(f"{'Metric':<10}{'NWM':<15}{'ML Corrected':<15}{'Improvement (%)':<15}")
    print("-" * 50)
    
    for metric in ['CC', 'RMSE', 'PBIAS', 'NSE', 'MAE']:
        nwm_value = metrics['NWM'][metric]
        ml_value = metrics['ML_Corrected'][metric]
        
        # Calculate improvement percentage (higher is better for CC and NSE, lower is better for others)
        if metric in ['CC', 'NSE']:
            improvement = ((ml_value - nwm_value) / abs(nwm_value)) * 100 if nwm_value != 0 else np.inf
        else:
            improvement = ((nwm_value - ml_value) / nwm_value) * 100 if nwm_value != 0 else np.inf
        
        print(f"{metric:<10}{nwm_value:<15.4f}{ml_value:<15.4f}{improvement:<15.2f}")
    
    print("-" * 50)

def plot_metrics_boxplot(all_metrics, by_watershed=False):
    """
    Create box plots of evaluation metrics.
    
    Parameters:
    -----------
    all_metrics : list of dict
        List of metric dictionaries for different stations/time periods
    by_watershed : bool
        Whether to group by watershed
        
    Returns:
    --------
    None, saves figure to disk
    """
    # Convert metrics to DataFrame
    metrics_df = []
    
    for i, metrics in enumerate(all_metrics):
        for model_type in ['NWM', 'ML_Corrected']:
            for metric_name, value in metrics[model_type].items():
                metrics_df.append({
                    'Station': i,
                    'Model': model_type,
                    'Metric': metric_name,
                    'Value': value
                })
    
    metrics_df = pd.DataFrame(metrics_df)
    
    # Create directory for figures
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot boxplots for each metric
    metrics_to_plot = ['CC', 'RMSE', 'PBIAS', 'NSE']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        metric_df = metrics_df[metrics_df['Metric'] == metric]
        
        sns.boxplot(x='Metric', y='Value', hue='Model', data=metric_df, ax=axes[i])
        axes[i].set_title(f"{metric} Distribution")
        axes[i].set_xlabel('')
        
        # Set y-axis limits based on metric
        if metric == 'NSE':
            axes[i].set_ylim(-0.5, 1.0)
        elif metric == 'CC':
            axes[i].set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'metrics_boxplots.png'))
    plt.close()

def main(predictions_file, observations_file):
    """
    Main evaluation function.
    
    Parameters:
    -----------
    predictions_file : str
        Path to predictions CSV file
    observations_file : str
        Path to observations CSV file
    """
    # Load data
    print(f"Loading predictions from {predictions_file}")
    pred_df = pd.read_csv(predictions_file)
    
    print(f"Loading observations from {observations_file}")
    obs_df = pd.read_csv(observations_file)
    
    # Check what columns are available in the observation data
    print(f"Columns in observations file: {list(obs_df.columns)}")
    
    # Determine the name of the column containing observed runoff values
    observed_runoff_column = None
    for possible_col in ['runoff_observed', 'runoff_usgs', 'observed_runoff']:
        if possible_col in obs_df.columns:
            observed_runoff_column = possible_col
            print(f"Found observed runoff data in column: '{observed_runoff_column}'")
            break
    
    if observed_runoff_column is None:
        print("ERROR: Could not find a column with observed runoff data.")
        print("Expected one of: 'runoff_observed', 'runoff_usgs', 'observed_runoff'")
        print("Available columns:", list(obs_df.columns))
        return
    
    # Rename column for consistency with the rest of the code
    obs_df = obs_df.rename(columns={observed_runoff_column: 'runoff_observed'})
    
    # Merge data
    try:
        eval_df = pd.merge(
            pred_df,
            obs_df[['datetime', 'station_id', 'runoff_observed']],
            on=['datetime', 'station_id'],
            how='inner'
        )
        print(f"Merged data contains {len(eval_df)} rows")
    except KeyError as e:
        print(f"ERROR during merge: {e}")
        print("Make sure both files have 'datetime' and 'station_id' columns")
        print("Predictions columns:", list(pred_df.columns))
        print("Observations columns:", list(obs_df.columns))
        return
    
    # Check if we have data after merging
    if len(eval_df) == 0:
        print("ERROR: No matching data found after merging predictions and observations.")
        print("Check that the datetime and station_id values match between files.")
        return
    
    # Group by station for station-wise evaluation
    all_metrics = []
    
    for station_id, group in eval_df.groupby('station_id'):
        obs = group['runoff_observed'].values
        
        # Check if 'runoff_nwm' exists in the data, otherwise use a fallback
        if 'runoff_nwm' in group.columns:
            nwm_pred = group['runoff_nwm'].values
        else:
            print(f"Warning: 'runoff_nwm' column not found for station {station_id}. Using zeros as placeholder.")
            nwm_pred = np.zeros_like(obs)
        
        # Check if 'runoff_predicted' exists in the data
        if 'runoff_predicted' in group.columns:
            ml_pred = group['runoff_predicted'].values
        else:
            print(f"Error: 'runoff_predicted' column not found for station {station_id}. Cannot evaluate ML model.")
            continue
        
        # Check how many NaN values we have
        nan_count = np.sum(np.isnan(obs) | np.isnan(nwm_pred) | np.isnan(ml_pred))
        total_count = len(obs)
        
        if nan_count > 0:
            print(f"Station {station_id}: Found {nan_count} NaN values out of {total_count} entries ({nan_count/total_count:.2%})")
            
            # If too many NaN values, skip this station
            if nan_count/total_count > 0.9:  # 90% threshold
                print(f"Skipping station {station_id} due to too many NaN values")
                continue
        
        metrics = evaluate_model(obs, nwm_pred, ml_pred)
        all_metrics.append(metrics)
        
        print(f"\nStation ID: {station_id}")
        print_evaluation_metrics(metrics)
    
    # Check if we have any valid metrics
    if not all_metrics:
        print("No valid metrics could be calculated for any station.")
        return
    
    # Plot metrics
    plot_metrics_boxplot(all_metrics)
    
    # Calculate and print aggregate metrics
    print("\nAggregate Metrics (All Stations):")
    
    # Filter out NaN values for aggregate metrics
    valid_rows = ~(
        np.isnan(eval_df['runoff_observed']) | 
        np.isnan(eval_df['runoff_predicted']) |
        (np.isnan(eval_df['runoff_nwm']) if 'runoff_nwm' in eval_df.columns else False)
    )
    
    if np.sum(valid_rows) == 0:
        print("WARNING: No valid data points for aggregate metrics")
        return
    
    filtered_df = eval_df[valid_rows]
    
    obs_all = filtered_df['runoff_observed'].values
    ml_pred_all = filtered_df['runoff_predicted'].values
    
    if 'runoff_nwm' in filtered_df.columns:
        nwm_pred_all = filtered_df['runoff_nwm'].values
    else:
        nwm_pred_all = np.zeros_like(obs_all)
    
    print(f"Using {len(filtered_df)} valid data points out of {len(eval_df)} total for aggregate metrics")
    
    metrics_all = evaluate_model(obs_all, nwm_pred_all, ml_pred_all)
    print_evaluation_metrics(metrics_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--observations', type=str, required=True, help='Path to observations CSV file')
    args = parser.parse_args()
    
    main(args.predictions, args.observations)
