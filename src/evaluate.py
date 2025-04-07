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
    metrics = {
        'NWM': {
            'CC': correlation_coefficient(obs, nwm_pred),
            'RMSE': rmse(obs, nwm_pred),
            'PBIAS': pbias(obs, nwm_pred),
            'NSE': nse(obs, nwm_pred),
            'MAE': mean_absolute_error(obs, nwm_pred)
        },
        'ML_Corrected': {
            'CC': correlation_coefficient(obs, ml_pred),
            'RMSE': rmse(obs, ml_pred),
            'PBIAS': pbias(obs, ml_pred),
            'NSE': nse(obs, ml_pred),
            'MAE': mean_absolute_error(obs, ml_pred)
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
    
    # Merge data
    eval_df = pd.merge(
        pred_df,
        obs_df,
        on=['datetime', 'station_id'],
        how='inner'
    )
    
    # Group by station for station-wise evaluation
    all_metrics = []
    
    for station_id, group in eval_df.groupby('station_id'):
        obs = group['runoff_observed'].values
        nwm_pred = group['runoff_nwm'].values
        ml_pred = group['runoff_predicted'].values
        
        metrics = evaluate_model(obs, nwm_pred, ml_pred)
        all_metrics.append(metrics)
        
        print(f"\nStation ID: {station_id}")
        print_evaluation_metrics(metrics)
    
    # Plot metrics
    plot_metrics_boxplot(all_metrics)
    
    # Calculate and print aggregate metrics
    print("\nAggregate Metrics (All Stations):")
    obs_all = eval_df['runoff_observed'].values
    nwm_pred_all = eval_df['runoff_nwm'].values
    ml_pred_all = eval_df['runoff_predicted'].values
    
    metrics_all = evaluate_model(obs_all, nwm_pred_all, ml_pred_all)
    print_evaluation_metrics(metrics_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--observations', type=str, required=True, help='Path to observations CSV file')
    args = parser.parse_args()
    
    main(args.predictions, args.observations)
