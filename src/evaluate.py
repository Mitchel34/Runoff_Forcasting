"""
Evaluate trained models and generate visualizations.
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import correlation_coefficient, rmse, pbias, nse, calculate_metrics_over_windows


def extract_true_values(X_test, y_test):
    """
    Extract true observed values, NWM forecasts, and errors from test data.
    
    Args:
        X_test: Input test sequences
        y_test: Target error sequences
        
    Returns:
        observed_flow: True observed flow values
        nwm_forecasts: NWM forecasted flow values
        true_errors: True error values between NWM and observed
    """
    # Extract NWM forecasts (last 18 elements of each sequence)
    # X shape is (samples, 42, 1) where 42 = window_size (24) + horizon (18)
    nwm_forecasts = X_test[:, -y_test.shape[1]:, 0]
    
    # True errors are directly in y_test
    true_errors = y_test
    
    # Observed flow = NWM forecast + true error
    observed_flow = nwm_forecasts + true_errors
    
    return observed_flow, nwm_forecasts, true_errors


def create_metric_box_plots(observed, nwm_forecasts, corrected_forecasts, station_id, output_dir):
    """
    Generate box plots for evaluation metrics (CC, RMSE, PBIAS, NSE).
    
    Args:
        observed: Observed flow values
        nwm_forecasts: Raw NWM forecasts
        corrected_forecasts: Corrected forecasts from model
        station_id: Station identifier
        output_dir: Directory to save plots
    """
    # Initialize metric arrays
    lead_times = list(range(1, 19))  # 1-18 hours
    metrics = {
        'CC': {'raw': [], 'corr': []},
        'RMSE': {'raw': [], 'corr': []},
        'PBIAS': {'raw': [], 'corr': []},
        'NSE': {'raw': [], 'corr': []}
    }
    
    # Calculate metrics for each lead time
    for i, lead_time in enumerate(lead_times):
        # Extract data for this lead time
        obs = observed[:, i]
        nwm = nwm_forecasts[:, i]
        corr = corrected_forecasts[:, i]
        
        # Calculate metrics over sliding windows
        window_size = min(30, len(obs) // 10)  # Use at least 10 windows
        
        raw_metrics = calculate_metrics_over_windows(obs, nwm, window_size)
        corr_metrics = calculate_metrics_over_windows(obs, corr, window_size)
        
        # Store all window metrics for box plots
        for metric_name in metrics:
            metrics[metric_name]['raw'].append(raw_metrics[metric_name])
            metrics[metric_name]['corr'].append(corr_metrics[metric_name])
    
    # Create metrics table
    metrics_df = pd.DataFrame(index=lead_times)
    metrics_df.index.name = 'lead_time'
    
    for metric_name in metrics:
        for i, lead_time in enumerate(lead_times):
            # Calculate mean of the windows for the table
            metrics_df.loc[lead_time, f'{metric_name}_raw'] = np.mean(metrics[metric_name]['raw'][i])
            metrics_df.loc[lead_time, f'{metric_name}_corr'] = np.mean(metrics[metric_name]['corr'][i])
    
    # Save metrics to CSV
    metrics_dir = os.path.join(project_root, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_dir, f'{station_id}_metrics.csv'))
    
    # Create box plots for each metric
    for metric_name in metrics:
        plt.figure(figsize=(15, 8))
        
        # Prepare data for box plots
        raw_data = [metrics[metric_name]['raw'][i] for i in range(len(lead_times))]
        corr_data = [metrics[metric_name]['corr'][i] for i in range(len(lead_times))]
        
        # Box plot positions
        positions = []
        for i in range(len(lead_times)):
            positions.extend([i*3, i*3+1])
        
        # Create box plots
        bplot = plt.boxplot(
            raw_data + corr_data,
            positions=positions,
            patch_artist=True,
            widths=0.6
        )
        
        # Set box colors
        for i, box in enumerate(bplot['boxes']):
            if i < len(lead_times):
                box.set_facecolor('lightblue')  # Raw NWM
            else:
                box.set_facecolor('lightgreen')  # Corrected
        
        # Set x-ticks at the center of each lead time group
        plt.xticks([i*3+0.5 for i in range(len(lead_times))], lead_times)
        plt.xlabel('Lead Time (hours)')
        plt.ylabel(metric_name)
        plt.title(f'Station {station_id}: {metric_name} by Lead Time')
        
        # Add legend
        plt.legend(
            [bplot['boxes'][0], bplot['boxes'][len(lead_times)]],
            ['Raw NWM', 'Corrected'],
            loc='best'
        )
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{station_id}_{metric_name}_boxplot.png'))
        plt.close()
    
    print(f"Generated metric box plots for station {station_id}")


def generate_runoff_box_plots(observed, nwm_forecasts, corrected_forecasts, station_id, output_dir):
    """
    Generate box plots of observed, NWM, and corrected runoff for each lead time.
    
    Args:
        observed: Observed flow values
        nwm_forecasts: Raw NWM forecasts
        corrected_forecasts: Corrected forecasts from model
        station_id: Station identifier
        output_dir: Directory to save plots
    """
    lead_times = list(range(1, 19))  # 1-18 hours
    
    plt.figure(figsize=(20, 12))
    
    # Prepare data for box plots
    data_to_plot = []
    for i in range(len(lead_times)):
        data_to_plot.extend([
            observed[:, i],
            nwm_forecasts[:, i],
            corrected_forecasts[:, i]
        ])
    
    # Box plot positions
    positions = []
    for i in range(len(lead_times)):
        positions.extend([i*4, i*4+1, i*4+2])
    
    # Create box plots
    bplot = plt.boxplot(
        data_to_plot,
        positions=positions,
        patch_artist=True,
        widths=0.6
    )
    
    # Set box colors
    for i, box in enumerate(bplot['boxes']):
        if i % 3 == 0:
            box.set_facecolor('lightblue')  # Observed
        elif i % 3 == 1:
            box.set_facecolor('lightcoral')  # Raw NWM
        else:
            box.set_facecolor('lightgreen')  # Corrected
    
    # Set x-ticks at the center of each lead time group
    plt.xticks([i*4+1 for i in range(len(lead_times))], lead_times)
    plt.xlabel('Lead Time (hours)')
    plt.ylabel('Runoff (cms)')
    plt.title(f'Station {station_id}: Runoff Distribution by Lead Time')
    
    # Add legend
    plt.legend(
        [bplot['boxes'][0], bplot['boxes'][1], bplot['boxes'][2]],
        ['Observed (USGS)', 'Raw NWM', 'Corrected'],
        loc='best'
    )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{station_id}_runoff_boxplot.png'))
    plt.close()
    
    print(f"Generated runoff box plot for station {station_id}")


def evaluate_station(station_id, data_dir, models_dir, output_dir):
    """
    Evaluate model for a specific station.
    
    Args:
        station_id: Station identifier
        data_dir: Directory with processed test data
        models_dir: Directory with trained models
        output_dir: Directory to save evaluation results
    """
    print(f"Evaluating model for station {station_id}...")
    
    # Load test data
    test_file = os.path.join(data_dir, 'test', f"{station_id}.npz")
    if not os.path.exists(test_file):
        print(f"Test data file not found: {test_file}")
        return False
    
    test_data = np.load(test_file)
    X_test, y_test = test_data['X'], test_data['y']
    
    # Load trained model
    model_path = os.path.join(models_dir, f"{station_id}.h5")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    predicted_errors = model.predict(X_test)
    
    # Extract true values
    observed_flow, nwm_forecasts, true_errors = extract_true_values(X_test, y_test)
    
    # Compute corrected forecasts
    corrected_forecasts = nwm_forecasts + predicted_errors
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate metric box plots
    create_metric_box_plots(observed_flow, nwm_forecasts, corrected_forecasts, station_id, output_dir)
    
    # Generate runoff box plots
    generate_runoff_box_plots(observed_flow, nwm_forecasts, corrected_forecasts, station_id, output_dir)
    
    print(f"Evaluated station {station_id}. Metrics saved to results/metrics/{station_id}_metrics.csv. Plots saved to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--station", type=str, required=False,
                        help="Station ID to evaluate (if not specified, evaluates both stations)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with processed test data")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--output-dir", type=str, default="results/plots",
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    if args.station:
        # Evaluate model for specified station
        evaluate_station(args.station, args.data_dir, args.models_dir, args.output_dir)
    else:
        # Evaluate models for both stations
        for station_id in ["21609641", "20380357"]:
            evaluate_station(station_id, args.data_dir, args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()
