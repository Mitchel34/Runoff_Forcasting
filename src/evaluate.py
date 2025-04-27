"""
Evaluate the trained model (LSTM or Transformer) on the test set.

Loads the best model, test data, and scalers.
Makes predictions, inverse-transforms results, calculates corrected runoff.
Computes evaluation metrics (CC, RMSE, PBIAS, NSE) for each lead time.
Generates box plots comparing NWM, Corrected NWM, and Observed USGS runoff.
Generates box plots comparing metrics for NWM vs. Corrected NWM.
Saves metrics and plots.
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load as joblib_load
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility functions and metrics
from utils import calculate_cc, calculate_rmse, calculate_pbias, calculate_nse

# Define paths (adjust if your structure differs)
PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')
MODELS_DIR = os.path.join('..', 'models')
RESULTS_DIR = os.path.join('..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
SCALERS_DIR = os.path.join(PROCESSED_DATA_DIR, 'scalers')

def load_test_data_and_metadata(station_id):
    """Loads test data (X, y) and original NWM/USGS values."""
    file_path = os.path.join(PROCESSED_DATA_DIR, 'test', f"{station_id}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed test data file not found: {file_path}")
    
    data = np.load(file_path)
    print(f"Loaded test data for station {station_id} from {file_path}")
    
    # Essential data: X_test, y_test (scaled errors)
    X_test = data['X']
    y_test_scaled = data['y'] # These are the scaled true errors
    
    # Metadata needed for evaluation: Original NWM forecasts and USGS observations
    # These should correspond to the time steps represented by y_test
    if 'nwm_test' not in data or 'usgs_test' not in data:
        raise KeyError("Test data file must contain 'nwm_test' and 'usgs_test' arrays.")
        
    nwm_test_original = data['nwm_test'] # Shape: (n_samples, n_lead_times)
    usgs_test_original = data['usgs_test'] # Shape: (n_samples, n_lead_times)
    
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test_scaled shape: {y_test_scaled.shape}")
    print(f"  nwm_test_original shape: {nwm_test_original.shape}")
    print(f"  usgs_test_original shape: {usgs_test_original.shape}")

    # Ensure shapes are consistent (samples and lead times)
    if y_test_scaled.shape != nwm_test_original.shape or y_test_scaled.shape != usgs_test_original.shape:
        raise ValueError("Shape mismatch between scaled errors (y_test), nwm_test, and usgs_test.")
        
    return X_test, y_test_scaled, nwm_test_original, usgs_test_original

def load_scaler(station_id, scaler_type='y'):
    """Loads the y-scaler for inverse transformation."""
    scaler_filename = f"{station_id}_{scaler_type.lower()}_scaler.joblib"
    scaler_path = os.path.join(SCALERS_DIR, scaler_filename)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}. Cannot inverse transform.")
    scaler = joblib_load(scaler_path)
    print(f"Loaded {scaler_type} scaler for station {station_id} from {scaler_path}")
    return scaler

def evaluate_model(station_id, model_type):
    """Evaluates the trained model and generates results."""
    print(f"\n--- Starting Evaluation for Station {station_id} ({model_type.upper()}) ---")
    
    # Create results directories
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # 1. Load Model
    model_filename = f"{station_id}_{model_type.lower()}_best.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # 2. Load Test Data and Scaler
    X_test, y_test_scaled, nwm_test_original, usgs_test_original = load_test_data_and_metadata(station_id)
    y_scaler = load_scaler(station_id, 'y')
    
    # 3. Make Predictions (Scaled Errors)
    print("Making predictions on the test set...")
    predicted_errors_scaled = model.predict(X_test)
    print(f"Predicted errors (scaled) shape: {predicted_errors_scaled.shape}")
    
    # 4. Inverse Transform Predictions and True Errors
    print("Inverse transforming predictions and true errors...")
    # Reshape if scaler expects 2D input (n_samples * n_features, 1) -> (n_samples, n_features)
    n_samples = predicted_errors_scaled.shape[0]
    n_lead_times = predicted_errors_scaled.shape[1]
    
    predicted_errors_unscaled = y_scaler.inverse_transform(predicted_errors_scaled.reshape(-1, n_lead_times))
    true_errors_unscaled = y_scaler.inverse_transform(y_test_scaled.reshape(-1, n_lead_times))
    
    # Reshape back if needed, although likely already (n_samples, n_lead_times)
    # predicted_errors_unscaled = predicted_errors_unscaled.reshape(n_samples, n_lead_times)
    # true_errors_unscaled = true_errors_unscaled.reshape(n_samples, n_lead_times)
    print(f"Predicted errors (unscaled) shape: {predicted_errors_unscaled.shape}")
    print(f"True errors (unscaled) shape: {true_errors_unscaled.shape}")

    # Verify true errors calculation (optional but recommended)
    # calculated_true_errors = nwm_test_original - usgs_test_original
    # error_diff = np.abs(true_errors_unscaled - calculated_true_errors)
    # print(f"Max difference between loaded true errors and calculated: {np.max(error_diff)}")
    # assert np.allclose(true_errors_unscaled, calculated_true_errors, atol=1e-5), "Mismatch in true error calculation!"

    # 5. Calculate Corrected NWM Forecasts
    # Corrected = NWM - Predicted_Error (since error = NWM - USGS)
    corrected_nwm_forecasts = nwm_test_original - predicted_errors_unscaled
    print(f"Corrected NWM forecasts shape: {corrected_nwm_forecasts.shape}")

    # 6. Calculate Metrics for each Lead Time
    print("Calculating evaluation metrics per lead time...")
    metrics = {'lead_time': list(range(1, n_lead_times + 1))}
    metric_funcs = {'CC': calculate_cc, 'RMSE': calculate_rmse, 'PBIAS': calculate_pbias, 'NSE': calculate_nse}
    
    for metric_name, func in metric_funcs.items():
        metrics[f'NWM_{metric_name}'] = []
        metrics[f'Corrected_{metric_name}'] = []
        for i in range(n_lead_times):
            obs = usgs_test_original[:, i]
            nwm_pred = nwm_test_original[:, i]
            corrected_pred = corrected_nwm_forecasts[:, i]
            
            metrics[f'NWM_{metric_name}'].append(func(obs, nwm_pred))
            metrics[f'Corrected_{metric_name}'].append(func(obs, corrected_pred))
            
    metrics_df = pd.DataFrame(metrics)
    metrics_filename = os.path.join(METRICS_DIR, f"{station_id}_{model_type.lower()}_evaluation_metrics.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Saved evaluation metrics to {metrics_filename}")
    print(metrics_df.head())

    # 7. Generate Visualizations
    print("Generating visualizations...")
    lead_times = list(range(1, n_lead_times + 1))

    # --- Box Plot 1: Runoff Comparison --- 
    plt.figure(figsize=(15, 8))
    plot_data = []
    for i in range(n_lead_times):
        plot_data.append(pd.DataFrame({
            'Runoff': usgs_test_original[:, i], 
            'Lead Time': lead_times[i], 
            'Type': 'Observed (USGS)'
        }))
        plot_data.append(pd.DataFrame({
            'Runoff': nwm_test_original[:, i], 
            'Lead Time': lead_times[i], 
            'Type': 'NWM Forecast'
        }))
        plot_data.append(pd.DataFrame({
            'Runoff': corrected_nwm_forecasts[:, i], 
            'Lead Time': lead_times[i], 
            'Type': f'Corrected ({model_type.upper()})'
        }))
    plot_df = pd.concat(plot_data)
    
    sns.boxplot(data=plot_df, x='Lead Time', y='Runoff', hue='Type', showfliers=False) # Hide outliers for clarity
    plt.title(f'Runoff Comparison by Lead Time - Station {station_id}')
    plt.xlabel('Lead Time (Hours)')
    plt.ylabel('Runoff (cms)') # Assuming units are cms, adjust if needed
    plt.xticks(rotation=45)
    plt.legend(title='Forecast Type')
    plt.tight_layout()
    plot_filename = os.path.join(PLOTS_DIR, f"{station_id}_{model_type.lower()}_runoff_boxplot.png")
    plt.savefig(plot_filename)
    print(f"Saved runoff comparison plot to {plot_filename}")
    plt.close()

    # --- Box Plot 2: Metrics Comparison --- 
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    metric_plot_names = ['CC', 'RMSE', 'PBIAS', 'NSE']
    
    for i, metric_name in enumerate(metric_plot_names):
        melted_df = pd.melt(metrics_df, 
                            id_vars=['lead_time'], 
                            value_vars=[f'NWM_{metric_name}', f'Corrected_{metric_name}'],
                            var_name='Metric Type', 
                            value_name=metric_name)
        melted_df['Metric Type'] = melted_df['Metric Type'].str.replace(f'_{metric_name}', '')
        
        sns.boxplot(data=melted_df, x='lead_time', y=metric_name, hue='Metric Type', ax=axes[i], showfliers=False)
        axes[i].set_title(f'{metric_name} Comparison')
        axes[i].set_xlabel('Lead Time (Hours)')
        axes[i].set_ylabel(metric_name)
        axes[i].legend(title='Forecast Type')
        axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Evaluation Metrics Comparison by Lead Time - Station {station_id} ({model_type.upper()})', y=1.02)
    plt.tight_layout()
    plot_filename = os.path.join(PLOTS_DIR, f"{station_id}_{model_type.lower()}_metrics_boxplot.png")
    plt.savefig(plot_filename)
    print(f"Saved metrics comparison plot to {plot_filename}")
    plt.close()

    print(f"--- Evaluation Complete for Station {station_id} ({model_type.upper()}) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model for runoff error correction.")
    parser.add_argument("--station_id", type=str, required=True, help="USGS Station ID (e.g., 21609641)")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'transformer'], help="Model type that was trained")
    
    args = parser.parse_args()
    
    evaluate_model(station_id=args.station_id, model_type=args.model_type)
    
    print("\nEvaluation script finished.")
