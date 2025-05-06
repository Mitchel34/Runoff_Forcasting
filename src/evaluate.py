"""
Evaluate the trained model (LSTM or Transformer) on the test set.

Loads the best model, test data, and scalers.
Makes predictions, inverse-transforms results, calculates corrected runoff.
Computes evaluation metrics (CC, RMSE, PBIAS, NSE) for each lead time.
Generates box plots comparing NWM, Corrected NWM, and Observed USGS runoff.
Generates line plots comparing metrics for NWM vs. Corrected NWM.
Generates box plots showing monthly metric distributions for NWM vs. Corrected forecasts.
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
import sys
from typing import Dict, List

# Import utility functions and metrics
try:
    from utils import calculate_cc, calculate_rmse, calculate_pbias, calculate_nse
except ImportError:
    print("Error: Could not import utility functions. Ensure 'utils.py' is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# Define paths relative to the script location (src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
SCALERS_DIR = os.path.join(PROCESSED_DATA_DIR, 'scalers')

def load_test_data_and_metadata(station_id):
    """Loads test data (X, y), original NWM/USGS values, and timestamps."""
    file_path = os.path.join(PROCESSED_DATA_DIR, 'test', f"{station_id}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed test data file not found: {file_path}")

    try:
        data = np.load(file_path, allow_pickle=True) # Allow pickle for timestamps
        print(f"Loaded test data for station {station_id} from {file_path}")
        print(f"  Available keys: {list(data.keys())}")

        # Use keys consistent with notebook/preprocessing output
        X_test = data['X_test']
        y_test_scaled = data['y_test_scaled'] # These are the scaled true errors
        nwm_test_original = data['nwm_test_original'] # Shape: (n_samples, n_lead_times)
        usgs_test_original = data['usgs_test_original'] # Shape: (n_samples, n_lead_times)
        if 'test_timestamps' not in data:
            raise KeyError("'test_timestamps' key not found in test data file. Please re-run preprocess.py.")
        test_timestamps = data['test_timestamps'] # Shape: (n_samples,)

    except KeyError as e:
        raise KeyError(f"Missing expected key in test data file {file_path}: {e}. Expected keys: 'X_test', 'y_test_scaled', 'nwm_test_original', 'usgs_test_original', 'test_timestamps'.")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {e}")

    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test_scaled shape: {y_test_scaled.shape}")
    print(f"  nwm_test_original shape: {nwm_test_original.shape}")
    print(f"  usgs_test_original shape: {usgs_test_original.shape}")
    print(f"  test_timestamps shape: {test_timestamps.shape}")

    # Ensure shapes are consistent (samples and lead times)
    if y_test_scaled.shape != nwm_test_original.shape or y_test_scaled.shape != usgs_test_original.shape:
        raise ValueError(f"Shape mismatch between scaled errors (y_test_scaled {y_test_scaled.shape}), nwm_test_original ({nwm_test_original.shape}), and usgs_test_original ({usgs_test_original.shape}).")

    # Ensure timestamps match the number of samples
    if len(test_timestamps) != X_test.shape[0]:
        raise ValueError(f"Shape mismatch between test_timestamps ({len(test_timestamps)}) and X_test samples ({X_test.shape[0]}).")

    # Ensure y_test_scaled has 2 dimensions (samples, lead_times)
    if len(y_test_scaled.shape) != 2:
        raise ValueError(f"Expected y_test_scaled to have 2 dimensions (samples, lead_times), but got shape {y_test_scaled.shape}")

    return X_test, y_test_scaled, nwm_test_original, usgs_test_original, test_timestamps

def load_scaler(station_id, scaler_type='y'):
    """Loads the y-scaler for inverse transformation."""
    scaler_filename = f"{station_id}_{scaler_type.lower()}_scaler.joblib"
    scaler_path = os.path.join(SCALERS_DIR, scaler_filename)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}. Cannot inverse transform.")
    scaler = joblib_load(scaler_path)
    print(f"Loaded {scaler_type} scaler for station {station_id} from {scaler_path}")
    return scaler

def calculate_metrics_by_month(usgs_original: np.ndarray, nwm_original: np.ndarray, corrected_forecasts: np.ndarray, timestamps: np.ndarray) -> Dict[str, List[List[float]]]:
    """
    Calculates evaluation metrics (CC, RMSE, PBIAS, NSE) grouped by month for each lead time.

    Args:
        usgs_original: Observed USGS data (samples, lead_times)
        nwm_original: Original NWM forecast data (samples, lead_times)
        corrected_forecasts: Corrected forecast data (samples, lead_times)
        timestamps: Timestamps corresponding to the samples (samples,)

    Returns:
        A dictionary where keys are metric names (e.g., 'NWM_CC', 'Corrected_RMSE')
        and values are lists of lists. The outer list corresponds to lead times (0-17),
        and the inner list contains the metric value for each month in the test period.
    """
    n_samples, n_lead_times = usgs_original.shape
    metric_funcs = {'CC': calculate_cc, 'RMSE': calculate_rmse, 'PBIAS': calculate_pbias, 'NSE': calculate_nse}
    
    # Create a DataFrame for easier grouping
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps)
    })
    df['month'] = df['timestamp'].dt.to_period('M')

    # Initialize results dictionary
    monthly_metrics = {}
    for model_prefix in ['NWM', 'Corrected']:
        for metric_name in metric_funcs:
            monthly_metrics[f'{model_prefix}_{metric_name}'] = [[] for _ in range(n_lead_times)]

    # Calculate metrics for each lead time and each month
    for lead_idx in range(n_lead_times):
        df['obs'] = usgs_original[:, lead_idx]
        df['nwm'] = nwm_original[:, lead_idx]
        df['corr'] = corrected_forecasts[:, lead_idx]

        grouped = df.groupby('month')

        for month, group in grouped:
            obs_month = group['obs'].values
            nwm_month = group['nwm'].values
            corr_month = group['corr'].values

            for metric_name, func in metric_funcs.items():
                # Calculate NWM metric for the month
                valid_mask_nwm = ~np.isnan(obs_month) & ~np.isnan(nwm_month)
                if np.any(valid_mask_nwm):
                    nwm_metric_val = func(obs_month[valid_mask_nwm], nwm_month[valid_mask_nwm])
                    if not np.isnan(nwm_metric_val):
                         monthly_metrics[f'NWM_{metric_name}'][lead_idx].append(nwm_metric_val)
                
                # Calculate Corrected metric for the month
                valid_mask_corr = ~np.isnan(obs_month) & ~np.isnan(corr_month)
                if np.any(valid_mask_corr):
                    corr_metric_val = func(obs_month[valid_mask_corr], corr_month[valid_mask_corr])
                    if not np.isnan(corr_metric_val):
                        monthly_metrics[f'Corrected_{metric_name}'][lead_idx].append(corr_metric_val)
                        
    # Clean up temporary columns
    del df['obs'], df['nwm'], df['corr'], df['month'], df['timestamp']

    return monthly_metrics

def create_improvement_boxplots(original_flows, corrected_flows, observed_flows, station_id, model_type):
    """Create boxplots showing percentage improvement in forecasts."""
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import os
    
    # First, make sure we're working with arrays, not DataFrames
    original_flows_array = original_flows.values if hasattr(original_flows, 'values') else np.array(original_flows)
    corrected_flows_array = corrected_flows.values if hasattr(corrected_flows, 'values') else np.array(corrected_flows)
    observed_flows_array = observed_flows.values if hasattr(observed_flows, 'values') else np.array(observed_flows)
    
    # Calculate absolute errors for each lead time separately
    original_errors = np.abs(original_flows_array - observed_flows_array)
    corrected_errors = np.abs(corrected_flows_array - observed_flows_array)
    
    # Calculate percentage improvement safely
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_improvement = ((original_errors - corrected_errors) / np.maximum(original_errors, 1e-10)) * 100
        
    # Replace inf and -inf with NaN
    pct_improvement = np.nan_to_num(pct_improvement, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # Create figure for plots
    plt.figure(figsize=(14, 10))
    
    # 1. Improvement by flow magnitude - use first lead time for categories
    plt.subplot(2, 2, 1)
    
    # Use the first lead time column for categorization
    observed_values = observed_flows_array[:, 0]  # First lead time
    improvements = pct_improvement[:, 0]  # Improvement for first lead time
    
    # Create flow magnitude bins safely with 1D arrays
    flow_categories = pd.cut(
        observed_values,
        bins=[0, 10, 50, 100, float('inf')],
        labels=['Low (<10)', 'Medium (10-50)', 'High (50-100)', 'Very High (>100)']
    )
    
    # Create dataframe for plotting
    flow_df = pd.DataFrame({
        'Improvement': improvements,
        'Flow Category': flow_categories
    }).dropna()
    
    # Plot
    sns.boxplot(x='Flow Category', y='Improvement', data=flow_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Forecast Improvement by Flow Magnitude (Lead Time 1)')
    plt.xlabel('Observed Flow Magnitude (cms)')
    plt.ylabel('Percentage Improvement (%)')
    plt.ylim(-50, 100)
    
    # 2. Improvement by lead time
    plt.subplot(2, 2, 2)
    
    # Prepare data for lead time analysis
    lead_data = []
    for lead in range(pct_improvement.shape[1]):
        lead_values = pct_improvement[:, lead]
        for val in lead_values[~np.isnan(lead_values)]:
            lead_data.append((lead+1, val))
    
    lead_df = pd.DataFrame(lead_data, columns=['Lead Time', 'Improvement'])
    
    # Create boxplot
    sns.boxplot(x='Lead Time', y='Improvement', data=lead_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Forecast Improvement by Lead Time')
    plt.xlabel('Forecast Lead Time (hours)')
    plt.ylabel('Percentage Improvement (%)')
    plt.ylim(-50, 100)
    
    # 3. Overall improvement distribution
    plt.subplot(2, 2, 3)
    
    # Flatten all improvements for overall distribution
    all_improvements = pct_improvement.flatten()
    all_improvements = all_improvements[~np.isnan(all_improvements)]
    
    plt.boxplot(all_improvements)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Overall Forecast Improvement Distribution')
    plt.xlabel('All Lead Times')
    plt.ylabel('Percentage Improvement (%)')
    plt.ylim(-50, 100)
    plt.xticks([1], ['All Data'])
    
    # Add overall statistics as text
    mean_imp = np.nanmean(pct_improvement)
    median_imp = np.nanmedian(pct_improvement)
    pct_positive = (pct_improvement > 0).sum() / (~np.isnan(pct_improvement)).sum() * 100
    
    plt.text(0.98, 0.02, 
             f'Mean: {mean_imp:.2f}%\nMedian: {median_imp:.2f}%\n% Improved: {pct_positive:.1f}%',
             transform=plt.gca().transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # 4. Median improvement by lead time
    plt.subplot(2, 2, 4)
    
    # Calculate median improvement for each lead time
    median_by_lead = [np.nanmedian(pct_improvement[:, i]) for i in range(pct_improvement.shape[1])]
    
    plt.plot(range(1, len(median_by_lead)+1), median_by_lead, marker='o', linestyle='-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Median Improvement by Lead Time')
    plt.xlabel('Lead Time (hours)')
    plt.ylabel('Median Percentage Improvement (%)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot using the PLOTS_DIR constant
    plt.suptitle(f'Forecast Percentage Improvement: Station {station_id}, {model_type.upper()} Model',
                fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Use the PLOTS_DIR constant for saving instead of hardcoded path
    PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
    plot_filename = os.path.join(PLOTS_DIR, f"{station_id}_{model_type}_pct_improvement_boxplots.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean percentage improvement: {mean_imp:.2f}%")
    print(f"Median percentage improvement: {median_imp:.2f}%")
    print(f"Percentage of forecasts improved: {pct_positive:.1f}%")
    
    return {
        'mean_pct_improvement': mean_imp,
        'median_pct_improvement': median_imp,
        'pct_forecasts_improved': pct_positive
    }

def evaluate_model(station_id, model_type):
    """Evaluates the trained model and generates results."""
    print(f"\n--- Starting Evaluation for Station {station_id} ({model_type.upper()}) ---")

    # Create results directories
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    try:
        # 1. Load Model
        model_filename = f"{station_id}_{model_type.lower()}_best.keras"
        model_path = os.path.join(MODELS_DIR, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        model.summary()

        # 2. Load Test Data and Scaler
        X_test, y_test_scaled, nwm_test_original, usgs_test_original, test_timestamps = load_test_data_and_metadata(station_id)
        y_scaler = load_scaler(station_id, 'y')

        # 3. Make Predictions (Scaled Errors)
        print("Making predictions on the test set...")
        predicted_errors_scaled = model.predict(X_test)
        print(f"Predicted errors (scaled) shape: {predicted_errors_scaled.shape}")

        # Ensure prediction shape matches y_test_scaled shape
        if predicted_errors_scaled.shape != y_test_scaled.shape:
            print(f"Warning: Shape mismatch between predicted ({predicted_errors_scaled.shape}) and true scaled errors ({y_test_scaled.shape}). Trying to reshape prediction.")
            if len(predicted_errors_scaled.shape) == 1 and len(y_test_scaled.shape) == 2:
                if predicted_errors_scaled.shape[0] == y_test_scaled.shape[0] * y_test_scaled.shape[1]:
                    predicted_errors_scaled = predicted_errors_scaled.reshape(y_test_scaled.shape)
                    print(f"Reshaped prediction to {predicted_errors_scaled.shape}")
                else:
                    raise ValueError("Cannot reshape prediction due to element count mismatch.")
            elif len(predicted_errors_scaled.shape) == 3 and predicted_errors_scaled.shape[-1] == 1 and len(y_test_scaled.shape) == 2:
                if predicted_errors_scaled.shape[0] == y_test_scaled.shape[0] and predicted_errors_scaled.shape[1] == y_test_scaled.shape[1]:
                    predicted_errors_scaled = predicted_errors_scaled.squeeze(-1)
                    print(f"Reshaped prediction to {predicted_errors_scaled.shape}")
                else:
                    raise ValueError("Cannot reshape prediction due to dimension mismatch.")
            else:
                raise ValueError(f"Unhandled shape mismatch: Predicted {predicted_errors_scaled.shape}, Expected {y_test_scaled.shape}")

        # 4. Inverse Transform Predictions and True Errors
        print("Inverse transforming predictions and true errors...")
        n_samples = predicted_errors_scaled.shape[0]
        n_lead_times = predicted_errors_scaled.shape[1]

        if hasattr(y_scaler, 'n_features_in_') and y_scaler.n_features_in_ != n_lead_times:
            raise ValueError(f"Scaler expected {y_scaler.n_features_in_} features, but data has {n_lead_times} lead times.")

        predicted_errors_unscaled = y_scaler.inverse_transform(predicted_errors_scaled)
        true_errors_unscaled = y_scaler.inverse_transform(y_test_scaled)

        print(f"Predicted errors (unscaled) shape: {predicted_errors_unscaled.shape}")
        print(f"True errors (unscaled) shape: {true_errors_unscaled.shape}")

        calculated_true_errors = nwm_test_original - usgs_test_original
        error_diff = np.abs(true_errors_unscaled - calculated_true_errors)
        max_diff = np.nanmax(error_diff)
        print(f"Max difference between loaded true errors and calculated: {max_diff:.6f}")
        if not np.allclose(true_errors_unscaled, calculated_true_errors, atol=1e-5, equal_nan=True):
            print("Warning: Mismatch detected between loaded true errors and calculated (NWM - USGS). Check preprocessing steps.")

        # 5. Calculate Corrected NWM Forecasts
        corrected_nwm_forecasts = nwm_test_original + predicted_errors_unscaled
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

                valid_mask = ~np.isnan(obs) & ~np.isnan(nwm_pred)
                nwm_metric = func(obs[valid_mask], nwm_pred[valid_mask]) if np.any(valid_mask) else np.nan

                valid_mask_corr = ~np.isnan(obs) & ~np.isnan(corrected_pred)
                corrected_metric = func(obs[valid_mask_corr], corrected_pred[valid_mask_corr]) if np.any(valid_mask_corr) else np.nan

                metrics[f'NWM_{metric_name}'].append(nwm_metric)
                metrics[f'Corrected_{metric_name}'].append(corrected_metric)

        metrics_df = pd.DataFrame(metrics)
        metrics_filename = os.path.join(METRICS_DIR, f"{station_id}_{model_type.lower()}_evaluation_metrics.csv")
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Saved evaluation metrics to {metrics_filename}")
        print("Metrics Summary (Head):")
        print(metrics_df.head())
        print("\nMetrics Description:")
        print(metrics_df.describe())

        # Calculate Monthly Metrics
        print("\nCalculating monthly evaluation metrics for distribution plots...")
        monthly_metrics_dict = calculate_metrics_by_month(
            usgs_test_original,
            nwm_test_original,
            corrected_nwm_forecasts,
            test_timestamps
        )
        print(f"Calculated monthly metrics for {len(monthly_metrics_dict['NWM_CC'][0])} months.")

        # 7. Generate Visualizations
        print("\nGenerating visualizations...")
        lead_times = list(range(1, n_lead_times + 1))

        # --- Box Plot 1: Runoff Comparison ---
        plt.figure(figsize=(18, 8))
        plot_data = []
        for i in range(n_lead_times):
            df_obs = pd.DataFrame({'Runoff': usgs_test_original[:, i], 'Lead Time': lead_times[i], 'Type': 'Observed (USGS)'})
            df_nwm = pd.DataFrame({'Runoff': nwm_test_original[:, i], 'Lead Time': lead_times[i], 'Type': 'NWM Forecast'})
            df_corr = pd.DataFrame({'Runoff': corrected_nwm_forecasts[:, i], 'Lead Time': lead_times[i], 'Type': f'Corrected ({model_type.upper()})'})
            plot_data.extend([df_obs, df_nwm, df_corr])

        plot_df = pd.concat(plot_data)
        plot_df.dropna(subset=['Runoff'], inplace=True)

        sns.boxplot(data=plot_df, x='Lead Time', y='Runoff', hue='Type', showfliers=False)
        plt.title(f'Runoff Comparison by Lead Time - Station {station_id}')
        plt.xlabel('Lead Time (Hours)')
        plt.ylabel('Runoff (cfs)') # Updated unit label
        plt.xticks(rotation=45)
        plt.legend(title='Forecast Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_filename = os.path.join(PLOTS_DIR, f"{station_id}_{model_type.lower()}_runoff_boxplot.png")
        plt.savefig(plot_filename)
        print(f"Saved runoff comparison plot to {plot_filename}")
        plt.close()

        # --- Line Plots 2: Individual Metrics Comparison ---
        metric_plot_names = ['CC', 'RMSE', 'PBIAS', 'NSE']

        for metric_name in metric_plot_names:
            nwm_col = f'NWM_{metric_name}'
            corrected_col = f'Corrected_{metric_name}'
            if nwm_col not in metrics_df.columns or corrected_col not in metrics_df.columns:
                 print(f"Warning: Skipping plot for metric '{metric_name}' - columns not found in DataFrame.")
                 continue

            plt.figure(figsize=(10, 6))

            plt.plot(metrics_df['lead_time'], metrics_df[nwm_col], marker='o', linestyle='-', label='NWM Forecast')
            plt.plot(metrics_df['lead_time'], metrics_df[corrected_col], marker='x', linestyle='--', label=f'Corrected ({model_type.upper()})')

            plt.title(f'{metric_name} Comparison by Lead Time - Station {station_id}')
            plt.xlabel('Lead Time (Hours)')
            plt.ylabel(metric_name)
            plt.xticks(lead_times)
            plt.legend(title='Forecast Type')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            plot_filename = os.path.join(PLOTS_DIR, f"{station_id}_{model_type.lower()}_{metric_name}_lineplot.png")
            plt.savefig(plot_filename)
            print(f"Saved {metric_name} comparison line plot to {plot_filename}")
            plt.close()

        # --- Box Plots 3: Metric Distribution Comparison ---
        print("\nGenerating metric distribution box plots...")
        for metric_name in metric_plot_names:
            plt.figure(figsize=(18, 8))
            
            nwm_metric_key = f'NWM_{metric_name}'
            corrected_metric_key = f'Corrected_{metric_name}'
            
            if not monthly_metrics_dict.get(nwm_metric_key) or not monthly_metrics_dict.get(corrected_metric_key):
                print(f"Warning: Skipping distribution plot for metric '{metric_name}' - monthly data not found.")
                continue
                
            nwm_monthly_data = monthly_metrics_dict[nwm_metric_key]
            corrected_monthly_data = monthly_metrics_dict[corrected_metric_key]
            
            nwm_plot_data = [data for data in nwm_monthly_data if len(data) > 1]
            corrected_plot_data = [data for data in corrected_monthly_data if len(data) > 1]
            valid_lead_times = [lt for lt, data in zip(lead_times, nwm_monthly_data) if len(data) > 1]
            
            if not nwm_plot_data or not corrected_plot_data or not valid_lead_times:
                 print(f"Warning: Skipping distribution plot for metric '{metric_name}' - insufficient monthly data points after filtering.")
                 plt.close()
                 continue

            positions_nwm = np.array(range(len(valid_lead_times))) * 2.0 - 0.3
            positions_corrected = np.array(range(len(valid_lead_times))) * 2.0 + 0.3

            bp_nwm = plt.boxplot(nwm_plot_data, 
                               positions=positions_nwm, 
                               widths=0.5, 
                               patch_artist=True, 
                               showfliers=False,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='blue'))

            bp_corrected = plt.boxplot(corrected_plot_data, 
                                     positions=positions_corrected, 
                                     widths=0.5, 
                                     patch_artist=True, 
                                     showfliers=False,
                                     boxprops=dict(facecolor='lightcoral', alpha=0.7),
                                     medianprops=dict(color='red'))

            plt.ylabel(metric_name)
            plt.xlabel('Lead Time (Hours)')
            plt.title(f'{metric_name} Distribution (Monthly) by Lead Time - Station {station_id}')
            plt.xticks(np.array(range(len(valid_lead_times))) * 2.0, valid_lead_times)
            plt.legend([bp_nwm["boxes"][0], bp_corrected["boxes"][0]], 
                       ['NWM Forecast', f'Corrected ({model_type.upper()})'], 
                       loc='best')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plot_filename = os.path.join(PLOTS_DIR, f"{station_id}_{model_type.lower()}_{metric_name}_distribution_boxplot.png")
            plt.savefig(plot_filename)
            print(f"Saved {metric_name} distribution plot to {plot_filename}")
            plt.close()

        print("\nFinished generating visualizations.")

        # --- Improvement Boxplots ---
        print("\nGenerating improvement boxplots...")
        improvement_stats = create_improvement_boxplots(
            original_flows=pd.DataFrame(nwm_test_original),
            corrected_flows=pd.DataFrame(corrected_nwm_forecasts),
            observed_flows=pd.DataFrame(usgs_test_original),
            station_id=station_id,
            model_type=model_type
        )
        print(f"Improvement statistics: {improvement_stats}")

        # Add these statistics to your results DataFrame
        for key, value in improvement_stats.items():
            metrics_df[key] = value

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"--- Evaluation Complete for Station {station_id} ({model_type.upper()}) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model for runoff error correction.")
    parser.add_argument("--station_id", type=str, required=True, help="USGS Station ID (e.g., 21609641)")
    parser.add_argument("--model_type", type=str, required=True, choices=['lstm', 'transformer'], help="Model type that was trained")

    args = parser.parse_args()

    evaluate_model(station_id=args.station_id, model_type=args.model_type)

    print("\nEvaluation script finished.")
