import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Create directories if they don't exist
os.makedirs('../results/plots', exist_ok=True)
os.makedirs('../results/metrics', exist_ok=True)

def load_test_data_and_model():
    """
    Load test data and trained model
    """
    print("Loading test data and model...")
    
    try:
        # Load test data
        test_data = pd.read_csv('../data/processed/test_data.csv')
        test_data['date'] = pd.to_datetime(test_data['date'])
        
        # Load scaler
        with open('../data/processed/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load model
        model = tf.keras.models.load_model('../results/models/final_model.h5')
        
        return test_data, scaler, model
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run preprocess.py and model.py first")
        return None, None, None

def create_test_sequences(test_data, seq_length, lead_times):
    """
    Create sequences for testing
    """
    print("Creating test sequences...")
    
    features = ['runoff_nwm', 'runoff_usgs']
    if 'precipitation' in test_data.columns:
        features.append('precipitation')
    
    feature_data = test_data[features].values
    
    X_test = []
    for i in range(len(test_data) - seq_length - max(lead_times)):
        X_test.append(feature_data[i:i+seq_length])
    
    return np.array(X_test), features

def generate_predictions(model, X_test, test_data, seq_length, lead_times, scaler):
    """
    Generate predictions on test data
    """
    print("Generating predictions...")
    
    # Predict residuals
    residuals_pred = model.predict(X_test)
    
    # Create a DataFrame to store results
    results = []
    
    for i, lead in enumerate(lead_times):
        # Get the start index for the predictions (accounting for sequence length)
        start_idx = seq_length
        
        # Get the end index (limited by available test data)
        end_idx = len(test_data) - max(lead_times)
        
        # Extract dates for this prediction window
        dates = test_data['date'].iloc[start_idx+lead-1:end_idx+lead].reset_index(drop=True)
        
        # Get observed values
        observed = test_data['runoff_usgs'].iloc[start_idx+lead-1:end_idx+lead].reset_index(drop=True)
        
        # Get NWM forecasts
        nwm_forecast = test_data['runoff_nwm'].iloc[start_idx+lead-1:end_idx+lead].reset_index(drop=True)
        
        # Inverse transform if needed (if scaler was applied)
        if 'runoff_nwm' in test_data.columns:
            # Get residuals for this lead time
            lead_residuals = residuals_pred[:, i]
            
            # Calculate corrected forecasts by adding predicted residuals to NWM forecasts
            corrected_forecast = nwm_forecast + lead_residuals[:len(nwm_forecast)]
            
            # Store results
            lead_results = pd.DataFrame({
                'date': dates,
                'lead_time': lead,
                'observed': observed,
                'nwm_forecast': nwm_forecast,
                'corrected_forecast': corrected_forecast
            })
            
            results.append(lead_results)
    
    # Combine results for all lead times
    all_results = pd.concat(results, ignore_index=True)
    
    # Save results
    all_results.to_csv('../results/predictions.csv', index=False)
    
    return all_results

def compute_metrics(obs, pred):
    """
    Compute evaluation metrics
    - Coefficient of Correlation (CC)
    - Root Mean Square Error (RMSE)
    - Percent Bias (PBIAS)
    - Nash-Sutcliffe Efficiency (NSE)
    """
    cc = np.corrcoef(obs, pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(obs, pred))
    pbias = 100 * (np.sum(pred - obs) / np.sum(obs))
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
    return cc, rmse, pbias, nse

def evaluate_predictions(results):
    """
    Evaluate predictions using metrics
    """
    print("Evaluating predictions...")
    
    # Get unique lead times
    lead_times = results['lead_time'].unique()
    
    # Create DataFrames to store metrics
    metrics_df = pd.DataFrame(index=lead_times, columns=[
        'cc_nwm', 'rmse_nwm', 'pbias_nwm', 'nse_nwm',
        'cc_corrected', 'rmse_corrected', 'pbias_corrected', 'nse_corrected'
    ])
    
    for lead in lead_times:
        # Filter data for this lead time
        lead_data = results[results['lead_time'] == lead]
        
        # Compute metrics for NWM forecasts
        cc_nwm, rmse_nwm, pbias_nwm, nse_nwm = compute_metrics(
            lead_data['observed'].values, lead_data['nwm_forecast'].values
        )
        
        # Compute metrics for corrected forecasts
        cc_corr, rmse_corr, pbias_corr, nse_corr = compute_metrics(
            lead_data['observed'].values, lead_data['corrected_forecast'].values
        )
        
        # Store metrics
        metrics_df.loc[lead, 'cc_nwm'] = cc_nwm
        metrics_df.loc[lead, 'rmse_nwm'] = rmse_nwm
        metrics_df.loc[lead, 'pbias_nwm'] = pbias_nwm
        metrics_df.loc[lead, 'nse_nwm'] = nse_nwm
        
        metrics_df.loc[lead, 'cc_corrected'] = cc_corr
        metrics_df.loc[lead, 'rmse_corrected'] = rmse_corr
        metrics_df.loc[lead, 'pbias_corrected'] = pbias_corr
        metrics_df.loc[lead, 'nse_corrected'] = nse_corr
    
    # Save metrics
    metrics_df.to_csv('../results/metrics/evaluation_metrics.csv')
    
    return metrics_df

def create_boxplots(results, metrics_df):
    """
    Create boxplots for visualization
    """
    print("Creating visualizations...")
    
    # 1. Box-plot of Observed, NWM, and Corrected Runoff
    plt.figure(figsize=(10, 6))
    data_to_plot = [
        results['observed'].values,
        results['nwm_forecast'].values,
        results['corrected_forecast'].values
    ]
    sns.boxplot(data=data_to_plot)
    plt.xticks([0, 1, 2], ['Observed', 'NWM', 'Corrected'])
    plt.title('Distribution of Runoff Values')
    plt.ylabel('Runoff')
    plt.tight_layout()
    plt.savefig('../results/plots/runoff_boxplot.png', dpi=300)
    
    # 2. Box-plots of metrics by lead time
    metrics = [
        ('CC', ['cc_nwm', 'cc_corrected']),
        ('RMSE', ['rmse_nwm', 'rmse_corrected']),
        ('PBIAS', ['pbias_nwm', 'pbias_corrected']),
        ('NSE', ['nse_nwm', 'nse_corrected'])
    ]
    
    for metric_name, columns in metrics:
        plt.figure(figsize=(12, 6))
        
        # Reshape data for seaborn
        plot_data = pd.DataFrame({
            'Lead Time': metrics_df.index.repeat(2),
            'Model': ['NWM'] * len(metrics_df) + ['Corrected'] * len(metrics_df),
            'Value': metrics_df[columns[0]].values.tolist() + metrics_df[columns[1]].values.tolist()
        })
        
        sns.boxplot(x='Lead Time', y='Value', hue='Model', data=plot_data)
        plt.title(f'{metric_name} by Lead Time')
        plt.xlabel('Lead Time (hours)')
        plt.ylabel(metric_name)
        plt.tight_layout()
        plt.savefig(f'../results/plots/{metric_name.lower()}_by_leadtime.png', dpi=300)
    
    # 3. Time series plot for a sample lead time (e.g., 6 hours)
    lead_time_sample = 6
    sample_data = results[results['lead_time'] == lead_time_sample].iloc[:500]  # Limit to 500 points for clarity
    
    plt.figure(figsize=(15, 6))
    plt.plot(sample_data['date'], sample_data['observed'], label='Observed', color='black')
    plt.plot(sample_data['date'], sample_data['nwm_forecast'], label='NWM Forecast', color='blue', alpha=0.7)
    plt.plot(sample_data['date'], sample_data['corrected_forecast'], label='Corrected Forecast', color='red', alpha=0.7)
    plt.title(f'Time Series Comparison (Lead Time: {lead_time_sample} hours)')
    plt.xlabel('Date')
    plt.ylabel('Runoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/plots/timeseries_comparison.png', dpi=300)

def main():
    # Load test data and model
    test_data, scaler, model = load_test_data_and_model()
    if test_data is None:
        return
    
    # Define parameters
    seq_length = 24
    lead_times = range(1, 19)  # 1-18 hours
    
    # Create test sequences
    X_test, features = create_test_sequences(test_data, seq_length, lead_times)
    
    # Generate predictions
    results = generate_predictions(model, X_test, test_data, seq_length, lead_times, scaler)
    
    # Evaluate predictions
    metrics_df = evaluate_predictions(results)
    
    # Create visualizations
    create_boxplots(results, metrics_df)
    
    print("Evaluation completed successfully.")
    print(f"Results saved to '../results/' directory")

if __name__ == "__main__":
    main()
