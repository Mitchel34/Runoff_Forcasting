import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class ForecastEvaluator:
    """
    Evaluates forecast performance using hydrologic metrics:
    - Coefficient of Correlation (CC)
    - Root Mean Square Error (RMSE)
    - Percent Bias (PBIAS)
    - Nash-Sutcliffe Efficiency (NSE)
    """
    
    def __init__(self):
        """Initialize the ForecastEvaluator"""
        pass
    
    def calculate_cc(self, observed, forecasted):
        """
        Calculate Coefficient of Correlation (CC).
        
        Parameters:
        -----------
        observed : array-like
            Observed values
        forecasted : array-like
            Forecasted values
            
        Returns:
        --------
        cc : float
            Coefficient of Correlation
        """
        if len(observed) < 2 or len(forecasted) < 2:
            return np.nan
        
        try:
            cc, _ = pearsonr(observed, forecasted)
            return cc
        except:
            return np.nan
    
    def calculate_rmse(self, observed, forecasted):
        """
        Calculate Root Mean Square Error (RMSE).
        
        Parameters:
        -----------
        observed : array-like
            Observed values
        forecasted : array-like
            Forecasted values
            
        Returns:
        --------
        rmse : float
            Root Mean Square Error
        """
        if len(observed) == 0 or len(forecasted) == 0:
            return np.nan
        
        return np.sqrt(np.mean((np.array(observed) - np.array(forecasted)) ** 2))
    
    def calculate_pbias(self, observed, forecasted):
        """
        Calculate Percent Bias (PBIAS).
        
        Parameters:
        -----------
        observed : array-like
            Observed values
        forecasted : array-like
            Forecasted values
            
        Returns:
        --------
        pbias : float
            Percent Bias
        """
        if len(observed) == 0 or len(forecasted) == 0:
            return np.nan
        
        obs_sum = np.sum(np.array(observed))
        if obs_sum == 0:
            return np.nan
        
        return 100 * np.sum(np.array(forecasted) - np.array(observed)) / obs_sum
    
    def calculate_nse(self, observed, forecasted):
        """
        Calculate Nash-Sutcliffe Efficiency (NSE).
        
        Parameters:
        -----------
        observed : array-like
            Observed values
        forecasted : array-like
            Forecasted values
            
        Returns:
        --------
        nse : float
            Nash-Sutcliffe Efficiency
        """
        if len(observed) == 0 or len(forecasted) == 0:
            return np.nan
        
        obs_mean = np.mean(observed)
        numerator = np.sum((np.array(observed) - np.array(forecasted)) ** 2)
        denominator = np.sum((np.array(observed) - obs_mean) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1 - (numerator / denominator)
    
    def evaluate_all_metrics(self, observed, forecasted):
        """
        Calculate all metrics at once.
        
        Parameters:
        -----------
        observed : array-like
            Observed values
        forecasted : array-like
            Forecasted values
            
        Returns:
        --------
        metrics : dict
            Dictionary with all calculated metrics
        """
        return {
            'CC': self.calculate_cc(observed, forecasted),
            'RMSE': self.calculate_rmse(observed, forecasted),
            'PBIAS': self.calculate_pbias(observed, forecasted),
            'NSE': self.calculate_nse(observed, forecasted)
        }
    
    def evaluate_forecasts(self, results_df, lead_times=range(1, 19)):
        """
        Evaluate forecasts for all lead times and forecast types.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame containing observed values and various forecasts
        lead_times : iterable
            Lead times to evaluate (default: 1-18 hours)
            
        Returns:
        --------
        evaluation : pandas.DataFrame
            DataFrame with evaluation metrics for all forecasts and lead times
        """
        # Prepare results container
        metrics = ['CC', 'RMSE', 'PBIAS', 'NSE']
        forecast_types = ['NWM', 'LSTM', 'Baseline']
        
        # Create MultiIndex for metrics
        index = pd.MultiIndex.from_product([forecast_types, lead_times], 
                                          names=['Forecast', 'Lead Time'])
        
        # Initialize results DataFrame
        results = pd.DataFrame(index=index, columns=metrics)
        
        # Calculate metrics for each forecast type and lead time
        for lead in lead_times:
            # Extract observed values
            observed = results_df['usgs_observed'].dropna().values
            
            # Original NWM forecasts
            if f'nwm_lead_{lead}' in results_df.columns:
                nwm_forecasts = results_df[f'nwm_lead_{lead}'].dropna().values
                if len(nwm_forecasts) == len(observed):
                    metrics_dict = self.evaluate_all_metrics(observed, nwm_forecasts)
                    results.loc[('NWM', lead), :] = list(metrics_dict.values())
            
            # LSTM corrected forecasts
            if f'lstm_corrected_lead_{lead}' in results_df.columns:
                lstm_forecasts = results_df[f'lstm_corrected_lead_{lead}'].dropna().values
                if len(lstm_forecasts) == len(observed):
                    metrics_dict = self.evaluate_all_metrics(observed, lstm_forecasts)
                    results.loc[('LSTM', lead), :] = list(metrics_dict.values())
            
            # Baseline corrected forecasts
            if f'baseline_corrected_lead_{lead}' in results_df.columns:
                baseline_forecasts = results_df[f'baseline_corrected_lead_{lead}'].dropna().values
                if len(baseline_forecasts) == len(observed):
                    metrics_dict = self.evaluate_all_metrics(observed, baseline_forecasts)
                    results.loc[('Baseline', lead), :] = list(metrics_dict.values())
        
        return results
    
    def summarize_by_lead_time(self, evaluation_df):
        """
        Summarize evaluation results by lead time.
        
        Parameters:
        -----------
        evaluation_df : pandas.DataFrame
            DataFrame with evaluation metrics
            
        Returns:
        --------
        summary : pandas.DataFrame
            DataFrame with summarized metrics by lead time
        """
        # Reset index to use groupby
        df_reset = evaluation_df.reset_index()
        
        # Group by lead time and calculate mean/std
        summary = df_reset.groupby(['Lead Time', 'Forecast']).mean().reset_index()
        
        # Convert to wide format for easier comparison
        summary_wide = summary.pivot(index='Lead Time', columns='Forecast')
        
        return summary_wide

if __name__ == "__main__":
    # Example usage
    from nwm_dl_postprocessing.src.preprocess import DataPreprocessor
    from nwm_dl_postprocessing.src.predict import ForecastPredictor
    from nwm_dl_postprocessing.src.baseline import PersistenceBaseline
    import os
    
    # Define base paths as absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_path = os.path.join(base_dir, "data", "raw")
    data_processed_path = os.path.join(base_dir, "data", "processed")
    models_path = os.path.join(base_dir, "models")
    reports_path = os.path.join(base_dir, "reports")
    figures_path = os.path.join(reports_path, "figures")
    
    # Ensure directories exist
    os.makedirs(data_processed_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(
        raw_data_path=data_raw_path,
        processed_data_path=data_processed_path,
        sequence_length=24
    )
    
    data = preprocessor.process_data(stream_ids=["20380357"])
    
    # Get test data
    test_data = data["20380357"]["test"]
    test_df = test_data["df"]
    
    # Generate LSTM predictions
    model_path = os.path.join(models_path, "test_model.keras")
    predictor = ForecastPredictor(model_path)
    predictor.set_scaler(data["20380357"]["scalers"]["target"])
    lstm_corrected_df = predictor.generate_corrected_forecasts(test_data)
    
    # Generate baseline predictions
    baseline = PersistenceBaseline()
    baseline.train(data["20380357"]["train_val"]["df"])
    baseline_corrected_df = baseline.predict_batch(test_df)
    
    # Combine results
    results_df = lstm_corrected_df.copy()
    for lead in range(1, 19):
        col = f'baseline_corrected_lead_{lead}'
        if col in baseline_corrected_df.columns:
            results_df[col] = baseline_corrected_df[col]
    
    # Evaluate forecasts
    evaluator = ForecastEvaluator()
    evaluation_df = evaluator.evaluate_forecasts(results_df)
    
    # Save evaluation results
    evaluation_file = os.path.join(reports_path, "forecast_evaluation.csv")
    evaluation_df.to_csv(evaluation_file)
    print(f"Evaluation results saved to {evaluation_file}")
    
    # Print summary
    summary = evaluator.summarize_by_lead_time(evaluation_df)
    print(summary)
    
    # Save summary
    summary_file = os.path.join(reports_path, "evaluation_summary.csv")
    summary.to_csv(summary_file)
    print(f"Evaluation summary saved to {summary_file}")