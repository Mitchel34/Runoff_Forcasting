import numpy as np
import pandas as pd

class PersistenceBaseline:
    """
    Simple persistence-based baseline model for error correction.
    Predicts the error for lead time L at time t+L as the observed error for lead time L at time t.
    """
    
    def __init__(self):
        """Initialize the persistence baseline model"""
        self.historical_errors = {}
    
    def train(self, train_df):
        """
        'Train' the persistence model by storing the last observed error for each lead time.
        This is not actual training but rather storing reference data.
        
        Parameters:
        -----------
        train_df : pandas.DataFrame
            DataFrame containing historical observed errors with columns 'error_lead_1', 'error_lead_2', etc.
        """
        # Store the last available error for each lead time
        for lead in range(1, 19):
            column_name = f'error_lead_{lead}'
            if column_name in train_df.columns:
                # Store the entire series for access by timestamp
                self.historical_errors[lead] = train_df[column_name].copy()
    
    def predict(self, current_time, forecast_times):
        """
        Generate error predictions for each lead time.
        For lead time L, predict the error at time t+L as the observed error for lead time L at time t.
        
        Parameters:
        -----------
        current_time : pandas.Timestamp
            Current timestamp (from which to generate predictions)
        forecast_times : list
            List of future timestamps to generate predictions for
            
        Returns:
        --------
        predictions : dict
            Dictionary with forecast timestamps as keys and predicted errors as values
        """
        predictions = {}
        
        for i, forecast_time in enumerate(forecast_times):
            lead_time = i + 1  # Lead times are 1-based (1 to 18 hours)
            
            if lead_time in self.historical_errors:
                # Get most recent error for this lead time
                if current_time in self.historical_errors[lead_time].index:
                    # Use the error at current time
                    predictions[forecast_time] = self.historical_errors[lead_time][current_time]
                else:
                    # Find the closest previous time
                    past_times = self.historical_errors[lead_time].index[
                        self.historical_errors[lead_time].index <= current_time
                    ]
                    if len(past_times) > 0:
                        most_recent_time = past_times[-1]
                        predictions[forecast_time] = self.historical_errors[lead_time][most_recent_time]
                    else:
                        # No past data available, use zero error
                        predictions[forecast_time] = 0.0
            else:
                # No data for this lead time, default to zero error
                predictions[forecast_time] = 0.0
        
        return predictions
    
    def predict_batch(self, test_df):
        """
        Generate error predictions for a batch of test data.
        
        Parameters:
        -----------
        test_df : pandas.DataFrame
            DataFrame containing NWM forecasts to correct
            
        Returns:
        --------
        corrected_df : pandas.DataFrame
            DataFrame with corrected forecasts
        """
        corrected_df = test_df.copy()
        
        # Add columns for baseline corrected forecasts
        for lead in range(1, 19):
            corrected_df[f'baseline_corrected_lead_{lead}'] = np.nan
        
        # For each timestamp in the test set
        for timestamp in test_df.index:
            # Find the last available timestamp in training data
            available_times = []
            for lead in range(1, 19):
                if lead in self.historical_errors:
                    past_times = self.historical_errors[lead].index[
                        self.historical_errors[lead].index < timestamp
                    ]
                    if len(past_times) > 0:
                        available_times.append(past_times[-1])
            
            if not available_times:
                # No historical data available before this timestamp
                continue
            
            most_recent_time = max(available_times)
            
            # For each lead time
            for lead in range(1, 19):
                nwm_col = f'nwm_lead_{lead}'
                error_col = f'error_lead_{lead}'
                corrected_col = f'baseline_corrected_lead_{lead}'
                
                if nwm_col in test_df.columns and lead in self.historical_errors:
                    # Get the NWM forecast
                    nwm_forecast = test_df.loc[timestamp, nwm_col]
                    
                    # Find the most recent error for this lead time
                    recent_errors = self.historical_errors[lead]
                    past_times = recent_errors.index[recent_errors.index < timestamp]
                    
                    if len(past_times) > 0:
                        last_time = past_times[-1]
                        predicted_error = recent_errors[last_time]
                        
                        # Correct the forecast
                        corrected_df.loc[timestamp, corrected_col] = nwm_forecast - predicted_error
                    else:
                        # No historical error, use original forecast
                        corrected_df.loc[timestamp, corrected_col] = nwm_forecast
        
        return corrected_df

if __name__ == "__main__":
    # Example usage
    from preprocess import DataPreprocessor
    import matplotlib.pyplot as plt
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed",
        sequence_length=24
    )
    
    data = preprocessor.process_data(stream_ids=["20380357"])
    
    # Extract DataFrames
    train_val_df = data["20380357"]["train_val"]["df"]
    test_df = data["20380357"]["test"]["df"]
    
    # Create and train baseline model
    baseline = PersistenceBaseline()
    baseline.train(train_val_df)
    
    # Generate predictions
    corrected_df = baseline.predict_batch(test_df)
    
    # Example plot for a single lead time
    lead = 6  # 6-hour lead time
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df[f'usgs_observed'], label='Observed (USGS)')
    plt.plot(test_df.index, test_df[f'nwm_lead_{lead}'], label=f'NWM Forecast ({lead}h)')
    plt.plot(corrected_df.index, corrected_df[f'baseline_corrected_lead_{lead}'], 
             label=f'Baseline Corrected ({lead}h)')
    plt.title(f'Comparison of Original vs Baseline-Corrected Forecasts ({lead}-hour Lead Time)')
    plt.xlabel('Time')
    plt.ylabel('Runoff')
    plt.legend()
    plt.savefig(f"../reports/figures/baseline_comparison_{lead}h.png")
    plt.close()