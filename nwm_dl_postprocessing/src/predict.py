import numpy as np
import pandas as pd
import os
import tensorflow as tf
from model import Seq2SeqLSTMModel

class ForecastPredictor:
    """
    Class for generating corrected runoff forecasts using trained Seq2Seq LSTM model.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the ForecastPredictor.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved model file. If provided, loads the model directly.
        """
        self.model = None
        self.target_scaler = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to saved model file
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def set_model(self, model):
        """
        Set the model directly from a Seq2SeqLSTMModel instance.
        
        Parameters:
        -----------
        model : Seq2SeqLSTMModel or tf.keras.Model
            Trained model instance
        """
        if isinstance(model, Seq2SeqLSTMModel):
            self.model = model.model
        else:
            self.model = model
    
    def set_scaler(self, target_scaler):
        """
        Set the scaler for inverse transforming predictions.
        
        Parameters:
        -----------
        target_scaler : sklearn.preprocessing.StandardScaler
            Scaler used to standardize target values during training
        """
        self.target_scaler = target_scaler
    
    def predict_errors(self, X_encoder, X_decoder):
        """
        Generate forecast error predictions using the trained model.
        
        Parameters:
        -----------
        X_encoder : numpy.ndarray
            Encoder input sequences
        X_decoder : numpy.ndarray
            Decoder input sequences (NWM forecasts)
            
        Returns:
        --------
        predicted_errors : numpy.ndarray
            Predicted error sequences
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() or set_model() first.")
        
        predicted_errors = self.model.predict([X_encoder, X_decoder])
        
        # Apply inverse scaling if a scaler is provided
        if self.target_scaler:
            # Reshape for inverse transform
            original_shape = predicted_errors.shape
            flattened = predicted_errors.reshape(-1, 1)
            # Inverse transform
            unscaled = self.target_scaler.inverse_transform(flattened)
            # Reshape back to original shape
            predicted_errors = unscaled.reshape(original_shape)
        
        return predicted_errors
    
    def correct_forecasts(self, nwm_forecasts, predicted_errors):
        """
        Apply error corrections to NWM forecasts.
        
        Parameters:
        -----------
        nwm_forecasts : numpy.ndarray or pandas.DataFrame
            Original NWM forecasts
        predicted_errors : numpy.ndarray
            Predicted forecast errors
            
        Returns:
        --------
        corrected_forecasts : numpy.ndarray or pandas.DataFrame
            Corrected NWM forecasts
        """
        # For numpy arrays
        if isinstance(nwm_forecasts, np.ndarray) and isinstance(predicted_errors, np.ndarray):
            return nwm_forecasts - predicted_errors
        
        # For pandas DataFrames
        elif isinstance(nwm_forecasts, pd.DataFrame):
            corrected = nwm_forecasts.copy()
            
            for lead in range(1, 19):
                nwm_col = f'nwm_lead_{lead}'
                corrected_col = f'lstm_corrected_lead_{lead}'
                
                if nwm_col in nwm_forecasts.columns:
                    # Get column index for this lead time in predicted errors
                    lead_idx = lead - 1
                    
                    # For each time step
                    for i in range(len(predicted_errors)):
                        if i < len(nwm_forecasts):
                            # Get the prediction for this lead time
                            prediction = predicted_errors[i, lead_idx]
                            # Apply correction
                            corrected.iloc[i, corrected.columns.get_loc(nwm_col)] = nwm_forecasts.iloc[i, nwm_forecasts.columns.get_loc(nwm_col)] - prediction
            
            return corrected
        else:
            raise ValueError("Unsupported input types. Use numpy arrays or pandas DataFrames.")
    
    def generate_corrected_forecasts(self, test_data, include_columns=None):
        """
        Generate corrected forecasts for test data.
        
        Parameters:
        -----------
        test_data : dict
            Dictionary containing test data with X_encoder, X_decoder, and df
        include_columns : list, optional
            List of columns to include in the output DataFrame
            
        Returns:
        --------
        corrected_df : pandas.DataFrame
            DataFrame with original and corrected forecasts
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() or set_model() first.")
        
        # Generate predictions
        X_encoder = test_data['X_encoder']
        X_decoder = test_data['X_decoder']
        test_df = test_data['df']
        
        # Predict errors
        predicted_errors = self.predict_errors(X_encoder, X_decoder)
        
        # Create output DataFrame
        result_df = test_df.copy()
        
        # Add columns for LSTM corrected forecasts
        for lead in range(1, 19):
            result_df[f'lstm_corrected_lead_{lead}'] = np.nan
        
        # Apply corrections
        for i in range(len(predicted_errors)):
            if i < len(result_df):
                for lead in range(1, 19):
                    lead_idx = lead - 1  # 0-based index
                    nwm_col = f'nwm_lead_{lead}'
                    
                    if nwm_col in result_df.columns:
                        # Get original forecast and predicted error
                        original = result_df.iloc[i][nwm_col]
                        error = predicted_errors[i, lead_idx]
                        
                        # Apply correction
                        result_df.iloc[i, result_df.columns.get_loc(f'lstm_corrected_lead_{lead}')] = original - error
        
        # Filter columns if specified
        if include_columns:
            cols_to_keep = ['usgs_observed'] + include_columns
            result_df = result_df[cols_to_keep]
        
        return result_df

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
    
    # Create predictor and load model
    predictor = ForecastPredictor("../models/nwm_lstm_model.keras")
    predictor.set_scaler(data["20380357"]["scalers"]["target"])
    
    # Generate corrected forecasts
    test_data = data["20380357"]["test"]
    corrected_df = predictor.generate_corrected_forecasts(test_data)
    
    # Save results
    corrected_df.to_csv("../data/processed/lstm_corrected_forecasts.csv")
    
    # Example plot for a single lead time
    lead = 6  # 6-hour lead time
    plt.figure(figsize=(12, 6))
    plt.plot(corrected_df.index, corrected_df['usgs_observed'], label='Observed (USGS)')
    plt.plot(corrected_df.index, corrected_df[f'nwm_lead_{lead}'], label=f'NWM Forecast ({lead}h)')
    plt.plot(corrected_df.index, corrected_df[f'lstm_corrected_lead_{lead}'], 
             label=f'LSTM Corrected ({lead}h)')
    plt.title(f'Comparison of Original vs LSTM-Corrected Forecasts ({lead}-hour Lead Time)')
    plt.xlabel('Time')
    plt.ylabel('Runoff')
    plt.legend()
    plt.savefig(f"../reports/figures/lstm_comparison_{lead}h.png")
    plt.close()