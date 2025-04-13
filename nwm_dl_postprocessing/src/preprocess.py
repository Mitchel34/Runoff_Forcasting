import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Class for loading, cleaning, and preparing NWM and USGS data for model training.
    Handles time alignment, missing value handling, feature engineering, and data splitting.
    """
    
    def __init__(self, raw_data_path, processed_data_path, sequence_length=24):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        raw_data_path : str
            Path to the directory containing raw data files
        processed_data_path : str
            Path to save processed data files
        sequence_length : int
            Length of input sequence for the encoder (default: 24 hours)
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.sequence_length = sequence_length
        self.scaler = None
        
        # Ensure processed directory exists
        os.makedirs(processed_data_path, exist_ok=True)
    
    def load_usgs_data(self, stream_id):
        """Load USGS observed runoff data for a specific stream"""
        usgs_files = [f for f in os.listdir(os.path.join(self.raw_data_path, str(stream_id))) 
                      if f.endswith('_Strt_*.csv')]
        
        if not usgs_files:
            raise FileNotFoundError(f"No USGS data files found for stream {stream_id}")
        
        usgs_df = pd.read_csv(os.path.join(self.raw_data_path, str(stream_id), usgs_files[0]))
        # Assuming the USGS file has datetime and value columns
        usgs_df['datetime'] = pd.to_datetime(usgs_df['datetime'])
        usgs_df.set_index('datetime', inplace=True)
        
        return usgs_df
    
    def load_nwm_data(self, stream_id):
        """Load NWM forecast data for a specific stream"""
        nwm_files = [f for f in os.listdir(os.path.join(self.raw_data_path, str(stream_id))) 
                     if f.startswith('streamflow_')]
        
        if not nwm_files:
            raise FileNotFoundError(f"No NWM data files found for stream {stream_id}")
        
        # Read and combine all NWM monthly files
        dfs = []
        for file in nwm_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, str(stream_id), file))
            dfs.append(df)
        
        nwm_df = pd.concat(dfs, ignore_index=True)
        # Assuming the NWM files have reference_time, value_time, and lead time columns
        nwm_df['reference_time'] = pd.to_datetime(nwm_df['reference_time'])
        nwm_df['value_time'] = pd.to_datetime(nwm_df['value_time'])
        
        return nwm_df
    
    def align_data(self, usgs_df, nwm_df):
        """Align NWM forecasts and USGS observations by datetime"""
        
        # Restructure NWM data by lead time
        lead_time_dfs = {}
        for lead in range(1, 19):  # 1-18 hour lead times
            # Filter for specific lead time
            lead_df = nwm_df[nwm_df['lead_time'] == lead].copy()
            # Index by value_time (when the prediction is for)
            lead_df.set_index('value_time', inplace=True)
            # Select only the streamflow column
            lead_df = lead_df[['streamflow']]
            lead_df.rename(columns={'streamflow': f'nwm_lead_{lead}'}, inplace=True)
            lead_time_dfs[lead] = lead_df
        
        # Combine all lead times with USGS observations
        aligned_df = usgs_df.rename(columns={'value': 'usgs_observed'})
        
        for lead, lead_df in lead_time_dfs.items():
            aligned_df = aligned_df.join(lead_df, how='outer')
        
        return aligned_df
    
    def calculate_errors(self, df):
        """Calculate NWM forecast errors for each lead time"""
        for lead in range(1, 19):
            df[f'error_lead_{lead}'] = df[f'nwm_lead_{lead}'] - df['usgs_observed']
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # For simplicity, we'll use forward fill followed by backward fill
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # Drop any remaining rows with NaN values
        df = df.dropna()
        
        return df
    
    def create_sequences(self, df):
        """
        Create sequences for the Seq2Seq model:
        - Encoder input: past observations, forecasts, and errors
        - Decoder input: current NWM forecasts for lead times 1-18
        - Target output: actual errors for lead times 1-18
        """
        X_encoder = []
        X_decoder = []
        y = []
        
        # Convert dataframe to numpy for easier sequence creation
        data_array = df.values
        
        # Create sequences
        for i in range(self.sequence_length, len(df) - 18):  # Need at least 18 hours ahead for targets
            # Encoder input: last sequence_length hours of [usgs_observed, nwm_lead_1, error_lead_1]
            enc_input = []
            for j in range(i-self.sequence_length, i):
                # Get USGS observation, NWM 1h lead forecast, and 1h forecast error
                enc_input.append([
                    df.iloc[j]['usgs_observed'],
                    df.iloc[j]['nwm_lead_1'],
                    df.iloc[j]['error_lead_1']
                ])
            
            # Decoder input: current NWM forecasts for lead times 1-18
            dec_input = []
            for lead in range(1, 19):
                dec_input.append(df.iloc[i][f'nwm_lead_{lead}'])
            
            # Target: actual errors for lead times 1-18 over the next 18 hours
            target = []
            for lead in range(1, 19):
                # For lead time L, get the error at time t+L
                target.append(df.iloc[i+lead-1][f'error_lead_{lead}'])
            
            X_encoder.append(np.array(enc_input))
            X_decoder.append(np.array(dec_input))
            y.append(np.array(target))
        
        return np.array(X_encoder), np.array(X_decoder), np.array(y)
    
    def scale_features(self, X_encoder_train, X_decoder_train, y_train, X_encoder_val=None, X_decoder_val=None):
        """Scale features using StandardScaler fit only on training data"""
        
        # Reshape data for scaling
        encoder_shape = X_encoder_train.shape
        X_encoder_flat = X_encoder_train.reshape(-1, X_encoder_train.shape[-1])
        
        decoder_shape = X_decoder_train.shape
        X_decoder_flat = X_decoder_train.reshape(-1, 1)
        
        y_shape = y_train.shape
        y_flat = y_train.reshape(-1, 1)
        
        # Fit scalers on training data
        encoder_scaler = StandardScaler()
        decoder_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        X_encoder_flat_scaled = encoder_scaler.fit_transform(X_encoder_flat)
        X_decoder_flat_scaled = decoder_scaler.fit_transform(X_decoder_flat)
        y_flat_scaled = target_scaler.fit_transform(y_flat)
        
        # Reshape back to original structure
        X_encoder_train_scaled = X_encoder_flat_scaled.reshape(encoder_shape)
        X_decoder_train_scaled = X_decoder_flat_scaled.reshape(decoder_shape)
        y_train_scaled = y_flat_scaled.reshape(y_shape)
        
        # Save scalers for later use
        self.encoder_scaler = encoder_scaler
        self.decoder_scaler = decoder_scaler
        self.target_scaler = target_scaler
        
        # If validation data is provided, scale it too
        if X_encoder_val is not None and X_decoder_val is not None:
            encoder_val_shape = X_encoder_val.shape
            X_encoder_val_flat = X_encoder_val.reshape(-1, X_encoder_val.shape[-1])
            X_encoder_val_scaled = encoder_scaler.transform(X_encoder_val_flat)
            X_encoder_val_scaled = X_encoder_val_scaled.reshape(encoder_val_shape)
            
            decoder_val_shape = X_decoder_val.shape
            X_decoder_val_flat = X_decoder_val.reshape(-1, 1)
            X_decoder_val_scaled = decoder_scaler.transform(X_decoder_val_flat)
            X_decoder_val_scaled = X_decoder_val_scaled.reshape(decoder_val_shape)
            
            return X_encoder_train_scaled, X_decoder_train_scaled, y_train_scaled, X_encoder_val_scaled, X_decoder_val_scaled
        
        return X_encoder_train_scaled, X_decoder_train_scaled, y_train_scaled
    
    def split_train_test(self, df):
        """
        Split data into training+validation and test sets based on date ranges:
        - Training/Validation: April 2021 – September 2022
        - Testing: October 2022 – April 2023
        """
        test_start_date = pd.Timestamp('2022-10-01')
        
        train_val_df = df[df.index < test_start_date].copy()
        test_df = df[df.index >= test_start_date].copy()
        
        return train_val_df, test_df
    
    def process_data(self, stream_ids=None):
        """
        Main method to process the data for all streams.
        
        Parameters:
        -----------
        stream_ids : list, optional
            List of stream IDs to process. If None, processes all streams in raw_data_path.
            
        Returns:
        --------
        dict : Dictionary containing preprocessed data for training, validation, and testing
        """
        if stream_ids is None:
            stream_ids = [d for d in os.listdir(self.raw_data_path) 
                         if os.path.isdir(os.path.join(self.raw_data_path, d))]
        
        all_data = {}
        
        for stream_id in stream_ids:
            print(f"Processing stream: {stream_id}")
            
            # Load data
            usgs_df = self.load_usgs_data(stream_id)
            nwm_df = self.load_nwm_data(stream_id)
            
            # Align and preprocess
            aligned_df = self.align_data(usgs_df, nwm_df)
            aligned_df = self.calculate_errors(aligned_df)
            aligned_df = self.handle_missing_values(aligned_df)
            
            # Split into train/val and test
            train_val_df, test_df = self.split_train_test(aligned_df)
            
            # Create sequences for each set
            X_enc_train_val, X_dec_train_val, y_train_val = self.create_sequences(train_val_df)
            X_enc_test, X_dec_test, y_test = self.create_sequences(test_df)
            
            # Scale features
            X_enc_train_val_scaled, X_dec_train_val_scaled, y_train_val_scaled = self.scale_features(
                X_enc_train_val, X_dec_train_val, y_train_val
            )
            
            # Scale test data with the same scalers fit on training data
            X_enc_test_scaled = self.encoder_scaler.transform(X_enc_test.reshape(-1, X_enc_test.shape[-1]))
            X_enc_test_scaled = X_enc_test_scaled.reshape(X_enc_test.shape)
            
            X_dec_test_scaled = self.decoder_scaler.transform(X_dec_test.reshape(-1, 1))
            X_dec_test_scaled = X_dec_test_scaled.reshape(X_dec_test.shape)
            
            # Store processed data
            all_data[stream_id] = {
                'train_val': {
                    'X_encoder': X_enc_train_val_scaled,
                    'X_decoder': X_dec_train_val_scaled,
                    'y': y_train_val_scaled,
                    'df': train_val_df
                },
                'test': {
                    'X_encoder': X_enc_test_scaled,
                    'X_decoder': X_dec_test_scaled,
                    'y': y_test,  # Unscaled for evaluation
                    'df': test_df
                },
                'scalers': {
                    'encoder': self.encoder_scaler,
                    'decoder': self.decoder_scaler,
                    'target': self.target_scaler
                }
            }
            
            # Save to disk
            np.save(os.path.join(self.processed_data_path, f'{stream_id}_train_val.npy'), {
                'X_encoder': X_enc_train_val_scaled,
                'X_decoder': X_dec_train_val_scaled,
                'y': y_train_val_scaled
            })
            
            np.save(os.path.join(self.processed_data_path, f'{stream_id}_test.npy'), {
                'X_encoder': X_enc_test_scaled,
                'X_decoder': X_dec_test_scaled,
                'y': y_test
            })
            
            # Save processed dataframes for reference
            train_val_df.to_csv(os.path.join(self.processed_data_path, 'train_validation_data.csv'))
            test_df.to_csv(os.path.join(self.processed_data_path, 'test_data.csv'))
        
        return all_data

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed",
        sequence_length=24
    )
    
    data = preprocessor.process_data(stream_ids=["20380357", "21609641"])
    print("Data preprocessing completed successfully.")