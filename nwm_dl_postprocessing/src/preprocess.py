import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Class for loading, cleaning, and preparing NWM and USGS data for model training and evaluation.
    Handles feature engineering, sequence creation, and data splitting.
    """
    
    def __init__(self, raw_data_path, processed_data_path, sequence_length=24):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        raw_data_path : str
            Path to raw data directory containing NWM and USGS files
        processed_data_path : str
            Path to save processed data
        sequence_length : int, optional
            Length of historical sequence for encoder input
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.sequence_length = sequence_length
        
        # Create processed data directory if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)
    
    def load_usgs_data(self, stream_id):
        """
        Load USGS observed runoff data for a stream.
        
        Parameters:
        -----------
        stream_id : str
            Stream identifier
            
        Returns:
        --------
        usgs_df : pandas.DataFrame
            DataFrame with USGS observed runoff
        """
        # Look for USGS files both in the expected raw_data_path and in the root directory
        usgs_files = glob(os.path.join(self.raw_data_path, str(stream_id), "*_Strt_*.csv"))
        
        # If not found in raw_data_path, try the root directory
        if not usgs_files:
            root_path = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.raw_data_path)))))
            usgs_files = glob(os.path.join(root_path, str(stream_id), "*_Strt_*.csv"))
            
        if not usgs_files:
            raise FileNotFoundError(f"No USGS data files found for stream {stream_id}")
        
        usgs_df = pd.read_csv(usgs_files[0])
        
        # Convert to datetime - use 'DateTime' column name (capital D and T) as shown in the data file
        usgs_df['datetime'] = pd.to_datetime(usgs_df['DateTime'])
        usgs_df.set_index('datetime', inplace=True)
        
        # Rename the flow value column to 'value' to match what's expected in other methods
        usgs_df = usgs_df.rename(columns={'USGSFlowValue': 'value'})
        
        return usgs_df
    
    def load_nwm_data(self, stream_id):
        """
        Load NWM forecast data for a stream.
        
        Parameters:
        -----------
        stream_id : str
            Stream identifier
            
        Returns:
        --------
        nwm_df : pandas.DataFrame
            DataFrame with NWM forecast data
        """
        # Look for NWM files both in the expected raw_data_path and in the root directory
        nwm_files = glob(os.path.join(self.raw_data_path, str(stream_id), "streamflow_*.csv"))
        
        # If not found in raw_data_path, try the root directory
        if not nwm_files:
            root_path = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.raw_data_path)))))
            nwm_files = glob(os.path.join(root_path, str(stream_id), "streamflow_*.csv"))
            
        if not nwm_files:
            raise FileNotFoundError(f"No NWM data files found for stream {stream_id}")
        
        dfs = []
        for file in nwm_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        nwm_df = pd.concat(dfs, ignore_index=True)
        
        # Rename columns to match what's expected in the code
        nwm_df = nwm_df.rename(columns={
            'model_initialization_time': 'reference_time',
            'model_output_valid_time': 'value_time',
            'streamflow_value': 'streamflow'
        })
        
        # Convert to datetime - handle the custom format with underscores
        # Format: YYYY-MM-DD_HH:MM:SS
        nwm_df['reference_time'] = pd.to_datetime(nwm_df['reference_time'].str.replace('_', ' '))
        nwm_df['value_time'] = pd.to_datetime(nwm_df['value_time'].str.replace('_', ' '))
        
        # Calculate lead time as the hours difference between value_time and reference_time
        nwm_df['lead_time'] = (nwm_df['value_time'] - nwm_df['reference_time']).dt.total_seconds() / 3600
        nwm_df['lead_time'] = nwm_df['lead_time'].astype(int)
        
        return nwm_df
    
    def clean_and_align_data(self, usgs_df, nwm_df):
        """
        Clean and align USGS and NWM data.
        
        Parameters:
        -----------
        usgs_df : pandas.DataFrame
            DataFrame with USGS observed runoff
        nwm_df : pandas.DataFrame
            DataFrame with NWM forecast data
            
        Returns:
        --------
        aligned_df : pandas.DataFrame
            DataFrame with aligned USGS observations and NWM forecasts
        """
        # Ensure both dataframes have timezone information standardized
        # Convert to timezone naive for consistent comparison
        if usgs_df.index.tzinfo is not None:
            usgs_df.index = usgs_df.index.tz_localize(None)
        if 'value_time' in nwm_df.columns and pd.api.types.is_datetime64_dtype(nwm_df['value_time']):
            if nwm_df['value_time'].dt.tz is not None:
                nwm_df['value_time'] = nwm_df['value_time'].dt.tz_localize(None)
        
        # Print some debug info
        print(f"  USGS data range: {usgs_df.index.min()} to {usgs_df.index.max()}")
        print(f"  USGS data shape: {usgs_df.shape}")
        print(f"  NWM data shape: {nwm_df.shape}")
        print(f"  NWM lead times: {nwm_df['lead_time'].unique()}")
        
        # Create a DataFrame with USGS observations
        # Handle potential duplicate timestamps by taking the last value
        aligned_df = pd.DataFrame({'usgs_observed': usgs_df['value']})
        aligned_df = aligned_df[~aligned_df.index.duplicated(keep='last')]
        
        # Add NWM forecasts for each lead time
        for lead in range(1, 19):  # Lead times 1-18 hours
            # Filter forecasts for this lead time
            lead_df = nwm_df[nwm_df['lead_time'] == lead].copy()
            
            if lead_df.empty:
                print(f"  No data for lead time {lead}")
                continue
                
            # Set index to value_time (when the forecast is for)
            lead_df.set_index('value_time', inplace=True)
            
            # Remove duplicate timestamps from the NWM data
            lead_df = lead_df[~lead_df.index.duplicated(keep='last')]
            
            # Add to aligned data with descriptive column name
            # Using more robust merge approach instead of reindex
            temp_df = aligned_df.copy()
            lead_col_name = f'nwm_lead_{lead}'
            lead_df_simple = pd.DataFrame({lead_col_name: lead_df['streamflow']})
            
            # Left join to keep all USGS observations
            aligned_df = pd.merge(temp_df, lead_df_simple, 
                                  left_index=True, right_index=True, 
                                  how='left')
            
            # Calculate forecast errors (residuals) for this lead time
            error_col = f'error_lead_{lead}'
            aligned_df[error_col] = aligned_df[lead_col_name] - aligned_df['usgs_observed']
        
        # Count non-null values for each column to debug alignment
        non_null_counts = aligned_df.count()
        print("  Column non-null counts after alignment:")
        for col, count in non_null_counts.items():
            print(f"    {col}: {count} / {len(aligned_df)}")
        
        # Drop any rows with missing values
        aligned_df_before_dropna = aligned_df.copy()
        aligned_df.dropna(inplace=True)
        
        # Debug info about how many rows were dropped
        dropped_rows = len(aligned_df_before_dropna) - len(aligned_df)
        print(f"  Dropped {dropped_rows} rows with missing values ({dropped_rows/len(aligned_df_before_dropna)*100:.1f}% of data)")
        
        if aligned_df.empty:
            print("  WARNING: All data was dropped due to missing values. Check date ranges and data alignment.")
            
        return aligned_df
    
    def split_data(self, df):
        """
        Split data into training+validation and testing sets based on time.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with aligned data
            
        Returns:
        --------
        train_val_df : pandas.DataFrame
            DataFrame for training and validation (Apr 2021 - Sep 2022)
        test_df : pandas.DataFrame
            DataFrame for testing (Oct 2022 - Apr 2023)
        """
        # Define split dates - ensure they have the same timezone as the DataFrame index
        if df.index.tzinfo is not None:
            train_val_start = pd.Timestamp('2021-04-01').tz_localize(df.index.tzinfo)
            train_val_end = pd.Timestamp('2022-09-30').tz_localize(df.index.tzinfo)
            test_start = pd.Timestamp('2022-10-01').tz_localize(df.index.tzinfo)
            test_end = pd.Timestamp('2023-04-30').tz_localize(df.index.tzinfo)
        else:
            # If the DataFrame index is timezone-naive, convert it to timezone-naive timestamps
            df.index = df.index.tz_localize(None) 
            train_val_start = pd.Timestamp('2021-04-01')
            train_val_end = pd.Timestamp('2022-09-30')
            test_start = pd.Timestamp('2022-10-01')
            test_end = pd.Timestamp('2023-04-30')
        
        # Create masks for the splits
        train_val_mask = (df.index >= train_val_start) & (df.index <= train_val_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        
        # Split the data
        train_val_df = df[train_val_mask].copy()
        test_df = df[test_mask].copy()
        
        return train_val_df, test_df
    
    def create_sequences(self, df):
        """
        Create encoder input sequences, decoder input sequences, and target output sequences.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with aligned data
            
        Returns:
        --------
        sequences : dict
            Dictionary containing:
            - X_encoder: Encoder input sequences (past observations and forecasts)
            - X_decoder: Decoder input sequences (current forecasts)
            - y: Target output sequences (forecast errors)
            - df: Original DataFrame
        """
        # Ensure data is sorted by time
        df = df.sort_index()
        
        # Lists to store sequences
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        
        # Create sequences
        for i in range(len(df) - self.sequence_length):
            # Encoder input: sequence_length hours of history
            encoder_input = []
            
            # For each timestep in the encoder sequence
            for j in range(self.sequence_length):
                # Get timestamp for this step
                idx = i + j
                
                # Features for each timestep: USGS observed, NWM 1h forecast, 1h error
                features = [
                    df.iloc[idx]['usgs_observed'],
                    df.iloc[idx]['nwm_lead_1'],
                    df.iloc[idx]['error_lead_1'],
                ]
                
                encoder_input.append(features)
            
            # Decoder input: Current NWM forecasts for lead times 1-18
            decoder_input = []
            for lead in range(1, 19):
                col_name = f'nwm_lead_{lead}'
                if col_name in df.columns:
                    decoder_input.append(df.iloc[i + self.sequence_length][col_name])
            
            # Target: Actual forecast errors for lead times 1-18
            target = []
            for lead in range(1, 19):
                error_col = f'error_lead_{lead}'
                if error_col in df.columns:
                    target.append(df.iloc[i + self.sequence_length][error_col])
            
            # Add sequences to lists
            if len(decoder_input) == 18 and len(target) == 18:
                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                targets.append(target)
        
        # Convert to numpy arrays
        X_encoder = np.array(encoder_inputs)
        X_decoder = np.array(decoder_inputs)
        y = np.array(targets)
        
        return {
            'X_encoder': X_encoder,
            'X_decoder': X_decoder,
            'y': y,
            'df': df
        }
    
    def scale_data(self, train_data, test_data=None):
        """
        Scale the data using StandardScaler.
        
        Parameters:
        -----------
        train_data : dict
            Dictionary with training data sequences
        test_data : dict, optional
            Dictionary with test data sequences
            
        Returns:
        --------
        scaled_train_data : dict
            Dictionary with scaled training data
        scaled_test_data : dict, optional
            Dictionary with scaled test data
        scalers : dict
            Dictionary with fitted scalers
        """
        # Check if there's any data to scale
        if train_data['X_encoder'].size == 0 or train_data['X_decoder'].size == 0 or train_data['y'].size == 0:
            print("  Warning: Training data is empty. Cannot scale.")
            return train_data, test_data, {}
            
        # Create scalers
        encoder_scaler = StandardScaler()
        decoder_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # Reshape data for scaling
        X_encoder_shape = train_data['X_encoder'].shape
        X_decoder_shape = train_data['X_decoder'].shape
        y_shape = train_data['y'].shape
        
        # Fit and transform training data
        X_encoder_scaled = encoder_scaler.fit_transform(
            train_data['X_encoder'].reshape(-1, X_encoder_shape[-1])
        ).reshape(X_encoder_shape)
        
        X_decoder_scaled = decoder_scaler.fit_transform(
            train_data['X_decoder'].reshape(-1, 1)
        ).reshape(X_decoder_shape)
        
        y_scaled = target_scaler.fit_transform(
            train_data['y'].reshape(-1, 1)
        ).reshape(y_shape)
        
        # Create scaled training data dictionary
        scaled_train_data = {
            'X_encoder': X_encoder_scaled,
            'X_decoder': X_decoder_scaled,
            'y': y_scaled,
            'df': train_data['df']
        }
        
        # Scale test data if provided
        scaled_test_data = None
        if test_data is not None:
            # Check if test data is empty
            if test_data['X_encoder'].size == 0 or test_data['X_decoder'].size == 0 or test_data['y'].size == 0:
                print("  Warning: Test data is empty. Cannot scale.")
                return scaled_train_data, test_data, {'encoder': encoder_scaler, 'decoder': decoder_scaler, 'target': target_scaler}
                
            X_encoder_test_shape = test_data['X_encoder'].shape
            X_decoder_test_shape = test_data['X_decoder'].shape
            y_test_shape = test_data['y'].shape
            
            X_encoder_test_scaled = encoder_scaler.transform(
                test_data['X_encoder'].reshape(-1, X_encoder_test_shape[-1])
            ).reshape(X_encoder_test_shape)
            
            X_decoder_test_scaled = decoder_scaler.transform(
                test_data['X_decoder'].reshape(-1, 1)
            ).reshape(X_decoder_test_shape)
            
            y_test_scaled = target_scaler.transform(
                test_data['y'].reshape(-1, 1)
            ).reshape(y_test_shape)
            
            scaled_test_data = {
                'X_encoder': X_encoder_test_scaled,
                'X_decoder': X_decoder_test_scaled,
                'y': y_test_scaled,
                'df': test_data['df']
            }
        
        # Store scalers
        scalers = {
            'encoder': encoder_scaler,
            'decoder': decoder_scaler,
            'target': target_scaler
        }
        
        return scaled_train_data, scaled_test_data, scalers
    
    def process_data(self, stream_ids):
        """
        Process all data for one or more streams.
        
        Parameters:
        -----------
        stream_ids : list
            List of stream identifiers
            
        Returns:
        --------
        processed_data : dict
            Dictionary with processed data for each stream
        """
        processed_data = {}
        
        for stream_id in stream_ids:
            print(f"Processing stream {stream_id}...")
            
            try:
                # Load data
                print("  Loading data...")
                usgs_df = self.load_usgs_data(stream_id)
                nwm_df = self.load_nwm_data(stream_id)
                
                # Clean and align data
                print("  Cleaning and aligning data...")
                aligned_df = self.clean_and_align_data(usgs_df, nwm_df)
                
                if aligned_df.empty:
                    print(f"  Warning: No aligned data found for stream {stream_id}. Skipping.")
                    continue
                
                # Split data
                print("  Splitting data...")
                train_val_df, test_df = self.split_data(aligned_df)
                
                if train_val_df.empty:
                    print(f"  Warning: No training/validation data found for stream {stream_id}.")
                
                if test_df.empty:
                    print(f"  Warning: No test data found for stream {stream_id}.")
                
                # Save processed DataFrames
                train_val_path = os.path.join(self.processed_data_path, f"{stream_id}_train_val_data.csv")
                test_path = os.path.join(self.processed_data_path, f"{stream_id}_test_data.csv")
                train_val_df.to_csv(train_val_path)
                test_df.to_csv(test_path)
                
                # Create sequences
                print("  Creating sequences...")
                train_val_sequences = self.create_sequences(train_val_df)
                test_sequences = self.create_sequences(test_df)
                
                # Scale data
                print("  Scaling data...")
                scaled_train_val, scaled_test, scalers = self.scale_data(train_val_sequences, test_sequences)
                
                # Store processed data
                processed_data[stream_id] = {
                    'train_val': scaled_train_val,
                    'test': scaled_test,
                    'scalers': scalers
                }
                
                print(f"  Done processing stream {stream_id}.")
            except Exception as e:
                print(f"  Error processing stream {stream_id}: {e}")
                continue
        
        return processed_data


if __name__ == "__main__":
    # Example usage
    raw_data_path = "../data/raw"
    processed_data_path = "../data/processed"
    
    # Create processor
    processor = DataPreprocessor(raw_data_path, processed_data_path)
    
    # Process data for all streams
    stream_ids = ["20380357", "21609641"]
    processed_data = processor.process_data(stream_ids)
    
    # Print data shapes for successfully processed streams
    for stream_id in stream_ids:
        if stream_id in processed_data:
            print(f"\nStream {stream_id} data shapes:")
            print(f"  Training/Validation:")
            print(f"    X_encoder: {processed_data[stream_id]['train_val']['X_encoder'].shape}")
            print(f"    X_decoder: {processed_data[stream_id]['train_val']['X_decoder'].shape}")
            print(f"    y: {processed_data[stream_id]['train_val']['y'].shape}")
            print(f"  Test:")
            print(f"    X_encoder: {processed_data[stream_id]['test']['X_encoder'].shape}")
            print(f"    X_decoder: {processed_data[stream_id]['test']['X_decoder'].shape}")
            print(f"    y: {processed_data[stream_id]['test']['y'].shape}")
        else:
            print(f"\nStream {stream_id} was not successfully processed.")