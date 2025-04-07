"""
Unit tests for preprocessing module.
"""
import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import align_data, create_features, split_data, prepare_sequence_data

class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample NWM data
        self.nwm_data = pd.DataFrame({
            'datetime': [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(100)],
            'station_id': ['USGS-01234567'] * 100,
            'runoff_nwm': np.random.uniform(10, 100, 100)
        })
        
        # Create sample USGS data
        self.usgs_data = pd.DataFrame({
            'datetime': [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(100)],
            'station_id': ['USGS-01234567'] * 100,
            'runoff_usgs': np.random.uniform(10, 100, 100)
        })
        
        # Add some additional data not in NWM data
        self.usgs_data = pd.concat([
            self.usgs_data,
            pd.DataFrame({
                'datetime': [datetime(2022, 1, 5) + timedelta(hours=i) for i in range(20)],
                'station_id': ['USGS-76543210'] * 20,
                'runoff_usgs': np.random.uniform(10, 100, 20)
            })
        ]).reset_index(drop=True)
    
    def test_align_data(self):
        """Test data alignment function."""
        aligned_df = align_data(self.nwm_data, self.usgs_data)
        
        # Check that alignment happened correctly
        self.assertEqual(len(aligned_df), 100)  # Should only keep matching rows
        self.assertIn('runoff_nwm', aligned_df.columns)
        self.assertIn('runoff_usgs', aligned_df.columns)
        
    def test_create_features(self):
        """Test feature creation function."""
        # First align data
        aligned_df = align_data(self.nwm_data, self.usgs_data)
        
        # Then create features
        featured_df = create_features(aligned_df)
        
        # Check that time-based features were created
        self.assertIn('hour', featured_df.columns)
        self.assertIn('day', featured_df.columns)
        self.assertIn('month', featured_df.columns)
        self.assertIn('dayofweek', featured_df.columns)
        self.assertIn('is_weekend', featured_df.columns)
        
        # Check that error metrics were calculated
        self.assertIn('nwm_error', featured_df.columns)
        self.assertIn('nwm_error_pct', featured_df.columns)
        
    def test_split_data(self):
        """Test data splitting function."""
        # Get data ready for splitting
        aligned_df = align_data(self.nwm_data, self.usgs_data)
        featured_df = create_features(aligned_df)
        
        # Use a custom split date to ensure we have data in both sets
        split_date = '2022-01-03'
        train_df, val_df, test_df = split_data(featured_df, test_start_date=split_date)
        
        # Check that data was split
        self.assertTrue(len(train_df) > 0)
        self.assertTrue(len(val_df) > 0)
        self.assertTrue(len(test_df) > 0)
        
        # Check that test data is all on or after split date
        self.assertTrue(all(test_df['datetime'] >= pd.to_datetime(split_date)))
        
        # Check that train/val data is all before split date
        self.assertTrue(all(train_df['datetime'] < pd.to_datetime(split_date)))
        self.assertTrue(all(val_df['datetime'] < pd.to_datetime(split_date)))
        
    def test_prepare_sequence_data(self):
        """Test sequence preparation function."""
        # Create sample data
        X = np.random.random((100, 10))  # 100 samples, 10 features
        y = np.random.random(100)  # 100 target values
        sequence_length = 24
        
        # Create sequences
        X_seq, y_seq = prepare_sequence_data(X, y, sequence_length)
        
        # Check sequence shape
        self.assertEqual(X_seq.shape, (100 - sequence_length, sequence_length, 10))
        self.assertEqual(y_seq.shape, (100 - sequence_length,))
        
        # Check a specific sequence
        for i in range(len(X_seq)):
            np.testing.assert_array_equal(X_seq[i], X[i:i+sequence_length])
            self.assertEqual(y_seq[i], y[i+sequence_length])

if __name__ == '__main__':
    unittest.main()
