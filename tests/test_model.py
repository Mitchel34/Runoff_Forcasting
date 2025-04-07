"""
Unit tests for model module.
"""
import unittest
import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import (
    create_lstm_model, create_gru_model, 
    create_transformer_model, create_hybrid_model
)

class TestModelCreation(unittest.TestCase):
    """Test cases for model creation functions."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.input_shape = (24, 10)  # 24 time steps, 10 features
        self.output_shape = 1
        
    def test_create_lstm_model(self):
        """Test LSTM model creation."""
        model = create_lstm_model(self.input_shape, self.output_shape)
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 24, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        
    def test_create_gru_model(self):
        """Test GRU model creation."""
        model = create_gru_model(self.input_shape, self.output_shape)
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 24, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        
    def test_create_transformer_model(self):
        """Test Transformer model creation."""
        model = create_transformer_model(self.input_shape, self.output_shape)
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 24, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        
    def test_create_hybrid_model(self):
        """Test hybrid CNN-LSTM model creation."""
        model = create_hybrid_model(self.input_shape, self.output_shape)
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 24, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

class TestModelPrediction(unittest.TestCase):
    """Test cases for model prediction functionality."""
    
    def setUp(self):
        """Set up common test parameters and a test model."""
        self.input_shape = (24, 10)
        self.batch_size = 16
        
        # Create a simple model for testing
        self.model = create_lstm_model(self.input_shape)
        
        # Create test data
        self.X_test = np.random.random((self.batch_size, 24, 10))
        
    def test_model_prediction(self):
        """Test that the model can make predictions."""
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (self.batch_size, 1))
        
    def test_prediction_range(self):
        """Test that predictions are within a reasonable range."""
        # Since we haven't trained the model, we just want to ensure
        # predictions don't explode and are of the right type
        predictions = self.model.predict(self.X_test)
        
        # Check prediction values
        self.assertTrue(np.all(np.isfinite(predictions)))  # No NaNs or infinities

if __name__ == '__main__':
    unittest.main()
