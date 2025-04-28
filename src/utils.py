<<<<<<< HEAD
import numpy as np


def correlation_coefficient(obs, pred):
    """Pearson correlation coefficient between obs and pred."""
    if len(obs) < 2:
        return np.nan
    return np.corrcoef(obs, pred)[0, 1]


def rmse(obs, pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((pred - obs)**2))


def pbias(obs, pred):
    """Percent bias."""
    # avoid division by zero
    denom = np.sum(obs)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(pred - obs) / denom


def nse(obs, pred):
    """Nash-Sutcliffe Efficiency."""
    denom = np.sum((obs - np.mean(obs))**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((obs - pred)**2) / denom
=======
"""
Utility functions for the runoff forecasting project, including evaluation metrics.
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union


def calculate_cc(observed, predicted):
    """Calculate the Pearson Correlation Coefficient (CC)."""
    if observed.ndim > 1: observed = observed.flatten()
    if predicted.ndim > 1: predicted = predicted.flatten()
    if len(observed) != len(predicted):
        raise ValueError("Observed and predicted arrays must have the same length.")
    if len(observed) < 2:
        return np.nan # Not defined for less than 2 points
    
    # Handle potential NaNs or constant series
    if np.all(np.isnan(observed)) or np.all(np.isnan(predicted)) or \
       np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
        
    # Mask NaNs if any exist
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(mask) < 2:
        return np.nan
        
    observed = observed[mask]
    predicted = predicted[mask]
    
    return np.corrcoef(observed, predicted)[0, 1]


def calculate_rmse(observed, predicted):
    """Calculate the Root Mean Square Error (RMSE)."""
    if observed.ndim > 1: observed = observed.flatten()
    if predicted.ndim > 1: predicted = predicted.flatten()
    if len(observed) != len(predicted):
        raise ValueError("Observed and predicted arrays must have the same length.")
        
    # Mask NaNs
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(mask) == 0:
        return np.nan
        
    observed = observed[mask]
    predicted = predicted[mask]
    
    return np.sqrt(np.mean((observed - predicted)**2))


def calculate_pbias(observed, predicted):
    """Calculate the Percent Bias (PBIAS)."""
    if observed.ndim > 1: observed = observed.flatten()
    if predicted.ndim > 1: predicted = predicted.flatten()
    if len(observed) != len(predicted):
        raise ValueError("Observed and predicted arrays must have the same length.")
        
    # Mask NaNs
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(mask) == 0:
        return np.nan
        
    observed = observed[mask]
    predicted = predicted[mask]
    
    sum_observed = np.sum(observed)
    if sum_observed == 0:
        # Avoid division by zero; PBIAS is undefined or could be considered infinite/NaN
        # depending on context. Returning NaN is safer.
        return np.nan 
        
    return 100 * np.sum(predicted - observed) / sum_observed


def calculate_nse(observed, predicted):
    """Calculate the Nash-Sutcliffe Efficiency (NSE)."""
    if observed.ndim > 1: observed = observed.flatten()
    if predicted.ndim > 1: predicted = predicted.flatten()
    if len(observed) != len(predicted):
        raise ValueError("Observed and predicted arrays must have the same length.")
        
    # Mask NaNs
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    if np.sum(mask) == 0:
        return np.nan
        
    observed = observed[mask]
    predicted = predicted[mask]
    
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - predicted)**2)
    denominator = np.sum((observed - mean_observed)**2)
    
    if denominator == 0:
        # Avoid division by zero. If observed is constant, NSE is undefined.
        # Can return -inf or NaN. Let's use NaN.
        return np.nan 
        
    return 1 - (numerator / denominator)


def calculate_metrics_over_windows(observed: np.ndarray, predicted: np.ndarray, 
                                  window_size: int = 30) -> Dict[str, List[float]]:
    """
    Calculate metrics (CC, RMSE, PBIAS, NSE) over sliding windows.
    
    Args:
        observed: Observed values
        predicted: Predicted values
        window_size: Size of sliding window
        
    Returns:
        Dictionary with metrics
    """
    n_samples = len(observed)
    metrics = {
        'CC': [],
        'RMSE': [],
        'PBIAS': [],
        'NSE': []
    }
    
    for i in range(0, n_samples - window_size + 1):
        o = observed[i:i+window_size]
        p = predicted[i:i+window_size]
        
        # Skip windows with constant values
        if np.std(o) == 0:
            continue
        
        metrics['CC'].append(calculate_cc(o, p))
        metrics['RMSE'].append(calculate_rmse(o, p))
        
        pbias_val = calculate_pbias(o, p)
        if not np.isnan(pbias_val):
            metrics['PBIAS'].append(pbias_val)
        
        nse_val = calculate_nse(o, p)
        if not np.isnan(nse_val):
            metrics['NSE'].append(nse_val)
    
    return metrics


def create_sequences(data_array: np.ndarray, window_size: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create input sequences from a NumPy array.
    
    Args:
        data_array: NumPy array with time series data (samples, features)
        window_size: Number of past time steps to use as input
        horizon: Number of future time steps (typically 1 when targets are selected separately)
        
    Returns:
        X: Input sequences (samples, window_size, features)
        y: Placeholder (empty array, as targets are handled separately)
        indices: Indices corresponding to the *end* of each sequence in the original data_array
    """
    X = []
    indices = []
    total_len = len(data_array)
    
    # Iterate up to the point where a full sequence (input window + horizon) can be formed
    for i in range(window_size, total_len - horizon + 1):
        # Input sequence is the window ending *before* the current step i
        input_seq = data_array[i-window_size:i]
        X.append(input_seq)
        # Store the index *at the end* of the input sequence (which corresponds to the target time step)
        indices.append(i - 1) # Index of the last element in the input window
    
    # Return X, an empty array for y (as it's handled outside), and the indices
    return np.array(X), np.array([]), np.array(indices)


def temporal_split(X: np.ndarray, y: np.ndarray, sequence_timestamps: pd.DatetimeIndex, 
                  split_date: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data temporally based on a date using provided timestamps.
    
    Args:
        X: Input sequences
        y: Target sequences
        sequence_timestamps: DatetimeIndex corresponding to each sequence (usually the end time)
        split_date: Date to split on
        
    Returns:
        X_train, y_train, X_test, y_test, train_indices, test_indices
    """
    if len(X) != len(sequence_timestamps) or len(y) != len(sequence_timestamps):
        raise ValueError("X, y, and sequence_timestamps must have the same length.")
        
    # Create mask based on split date
    train_mask = sequence_timestamps < split_date
    test_mask = sequence_timestamps >= split_date # Use >= for test set
    
    # Get original indices for saving
    original_indices = np.arange(len(sequence_timestamps))
    train_indices = original_indices[train_mask]
    test_indices = original_indices[test_mask]

    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    return X_train, y_train, X_test, y_test, train_indices, test_indices


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings for transformer architectures.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding array
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(positions * div_term)
    pos_enc[:, 1::2] = np.cos(positions * div_term)
    
    return pos_enc
>>>>>>> lstm_transformer
