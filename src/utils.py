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
