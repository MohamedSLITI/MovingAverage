import pandas as pd
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> bool:
    """
    Perform Augmented Dickey-Fuller test to check stationarity.

    Parameters:
        series (pd.Series): Time series data
        alpha (float): Significance level

    Returns:
        bool: True if series is stationary
    """
    result = adfuller(series)
    p_value = result[1]
    return p_value < alpha


def difference_series(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Difference a series to remove trend for stationarity.

    Parameters:
        series (pd.Series): Time series data
        periods (int): Number of periods to difference

    Returns:
        pd.Series: Differenced series
    """
    return series.diff(periods=periods).dropna()
