import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def fit_ma_model(series: pd.Series, q: int):
    """
    Fit a Moving Average (utils) model of order q.

    Parameters:
        series (pd.Series): Time series data
        q (int): utils order

    Returns:
        ARIMAResults: Fitted model
    """
    model = ARIMA(series, order=(0, 0, q))
    results = model.fit()
    return results


def select_best_ma_order(series: pd.Series, max_q: int = 20):
    """
    Select the best utils order using AIC.

    Parameters:
        series (pd.Series): Time series data
        max_q (int): Maximum utils order to try

    Returns:
        int: Best order q
    """
    best_aic = np.inf
    best_order = 0
    for q in range(1, max_q + 1):
        try:
            results = fit_ma_model(series, q)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = q
        except Exception:
            continue
    return best_order


def walk_forward_validation(series: pd.Series, q: int, train_size: float = 0.8):
    """
    Walk-forward validation for utils model.

    Parameters:
        series (pd.Series): Time series data
        q (int): utils order
        train_size (float): Fraction of data used for training

    Returns:
        pd.Series: Predicted values
    """
    train_len = int(len(series) * train_size)
    train, test = series[:train_len], series[train_len:]
    history = train.tolist()
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(0, 0, q))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
    return pd.Series(predictions, index=test.index)
