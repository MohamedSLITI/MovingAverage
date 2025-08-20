import pandas as pd


def forecast_ma_model(model, steps: int):
    """
    Forecast future values using a fitted utils model.

    Parameters:
        model: Fitted ARIMA model
        steps (int): Number of steps to forecast

    Returns:
        pd.Series: Forecasted values
    """
    forecasted_values = model.forecast(steps=steps)
    return pd.Series(forecasted_values)
