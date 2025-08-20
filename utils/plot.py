import matplotlib.pyplot as plt
import pandas as pd


def plot_series(series: pd.Series, rolling: pd.Series = None, forecast: pd.Series = None,
                title: str = "Time Series Analysis"):
    """
    Plot original series, rolling average, and forecast.

    Parameters:
        series (pd.Series): Original time series
        rolling (pd.Series, optional): Rolling average
        forecast (pd.Series, optional): Forecasted values
        title (str): Plot title
    """
    plt.figure(figsize=(15, 5))
    plt.plot(series, label='Original', color='blue')
    if rolling is not None:
        plt.plot(rolling, label='Rolling Avg', color='orange')
    if forecast is not None:
        plt.plot(forecast, label='Forecast', color='red')
    plt.title(title)
    plt.legend()
    plt.show()
