import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (15, 5)

# -----------------------------
# Step 1: Download AMD stock data
# -----------------------------
AMD = yf.Ticker("AMD")
AMD_values = AMD.history(start="2020-01-01")

# -----------------------------
# Step 2: Fix DatetimeIndex frequency
# -----------------------------
AMD_values.index = pd.DatetimeIndex(AMD_values.index)
AMD_values = AMD_values.asfreq('B').fillna(method='ffill')  # B = business day

# -----------------------------
# Step 3: Plot closing price
# -----------------------------
AMD_values[['Close']].plot(title='AMD Closing Price')
plt.show()

# -----------------------------
# Step 4: Calculate 10-day rolling average
# -----------------------------
AMD_values['rolling_av'] = AMD_values['Close'].rolling(10).mean()
AMD_values[['Close', 'rolling_av']].plot(title='AMD Closing Price & 10-day Rolling Avg')
plt.show()


# -----------------------------
# Step 5: Plot ACF and PACF
# -----------------------------
def plot_acf_pacf(timeseries):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(timeseries, ax=ax1, lags=75)
    plot_pacf(timeseries, ax=ax2, lags=75)
    plt.show()


plot_acf_pacf(AMD_values['Close'])

# -----------------------------
# Step 6: Fit MA model
# -----------------------------
# Based on ACF, choose order q=55
MA_model = ARIMA(endog=AMD_values['Close'], order=(0, 0, 55))
results = MA_model.fit()

print(results.summary())

# -----------------------------
# Step 7: Predict past data for comparison
# -----------------------------
start_date = '2023-12-15'
end_date = '2024-02-05'
AMD_values['prediction'] = results.predict(start=start_date, end=end_date)

print(AMD_values[['Close', 'rolling_av', 'prediction']].tail(14))

# -----------------------------
# Step 8: Forecast future values
# -----------------------------
forecast_steps = 30  # next 30 business days
forecast_index = pd.date_range(start=AMD_values.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast = results.forecast(steps=forecast_steps)

# Plot original, rolling avg, predicted, and forecast
AMD_values[['Close', 'rolling_av', 'prediction']].plot(title='AMD Stock Forecast')
plt.plot(forecast_index, forecast, color='red', label='Forecast (next 30 days)')
plt.legend()
plt.show()
