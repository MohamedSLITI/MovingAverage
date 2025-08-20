import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data import fetch_stock_data
from utils.preprocessing import difference_series, check_stationarity
from utils.ma_model import select_best_ma_order, fit_ma_model, walk_forward_validation
from utils.plot import plot_series
import pandas as pd

# Fetch data
df = fetch_stock_data("AMD", start="2020-01-01")
close = df['Close']

# Check stationarity
if not check_stationarity(close):
    close_diff = difference_series(close)
else:
    close_diff = close

# Select best utils order
best_q = select_best_ma_order(close_diff, max_q=60)
print(f"Best MA order: {best_q}")

# Fit model
model = fit_ma_model(close_diff, best_q)

# Walk-forward validation
predictions = walk_forward_validation(close_diff, best_q)

# Rolling mean
rolling = close_diff.rolling(window=10).mean()

# Plot
plot_series(close_diff, rolling=rolling, forecast=predictions)
