# AMD Stock Price Analysis with Moving Average (MA) Model

This repository demonstrates **time series analysis** and **forecasting** of AMD stock prices using the **Moving Average (MA) model** in Python. The project fetches historical stock data from Yahoo Finance and applies **ARIMA (MA) modeling** to analyze trends and forecast future prices.

---

## Features

* Download historical AMD stock price data using **`yfinance`**
* Calculate and visualize **rolling averages** to smooth fluctuations
* Plot **ACF (Autocorrelation Function)** and **PACF (Partial Autocorrelation Function)** to analyze dependencies
* Fit **Moving Average (MA) model** using **`statsmodels`**
* Predict historical data to validate model accuracy
* Forecast stock prices for the next 30 business days
* Clear visualizations of original data, rolling averages, predictions, and forecast

---

## Technologies Used

* Python 3.8+
* `pandas` for data manipulation
* `numpy` for numerical computation
* `matplotlib` for visualization
* `yfinance` for historical stock data
* `statsmodels` for time series modeling (ARIMA/MA)
* `warnings` to filter unnecessary output

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/MohamedSLITI/MovingAverage.git
cd MovingAverage
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo script:

```bash
python examples/demo_ma.py
```

4. The script will generate:

* Closing price plot
* Rolling average plot
* ACF and PACF plots
* Predictions vs actual stock prices
* Forecast plot for the next 30 business days

---

## Example Output

* **Green line:** MA model predicted values
* **Blue line:** Original AMD stock price
* **Orange line:** Rolling average
* **Red line:** Forecasted future values

---

## Notes

* The MA model works best for **short-term forecasting**.
* Ensure a stable internet connection for fetching stock data from Yahoo Finance.
* Python 3.8+ is recommended due to library compatibility.

---

## Dependencies

Create a `requirements.txt` with:

```
pandas
numpy
matplotlib
statsmodels
yfinance
```

Install using:

```bash
pip install -r requirements.txt
```

---

## References

* [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/tsa.html)
* [Yahoo Finance API (`yfinance`)](https://pypi.org/project/yfinance/)
* [Time Series Analysis Concepts](https://www.geeksforgeeks.org/machine-learning/understanding-the-moving-average-ma-in-time-series-data/)

---


