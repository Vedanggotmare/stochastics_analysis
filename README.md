# 📈 Volatility Forecasting on Irregular Financial Time Series

This project explores a hybrid modeling approach using **Lomb–Scargle Periodograms** (typically used in astronomy) and **GARCH(1,1)** models to analyze and forecast volatility in irregularly sampled financial time series.

## 💡 Motivation

Financial data is often assumed to be evenly spaced (daily, hourly), but in real-world scenarios — especially high-frequency trading or incomplete datasets — **irregular time series** frequently occur.

We leverage the **Lomb–Scargle Periodogram**, which excels at extracting periodic signals from irregular time series, to preprocess the data before applying a **GARCH(1,1)** model for volatility forecasting.

## 🛠️ Tools Used

- `R`
- `quantmod` – for stock data retrieval
- `lomb` – to compute Lomb–Scargle Periodograms
- `rugarch` – to implement GARCH(1,1) models
- `tseries`, `xts` – for time series handling

## 🔁 Workflow

1. **Pull TSLA stock data** from Yahoo Finance.
2. **Simulate irregular sampling** by randomly removing 40% of the dates.
3. **Apply Lomb–Scargle Periodogram** to extract dominant frequency components.
4. **Compute log returns** and fit a **GARCH(1,1)** model on the irregular data.
5. **Visualize conditional volatility** and analyze model fit.

## 📊 Results

- The **Lomb–Scargle Periodogram** identifies dominant periodicity in irregularly sampled financial returns.
- The **GARCH model**, trained on the preprocessed log returns, captures volatility clustering effectively.
- This method improves robustness compared to naive interpolation, especially under data sparsity.

## 🚀 Run This Project

1. Install R and RStudio (if not already installed).
2. Install required packages:

```r
install.packages(c("quantmod", "lomb", "rugarch", "tseries", "xts"))
