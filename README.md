# ARIMA Crypto Price Forecast 📈

A time series forecasting project that uses the **ARIMA(1,1,1)** model to predict 
**Bitcoin (BTC)** and **Ethereum (ETH)** prices 14 days into the future.

## Project Overview
This project walks through the full data science workflow:
- Fetching live crypto price data using yfinance
- Testing for stationarity using the ADF test
- Differencing the series to achieve stationarity
- Identifying model parameters using ACF and PACF plots
- Fitting an ARIMA(1,1,1) model for both BTC and ETH
- Generating a 14-day forecast with 95% confidence intervals

## Charts Generated
| Chart | Description |
|-------|-------------|
| step2_raw_prices.png | Raw BTC and ETH historical prices |
| step4_differenced.png | First differenced series (stationary) |
| step5_acf_pacf.png | ACF and PACF plots for parameter selection |
| step8_final_forecast.png | Final 14-day forecast with confidence intervals |

## Technologies Used
- Python 3.11
- yfinance
- statsmodels
- pandas
- numpy
- matplotlib

## How to Run
1. Install dependencies:
```
pip install yfinance pandas numpy matplotlib statsmodels
```
2. Run the script:
```
python arima_crypto.py
```

## Results
- **BTC** forecast: ~$71,200 for the next 14 days
- **ETH** forecast: ~$2,203 for the next 14 days
- Confidence intervals widen daily reflecting growing uncertainty

## Disclaimer
This project is for educational purposes only. 
Do not use ARIMA forecasts as financial advice.
