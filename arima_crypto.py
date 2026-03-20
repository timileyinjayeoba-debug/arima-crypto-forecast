import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pull 1 year of daily BTC and ETH closing prices
btc = yf.download("BTC-USD", start="2025-01-01", end="2026-03-19")["Close"]
eth = yf.download("ETH-USD", start="2025-01-01", end="2026-03-19")["Close"]

# Quick look
print("=== BTC - Last 5 rows ===")
print(btc.tail())

print("\n=== ETH - Last 5 rows ===")
print(eth.tail())

print("\n=== Missing values? ===")
print("BTC nulls:", btc.isnull().sum())
print("ETH nulls:", eth.isnull().sum())

print("\n=== Shape of data ===")
print("BTC rows:", btc.shape[0])
print("ETH rows:", eth.shape[0])






# ── STEP 2: Visualize raw prices ──────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# BTC plot
axes[0].plot(btc, color='orange', linewidth=1.5)
axes[0].set_title('Bitcoin (BTC-USD) Daily Closing Price', fontsize=14)
axes[0].set_ylabel('Price (USD)')
axes[0].grid(True, alpha=0.3)

# ETH plot
axes[1].plot(eth, color='blue', linewidth=1.5)
axes[1].set_title('Ethereum (ETH-USD) Daily Closing Price', fontsize=14)
axes[1].set_ylabel('Price (USD)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step2_raw_prices.png')
plt.show()
plt.close()
print("Step 2 chart saved!")



# ── STEP 3: Stationarity Test (ADF) ───────────────────────────────────────
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f"\n=== ADF Test: {name} ===")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-value       : {result[1]:.4f}")
    print(f"Critical Values:")
    for key, val in result[4].items():
        print(f"   {key}: {val:.4f}")
    if result[1] < 0.05:
        print(f">> RESULT: {name} is STATIONARY ✅")
    else:
        print(f">> RESULT: {name} is NOT STATIONARY ❌ — differencing needed")

adf_test(btc, "BTC")
adf_test(eth, "ETH")




# ── STEP 4: Differencing to achieve stationarity ──────────────────────────

# First difference
btc_diff = btc.diff().dropna()
eth_diff = eth.diff().dropna()

# Run ADF test again on differenced series
print("\n--- After First Differencing ---")
adf_test(btc_diff, "BTC (differenced)")
adf_test(eth_diff, "ETH (differenced)")

# Plot the differenced series
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(btc_diff, color='orange', linewidth=1)
axes[0].set_title('BTC - First Difference (Returns)', fontsize=14)
axes[0].set_ylabel('Price Change (USD)')
axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(eth_diff, color='blue', linewidth=1)
axes[1].set_title('ETH - First Difference (Returns)', fontsize=14)
axes[1].set_ylabel('Price Change (USD)')
axes[1].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step4_differenced.png')
plt.show()
plt.close()
print("Step 4 chart saved!")


# ── STEP 5: ACF and PACF plots ─────────────────────────────────────────────
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# BTC ACF and PACF
plot_acf(btc_diff, lags=30, ax=axes[0, 0], title='BTC - ACF (differenced)')
plot_pacf(btc_diff, lags=30, ax=axes[0, 1], title='BTC - PACF (differenced)')

# ETH ACF and PACF
plot_acf(eth_diff, lags=30, ax=axes[1, 0], title='ETH - ACF (differenced)')
plot_pacf(eth_diff, lags=30, ax=axes[1, 1], title='ETH - PACF (differenced)')

plt.tight_layout()
plt.savefig('step5_acf_pacf.png')
plt.show()
plt.close()
print("Step 5 chart saved!")







# ── STEP 6: Fit ARIMA(1,1,1) model ────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA

print("\n========================================")
print("Fitting ARIMA(1,1,1) for BTC...")
print("========================================")
btc_model = ARIMA(btc, order=(1, 1, 1))
btc_fitted = btc_model.fit()
print(btc_fitted.summary())

print("\n========================================")
print("Fitting ARIMA(1,1,1) for ETH...")
print("========================================")
eth_model = ARIMA(eth, order=(1, 1, 1))
eth_fitted = eth_model.fit()
print(eth_fitted.summary())




# ── STEP 7: Forecast 14 days ───────────────────────────────────────────────
import pandas as pd

# BTC Forecast
btc_forecast = btc_fitted.get_forecast(steps=14)
btc_forecast_df = btc_forecast.summary_frame(alpha=0.05)
btc_forecast_df.index = pd.date_range(start=btc.index[-1] + pd.Timedelta(days=1), periods=14)

# ETH Forecast
eth_forecast = eth_fitted.get_forecast(steps=14)
eth_forecast_df = eth_forecast.summary_frame(alpha=0.05)
eth_forecast_df.index = pd.date_range(start=eth.index[-1] + pd.Timedelta(days=1), periods=14)

# Print forecast tables
print("\n========================================")
print("BTC 14-Day Forecast")
print("========================================")
print(btc_forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].round(2))

print("\n========================================")
print("ETH 14-Day Forecast")
print("========================================")
print(eth_forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].round(2))




# ── STEP 8: Final Forecast Chart ──────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# ── BTC Chart ──
# Plot last 60 days of historical data
btc_tail = btc.tail(60)
axes[0].plot(btc_tail, color='orange', linewidth=2, label='Historical BTC')

# Plot forecast mean
axes[0].plot(btc_forecast_df['mean'], color='red', linewidth=2,
             linestyle='--', label='Forecast')

# Plot confidence interval shading
axes[0].fill_between(btc_forecast_df.index,
                     btc_forecast_df['mean_ci_lower'],
                     btc_forecast_df['mean_ci_upper'],
                     color='red', alpha=0.15, label='95% Confidence Interval')

axes[0].set_title('Bitcoin (BTC) — ARIMA(1,1,1) 14-Day Forecast', fontsize=14)
axes[0].set_ylabel('Price (USD)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── ETH Chart ──
# Plot last 60 days of historical data
eth_tail = eth.tail(60)
axes[1].plot(eth_tail, color='blue', linewidth=2, label='Historical ETH')

# Plot forecast mean
axes[1].plot(eth_forecast_df['mean'], color='red', linewidth=2,
             linestyle='--', label='Forecast')

# Plot confidence interval shading
axes[1].fill_between(eth_forecast_df.index,
                     eth_forecast_df['mean_ci_lower'],
                     eth_forecast_df['mean_ci_upper'],
                     color='red', alpha=0.15, label='95% Confidence Interval')

axes[1].set_title('Ethereum (ETH) — ARIMA(1,1,1) 14-Day Forecast', fontsize=14)
axes[1].set_ylabel('Price (USD)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout(pad=4.0)
plt.savefig('step8_final_forecast.png')
plt.show()
plt.close()
print("\n✅ All done! Final forecast chart saved as step8_final_forecast.png")