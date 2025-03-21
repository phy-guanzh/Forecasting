#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- ARIMA Forecasting for Global Temperature Anomaly (MSTA)
--
-- Description:
-- This script applies the ARIMA (AutoRegressive Integrated Moving Average) model to forecast the Global Mean Surface
-- Temperature Anomaly (MSTA) time series. The process includes:
--     - Box-Cox transformation to stabilize variance
--     - Augmented Dickey-Fuller (ADF) test for stationarity
--     - ACF/PACF plots to determine model order
--     - Grid search with cross-validation to identify optimal ARIMA(p,d,q)
--     - Final ARIMA model fitting and forecasting
--     - Visualization of forecast results and export to Excel
--
-- Content:
--     0. Set-up: Importing Necessary Libraries
--     1. Preprocessing: Box-Cox transformation and stationarity testing
--     2. ACF/PACF visualization
--     3. Grid Search: ARIMA model selection via AIC and CV MSE
--     4. Forecasting and Plotting
--
-- Maintainer:  Zhe GUAN
-- Contact:     zg2u24@soton.ac.uk
--
------------------------------------------------------------------------------------------------------------------------
"""

########################################################################
# 0. Importing Necessary Libraries
########################################################################
import sys, os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

########################################################################
# 1. Utility Function: Non-Positive Value Transformation
########################################################################
def non_positive_trans(ori_data: pd.DataFrame):
    min_val = ori_data.min()
    ori_data_adj = ori_data.copy()
    if min_val <= 0:
        shift_value = abs(min_val) + 1
        ori_data_adj += shift_value
    return ori_data_adj

########################################################################
# 2. Utility Function: ACF & PACF Plotting
########################################################################
def acf_pacf(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, ax=axes[0])
    plot_pacf(data, ax=axes[1])
    axes[0].set_title("ACF Plot")
    axes[1].set_title("PACF Plot")
    plt.savefig("ARIMA/ACF_PACF_after_diff.pdf")
    plt.show()
    return True

########################################################################
# 3. Utility Function: ADF Stationarity Test
########################################################################
def adf_test(data):
    result = adfuller(data)
    print("ADF Test Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("→ Series is Stationary")
    else:
        print("→ Series is Non-Stationary")
    return True

########################################################################
# 4. Train ARIMA Model with CV
########################################################################
def train_arima(data, order, tscv):
    cv_mse = []
    for train_idx, test_idx in tscv.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test, forecast)
            cv_mse.append(mse)
            if len(cv_mse) == 1:
                aic = model_fit.aic
        except:
            return np.inf, np.inf
    return np.mean(cv_mse), aic

########################################################################
# 5. Grid Search for Best ARIMA(p,d,q)
########################################################################
def search_best_arima(data, p_values, d_values, q_values, n_splits=5):
    pdq_combinations = list(itertools.product(q_values, p_values, d_values))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_aic, best_mse = float("inf"), float("inf")
    best_order, best_model = None, None
    results = []

    print("Starting ARIMA parameter search...")
    for q, p, d in pdq_combinations:
        order = (p, d, q)
        mean_cv_mse, aic = train_arima(data, order, tscv)
        if np.isinf(aic) or np.isinf(mean_cv_mse): continue
        results.append((q, p, d, aic, mean_cv_mse))
        print(f"ARIMA{order} - AIC: {aic:.2f}, CV MSE: {mean_cv_mse:.5f}")
        if aic < best_aic and mean_cv_mse < best_mse:
            best_aic, best_mse = aic, mean_cv_mse
            best_order = order
            best_model = ARIMA(data, order=order).fit()

    results_df = pd.DataFrame(results, columns=["q", "p", "d", "AIC", "MSE"]).sort_values(by=["q", "p", "d"])
    print("Best ARIMA Model:", best_order)
    print("Best AIC:", best_aic)
    print("Best Cross-Validation MSE:", best_mse)
    print(best_model.summary())

    # Plot AIC and MSE trends
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(["AIC", "MSE"], 1):
        plt.subplot(2, 1, i)
        for q in q_values:
            subset = results_df[results_df["q"] == q]
            plt.plot(subset["p"], subset[metric], marker="o", label=f"q={q}")
        plt.xlabel("p")
        plt.ylabel(metric)
        plt.title(f"{metric} Trend Across ARIMA Parameters")
        plt.legend()
    plt.tight_layout()
    plt.savefig("ARIMA/ARIMA_trends_parameters.pdf")
    plt.show()

    return best_order, best_model

########################################################################
# 6. Main Execution: Loading, Transforming, Fitting, Forecasting
########################################################################
def main(args):
    # Load Dataset
    MSTA = pd.read_excel("Data_36516473.xlsx", sheet_name=["MSTA"], parse_dates=["Date"], index_col="Date")['MSTA']['Temperature(C)']

    # Apply transformation if necessary
    if MSTA.min() <= 0:
        MSTA = non_positive_trans(MSTA.copy())

    # Box-Cox transformation
    transformed_MSTA, lambda_MSTA = stats.boxcox(MSTA)
    transformed_MSTA = pd.Series(transformed_MSTA, index=MSTA.index)
    print("Lambda (Box-Cox):", lambda_MSTA)

    # Visualize variance stabilization
    plt.figure(figsize=(12,6))
    plt.plot(MSTA.rolling(12).std(), label="Original STD", color="red")
    plt.plot(transformed_MSTA.rolling(12).std(), label="Box-Cox STD", color="green")
    plt.title("Rolling STD Before and After Box-Cox")
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.savefig("ARIMA/SD_box_cox.pdf")
    plt.show()

    # Stationarity testing
    adf_test(transformed_MSTA)
    transformed_MSTA_diff1 = transformed_MSTA.diff().dropna()
    print("\nADF Test After First Differencing:")
    adf_test(transformed_MSTA_diff1)

    # ACF/PACF for differenced series
    acf_pacf(transformed_MSTA_diff1)

    # Optional: Grid search
    best_order, best_model = search_best_arima(transformed_MSTA, range(1, 12), [1], range(1, 3))

    # Final ARIMA fitting (manual order)
    model = ARIMA(MSTA, order=(8, 1, 2), trend='t')
    model_fit = model.fit()

    # Forecasting
    forecast = model_fit.forecast(steps=12)

    # Plot forecast results
    plt.figure(figsize=(12, 6))
    plt.plot(MSTA, label="Original Data")
    plt.plot(forecast, label="MSTA Forecast")
    plt.title("Global Mean Surface Temperature Anomaly Forecast with ARIMA")
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.savefig("ARIMA/Forecast_ARIMA_MSTA.pdf")
    plt.show()

    # Export forecast
    forecast.to_excel("MSTA_forecasts_ARIMA.xlsx")
    return "ARIMA Forecast Done!"

########################################################################
# Script Execution
########################################################################
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
