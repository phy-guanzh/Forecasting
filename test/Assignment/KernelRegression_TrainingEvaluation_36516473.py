#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- Global Temperature Anomaly Forecasting using Kernel Ridge Regression (KRR)
--
-- Description:
-- This script builds a Kernel Ridge Regression model to forecast Global Mean Surface Temperature Anomalies (MSTA).
-- It leverages three predictor time series:
--     - Atmospheric CH4 levels
--     - UK outbound tourism (GMAF)
--     - UK energy consumption (ET12)
--
-- Key Steps:
--     1. Load and preprocess historical data (scaling)
--     2. Use GridSearchCV with TimeSeriesSplit to optimize KRR hyperparameters (kernel, alpha, gamma, degree)
--     3. Fit model on full data and evaluate on historical set
--     4. Forecast the next 12 months using smoothed forecasts of predictors (from exponential smoothing models)
--     5. Plot historical fit and forward forecasts for visual comparison
--
-- Content:
--     0. Data Loading and Scaling
--     1. Model Selection via Grid Search
--     2. Forecasting Future MSTA using Predicted Inputs
--     3. Visualization and Comparison
--
-- Maintainer:  Zhe GUAN
-- Contact:     zg2u24@soton.ac.uk
--
------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

########################################################################
# 0. Data Loading and Preprocessing
########################################################################
# Load aligned regression dataset
df = pd.read_excel("./Exported_Data/regression_data.xlsx", parse_dates=["Date"], index_col="Date")

# Select predictors and target
X = df[["CH4", "GMAF", "ET12"]]  # Independent variables
y = df["MSTA"]                   # Target variable

# Use TimeSeriesSplit for time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Standardize features for kernel-based methods
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

########################################################################
# 1. Kernel Ridge Regression with Grid Search
########################################################################
# Define parameter grid
param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1, 10],    # used by rbf/poly/sigmoid
    'degree': [2, 3, 4]                    # only used for poly
}

# Initialize model and grid search
krr = KernelRidge()
grid_search = GridSearchCV(
    estimator=krr,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

# Perform grid search
grid_search.fit(X_scaled, y)
print("Best Parameters:", grid_search.best_params_)
print("Best Score (neg MSE):", grid_search.best_score_)

# Fit the best model on full historical data
best_model = grid_search.best_estimator_
best_model.fit(X_scaled, y)

with open('Models/model_with_kernel.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Evaluate model
y_pred = best_model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Final MSE:", mse)
print("Final R^2:", r2)

y_pred_series = pd.Series(y_pred, index=X.index, name="Predicted_MSTA")

########################################################################
# 2. Forecasting Future Values Using Forecasted Inputs
########################################################################
# Define future prediction index
future_index = pd.date_range(start='2023-11-01', periods=26, freq='MS')

# Load forecasted CH4, GMAF, and ET12 from exponential smoothing outputs
ch4_forecast = pd.read_excel("./Exported_Data/CH4_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2023-10-01':, 'HWES']
gmaf_forecast = pd.read_excel("./Exported_Data/GMAF_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2023-10':, 'HWES']
et12_forecast = pd.read_excel("./Exported_Data/ET12_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2023-10':, 'HWES']
msta_forecast = pd.read_excel("./Exported_Data/MSTA_forecasts_ARIMA.xlsx", index_col=0, parse_dates=True).loc['2023-10':, 'predicted_mean']

# Construct future input matrix
X_future_raw = pd.DataFrame({
    "CH4": ch4_forecast.loc[future_index],
    "GMAF": gmaf_forecast.loc[future_index],
    "ET12": et12_forecast.loc[future_index]
}, index=future_index)

# Scale future inputs using the same scaler
X_future_scaled = scaler.transform(X_future_raw)

# Predict MSTA using kernel model
y_future_kernel_pred = best_model.predict(X_future_scaled)
y_future_kernel_series = pd.Series(y_future_kernel_pred, index=future_index, name="KRR Forecast")

########################################################################
# 3. Visualization of In-Sample Fit and Forecast
########################################################################
plt.figure(figsize=(12, 6))
plt.plot(X.index, y, label="Observed MSTA", color="black")
plt.plot(X.index, y_pred, label="KRR Fit (In-sample)", color="orange")
plt.plot(y_future_kernel_series.index, y_future_kernel_series, label="KRR Forecast (12 months)", linestyle="--", color="blue")
plt.axvline(x=y.index[-1], color='gray', linestyle=':', label="Forecast Start")
plt.title("Forecasting MSTA using Kernel Ridge Regression")
plt.xlabel("Date")
plt.ylabel("MSTA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Regression/MSTA_forecast_comparison.pdf")
plt.show()


