#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- Forecasting Global Temperature Anomalies Using Pre-trained Regression Model
--
-- Description:
-- This script applies a previously trained linear regression model (with lagged interaction terms)
-- to forecast Global Mean Surface Temperature Anomaly (MSTA) for the year 2025. It integrates
-- exponential smoothing-based forecasts of the predictors (CH4, GMAF, ET12), applies required
-- lagging and interaction transformations, standardizes the features, and predicts MSTA.
--
-- It then compares the regression-based forecast with an independent ARIMA-based forecast of MSTA.
-- Additionally, it visualizes predictions for both 2025 and the full historical period since 1960.
--
-- Content:
--    0. Load Data: Historical + forecasted values of CH4, GMAF, ET12, and ARIMA MSTA
--    1. Combine and align predictor forecasts with historical series
--    2. Apply lag and interaction transformations as used in model training
--    3. Standardize forecast data and generate prediction input
--    4. Load trained regression model and produce MSTA forecasts
--    5. Plot and compare with ARIMA forecast for 2025
--    6. Plot full timeline of model fit vs. actual MSTA (from 1960 onwards)
--
-- Maintainer:  Zhe GUAN
-- Contact:     zg2u24@soton.ac.uk
--
------------------------------------------------------------------------------------------------------------------------
"""

import sys
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main(args):
    # Load datasets from Excel (historical and forecasted data)
    sheets_name = ['MSTA', "CH4", 'GMAF', "ET12"]
    data_dict = pd.read_excel("./Exported_Data/Data_36516473.xlsx", sheet_name=sheets_name, parse_dates=["Date"], index_col=0)

    # Extract individual series
    MSTA = data_dict['MSTA']['Temperature(C)'].squeeze()
    CH4 = data_dict['CH4']['CH4(ppb)'].squeeze()
    GMAF = data_dict['GMAF']['Visitors(GMAF)'].squeeze()
    ET12 = data_dict['ET12']['Energy Consumption'].squeeze()

    # Load forecasted CH4, GMAF, ET12 (from HWES) and MSTA (from ARIMA)
    ch4_forecast = pd.read_excel("./Exported_Data/CH4_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2024-10-01':,'HWES'].squeeze()
    gmaf_forecast = pd.read_excel("./Exported_Data/GMAF_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2024':,'HWES']
    et12_forecast = pd.read_excel("./Exported_Data/ET12_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2025':,'HWES'].squeeze()
    msta_forecast = pd.read_excel("./Exported_Data/MSTA_forecasts_ARIMA.xlsx", index_col=0, parse_dates=True).loc['2025':,'predicted_mean'].squeeze()

    # Combine historical and forecasted data
    CH4_combined = pd.concat([CH4, ch4_forecast])
    GMAF_combined = pd.concat([GMAF, gmaf_forecast])
    ET12_combined = pd.concat([ET12, et12_forecast])

    # Load pre-trained regression model (with lags and interaction terms)
    with open('Models/model_with_lags.pkl', 'rb') as f:
        model_with_lags = pickle.load(f)

    # Apply lag transformation based on training configuration
    lags = {'CH4': 0, 'GMAF': 18, 'ET12': 18}
    forecast_data = pd.DataFrame({'CH4': CH4_combined, 'GMAF': GMAF_combined, 'ET12': ET12_combined})

    for feature, lag in lags.items():
        if lag > 0:
            forecast_data[f'{feature}_lag{lag}'] = forecast_data[feature].shift(lag)

    forecast_data = forecast_data.dropna()

    # Add interaction terms (same as training)
    forecast_data['CH4_x_ET12_lag18'] = forecast_data['CH4'] * forecast_data['ET12_lag18']
    forecast_data['CH4_x_GMAF_lag18'] = forecast_data['CH4'] * forecast_data['GMAF_lag18']
    forecast_data['GMAF_lag18_x_ET12_lag18'] = forecast_data['GMAF_lag18'] * forecast_data['ET12_lag18']
    forecast_data = forecast_data.drop(['GMAF', 'ET12'], axis=1)

    # Standardize features (fresh scaler for evaluation, same logic as in training)
    scaler = StandardScaler()
    scaled_forecast_data = pd.DataFrame(
        scaler.fit_transform(forecast_data),
        columns=forecast_data.columns,
        index=forecast_data.index
    )

    # Add intercept term
    X_forecast = sm.add_constant(scaled_forecast_data)

    # Predict MSTA using regression model
    forecast_predictions = model_with_lags.predict(X_forecast).loc["2025-02":]

    # Compare regression-based predictions with ARIMA forecast
    comparison_df = pd.DataFrame({
        'MSTA Forecast': msta_forecast.loc[forecast_predictions.index],
        'Model Prediction': forecast_predictions
    })

    # Plot comparison (ARIMA vs. regression)
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df.index, comparison_df['MSTA Forecast'], label='MSTA Forecast (ARIMA)', color='blue')
    plt.plot(comparison_df.index, comparison_df['Model Prediction'], label='Regression Model', color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('MSTA Forecast vs. Regression Model Prediction (2025)')
    plt.savefig("Regression/Regression1_forecast.pdf")
    plt.show()

    # Plot full series (1960–2025) model prediction vs actual
    msta_1960_onwards = MSTA.loc['1960':]
    forecast_full = model_with_lags.predict(X_forecast).loc['1960':]

    comparison_full = pd.DataFrame({
        'MSTA Forecast': msta_1960_onwards,
        'Model Prediction': forecast_full
    })

    plt.figure(figsize=(10, 6))
    plt.plot(comparison_full.index, comparison_full['MSTA Forecast'], label='MSTA (True)', color='blue')
    plt.plot(comparison_full.index, comparison_full['Model Prediction'], label='Regression Model', color='red', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Full Historical Comparison: MSTA vs. Regression Model (1960–2025)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Regression/Regression1_forecast_1990.pdf")
    plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
