#!/usr/bin/env python
"""
Description:

"""
import sys
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import joblib  # 新增模型保存库
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


def main(args):
    sheets_name = ['MSTA', "CH4", 'GMAF', "ET12"]
    data_dict = pd.read_excel("Data_36516473.xlsx", sheet_name=sheets_name, parse_dates=["Date"],
                                          index_col=0)

    MSTA = data_dict['MSTA']['Temperature(C)'].squeeze()
    CH4 = data_dict['CH4']['CH4(ppb)'].squeeze()
    GMAF = data_dict['GMAF']['Visitors(GMAF)'].squeeze()
    ET12 = data_dict['ET12']['Energy Consumption'].squeeze()

    # 打印每个时间序列的索引
    print(MSTA)
    print(CH4)
    print(GMAF)
    print(ET12)

    ch4_forecast = pd.read_excel("CH4_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2024-10-01':,'HWES'].squeeze()
    gmaf_forecast = pd.read_excel("GMAF_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2024':,'HWES']
    et12_forecast = pd.read_excel("ET12_forecasts_smooth.xlsx", index_col=0, parse_dates=True).loc['2025':,'HWES'].squeeze()
    msta_forecast = pd.read_excel("MSTA_forecasts_ARIMA.xlsx", index_col=0, parse_dates=True).loc['2025':,'predicted_mean'].squeeze()



    print(ch4_forecast)
    print(gmaf_forecast)
    print(et12_forecast)
    print(msta_forecast)

    #combine data

    # 然后按日期索引进行合并
    print(CH4, ch4_forecast)

    CH4_combined = pd.concat([CH4, ch4_forecast], axis = 0)
    GMAF_combined = pd.concat([GMAF, gmaf_forecast], axis = 0)
    ET12_combined = pd.concat([ET12, et12_forecast], axis = 0)
    print(CH4_combined)


    with open('model_with_lags.pkl', 'rb') as f:
        model_with_lags = pickle.load(f)
    lags = {'CH4': 0, 'GMAF': 18, 'ET12': 18}  # Example lags for demonstration
    forecast_data = pd.DataFrame({
        'CH4': CH4_combined,
        'GMAF': GMAF_combined,
        'ET12': ET12_combined
    })

    # Step 5: Apply the lags to the forecast data (shift the data by the lag values)
    for feature, lag in lags.items():
        if lag > 0:
            forecast_data[f'{feature}_lag{lag}'] = forecast_data[feature].shift(lag)

    # Drop NaN values after lagging
    forecast_data = forecast_data.dropna()

    forecast_data['CH4_x_ET12_lag18'] = forecast_data['CH4'] * forecast_data['ET12_lag18']
    forecast_data['CH4_x_GMAF_lag18'] = forecast_data['CH4'] * forecast_data['GMAF_lag18']
    forecast_data['GMAF_lag18_x_ET12_lag18'] = forecast_data['GMAF_lag18'] * forecast_data['ET12_lag18']

    forecast_data = forecast_data.drop(['GMAF', 'ET12'],axis=1)

    print(forecast_data)
    # Standardize the forecast data using the same scaler as before
    scaler = StandardScaler()
    scaled_forecast_data = pd.DataFrame(
        scaler.fit_transform(forecast_data),
        columns=forecast_data.columns,
        index=forecast_data.index
    )

    # Step 6: Generate the features for prediction (same as used in training)
    X_forecast = sm.add_constant(scaled_forecast_data)
    print(X_forecast.columns)

    # Step 7: Make predictions using the loaded model
    forecast_predictions = model_with_lags.predict(X_forecast).loc["2025-02":]

    print("result:",forecast_predictions)
    print("msta:",msta_forecast)

    # Step 8: Compare the predictions with the actual MSTA forecast
    comparison_df = pd.DataFrame({
        'MSTA Forecast': msta_forecast.loc[forecast_predictions.index],
        'Model Prediction': forecast_predictions
    })

    # Display the comparison
    print(comparison_df)

    # Optional: Plot the results for visualization

    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df.index, comparison_df['MSTA Forecast'], label='MSTA Forecast', color='blue')
    plt.plot(comparison_df.index, comparison_df['Model Prediction'], label='Model Prediction', color='red',
             linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Temperature (C)')
    plt.title('MSTA Forecast vs. Model Prediction')
    plt.savefig("Regression/Regression1_forecast.pdf")
    plt.show()

    msta_1990_onwards = MSTA.loc['1960':]
    forecast_1990_onwards = model_with_lags.predict(X_forecast).loc['1960':]

    # Combine MSTA and model predictions for comparison
    comparison_1990_onwards = pd.DataFrame({
        'MSTA Forecast': msta_1990_onwards,
        'Model Prediction': forecast_1990_onwards
    })

    # Plot the comparison for the year 1990 onwards
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_1990_onwards.index, comparison_1990_onwards['Model Prediction'], label='Model Prediction',
            color='red', linestyle='--')
    plt.plot(comparison_1990_onwards.index, comparison_1990_onwards['MSTA Forecast'], label='MSTA Forecast',
             color='blue')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title('MSTA Forecast vs. Regression Model Prediction (1990 onwards)')
    plt.grid(True)
    plt.savefig("Regression/Regression1_forecast_1990.pdf")
    plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
