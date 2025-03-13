#!/usr/bin/env python
"""
Description:
    apply different smoothing functions for four preprocessed datasets,
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import SimpleExpSmoothing

def compare_SES_HES_HWES(ori_data: pd.DataFrame,
                         SES_params: dict = {None},
                         HES_params: dict = {None},
                         HWES_params: dict = {None},
                         pred_period: int = 12):

    # Apply SES, HES, and HWES
    print(SES_params.get('fit'))
    fit_SES = SimpleExpSmoothing(ori_data).fit(**SES_params.get('fit'))
    fcast_SES = fit_SES.forecast(pred_period).rename("SES")

    fit_HES = Holt(ori_data, **HES_params.get('model')).fit(**HES_params.get('fit'))
    fcast_HES = fit_HES.forecast(pred_period).rename("HES")

    fit_HWES = ExponentialSmoothing(ori_data, **HWES_params.get('model')).fit(**HWES_params.get('fit'))
    fcast_HWES = fit_HWES.forecast(pred_period).rename("HWES")

    #Calculate error
    SES_MSE = mean_squared_error(fit_SES.fittedvalues, ori_data)
    HES_MSE = mean_squared_error(fit_HES.fittedvalues, ori_data)
    HWES_MSE = mean_squared_error(fit_HWES.fittedvalues, ori_data)

    print('Summary of paramters and errors from each of the models:')
    params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
    results = pd.DataFrame(index=[r"alpha", r"beta", r"phi", r"l0", r"b0", "MSE"],
                           columns=["SES model", "HES model", "HWES model"])

    results["SES model"] = [fit_SES.params[p] for p in params] + [SES_MSE]
    results["HES model"] = [fit_HES.params[p] for p in params] + [HES_MSE]
    results["HWES model"] = [fit_HWES.params[p] for p in params] + [HWES_MSE]
    print(results)
    print(ori_data)
    forecasts = pd.concat([fcast_SES, fcast_HES, fcast_HWES], axis=1)
    forecasts = forecasts.sort_index()
    print(forecasts)

    return results, forecasts


def main(args):
    # load the datasets
    sheets_name = ['MSTA', "CH4",'GMAF', "ET12" ]
    MSTA, CH4, GMAF, ET12 = pd.read_excel("Data_36516473.xlsx", sheet_name=sheets_name, parse_dates=["Date"], index_col= "Date").values()
    print(MSTA.index, CH4.index, GMAF.index, ET12.index)

    #Define model parameters
    optimize_options = {
        'SES':  True,
        #'SES': False,
        'HES': True,
        #'HES': False,
        'HWES': True,
        #'HWES': False
    }

    SES_params = {'fit':{'smoothing_level': 0.2} if not optimize_options.get('SES')
                  else {'optimized': True}}
    HES_params = {'model':{'exponential': False, 'damped_trend': False,
                           'initialization_method': 'estimated'},
                  'fit':{'smoothing_level': 0.2, 'smoothing_slope': 0.1, 'damping_slope': 0.1} if not optimize_options.get('HES')
                  else {'optimized': True}}
    HWES_params = {'model':{'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12, 'damped_trend':False},
                   'fit':{'smoothing_level': 0.9, 'smoothing_slope': 0.00001,
                       'smoothing_seasonal': 0.1, 'damping_slope': 0.1} if not optimize_options.get('HWES')
                   else {'optimized': True}}


    print(SES_params, HES_params, HWES_params)
    ori_data = MSTA.loc['1960':, 'Temperature(C)']
    MSE, forecasts = compare_SES_HES_HWES(ori_data, SES_params, HES_params, HWES_params, pred_period=50)
    # plot the forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(ori_data, label="Original Data", color="black", linestyle="--")
    plt.plot(forecasts["SES"], label="SES Forecast: optimization = " + str(optimize_options.get('SES')))
    plt.plot(forecasts["HES"], label="HES Forecast: optimization = " + str(optimize_options.get('HES')))
    plt.plot(forecasts["HWES"], label="HWES Forecast: optimization = "+ str(optimize_options.get('HWES')) )

    plt.title("Forecast Comparison")
    plt.xlim(datetime(1960, 1, 1), datetime(2029, 1, 1))
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(MSE)

    # Decompose the MSTA dataset
    # Additive model because the data has a trend and seasonality components.

    MSTA_decomposed =  seasonal_decompose(MSTA['Temperature(C)'], model = "addictive")

    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    # Plot the original data and decomposition components
    axes[0].plot(MSTA_decomposed.trend, label="Trend")
    axes[0].set_title("Trend Component")
    axes[1].plot(MSTA_decomposed.seasonal, label="Seasonality")
    axes[1].set_title("Seasonal Component")
    axes[1].set_xlim(datetime(1990, 1, 1), datetime(1995, 1, 1))
    axes[2].plot(MSTA_decomposed.resid, label="Residuals")
    axes[2].set_title("Residuals")
    axes[3].plot(MSTA_decomposed.observed, label="Original Series")
    axes[3].set_title("Original Series")

    plt.tight_layout()
    plt.show()

    MSTA["Diff"] = MSTA["Temperature(C)"].diff()
    plot_acf(MSTA["Temperature(C)"], lags=50)
    plt.show()

    # Smoothing functions




    # ACF plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(MSTA['Temperature(C)'], ax=axs[0, 0], lags=50)
    plot_acf(CH4['CH4(ppb)'], ax=axs[0, 1], lags=50)
    plot_acf(GMAF['Visitors(GMAF)'], ax=axs[1, 0], lags=50)
    plot_acf(ET12['Energy Consumption'], ax=axs[1, 1], lags=50)

    plt.show()

    # Smoothing functions
    # 1. Moving Average (MA)
    MSTA_ma = MSTA.rolling(window=7).mean()
    CH4_ma = CH4.rolling(window=7).mean()
    GMAF_ma = GMAF.rolling(window=7).mean()



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
