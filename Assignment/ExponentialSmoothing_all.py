#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- Time Series Forecasting Using Exponential Smoothing Models
--
-- Description:
-- Framework is provided for time series forecasting comparison using three different exponential smoothing models:
--    - Simple Exponential Smoothing (SES) for data without trend or seasonality.
--    - Holt's Exponential Smoothing (HES) for data with a trend component.
--    - Holt-Winters Exponential Smoothing (HWES) for data with both trend and seasonality.
--
-- The script loads 4 time series datasets(MSTA, CH4, GMAF, ET12), applies these models, evaluates their performance,
-- and visualizes the results. Note: we also handles special cases, such as non-positive values, which require
-- adjustments when using a multiplicative trend or seasonal component in HWES.
--
-- Content:
--    0. Set-up (Importing Necessary Libraries)
--    1. Function1: compare_SES_HES_HWES() - Model Training and Evaluation
--    2. Function2: Visualization
--    3. Main Function: Executing the Workflow
--
-- Maintainer:  Zhe GUAN
-- Contact:     zg2u24@soton.ac.uk
--
------------------------------------------------------------------------------------------------------------------------
"""

########################################################################
#Importing Necessary Libraries
########################################################################

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import SimpleExpSmoothing

########################################################################
#Function1: compare_SES_HES_HWES() - Model Training and Evaluation
########################################################################

def compare_SES_HES_HWES(ori_data: pd.DataFrame,
                         SES_params: dict = None,
                         HES_params: dict = None,
                         HWES_params: dict = None,
                         pred_period: int = 12,
                         title : str = None):

    # Apply SES, HES, and HWES
    print(SES_params.get('fit'))
    fit_SES = SimpleExpSmoothing(ori_data).fit(**SES_params.get('fit'))
    fcast_SES = fit_SES.forecast(pred_period).rename("SES")

    fit_HES = Holt(ori_data, **HES_params.get('model')).fit(**HES_params.get('fit'))
    fcast_HES = fit_HES.forecast(pred_period).rename("HES")

    #Handling Negative Values for HWES
    min_val = ori_data.min()
    ori_data_adj = ori_data.copy()
    operation = False
    if min_val <= 0:
        shift_value = abs(min_val) + 1
        ori_data_adj += shift_value
        operation = True

    fit_HWES = ExponentialSmoothing(ori_data_adj, **HWES_params.get('model')).fit(**HWES_params.get('fit'))
    fcast_HWES = fit_HWES.forecast(pred_period).rename("HWES")
    #Fits the HWES model and applies a shift correction if necessary.
    if operation:
        fcast_HWES -= shift_value

    #Calculate MSE, MAE and R^2
    SES_MSE = mean_squared_error(fit_SES.fittedvalues, ori_data)
    HES_MSE = mean_squared_error(fit_HES.fittedvalues, ori_data)
    HWES_MSE = mean_squared_error(fit_HWES.fittedvalues, ori_data)

    SES_MAE = mean_absolute_error(fit_SES.fittedvalues, ori_data)
    HES_MAE = mean_absolute_error(fit_HES.fittedvalues, ori_data)
    HWES_MAE = mean_absolute_error(fit_HWES.fittedvalues, ori_data)

    SES_R2 = r2_score(fit_SES.fittedvalues, ori_data)
    HES_R2 = r2_score(fit_HES.fittedvalues, ori_data)
    HWES_R2 = r2_score(fit_HWES.fittedvalues, ori_data)

    print(f'Summary of paramters and errors from each of the models for {title}:')
    params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
    results = pd.DataFrame(index=[r"alpha", r"beta", r"phi", r"l0", r"b0", "MSE", 'MAE', "R2"],
                           columns=["SES model", "HES model", "HWES model"])

    results["SES model"] = [fit_SES.params[p] for p in params] + [SES_MSE] + [SES_MAE] + [SES_R2]
    results["HES model"] = [fit_HES.params[p] for p in params] + [HES_MSE] + [HES_MAE] + [HES_R2]
    results["HWES model"] = [fit_HWES.params[p] for p in params] + [HWES_MSE] + [HWES_MAE] + [HWES_R2]
    print(results)

    ses_series = pd.concat([fit_SES.fittedvalues, fcast_SES]).rename("SES")
    hes_series = pd.concat([fit_HES.fittedvalues, fcast_HES]).rename("HES")
    hwes_series = pd.concat([fit_HWES.fittedvalues, fcast_HWES]).rename("HWES")


    forecasts = pd.concat([ses_series, hes_series, hwes_series], axis=1)
    forecasts = forecasts.sort_index()
    print(forecasts)

    return results, forecasts

################################################################
# Function2: Visualization
################################################################

def plots_series(ori_data, forecasts, optimize_options, title):

    #define titles for different datasets
    dict_title = {
        'MSTA': ['Global Mean Surface Temperature Anomaly Forecast', '\u00B0C'],
        'CH4': ['Global Monthly Atmospheric Carbon Dioxide Levels', 'parts per billion (ppb)'],
        'GMAF': ['UK visits abroad:All visits Thousands', "thousands"],
        'ET12': ['UK Inland Monthly Energy Consumption', 'million tonnes of oil equivalent (Mtoe)']
    }

    #Plotting Forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(ori_data, label="Original Data", color="black", linestyle="--")
    plt.plot(forecasts["SES"], label="SES Model: optimization = " + str(optimize_options.get('SES')))
    plt.plot(forecasts["HES"], label="HES Model: optimization = " + str(optimize_options.get('HES')))
    plt.plot(forecasts["HWES"], label="HWES Model: optimization = "+ str(optimize_options.get('HWES')) )

    # adjusting the x-axis label to date format
    plt.title(dict_title.get(title)[0])
    plt.xlim(pd.to_datetime(ori_data.index.min().value * 0.95),
             pd.to_datetime(ori_data.index.max().value * 1.05))
    plt.xlabel("Year")
    plt.ylabel(dict_title.get(title)[1])
    plt.legend()
    plt.grid(True)
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title}_forecast.pdf"))
    plt.show()

########################################################################
# Main Function: Executing the Workflow
########################################################################

def main(args):

    # Loading Data
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
    HWES_params = {'model':{'trend': 'mul', 'seasonal': 'mul', 'seasonal_periods': 12, 'damped_trend':False},
                   'fit':{'smoothing_level': 0.9, 'smoothing_slope': 0.00001,
                       'smoothing_seasonal': 0.1, 'damping_slope': 0.1} if not optimize_options.get('HWES')
                   else {'optimized': True}}


    #select datasets
    print(SES_params, HES_params, HWES_params)
    MSTA_selected = MSTA.loc['1960':, 'Temperature(C)']
    CH4_selected = CH4.loc[:, 'CH4(ppb)']
    GMAF_selected = GMAF.loc[:,"Visitors(GMAF)"]
    ET12_selected = ET12.loc[:,"Energy Consumption"]

    #Forecasting Each Dataset
    metrics_MSTA, forecasts_MSTA = compare_SES_HES_HWES(MSTA_selected, SES_params, HES_params, HWES_params, pred_period=13, title = "MSTA")
    metrics_CH4, forecast_CH4 = compare_SES_HES_HWES(CH4_selected, SES_params, HES_params, HWES_params, pred_period=13, title ="CH4")
    metrics_GMAF, forecasts_GMAF = compare_SES_HES_HWES(GMAF_selected, SES_params, HES_params, HWES_params, pred_period=13,title = "GMAF")
    metrics_ET12, forecasts_ET12 = compare_SES_HES_HWES(ET12_selected, SES_params, HES_params, HWES_params, pred_period=13, title ="ET12")

    #Generating Plots
    plots_series(MSTA_selected, forecasts_MSTA, optimize_options, "MSTA")
    plots_series(CH4_selected, forecast_CH4, optimize_options, "CH4")
    plots_series(GMAF_selected, forecasts_GMAF, optimize_options,"GMAF")
    plots_series(ET12_selected, forecasts_ET12, optimize_options,"ET12")



#Script Execution
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


########################################################################################################################
# ** Publisher's Imprint **
########################################################################################################################
__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
