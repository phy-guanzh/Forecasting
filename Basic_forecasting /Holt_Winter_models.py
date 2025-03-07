#!/usr/bin/env python
"""
Description:
Holt_winter models for CementProduction
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Exponential Smoothing
from statsmodels.tsa.api import ExponentialSmoothing

# Forecasting error
from sklearn.metrics import mean_squared_error

# ACF
from statsmodels.graphics.tsaplots import plot_acf

def main(args):
    # Load data
    data = pd.read_excel("./CementProduction.xls", index_col=0, header=None, parse_dates=True)[:-5].squeeze()
    #data.columns = ["Data"]
    data = pd.to_numeric(data)

    freq = pd.infer_freq(data.index)
    data.index = pd.DatetimeIndex(data.index.values, freq=freq)

    # Holt-Winters model
    fit = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12).fit()
    fcast = fit.forecast(12).rename(r'Holt-Winters Exponential Smoothing add-add')

    fit1 = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods=12).fit()
    fcast1 = fit1.forecast(12).rename(r'Holt-Winters Exponential Smoothing mul-mul')


    # Plotting
    fig, ax = plt.subplots(4, 1, figsize=(12, 10))
    data.plot(ax=ax[0], title='Cement Production', label='Original data', color="red")

    fit.fittedvalues.plot(ax=ax[0], color="blue")
    fcast.plot(ax=ax[0], color="blue", legend=True)

    fit1.fittedvalues.plot(ax=ax[0], color="green")
    fcast1.plot(ax=ax[0], color="green", legend=True)

    # ACF
    plot_acf(data, ax=ax[1])

    residuals = data - fit.fittedvalues
    residuals1 = data - fit1.fittedvalues
    plot_acf(residuals, ax=ax[2])
    plot_acf(residuals1, ax=ax[3])
    plt.tight_layout()
    plt.show()

    #MSE Mean Squared Error
    mse = mean_squared_error(data, fit.fittedvalues)
    print(f"Mean Squared Error add-add: {mse}")

    mse1 = mean_squared_error(data, fit1.fittedvalues)
    print(f"Mean Squared Error mul-mul: {mse1}")



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
