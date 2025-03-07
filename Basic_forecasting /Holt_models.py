#!/usr/bin/env python
"""
Description:
Holt models for smoothing
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error


def main(args):
    # Load data
    data = pd.read_excel("EmploymentPrivateServices.xls", index_col=0, header=None, parse_dates=True).squeeze()
    data.columns = ["Data"]

    print(data)
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
    # Holt Models
    fit = Holt(data).fit(smoothing_level=0.5, smoothing_trend=0.5, optimized=False)
    fcast = fit.forecast(12).rename(r"holt ($\alpha$ = 0.5, $\beta$ = 0.5)")

    fit1 = Holt(data).fit(smoothing_level=0.8, smoothing_trend=0.8, optimized=False)
    fcast1 = fit.forecast(12).rename(r"holt ($\alpha$ = 0.8, $\beta$ = 0.2)")

    fit2 = Holt(data, damped_trend=True).fit()
    fcast2 = fit.forecast(12).rename(r"damped_trend")

    fit3 = Holt(data, exponential=True).fit()
    fcast3 = fit.forecast(12).rename(r"exponential")

    fit4 = Holt(data, exponential=True).fit(optimized=True)
    fcast4 = fit.forecast(12).rename(r"exponential")

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    data.plot(ax=ax, title='Employment in Private Services', label='Original data', color="red")
    fit.fittedvalues.plot(ax=ax, color="black")
    fcast.plot(ax=ax, color="blue", legend=True)

    fit1.fittedvalues.plot(ax=ax, color="red")
    fcast1.plot(ax=ax, color="red", legend=True)

    fit2.fittedvalues.plot(ax=ax, color="orange")
    fcast2.plot(ax=ax, color="orange", legend=True)

    fit3.fittedvalues.plot(ax=ax, color="yellow")
    fcast3.plot(ax=ax, color="yellow", legend=True)

    fit4.fittedvalues.plot(ax=ax, color="pink")
    fcast4.plot(ax=ax, color="pink", legend=True)

    plt.legend()
    plt.show()

    # mean squared error
    mse_holt = mean_squared_error(data.values, fit.fittedvalues.values)
    mse_holt1 = mean_squared_error(data.values, fit1.fittedvalues.values)
    mse_holt2 = mean_squared_error(data.values, fit2.fittedvalues.values)
    mse_holt3 = mean_squared_error(data.values, fit3.fittedvalues.values)
    mse_holt4 = mean_squared_error(data.values, fit4.fittedvalues.values)

    print(mse_holt)
    print('Summary of paramters and errors from each of the models:')
    params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
    results = pd.DataFrame(index=[r"alpha", r"beta", r"phi", r"l0", r"b0", "MSE"],
                           columns=["Holt model 0", "Holt model 1", "Holt model 2", "Holt model 3", "Holt model 4"])

    results["Holt model 0"] = [fit.params[p] for p in params] + [mse_holt]
    results["Holt model 1"] = [fit1.params[p] for p in params] + [mse_holt1]
    results["Holt model 2"] = [fit2.params[p] for p in params] + [mse_holt2]
    results["Holt model 3"] = [fit3.params[p] for p in params] + [mse_holt3]
    results["Holt model 4"] = [fit4.params[p] for p in params] + [mse_holt4]

    print(results)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


__maintainer__ = "Alain Zemkoho"
__email__ = "A.B.Zemkoho@soton.ac.uk"