#!/usr/bin/env python
"""
Description:
    apply different smoothing functions for four preprocessed datasets
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def main(args):
    # load the datasets
    sheets_name = ['MSTA', "CH4",'GMAF', "ET12" ]
    MSTA, CH4, GMAF, ET12 = pd.read_excel("Data_36516473.xlsx", sheet_name=sheets_name, parse_dates=["Date"], index_col= "Date").values()
    print(MSTA.index, CH4.index, GMAF.index, ET12.index)
    MSTA_decomposed =  seasonal_decompose(MSTA['Temperature(C)'], model = "addictive")

    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    # Plot the original data and decomposition components
    axes[0].plot(MSTA_decomposed.trend, label="Trend")
    axes[0].set_title("Trend Component")
    axes[1].plot(MSTA_decomposed.seasonal, label="Seasonality")
    axes[1].set_title("Seasonal Component")
    axes[2].plot(MSTA_decomposed.resid, label="Residuals")
    axes[2].set_title("Residuals")
    axes[3].plot(MSTA_decomposed.observed, label="Original Series")
    axes[3].set_title("Original Series")

    plt.tight_layout()
    plt.show()

    MSTA["Diff"] = MSTA["Temperature(C)"].diff()
    plot_acf(MSTA["Diff"].dropna(), lags=50)
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
