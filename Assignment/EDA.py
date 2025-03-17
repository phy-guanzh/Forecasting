#!/usr/bin/env python
"""
Description:

"""
import sys, os
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def non_positive_trans(ori_data: pd.DataFrame):
    # Apply non-positive transformation
    min_val = ori_data.min()
    ori_data_adj = ori_data.copy()
    if min_val <= 0:
        shift_value = abs(min_val) + 1
        ori_data_adj += shift_value
    return ori_data_adj

def plot_decomposition(data, model, title):

    if data.min() <=0 and model == "multiplicative":
        data_adj = non_positive_trans(data.copy())
    else:
        data_adj = data.copy()

    # Plot the decomposition
    decomposed = seasonal_decompose(data_adj, model=model)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    # Plot the original data and decomposition components
    axes[1].plot(decomposed.trend, label="Trend")
    axes[1].set_title(f"Trend Component ({model} decomposition)")
    axes[1].set_xlabel("Year")
    axes[2].plot(decomposed.seasonal, label="Seasonality")
    axes[2].set_title(f"Seasonal Component ({model} decomposition)")
    if title == "MSTA": axes[2].set_xlim(datetime(1990, 1, 1), datetime(1995, 1, 1))
    axes[2].set_xlabel("Year")
    axes[3].plot(decomposed.resid, label="Residuals")
    axes[3].set_title(f"Residuals ({model} decomposition)")
    axes[3].set_xlabel("Year")
    axes[0].plot(decomposed.observed, label="Original Series")
    axes[0].set_title("Original Series of "+ title)
    axes[0].set_xlabel("Year")
    plt.tight_layout()
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{title}_{model}_decomposition.pdf"))
    plt.show()

    return True

def main(args):
    sheets_name = ['MSTA', "CH4", 'GMAF', "ET12"]
    MSTA, CH4, GMAF, ET12 = pd.read_excel("Data_36516473.xlsx", sheet_name=sheets_name, parse_dates=["Date"],
                                          index_col="Date").values()


    plot_decomposition(MSTA['Temperature(C)'], model = "multiplicative", title = "MSTA")
    plot_decomposition(MSTA['Temperature(C)'], model = "additive", title = "MSTA")
    plot_decomposition(CH4['CH4(ppb)'], model = "additive", title = "CH4")
    plot_decomposition(CH4['CH4(ppb)'], model="multiplicative", title="CH4")
    plot_decomposition(GMAF["Visitors(GMAF)"], model = "additive", title = "GMAF")
    plot_decomposition(GMAF["Visitors(GMAF)"], model = "multiplicative", title = "GMAF")
    plot_decomposition(ET12["Energy Consumption"], model = "additive", title = "ET12")
    plot_decomposition(ET12["Energy Consumption"], model = "multiplicative", title = "ET12")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
