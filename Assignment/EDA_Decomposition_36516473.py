#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- Time Series Decomposition Using Additive and Multiplicative Models
--
-- Description:
-- This script performs seasonal decomposition of four key time series datasets (MSTA, CH4, GMAF, ET12),
-- using both additive and multiplicative models depending on the nature of the data.
--
-- Additive decomposition assumes components are linearly additive (trend + seasonality + residual),
-- while multiplicative assumes proportional relationships (trend × seasonality × residual).
--
-- The script automatically applies a non-negative transformation for multiplicative decomposition
-- if non-positive values are detected in the input series.
--
-- Content:
--    0. Set-up: Importing Necessary Libraries
--    1. Function1: non_positive_trans() - Adjusts data for multiplicative decomposition
--    2. Function2: plot_decomposition() - Performs and visualizes seasonal decomposition
--    3. Main Function: Loads data and applies both additive and multiplicative decomposition
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
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

########################################################################
# 1. Function: non_positive_trans()
#    Adjusts series by shifting if non-positive values are found,
#    allowing multiplicative decomposition.
########################################################################
def non_positive_trans(ori_data: pd.DataFrame):
    min_val = ori_data.min()
    ori_data_adj = ori_data.copy()
    if min_val <= 0:
        shift_value = abs(min_val) + 1
        ori_data_adj += shift_value
    return ori_data_adj

########################################################################
# 2. Function: plot_decomposition()
#    Decomposes the time series using specified model ("additive" or "multiplicative")
#    and saves the resulting plots into a PDF file.
########################################################################
def plot_decomposition(data, model, title):

    if data.min() <= 0 and model == "multiplicative":
        data_adj = non_positive_trans(data.copy())
    else:
        data_adj = data.copy()

    # Perform seasonal decomposition
    decomposed = seasonal_decompose(data_adj, model=model)

    # Plot decomposition results
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    axes[0].plot(decomposed.observed, label="Original Series")
    axes[0].set_title("Original Series of " + title)
    axes[0].set_xlabel("Year")

    axes[1].plot(decomposed.trend, label="Trend")
    axes[1].set_title(f"Trend Component ({model} decomposition)")
    axes[1].set_xlabel("Year")

    axes[2].plot(decomposed.seasonal, label="Seasonality")
    axes[2].set_title(f"Seasonal Component ({model} decomposition)")
    axes[2].set_xlabel("Year")

    axes[3].plot(decomposed.resid, label="Residuals")
    axes[3].set_title(f"Residuals ({model} decomposition)")
    axes[3].set_xlabel("Year")

    plt.tight_layout()
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title}_{model}_decomposition.pdf"))
    plt.show()

    return True

########################################################################
# 3. Main Function: Executing the Workflow
########################################################################
def main(args):
    # Load all sheets
    sheets_name = ['MSTA', "CH4", 'GMAF', "ET12"]
    MSTA, CH4, GMAF, ET12 = pd.read_excel("./Exported_Data/Data_36516473.xlsx", sheet_name=sheets_name, parse_dates=["Date"],
                                          index_col="Date").values()

    # Apply decomposition to all datasets using both models
    plot_decomposition(MSTA['Temperature(C)'], model="multiplicative", title="MSTA")
    plot_decomposition(MSTA['Temperature(C)'], model="additive", title="MSTA")

    plot_decomposition(CH4['CH4(ppb)'], model="additive", title="CH4")
    plot_decomposition(CH4['CH4(ppb)'], model="multiplicative", title="CH4")

    plot_decomposition(GMAF["Visitors(GMAF)"], model="additive", title="GMAF")
    plot_decomposition(GMAF["Visitors(GMAF)"], model="multiplicative", title="GMAF")

    plot_decomposition(ET12["Energy Consumption"], model="additive", title="ET12")
    plot_decomposition(ET12["Energy Consumption"], model="multiplicative", title="ET12")

########################################################################
# Script Execution
########################################################################
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
