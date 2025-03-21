#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- Data Exploration and Preprocessing of Global Environmental Datasets
--
-- Description:
-- This script processes four global or UK-based time series datasets:
--     1. Global Mean Surface Temperature Anomaly (MSTA)
--     2. Atmospheric Methane Concentration (CH4)
--     3. UK International Travel (GMAF)
--     4. UK Monthly Energy Consumption (ET12)
--
-- For each dataset, the script:
--     - Parses and cleans the raw input files from various formats (CSV/Excel).
--     - Handles date formatting and missing/invalid values.
--     - Visualizes each dataset as a time series with shaded uncertainty (where applicable).
--     - Stores the cleaned datasets into a unified Excel file for further modeling and forecasting.
--
-- Content:
--     0. Set-up: Importing Necessary Libraries
--     1. Main Function: Data Loading, Cleaning, Visualization, and Export
--
-- Maintainer:  Zhe GUAN
-- Contact:     zg2u24@soton.ac.uk
--
------------------------------------------------------------------------------------------------------------------------
"""

########################################################################
# 0. Importing Necessary Libraries
########################################################################
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

########################################################################
# 1. Main Function: Data Preprocessing and Visualization
########################################################################
def main(args):

    # Load raw datasets
    MSTA = pd.read_csv("Ori_Data/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv", parse_dates=["Time"])
    CH4 = pd.read_csv("Ori_Data/ch4_NOAA CH4.csv")
    GMAF = pd.read_csv("Ori_Data/ott.csv")
    ET12 = pd.read_excel("./Ori_Data/ET_1.2_FEB_25.xlsx", sheet_name="Month", header=5, usecols=[0,1], parse_dates=True)

    ####################################################################
    # MSTA Dataset (Temperature Anomaly) - Global
    ####################################################################
    MSTA['Date'] = pd.to_datetime(MSTA['Time'], format="%Y-%m")
    MSTA = MSTA.rename(columns={
        'Anomaly (deg C)': "Temperature(C)",
        'Lower confidence limit (2.5%)': 'Lower(2.5%)',
        'Upper confidence limit (97.5%)': 'Upper(97.5%)'
    })[["Date", "Temperature(C)", "Lower(2.5%)", "Upper(97.5%)"]]

    ####################################################################
    # CH4 Dataset (Methane Levels) - Global
    ####################################################################
    CH4 = CH4.fillna(0.)
    CH4['Date'] = pd.to_datetime(CH4['Year'].astype(str) + '-' + CH4['Month'].astype(str))
    CH4 = CH4.rename(columns={
        'NOAA CH4 (ppb)': "CH4(ppb)",
        "NOAA CH4 uncertainty": "uncertainty"
    })
    CH4['Upper'] = CH4['CH4(ppb)'] + CH4['uncertainty']
    CH4['Lower'] = CH4['CH4(ppb)'] - CH4['uncertainty']
    CH4 = CH4[['Date', "CH4(ppb)", "Upper", "Lower"]]

    ####################################################################
    # GMAF Dataset (UK International Travel)
    ####################################################################
    GMAF = GMAF[226:].rename(columns={
        'Title': 'Date',
        'UK visits abroad:All visits Thousands-NSA': 'Visitors(GMAF)'
    }).reset_index(drop=True)
    GMAF["Date"] = pd.to_datetime(GMAF["Date"])
    GMAF["Visitors(GMAF)"] = pd.to_numeric(GMAF["Visitors(GMAF)"])
    GMAF = GMAF[["Date", "Visitors(GMAF)"]]

    ####################################################################
    # ET12 Dataset (UK Monthly Energy Consumption)
    ####################################################################
    ET12 = ET12[5:].rename(columns={
        'Month': "Date",
        'Unadjusted total [note 1]': "Energy Consumption"
    })
    ET12.iloc[-1, 0] = 'December 2024'  # Fix known issue
    ET12["Date"] = pd.to_datetime(ET12["Date"])

    ####################################################################
    # Visualization of All Four Time Series
    ####################################################################
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 14))

    # MSTA Plot
    sns.lineplot(x="Date", y="Temperature(C)", data=MSTA, ax=ax[0], label="Temperature")
    ax[0].fill_between(MSTA["Date"], MSTA["Lower(2.5%)"], MSTA["Upper(97.5%)"], alpha=0.2, color="green", label="Confidence Interval")
    ax[0].set_ylabel("Temperature (°C)", fontsize=18)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[0].legend()

    # CH4 Plot
    sns.lineplot(x="Date", y="CH4(ppb)", data=CH4, ax=ax[1], label="CH4 Levels (ppb)")
    ax[1].fill_between(CH4["Date"], CH4["Lower"], CH4["Upper"], alpha=0.2, color="green", label="Uncertainty")
    ax[1].set_ylabel("CH4 (ppb)", fontsize=18)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[1].legend()

    # GMAF Plot
    sns.lineplot(x="Date", y="Visitors(GMAF)", data=GMAF, ax=ax[2], label="UK Visitors Abroad")
    ax[2].set_ylabel("Visitors (Thousands)", fontsize=18)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[2].legend()

    # ET12 Plot
    sns.lineplot(x="Date", y="Energy Consumption", data=ET12, ax=ax[3], label="Energy Consumption")
    ax[3].set_ylabel("Energy Consumption (Mtoe)", fontsize=18)
    ax[3].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[3].set_xlabel("Date", fontsize=20)
    ax[3].legend()

    plt.tight_layout()
    plt.savefig("plots/Distributions_four_datasets.pdf")
    plt.savefig("plots/Distributions_four_datasets.png", dpi=500)
    plt.show()

    ####################################################################
    # Export Cleaned Data to Excel
    ####################################################################
    filename = "./Data_36516473.xlsx"
    for df in [MSTA, CH4, GMAF, ET12]:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m")

    with pd.ExcelWriter(filename) as writer:
        MSTA.to_excel(writer, sheet_name='MSTA', index=False)
        CH4.to_excel(writer, sheet_name='CH4', index=False)
        GMAF.to_excel(writer, sheet_name='GMAF', index=False)
        ET12.to_excel(writer, sheet_name='ET12', index=False)

    return "Preprocessing Done!"

########################################################################
# Script Execution
########################################################################
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
