#!/usr/bin/env python
"""
Description:
Explore data analysis for four different datasets
including:
1. Global Mean Surface Temperature Anomaly (MSTA) in C
2. Global Monthly Atmospheric Carbon Dioxide Levels (CH4)
3.International Passenger Survey, UK visits abroad (GMAF)
4. UK Inland Monthly Energy Consumption (ET12), in million tonnes of oil equivalent.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def main(args):

    #load the datasets
    MSTA = pd.read_csv("HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv")
    CH4 = pd.read_csv("ch4_NOAA CH4.csv")
    GMAF = pd.read_csv("ott.csv")
    #ET12 = pd.read_excel("./ET_1.2_FEB_25.xlsx", sheet_name="Month")

    #MSTA data preprocessing
    MSTA['Date'] = pd.to_datetime(MSTA['Time'])
    MSTA = MSTA.rename(columns={'Anomaly (deg C)':"Temperature(C)",
                                'Lower confidence limit (2.5%)':'Lower(2.5%)',
                                'Upper confidence limit (97.5%)':'Upper(97.5%)'})

    #print(MSTA)
    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize=(14, 10))
    sns.lineplot(x='Date', y='Temperature(C)', data=MSTA, ax=ax[0], label="Temperature")
    ax[0].fill_between(x = MSTA['Date'], y1 = MSTA["Lower(2.5%)"], y2 = MSTA["Upper(97.5%)"], alpha = 0.2, color = "green", label = "Confidence Interval" )
    ax[0].legend()

    # CH4 data preprocessing
    CH4 = CH4.fillna(0.)
    CH4['Date'] = pd.to_datetime(CH4['Year'].astype(str)+'-'+CH4['Month'].astype(str))
    CH4 = CH4.rename(columns = {'NOAA CH4 (ppb)': "CH4(ppb)", "NOAA CH4 uncertainty": "uncertainty"})
    CH4['Upper'] = CH4['CH4(ppb)'] + CH4['uncertainty']
    CH4['Lower'] = CH4['CH4(ppb)'] - CH4['uncertainty']
    x_min, x_max = CH4['Date'].min(), CH4['Date'].max()
    y_min, y_max = CH4['CH4(ppb)'].min(), CH4['CH4(ppb)'].max()

    sns.lineplot(x='Date', y='CH4(ppb)', data=CH4, ax=ax[1], label="CH4 Levels (ppb)")
    ax[1].set_xlim(x_min , x_max)
    ax[1].set_ylim(y_min , y_max)
    ax[1].fill_between(x=MSTA['Date'], y1=MSTA["Lower(2.5%)"], y2=MSTA["Upper(97.5%)"], alpha=0.2, color="green",
                       label="Uncertainty")
    ax[1].legend()

    # GMAF data preprocessing
    GMAF = GMAF[227:].rename(columns = {
        'Title': 'Date',
        'UK visits abroad:All visits Thousands-NSA': 'Visitors(GMAF)'
    }).reset_index(drop=True)
    GMAF["Date"] = pd.to_datetime(GMAF["Date"])

    sns.lineplot(data = GMAF, x = "Date", y = "Visitors(GMAF)", ax = ax[2], label = "The number of Visitors")
    ax[2].yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax[2].invert_yaxis()
    print(GMAF["Visitors(GMAF)"])
    ax[2].legend()

    plt.show()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
