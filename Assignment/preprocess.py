#!/usr/bin/env python
"""
Description:
Explore data analysis for four different datasets
including:
1. Global Mean Surface Temperature Anomaly (MSTA) in \u00B0C
2. Global Monthly Atmospheric Carbon Dioxide Levels (CH4)
3. International Passenger Survey, UK visits abroad (GMAF)
4. UK Inland Monthly Energy Consumption (ET12), in million tonnes of oil equivalent.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def main(args):

    #load the datasets
    MSTA = pd.read_csv("Ori_Data/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv", parse_dates=["Time"])
    CH4 = pd.read_csv("Ori_Data/ch4_NOAA CH4.csv")
    GMAF = pd.read_csv("Ori_Data/ott.csv")
    ET12 = pd.read_excel("./Ori_Data/ET_1.2_FEB_25.xlsx", sheet_name="Month", header=5, usecols=[0,1], parse_dates = True)

    #MSTA data preprocessing
    MSTA['Date'] = pd.to_datetime(MSTA['Time'], format="%Y-%m" )
    MSTA['Date'] = pd.to_datetime(MSTA['Date'])
    MSTA = MSTA.rename(columns={'Anomaly (deg C)':"Temperature(C)",
                                'Lower confidence limit (2.5%)':'Lower(2.5%)',
                                'Upper confidence limit (97.5%)':'Upper(97.5%)'})

    MSTA = MSTA.loc[:,['Date',"Temperature(C)","Lower(2.5%)","Upper(97.5%)"]]
    print(MSTA['Date'].dtype)
    fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize=(12, 14))
    sns.lineplot(x='Date', y='Temperature(C)', data=MSTA, ax=ax[0], label="Temperature")
    ax[0].fill_between(x = MSTA['Date'], y1 = MSTA["Lower(2.5%)"], y2 = MSTA["Upper(97.5%)"], alpha = 0.2, color = "green", label = "Confidence Interval" )
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Temperature(\u00B0C)", fontsize=18, labelpad=20)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[0].legend()

    # CH4 data preprocessing
    CH4 = CH4.fillna(0.)
    CH4['Date'] = pd.to_datetime(CH4['Year'].astype(str)+'-'+CH4['Month'].astype(str))
    CH4 = CH4.rename(columns = {'NOAA CH4 (ppb)': "CH4(ppb)", "NOAA CH4 uncertainty": "uncertainty"})
    CH4['Upper'] = CH4['CH4(ppb)'] + CH4['uncertainty']
    CH4['Lower'] = CH4['CH4(ppb)'] - CH4['uncertainty']
    x_min, x_max = CH4['Date'].min(), CH4['Date'].max()
    y_min, y_max = CH4['CH4(ppb)'].min(), CH4['CH4(ppb)'].max()

    CH4 = CH4.loc[:, ['Date',"CH4(ppb)","Upper","Lower"]]

    sns.lineplot(x='Date', y='CH4(ppb)', data=CH4, ax=ax[1], label="CH4 Levels (ppb)")
    ax[1].set_xlim(x_min , x_max)
    ax[1].set_ylim(y_min , y_max)
    ax[1].fill_between(x=MSTA['Date'], y1=MSTA["Lower(2.5%)"], y2=MSTA["Upper(97.5%)"], alpha=0.2, color="green",
                       label="Uncertainty")
    ax[1].set_xlabel("")
    ax[1].set_ylabel("CH4(pbb)", fontsize=18, labelpad = 20)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[1].legend()

    # GMAF data preprocessing
    GMAF = GMAF[226:].rename(columns = {
        'Title': 'Date',
        'UK visits abroad:All visits Thousands-NSA': 'Visitors(GMAF)'
    }).reset_index(drop = True)
    GMAF["Date"] = pd.to_datetime(GMAF["Date"])
    GMAF["Visitors(GMAF)"] = pd.to_numeric(GMAF["Visitors(GMAF)"])
    x_min, x_max = GMAF['Date'].min(), GMAF['Date'].max()
    y_min, y_max = GMAF['Visitors(GMAF)'].min(), GMAF['Visitors(GMAF)'].max()
    GMAF = GMAF.loc[:,['Date','Visitors(GMAF)']]
    sns.lineplot(data = GMAF, x = "Date", y = "Visitors(GMAF)", ax = ax[2], label = "The number of Visitors")
    ax[2].set_xlim(x_min, x_max)
    ax[2].set_ylim(y_min, y_max)
    ax[2].yaxis.set_major_locator(ticker.MaxNLocator(10))
    ax[2].set_xlabel("")
    ax[2].set_ylabel("Visitors(GMAF)", fontsize = 18, labelpad=18)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[2].legend()

    #ET12 data preprocessing
    ET12 = ET12[5:].rename(columns= {'Month':"Date",
                                     'Unadjusted total [note 1]':"Energy Consumption"})
    ET12.iloc[-1, 0] = 'December 2024'
    ET12['Date'] = pd.to_datetime(ET12['Date'])
    sns.lineplot(data = ET12, x = "Date", y = "Energy Consumption", label = "Energy consumption", ax = ax[3])
    x_min, x_max = ET12['Date'].min(), ET12['Date'].max()
    y_min, y_max = ET12['Energy Consumption'].min(), ET12['Energy Consumption'].max()
    ax[3].legend()
    ax[3].set_xlim(x_min, x_max)
    ax[3].set_ylim(y_min, y_max)
    ax[3].set_xlabel("Date", fontsize = 25, labelpad=20)
    ax[3].set_ylabel("Energy Consumption", fontsize = 18, labelpad=20)
    ax[3].xaxis.set_major_locator(mdates.MonthLocator( interval = 24))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #plt.setp(ax[3].get_xticklabels(), rotation = 45)
    plt.tight_layout()
    plt.savefig("plots/Distributions_four_datasets.pdf")
    plt.savefig("plots/Distributions_four_datasets.png", dpi = 500)
    plt.show()

    #store cleaned data
    filename = "./Data_36516473.xlsx"

    MSTA['Date'] = MSTA['Date'].dt.strftime('%Y-%m')
    CH4['Date'] = CH4['Date'].dt.strftime("%Y-%m")
    GMAF['Date'] = GMAF['Date'].dt.strftime("%Y-%m")
    ET12['Date'] = ET12['Date'].dt.strftime("%Y-%m")


    with pd.ExcelWriter(filename) as writer:
        MSTA.to_excel(writer, sheet_name='MSTA', index=False)
        CH4.to_excel(writer, sheet_name='CH4', index=False)
        GMAF.to_excel(writer, sheet_name='GMAF', index=False)
        ET12.to_excel(writer, sheet_name='ET12', index=False)

    return "PreProcess Done!"


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
