import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os


os.chdir("/Users/zheguan/DDA/Forecasting/Data and Codes for all Chapters/")

sliced_data = pd.read_excel(
    "./Data and Codes for all Chapters/Chapter 1 Data and Codes 2024/Chapter 1 Workshop Data/BuildingMaterials.xls"
    , names=["Date", "Quantity"]
    , parse_dates=["Date"]).iloc[0:263]

sliced_data2 = pd.read_excel(
    "./Data and Codes for all Chapters/Chapter 1 Data and Codes 2024/Chapter 1 Workshop Data/CementProduction.xls"
    , names=["Date", "Quantity"]
    , parse_dates=["Date"]
    ).iloc[0:88]


fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratio': [2, 1]})

ax[0].plot(sliced_data["Date"], sliced_data["Quantity"])
ax[0].plot(sliced_data["Date"], sliced_data["Quantity"])
ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[1].plot(sliced_data2["Date"], sliced_data2["Quantity"])
ax[1].plot(sliced_data2["Date"], sliced_data2["Quantity"])
ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
plt.plot()
ax[0].tick_params(axis='x', rotation=45)
ax[1].tick_params(axis='x', rotation=45)

plt.show()