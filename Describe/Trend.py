import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Loading data
df_materials = pd.read_excel('BuildingMaterials.xls',index_col=0, parse_dates=True, date_format="%Y %b")[:-4].squeeze()
df_cement = pd.read_excel('CementProduction.xls',index_col=0, parse_dates=True, date_format="%Y %b")[:-4].squeeze()

df_materials.index = pd.to_datetime(df_materials.index,format="%Y %b",errors='coerce')
df_cement.index = pd.to_datetime(df_cement.index,format="%Y %b",errors='coerce')

print(type(df_materials.index))
print(df_materials.tail() ) # Show last 5 rows
years = df_materials.index.year.unique()
years2 = df_cement.index.year.unique()
months = df_materials.index.month.unique()
months2 = df_cement.index.month.unique()

df_year = pd.DataFrame(index=months, columns=years)

print(years)

df_year = df_materials.groupby(by=[df_materials.index.month, df_materials.index.year]).sum().unstack()
print(df_year)

df_year2 = df_cement.groupby(by=[df_cement.index.month, df_cement.index.year]).sum().unstack()
print(df_year2)

fig, ax = plt.subplots(2,1, figsize=(12,9),gridspec_kw={'height_ratios':[1,1]})
print(df_year)
df_year.sort_index(inplace=True)
print(df_year)

for year in years:
    ax[0].plot(df_year.index, df_year[year], label=str(year))
    ax[0].legend()
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel('Quantity')
for year in years2:
    ax[1].plot(df_year2.index, df_year2[year], label=str(year))
    ax[1].legend()
    ax[1].set_xlabel('Month')
    ax[1].set_ylabel('Quantity')

plt.savefig('monthly_plots.png',dpi=300)
plt.show()