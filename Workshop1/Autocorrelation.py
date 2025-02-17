import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.pyplot import figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Loading data
df_materials = pd.read_excel('BuildingMaterials.xls',index_col=0, parse_dates=True, date_format="%Y %b")[:-4].squeeze()
df_cement = pd.read_excel('CementProduction.xls',index_col=0, parse_dates=True, date_format="%Y %b")[:-4].squeeze()

df_materials.index = pd.to_datetime(df_materials.index,format="%Y %b",errors='coerce')
df_cement.index = pd.to_datetime(df_cement.index,format="%Y %b",errors='coerce')

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,9), gridspec_kw={'height_ratios':[1,1]})
plot_acf(df_materials, lags=60, title='Autocorrelation Function for Building Materials', ax=ax1)
plot_acf(df_cement, lags=60, title='Autocorrelation Function for Cement Production', ax=ax2)
plt.savefig('acf_plots.png',dpi=300)
plt.show()