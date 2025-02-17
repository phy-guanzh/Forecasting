import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.pyplot import figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
# Loading data
df_materials = pd.read_excel('BuildingMaterials.xls',index_col=0, parse_dates=True, date_format="%Y %b")[:-4].squeeze()
df_cement = pd.read_excel('CementProduction.xls',index_col=0, parse_dates=True, date_format="%Y %b")[:-4].squeeze()

df_materials.index = pd.to_datetime(df_materials.index,format="%Y %b",errors='coerce')
df_cement.index = pd.to_datetime(df_cement.index,format="%Y %b",errors='coerce')

df_12MA = df_materials.rolling(window=12).mean()
df_2_12MA = df_12MA.rolling(window=2).mean()
df_7MA = df_materials.rolling(window=7).mean()


print(df_12MA.tail())
print(df_2_12MA.tail())
print(df_materials.tail())

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df_materials, label='Original', color='blue')
ax.plot(df_12MA, label='12-Month MA', color='red')
ax.plot(df_2_12MA, label='2-12-Month MA', color='green')
ax.plot(df_7MA, label='7-Month MA', color='orange')

plt.legend(loc='best')
plt.title('Building Materials 12-Month Moving Averages')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.savefig('moving_averages.png', dpi=300)
plt.show()