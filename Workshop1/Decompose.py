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


additive = seasonal_decompose(df_materials, model='additive')
fig1 = additive.plot()
fig1.suptitle("Additive Decomposition", fontsize=14)
fig1.savefig("additive_decompose.png", dpi=300, bbox_inches="tight")

multiplicative = seasonal_decompose(df_materials, model='multiplicative')
fig2 = multiplicative.plot()
fig2.suptitle("Multiplicative Decomposition", fontsize=14)
fig2.savefig("multiplicative_decompose.png", dpi=300, bbox_inches="tight")

plt.show()
