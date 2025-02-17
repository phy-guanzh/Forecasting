import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

#Loading data
df_materials = pd.read_excel('BuildingMaterials.xls',index_col=0, parse_dates=True, date_format="%Y-%m")[:-4].squeeze()
df_cement = pd.read_excel('CementProduction.xls',index_col=0, parse_dates=True, date_format="%Y-%m")[:-4].squeeze()

print(df_materials.tail() ) # Show last 5 rows)
print(df_cement.tail() )
#Plotting data
fig, ax = plt.subplots(2,1, figsize=(12,9), gridspec_kw={'height_ratios':[2,1]})

ax[0].plot(df_materials.index, df_materials.values, color = "red", label = "material")
ax[0].legend()
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Quantity')
ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
#ax[0].tick_params(axis='x', rotation=45)
ax[1].plot(df_cement.index, df_cement.values, color = "blue", label = "cement")
ax[1].legend()
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Quantity')
ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
#ax[1].tick_params(axis='x', rotation=45)

plt.show()

#ax[0].legend(['Materials', 'Cement'])

