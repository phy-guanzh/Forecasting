import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
data = pd.read_excel("../Describe/BuildingMaterials.xls", index_col=0, parse_dates= True, date_format="%Y %b", header=None, names=["Date", "Amount"] )[:-4]
data.index = pd.to_datetime(data.index, format="%Y %b", errors='coerce')

print(data.head())
#NF1 prediction
data_NF1 = data.copy()
data_NF1["Amount"] = data["Amount"].shift(-1)

print(data_NF1.head())
# Decompose the data into trend, seasonal, and residual components
addictive = seasonal_decompose(data,model="additive", period=12)
seasonal_mat = addictive.seasonal
print(seasonal_mat)

data_NF2 = data.copy()

# Decomposition seasonal components
data_NF2["Amount"] = data_NF2["Amount"] - seasonal_mat

# Add t-12 + 1
data_NF2["Amount"]  += seasonal_mat.shift(-11)

print(data_NF2.head())

# Plot the data
plt.figure(figsize=(10,6))
plt.plot(data["Amount"], label="Original", color="black")
plt.plot(data_NF1["Amount"], label="Naive Forecast NF1", color="blue")
plt.plot(data_NF2["Amount"], label="Seasonal Naive Forecast NF2",color="red")
plt.legend()
plt.title("Building Materials Sales Naive Forecasting")


plt.show()




print(addictive)
