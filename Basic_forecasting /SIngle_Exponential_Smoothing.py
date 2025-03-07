import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import SimpleExpSmoothing

#Load data
data = pd.read_excel("EmploymentPrivateServices.xls", index_col=0, header = None, parse_dates=True)
data.columns = ["Data"]

print(data)

fit = SimpleExpSmoothing(data).fit(smoothing_level = 0.5, optimized = False)
fcast = fit.forecast(10).rename(r"$\alpha = 0.5$")

fit1 = SimpleExpSmoothing(data).fit(smoothing_level = 0.7, optimized = False)
fcast1 = fit.forecast(10).rename(r"$\alpha = 0.7$")

fit2 = SimpleExpSmoothing(data).fit()
fcast2 = fit.forecast(10).rename(r"$\alpha = 1$")

print(fcast1)
#plotting
fig, ax = plt.subplots(figsize=(15, 10))

fit.fittedvalues.plot(color='blue', ax=ax)
fcast.plot(color='blue', legend=True, ax=ax)

fit1.fittedvalues.plot(color='red', ax=ax)
fcast1.plot(color='red', legend=True, ax=ax)

fit2.fittedvalues.plot(color='black', ax=ax)
fcast2.plot(color='black', legend=True, ax=ax)

data.plot(color='green',legend=True, ax = ax)
ax.set_xlabel('Years', fontsize = 14)
ax.set_ylabel('Number of Employees', fontsize = 14)
plt.title('Employment data with SES forecasts', fontsize = 14)

plt.show()


