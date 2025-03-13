#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Employment data - Holt Models.
--
-- Description: In this script the Employment data is used to fit serval Holt models.
--
-- Content:     0. Set-up
--              1. Data
--              2. Holt Models
--              3. Plotting
--              4. Forecasting Error
--              5. Publisher's Imprint
--
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
Version  Date        Author    Major Changes
1.0      2020-02-04  ABZ       Initialization
1.1      2024-01-02  MLT       Updated version
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
########################################################################################################################
# 0. Set-ups
########################################################################################################################

# Read in data
import pandas as pd

# Plotting
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

# Holt
from statsmodels.tsa.api import Holt

# Forecasting Error
from sklearn.metrics import mean_squared_error

########################################################################################################################
# 1. Data
########################################################################################################################

# Define file path
file = "./EmploymentPrivateServices.xls"

# Load employment data
series = pd.read_excel(file, header=0, index_col=0, parse_dates=True).squeeze()

# Add frequency to index
series.index = pd.DatetimeIndex(series.index.values, freq=series.index.inferred_freq)

########################################################################################################################
# 2. Holt Models
########################################################################################################################

# Holt model 1: alpha = 0.5, beta=0.5
fit1 = Holt(series).fit(smoothing_level=0.5, smoothing_trend=0.5, optimized=False)
fcast1 = fit1.forecast(12).rename("Model 1: Holt's linear trend")

fit2 = Holt(series, exponential=True).fit(smoothing_level=0.2, smoothing_trend=0.4, optimized=False)
fcast2 = fit2.forecast(12).rename("Model 2: Exponential trend")

fit3 = Holt(series, damped_trend=True).fit()
fcast3 = fit3.forecast(12).rename("Model 3: Damped trend + optimized")

fit4 = Holt(series).fit(optimized=True)
fcast4 = fit4.forecast(12).rename("Model 4: Linear trend + optimized")

########################################################################################################################
# 3. Plotting
########################################################################################################################

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

fit1.fittedvalues.plot(color='blue', ax=ax)
fcast1.plot(color='blue', legend=True, ax=ax)

fit2.fittedvalues.plot(color='red', ax=ax)
fcast2.plot(color='red', legend=True, ax=ax)

fit3.fittedvalues.plot(color='green', ax=ax)
fcast3.plot(color='green', legend=True, ax=ax)

fit4.fittedvalues.plot(color='yellow', ax=ax)
fcast4.plot(color='yellow', legend=True, ax=ax)

series.plot(color='black', legend=True, ax=ax)

plt.xlabel('Dates')
plt.ylabel('Values')
plt.title("Holt's method-based forecasts for Employment Services")

plt.tight_layout()
plt.show()

########################################################################################################################
# 4. Forecasting Error
########################################################################################################################

# Compute MSE per model
MSE1 = mean_squared_error(fit1.fittedvalues, series)
MSE2 = mean_squared_error(fit2.fittedvalues, series)
MSE3 = mean_squared_error(fit3.fittedvalues, series)
MSE4 = mean_squared_error(fit4.fittedvalues, series)

print('Summary of paramters and errors from each of the models:')
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
results = pd.DataFrame(index=[r"alpha", r"beta", r"phi", r"l0", r"b0", "MSE"],
                       columns=["Holt model 1", "Holt model 2", "Holt model 3", "Holt model 4"])

results["Holt model 1"] = [fit1.params[p] for p in params] + [MSE1]
results["Holt model 2"] = [fit2.params[p] for p in params] + [MSE2]
results["Holt model 3"] = [fit3.params[p] for p in params] + [MSE3]
results["Holt model 4"] = [fit4.params[p] for p in params] + [MSE4]

print(results)

########################################################################################################################
# 5. Publisher's Imprint
########################################################################################################################

__author__ = ["Alain Zemkoho"]
__credits__ = ["Marah-Lisanne Thormann"]
__version__ = "1.1"
__maintainer__ = "Alain Zemkoho"
__email__ = "A.B.Zemkoho@soton.ac.uk"

########################################################################################################################
########################################################################################################################
