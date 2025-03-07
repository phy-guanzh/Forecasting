#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
--
-- Employment data - Simple Exponential Smoothing.
--
-- Description: In this script the Employment data is used to fit models based on simple exponential smoothing.
--
-- Content:     0. Set-up
--              1. Data
--              2. Simple Exponential Smoothing
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
#atplotlib.use('Qt5Agg')
sns.set_style("whitegrid")

# Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing

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
# 2. Simple Exponential Smoothing
########################################################################################################################

# SES model 1: alpha = 0.5
fit1 = SimpleExpSmoothing(series).fit(smoothing_level=0.5, optimized=False)
fcast1 = fit1.forecast(10).rename(r'$\alpha=0.5$')

# SES model 2: alpha = 0.7
fit2 = SimpleExpSmoothing(series).fit(smoothing_level=0.7, optimized=False)
fcast2 = fit2.forecast(10).rename(r'$\alpha=0.7$')

# SES model 3: alpha automatically selected by the built-in optimization software
fit3 = SimpleExpSmoothing(series).fit()
fcast3 = fit3.forecast(10).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

########################################################################################################################
# 3. Plotting
########################################################################################################################

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot of fitted values and forecast of next 10 values, respectively
fit1.fittedvalues.plot(color='blue', ax=ax)
fcast1.plot(color='blue', legend=True, ax=ax)

# Plot of fitted values and forecast of next 10 values, respectively
fcast2.plot(color='red', legend=True, ax=ax)
fit2.fittedvalues.plot(color='red', ax=ax)

# Plot of fitted values and forecast of next 10 values, respectively
fcast3.plot(color='green', legend=True, ax=ax)
fit3.fittedvalues.plot(color='green', ax=ax)

# Plotting the original data together with the 3 forecast plots
series.plot(color='black', legend=True, ax=ax)

plt.title('SES method-based forecasts for Employment Services')
plt.tight_layout()
plt.show()

########################################################################################################################
# 4. Forecasting Error
########################################################################################################################

MSE1 = mean_squared_error(fit1.fittedvalues, series)
MSE2 = mean_squared_error(fit2.fittedvalues, series)
MSE3 = mean_squared_error(fit3.fittedvalues, series)

print('Summary of errors resulting from SES models 1, 2 & 3:')
summary = {'Model': ['MSE'],
           'SES model 1': [MSE1],
           'SES model 2': [MSE2],
           'SES model 3': [MSE3]
           }
AllErrors = pd.DataFrame(summary, columns=['Model', 'SES model 1', 'SES model 2', 'SES model 3'])
print(AllErrors)

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
