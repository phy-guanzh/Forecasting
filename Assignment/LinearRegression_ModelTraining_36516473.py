#!/usr/bin/env python
"""
Description:
------------------------------------------------------------------------------------------------------------------------
--
-- Global Temperature Anomaly Prediction Using Linear Regression with Lagged Features and Interaction Terms
--
-- Description:
-- This script implements a linear regression model to predict the Global Mean Surface Temperature Anomaly (MSTA)
-- using three key predictors:
--     - Global Atmospheric Methane Concentration (CH4)
--     - UK Outbound Tourism Statistics (GMAF)
--     - UK Inland Energy Consumption (ET12)
--
-- Key Features:
--     - Automatic lag selection for each feature using AIC
--     - Standardization of features before modeling
--     - Generation of interaction terms and removal of multicollinearity (via VIF)
--     - Comparison of two models: with and without lagged interaction terms
--     - Outputs include full regression summaries and performance metrics (R², AIC, MSE)
--
-- Content:
--     0. Set-up: Importing Libraries
--     1. Function: find_optimal_lag() – AIC-based lag selection
--     2. Function: generate_interactions() – Create and filter interaction terms
--     3. Main: Data Loading, Lagged Modeling, Comparison, and Export
--
-- Maintainer:  Zhe GUAN
-- Contact:     zg2u24@soton.ac.uk
--
------------------------------------------------------------------------------------------------------------------------
"""

########################################################################
# 0. Importing Necessary Libraries
########################################################################
import sys
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error

########################################################################
# 1. Function: Select Optimal Lag for Each Predictor via AIC
########################################################################
def find_optimal_lag(target, feature, max_lag=24, method='aic'):
    aligned = pd.concat([target, feature], axis=1).dropna()
    X = aligned.iloc[:, 1]
    y = aligned.iloc[:, 0]

    best_lag = 0
    best_score = np.inf

    for lag in range(0, max_lag + 1):
        X_lagged = X.shift(lag).dropna()
        y_aligned = y.loc[X_lagged.index]
        if len(y_aligned) < 10:
            continue
        X_sm = sm.add_constant(X_lagged)
        model = sm.OLS(y_aligned, X_sm).fit()
        score = model.aic
        if score < best_score:
            best_score = score
            best_lag = lag
    return best_lag

########################################################################
# 2. Function: Generate Interaction Terms and Remove High Multicollinearity
########################################################################
def generate_interactions(df, features):
    interaction_pairs = list(combinations(features, 2))
    new_features = []

    for (f1, f2) in interaction_pairs:
        col_name = f"{f1}_x_{f2}"
        df[col_name] = df[f1] * df[f2]
        new_features.append(col_name)

    vif = pd.DataFrame()
    vif["Variable"] = features + new_features
    vif["VIF"] = [variance_inflation_factor(df[vif["Variable"]].values, i)
                  for i in range(len(vif["Variable"]))]

    selected_features = vif[vif["VIF"] <= 10]["Variable"].tolist()
    print("\nVIF Check for Interaction Terms:")
    print(vif.round(1))
    return df[selected_features]

########################################################################
# 3. Main Function: Data Loading, Preprocessing, Modeling, and Evaluation
########################################################################
def main(args):
    try:
        # Load data from Excel file
        sheets_name = ['MSTA', "CH4", 'GMAF', "ET12"]
        data_dict = pd.read_excel(
            "Data_36516473.xlsx",
            sheet_name=sheets_name,
            parse_dates=["Date"],
            index_col="Date"
        )

        # Extract and align data
        MSTA = data_dict['MSTA'].loc['1990':, 'Temperature(C)'].squeeze()
        CH4 = data_dict['CH4'].loc['1990':, 'CH4(ppb)'].squeeze()
        GMAF = data_dict['GMAF'].loc['1990':, "Visitors(GMAF)"].squeeze()
        ET12 = data_dict['ET12'].loc['1960':, "Energy Consumption"].squeeze()

        start_date = max(MSTA.index.min(), CH4.index.min(), GMAF.index.min(), ET12.index.min())
        end_date = min(MSTA.index.max(), CH4.index.max(), GMAF.index.max(), ET12.index.max())

        aligned_data = pd.DataFrame({
            'MSTA': MSTA.loc[start_date:end_date],
            'CH4': CH4.loc[start_date:end_date],
            'GMAF': GMAF.loc[start_date:end_date],
            'ET12': ET12.loc[start_date:end_date]
        }).dropna()

        aligned_data.to_excel("regression_data.xlsx")

        # Find optimal lags for each feature
        max_lag = 36
        optimal_lags = {
            feature: find_optimal_lag(aligned_data['MSTA'], aligned_data[feature], max_lag=max_lag)
            for feature in ['CH4', 'GMAF', 'ET12']
        }
        print("Optimal Lags:", optimal_lags)

        # Scale features
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(aligned_data),
            columns=aligned_data.columns,
            index=aligned_data.index
        )

        for feature, lag in optimal_lags.items():
            if lag > 0:
                scaled_data[f'{feature}_lag{lag}'] = scaled_data[feature].shift(lag)
        scaled_data = scaled_data.dropna()

        # Build regression with interaction terms
        base_features = [f'{k}_lag{v}' if v > 0 else k for k, v in optimal_lags.items()]
        X_lag = generate_interactions(scaled_data.copy(), base_features)
        X_lag = sm.add_constant(X_lag)
        y_lag = scaled_data['MSTA']

        model_lag = sm.OLS(y_lag, X_lag).fit()

        # Build baseline regression without lags
        X_no_lag = sm.add_constant(scaled_data[['CH4', 'GMAF', 'ET12']])
        y_no_lag = scaled_data['MSTA']
        model_no_lag = sm.OLS(y_no_lag, X_no_lag).fit()

        # Evaluate models
        mse_with = mean_squared_error(y_lag, model_lag.predict(X_lag))
        mse_without = mean_squared_error(y_no_lag, model_no_lag.predict(X_no_lag))

        comparison_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adj. R-squared', 'AIC', 'BIC', 'DW Stat', 'MSE'],
            'With Lags': [
                model_lag.rsquared,
                model_lag.rsquared_adj,
                model_lag.aic,
                model_lag.bic,
                sm.stats.durbin_watson(model_lag.resid),
                mse_with
            ],
            'Without Lags': [
                model_no_lag.rsquared,
                model_no_lag.rsquared_adj,
                model_no_lag.aic,
                model_no_lag.bic,
                sm.stats.durbin_watson(model_no_lag.resid),
                mse_without
            ]
        }).set_index("Metric")

        print("\n" + "=" * 60)
        print("Regression Summary (With Lags):")
        print(model_lag.summary())
        print("\nRegression Summary (Without Lags):")
        print(model_no_lag.summary())

        print("\n" + "=" * 60)
        print("Model Comparison Summary:")
        print(comparison_df.round(3))

        # Save models
        with open('model_with_lags.pkl', 'wb') as f:
            pickle.dump(model_lag, f)
        with open('model_no_lags.pkl', 'wb') as f:
            pickle.dump(model_no_lag, f)

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

########################################################################
# Script Execution
########################################################################
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"
