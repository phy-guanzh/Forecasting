#!/usr/bin/env python
"""
Global Temperature Anomaly Prediction Model with Optimal Lag Selection
Developed for GCPA project
"""
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import ccf
from sklearn.model_selection import TimeSeriesSplit


def find_optimal_lag(target, feature, max_lag=24, method='aic'):
    """
    为单个特征寻找最优滞后阶数
    :param target: 目标变量 (pd.Series)
    :param feature: 特征变量 (pd.Series)
    :param max_lag: 最大滞后月数
    :param method: 选择方法 (aic/ccf/cv)
    :return: 最优滞后阶数
    """
    aligned = pd.concat([target, feature], axis=1).dropna()
    X = aligned.iloc[:, 1]
    y = aligned.iloc[:, 0]

    if method == 'ccf':
        ccf_vals = ccf(y, X, adjusted=False)[:max_lag + 1]
        return np.argmax(np.abs(ccf_vals))

    best_lag = 0
    best_score = np.inf if method in ['aic', 'cv'] else -np.inf

    for lag in range(0, max_lag + 1):
        X_lagged = X.shift(lag).dropna()
        y_aligned = y.loc[X_lagged.index]

        if len(y_aligned) < 10:  # 确保最小样本量
            continue

        if method == 'aic':
            X_sm = sm.add_constant(X_lagged)
            model = sm.OLS(y_aligned, X_sm).fit()
            score = model.aic
        elif method == 'cv':
            tscv = TimeSeriesSplit(n_splits=3)
            rmse_scores = []
            for train_idx, test_idx in tscv.split(X_lagged):
                X_train, X_test = X_lagged.iloc[train_idx], X_lagged.iloc[test_idx]
                y_train, y_test = y_aligned.iloc[train_idx], y_aligned.iloc[test_idx]
                model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
                y_pred = model.predict(sm.add_constant(X_test))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            score = np.mean(rmse_scores) if rmse_scores else np.inf
        else:
            raise ValueError("Invalid method")

        if (method in ['aic', 'cv'] and score < best_score) or \
                (method == 'ccf' and score > best_score):
            best_score = score
            best_lag = lag

    return best_lag


def main(args):
    try:
        sheets_name = ['MSTA', "CH4", 'GMAF', "ET12"]
        data_dict = pd.read_excel(
            "Data_36516473.xlsx",
            sheet_name=sheets_name,
            parse_dates=["Date"],
            index_col="Date"
        )

        # 提取时间序列（修复列名引用）
        MSTA = data_dict['MSTA'].loc['1960':, 'Temperature(C)'].squeeze()
        CH4 = data_dict['CH4'].loc[:, 'CH4(ppb)'].squeeze()
        GMAF = data_dict['GMAF'].loc[:,"Visitors(GMAF)"].squeeze()
        ET12 = data_dict['ET12'].loc[:,"Energy Consumption"].squeeze()

        # 对齐时间范围
        start_date = max(MSTA.index.min(), CH4.index.min(),
                         GMAF.index.min(), ET12.index.min())
        end_date = min(MSTA.index.max(), CH4.index.max(),
                       GMAF.index.max(), ET12.index.max())

        aligned_data = pd.DataFrame({
            'MSTA': MSTA.loc[start_date:end_date],
            'CH4': CH4.loc[start_date:end_date],
            'GMAF': GMAF.loc[start_date:end_date],
            'ET12': ET12.loc[start_date:end_date]
        }).dropna()

        # 自动寻找各变量最优滞后
        max_lag = 24
        optimal_lags = {}
        for feature in ['CH4', 'GMAF', 'ET12']:
            optimal_lag = find_optimal_lag(
                target=aligned_data['MSTA'],
                feature=aligned_data[feature],
                max_lag=max_lag,
                method='aic'
            )
            optimal_lags[feature] = optimal_lag
            print(f"Optimal lag for {feature}: {optimal_lag} months")

        # 标准化与特征生成
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

        # 构建回归模型
        features = [f'{k}_lag{v}' if v > 0 else k for k, v in optimal_lags.items()]
        X = sm.add_constant(scaled_data[features])
        y = scaled_data['MSTA']

        model = sm.OLS(y, X).fit()
        print("\n" + "=" * 50)
        print(model.summary())
        print("Durbin-Watson statistic:", sm.stats.durbin_watson(model.resid))

        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

__maintainer__ = "Zhe GUAN"
__email__ = "zg2u24@soton.ac.uk"