import numpy as np
import os, sys
import math
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt
from modules.preprocessing import *
from modules.statistics import *
from modules.learning import *
from modules.io import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def calculate_pi(weeks_train, start_date, end_date, path, dataset_id,
                cp_starts, cp_ends, query_modelar=False):

    filename = path

    res_df = load_power_index_cql(start_date=start_date, end_date='end_date',
                                  dataset=dataset_id, cp_starts=cp_starts, cp_ends=cp_ends,
                                  weeks_train=weeks_train, query_modelar=query_modelar)

    if res_df is not None:
        # if power index is stored in cassandra dont recalculate
        res_df.set_index('timestamp', inplace=True)
        res_df.index = pd.DatetimeIndex(res_df.index)
        res_df.columns = ['power_index', 'estimated_power_lost']
        res_df.sort_index(inplace=True)
        return res_df



    if not query_modelar:
        df = pd.read_csv(filename, index_col = 'timestamp')
    else:
        print("Use modelar")
        df = load_df_modelar([1, 2, 3], ['irradiance', 'power', 'mod_temp'],
                             hostname='localhost', limit=10**6)

    df = df.dropna()
    df.index = pd.DatetimeIndex(df.index)
    feats = ['irradiance', 'mod_temp']
    target = 'power'
    w_train = weeks_train * 1

    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()

    df = filter_dates(df, start_date, end_date)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    ref_points = pd.Index(pd.Series(cp_ends))
    model1, _, _ = train_on_reference_points(df_scaled, w_train, ref_points, feats, target)

    est = predict(df_scaled, model1, feats, target)
    df_est = pd.DataFrame(est, columns=['estimated_power'], index=df_scaled.index)
    pi_daily = (df_scaled.power/df_est.estimated_power).resample("1D").median()
    pi_daily = np.clip(pi_daily, 0, 1)
    pi_daily = pi_daily.ffill()
    df_daily = df.resample("1D").sum()
    df_daily = df_daily.ffill()
    daily_loss = calculate_daily_loss(pi_daily, df_daily)
    df_result = pd.concat([pi_daily, daily_loss], axis=1)
    df_result.columns = ['power_index', 'estimated_power_lost']
    save_power_index_cql(df_result, start_date=start_date, end_date='end_date',
                         dataset=dataset_id, cp_starts=cp_starts, cp_ends=cp_ends,
                         weeks_train=weeks_train, query_modelar=query_modelar)

    return df_result


def calculate_daily_loss(pi_daily, df_daily):
    daily_loss = (df_daily.power/pi_daily) - df_daily.power
    return daily_loss


