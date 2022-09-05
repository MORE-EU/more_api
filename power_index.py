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


def calculate_pi(weeks_train, start_date, end_date,
                 path, cp_starts, cp_ends, query_modelar=False):

    filename = path

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
    #print(df.shape)

    ref_points = pd.Index(pd.Series(cp_ends))
    ### debugging print in comments
    #print(cp_ends)
    model1, _, _ = train_on_reference_points(df_scaled, w_train, ref_points, feats, target)

    est = predict(df_scaled, model1, feats, target)
    #print(df_scaled.shape)
    #est = np.clip(est, 0, 1)
    #negatives = [x for x in est if x == 0]
    #print(f"Shape of estimations {est.shape}")
    #print(f"Number of neg {len(negatives)}")
    df_est = pd.DataFrame(est, columns=['estimated_power'], index=df_scaled.index)
    #print(df_scaled.power.resample("1D").median().head(5))
    #print(df_scaled.power.head(60))
    #print(f"est nans = {df_est.estimated_power.isna().sum()}")
    pi_daily = (df_scaled.power/df_est.estimated_power).resample("1D").median()
    #print(f"pi nans = {pi_daily.isna().sum()}")
    pi_daily = np.clip(pi_daily, 0, 1)
    pi_daily = pi_daily.ffill()
    ### try a smoothened version
    #pi_daily = pi_daily.ffill().rolling('7D').median()
    ### debugging plot
    #daily_derate = df.soiling_derate.resample("1D").median()
    #daily_derate = daily_derate.ffill()
    df_daily = df.resample("1D").sum()
    df_daily = df_daily.ffill()
    #plt.figure(figsize=(12,8))
    #plt.plot(pi_daily.values, label='pred')
    #plt.plot(daily_derate.values)
    #plt.legend()
    #plt.savefig('./test_derate.png')
    #print(daily_derate)
    ### calculate aggregate power loss
    daily_loss = calculate_daily_loss(pi_daily, df_daily)
    df_result = pd.concat([pi_daily, daily_loss], axis=1)
    df_result.columns = ['power_index', 'estimated_power_lost']
    return df_result


def calculate_daily_loss(pi_daily, df_daily):
    daily_loss = (df_daily.power/pi_daily) - df_daily.power
    #print(pi_daily.values)
    #print(daily_loss.sum())
    return daily_loss


