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


def run_cp_detection(extr_rains=True, method="method1"):
    filename = './input_panel.csv'
    dates_wash_start = pd.to_datetime(pd.Series(['2013-03-11 00:00:00', '2013-07-10 00:00:00', '2013-08-14 00:00:00', '2013-08-21 00:00:00', '2013-08-26 00:00:00']))
    dates_wash_stop = pd.to_datetime(pd.Series(['2013-03-12 00:00:00', '2013-07-11 00:00:00', '2013-08-15 00:00:00', '2013-08-22 00:00:00','2013-08-27 00:00:00']))
    df = pd.read_csv('./input_panel.csv', index_col = 'timestamp')
    df = df.dropna()
    df.index = pd.DatetimeIndex(df.index)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    if extr_rains:
        if df.precipitation.iloc[0]>0:
            precipitation = pd.concat([pd.Series({min(df.index)-pd.Timedelta('1s'): 0}),df.precipitation])
        else:
            precipitation = df.precipitation

        precipitation.index = pd.to_datetime(precipitation.index)
        df_dates = pd.DataFrame(index = precipitation.index)
        df_dates["rain_start"] = precipitation[(precipitation.shift(-1) > 0) & (precipitation == 0)] # compare current to next
        df_dates["rain_stop"] = precipitation[(precipitation.shift(1) > 0) & (precipitation == 0)] # compare current to prev
        dates_rain_start = pd.Series(df_dates.rain_start.index[df_dates.rain_start.notna()])
        dates_rain_stop = pd.Series(df_dates.rain_stop.index[df_dates.rain_stop.notna()])

    # filter light rains
    x = 0.1
    ids = []
    for idx in range(dates_rain_start.size):
        d1 = dates_rain_start[idx]
        d2 = dates_rain_stop[idx]
        if np.max(precipitation.loc[d1:d2]) >= x:
            ids.append(idx)
    dates_rain_start_filtered = dates_rain_start[ids]
    dates_rain_stop_filtered = dates_rain_stop[ids]


    # initialize params
    wa1 = 10  # window-length of days to train (before the rain), for Method 1
    wa2 = 5 # window-length of days to validate (before the rain), for Method 1
    wa3 = 10 # window-length of days to test (after the rain), for Method 1
    wb1 = 10 # window-length of days to test (before the rain), for Method 2
    wb2 = 5 # window-length of days to test (after the rain), for Method 2
    error_br_column = 5 #0=r_squared, 1=mae, 2=me, 3=mape, 4=mpe, 5=median error
    error_ar_column = 5
    thrsh = 1
    errors_br1 = np.empty((dates_rain_start_filtered.size, 6))
    errors_ar1 = np.empty((dates_rain_start_filtered.size, 6))
    scores = np.empty((dates_rain_start_filtered.size))
    indices = np.empty(len(scores), dtype=int)
    error_names = {0: "r_squared", 1: "MAE", 2: "ME (true-pred)", 3: "MAPE", 4: "MPE (true-pred)", 5: "Median error"}


    # detect changepoints
    p_changepoints_start = (pd.Series(dates_rain_start_filtered.tolist() + dates_wash_start.tolist()).sort_values())
    p_changepoints_stop = (pd.Series(dates_rain_stop_filtered.tolist() + dates_wash_stop.tolist()).sort_values())
    target = 'power'
    feats = ['irradiance', 'mod_temp']
    error_name_br = error_names[error_br_column]
    error_name_ar = error_names[error_ar_column]

    errors_br1, errors_ar1 = calc_changepoints_many_models(df_scaled, p_changepoints_start, p_changepoints_stop, target, feats, wa1, wa2, wa3 )

    #set threshold on MAPE error before rain
    mask1 = (errors_br1[:,3]<= 0.05)

    #compute scores for the remaining
    scores1 = -(errors_br1[:, error_br_column]-errors_ar1[:, error_ar_column])/np.abs(errors_br1[:, error_ar_column])
    scores1[(~mask1)] = np.finfo('d').min


    #sort
    indices1 = np.argsort(-scores1)

    #compute final output
    precip = df.precipitation
    df_events_output1=pd.DataFrame(output_changepoints(scores1, indices1, p_changepoints_start, p_changepoints_stop,
                                   errors_br1[:, error_br_column], errors_ar1[:, error_ar_column], error_name_br, error_name_ar, precip))

    # effective changepoints = the ones with score at least thrsh
    mask1 = (df_events_output1['Score']>thrsh)

    effective_cp1 = df_events_output1[mask1]["id"].values

    print(f'Number of "effective" changepoints using Method 1: {len(set(effective_cp1))}')
    if method == "method1":
        ref_points = pd.Index(pd.Series(p_changepoints_stop.iloc[list(effective_cp1)]))
        res = method1_segmentation(df=df, df_scaled=df_scaled ,p_changepoints_start=p_changepoints_start,
                                   p_changepoints_stop=p_changepoints_stop, effective_cp1=effective_cp1,
                                   w_train=30, ref_points=ref_points, feats=feats, target=target)
        return res
    else:
        print("Not implemented")

def method1_segmentation(df, df_scaled, p_changepoints_start, p_changepoints_stop, effective_cp1, ref_points, feats, target, w_train=30):
    # train model
    model1, training_error, validation_error = train_on_reference_points(df_scaled, w_train, ref_points, feats, target)


    dates_changepoints = pd.DatetimeIndex(p_changepoints_stop.iloc[list(effective_cp1)])
    dates_start = (pd.Index(pd.Series(p_changepoints_start.iloc[list(effective_cp1)].tolist()).sort_values()))
    dates_stop = (pd.Index(pd.Series(p_changepoints_stop.iloc[list(effective_cp1)].tolist()).sort_values()))
    dates_start = dates_start.union([max(df.index)])
    dates_stop = dates_stop.union([min(df.index)])

    # assign scores
    ref_model = model1
    slopes = []
    mpe_scores = []
    sign_slopes = []
    for j, d2 in enumerate(dates_start):
        d1 = dates_stop[dates_stop.get_loc(d2, method='pad')]
        try:
            y_pred = predict(df_scaled.loc[d1:d2], ref_model, feats, target)
            diff = (df_scaled.loc[d1:d2].power - y_pred).values
            line , slope, _ = get_ts_line_and_slope(diff)
            mpe_scores.append(st.score_segment(df_scaled.loc[d1:d2].power, y_pred)[0])
            slopes.append(-slope)
            sign_slopes.append(np.sign(-slope))
        except:
            sign_slopes.append(np.finfo('d').min)
            slopes.append(np.finfo('d').min)
            mpe_scores.append(1)

    # format scores for output
    indices = np.argsort(np.array(mpe_scores)*(np.array(sign_slopes)))
    all_scores = []
    all_dates_start = []
    all_dates_end = []
    for j in indices:
        d2 = dates_start[j]
        d1 = dates_stop[dates_stop.get_loc(d2, method='pad')]
        try:
            d3 = dates_stop[dates_stop.get_loc(d2, method='bfill')]
            d4 = d3 + pd.to_timedelta(d2-d1)
        except:
            d3 = d2
            d4 = d2
        if slopes[j] > 0 and mpe_scores[j]<0 and len(df_scaled.loc[d1:d2])>0:
            y_pred = predict(df_scaled.loc[d1:d2], model1, feats, target)
            all_dates_start.append(d1)
            all_dates_end.append(d2)
            all_scores.append(-sign_slopes[j]*mpe_scores[j])
            diff = ((df_scaled.loc[d1:d2].power - y_pred))
            line , slope, _ = get_ts_line_and_slope(diff.values)

    df_segments_output = (pd.DataFrame.from_dict({"Score": all_scores, "Starting date": all_dates_start, "Ending date": all_dates_end}))
    print(df_segments_output)
    return df_segments_output
