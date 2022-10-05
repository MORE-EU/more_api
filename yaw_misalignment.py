from yaw_estimation import DirectYawEstimator
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import seaborn as sns
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.preprocessing import *
from modules.io import *
from modules.learning import *
from modules.patterns import *
from modules.statistics import *
from modules.plots import *
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from tqdm.notebook import tqdm
import seaborn
import pickle
from copy import deepcopy
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from lightgbm import LGBMRegressor as lgbmr
import joblib
from functools import partial
from feature_selection_config import SelectPercentilePd
from sklearn.feature_selection import mutual_info_regression, r_regression

### Note: the code of train() was used to save a pre-trained model in Cassandra for demo peurposes
### Both functions are still under development
def train():
    turbines = [f'/data/data2/engie_initial/post_treated_data/BEZ/BEBEZE01_scada_high_frequency.parquet',
                f'/data/data2/engie_initial/post_treated_data/BEZ/BEBEZE03_scada_high_frequency.parquet']


    df_dict = {}


    for i, t in enumerate(turbines):
        dataset_file = t
        df_temp = load_df(dataset_file)
        df_temp = df_temp.dropna(axis=1, how='all')
        df_temp.columns = df_temp.columns.str.replace('cor. ', '', regex=False)
        cols = ['wind speed', 'pitch angle', 'rotor speed', 'active power',
                'nacelle direction', 'wind direction']
        df_temp = df_temp[cols]
        key = f"{i+1}/" + os.path.basename(t)
        df_dict[key] = df_temp
        print(f"Data from turbine {key} loaded.")


    granularity = '1min'
    for t, df in df_dict.items():
        # Resample with 60second granularity
        df=change_granularity(df,granularity=granularity)

        # calculate dynamic yaw misalignment and take a 60-min rolling mean
        df["theta_d"] = (df['wind direction'] - df['nacelle direction']) % 360
        df["theta_d"][df["theta_d"] > +180] -= 360
        df["theta_d"][df["theta_d"] < -180] += 360
        df["theta_d"] = df["theta_d"].rolling(60).mean()
        df = df.dropna()

        df_initial = df.copy()

        # Perform IQR outlier removal
        df = outliers_IQR(df)

        # drop values of wind speed under 5m/s
        df=filter_col(df, 0, less_than=5, bigger_than=11)

        # drop values with pitch angle higher than 2 degrees or lower than -2 degrees
        df=filter_col(df, 1, less_than=-2, bigger_than=2)

        # drop values of rotor speed under 8rpm
        df=filter_col(df, 2, less_than=8, bigger_than=15)

        # drop values of power near the power_limit and near or below zero

        power_limit = 2050
        df=filter_col(df, 4, less_than=1e-4, bigger_than= 1 * power_limit)


        # Keep only dates with lidar measurements
        start = '2018-06-02'
        end = '2019-01-11'
        df = filter_dates(df, start, end)

        # Remove Outliers using LoF
        # df = outliers_LoF(df).copy()

        # Add resulting dataframe to the dict of all dataframes
        print(df.shape)
        df_dict[t] = df


    df_labels = pd.read_csv('/data/data2/panos/Yaw.csv')
    df_angles_dict = {}
    for i, t in enumerate(turbines):
        key = f"{i+1}/" + os.path.basename(t)
        df_turbine = df_dict[key]
        df_turbine["y"] = np.nan
        df_l = df_labels[df_labels.Turbines == t]
        for start, end, static in zip(df_l.StartDate, df_l.EndDate, df_l.StaticYaw):
            df_turbine.loc[start:end, 'y'] = static
        df_turbine = df_turbine.dropna()
        df_angles_dict[key] = list(df_l.StaticYaw.values)
        df_dict[key] = df_turbine


# Scale the datasets to 0-1 using minmax scaling
    df_scaled_dict = {}
    scaler = create_scaler(list(df_dict.values()))


    test_list = []
    for t, df in df_dict.items():
        df_scaled = df.copy()
        df_scaled_dict[t] = df_scaled
        df_scaled[df_scaled.columns] = scaler.transform(df_scaled)
        df_scaled['y'] = np.abs(df_dict[t]['y']) # use the absolute value of the non-scaled df
# save the scaling parameters to be used on test data
    with open('select_bins_yaw_scaler.pickle', 'wb') as file:
        pickle.dump(scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

    df = pd.concat([d for _, d in df_scaled_dict.items()])
    df = df.sort_index()
    df.describe()

    features = ['wind speed', 'pitch angle', 'active power', 'rotor speed', 'nacelle direction', 'wind direction', 'theta_d']
    target = 'y'
    df_unscaled = pd.concat([d for _, d in df_dict.items()])
    df_unscaled = df_unscaled.sort_index()

    print(df.shape, df_unscaled.shape)

    X_train = df[features]
    y_train = df[target]
    y_train = np.abs(y_train)


    selector = SelectPercentilePd(percentile = 100,
                                  score_func = partial(mutual_info_regression,
                                                       n_neighbors=9))

    base_est =lgbmr(n_jobs=32)

    num_bins = 6 # use 6 bins TODO add this as a parameter later
    e = 1e-5
    min_speed = df["wind speed"].min() - e
    max_speed = df["wind speed"].max() + e
    bin_size = (max_speed - min_speed) / num_bins

    est = DirectYawEstimator(base_estimator=base_est,
                             bin_size=bin_size,
                             min_speed=min_speed,
                             max_speed=max_speed)

    params = {'base_estimator__subsample_freq': 20,
              'base_estimator__subsample': 0.9,
              'base_estimator__reg_lambda': 0.01,
              'base_estimator__reg_alpha': 0.0001,
              'base_estimator__num_leaves': 15,
              'base_estimator__n_estimators': 400,
              'base_estimator__min_child_samples': 10,
              'base_estimator__max_depth': 3,
              'base_estimator__max_bin': 255,
              'base_estimator__learning_rate': 0.2677777777777778,
              'base_estimator__extra_trees': False,
              'base_estimator__colsample_bytree': 0.7}

    est.set_params(**params)

    pipe = Pipeline([("selector", selector),
                     ("estimator", est)])

    pipe.fit(X_train, y_train)

    save_model_scaler_cql(pipe, scaler, name='demo_model')


def estimate_yaw(path='', start_date='2018-06-02',
                 end_date='2019-01-11', window=2,
                 query_modelar=False, dataset_id='bbz2'):
    path = f'/data/data2/engie_initial/post_treated_data/BEZ/BEBEZE02_scada_high_frequency.parquet'
    turbines = [path]
    df_dict = {}
    if start_date is None:
        start_date = '2018-06-02'
    if end_date is None:
        end_date = '2019-01-11'


    for i, t in enumerate(turbines):
        dataset_file = t
        df_temp = load_df(dataset_file)
        df_temp = df_temp.dropna(axis=1, how='all')
        df_temp.columns = df_temp.columns.str.replace('cor. ', '', regex=False)
        cols = ['wind speed', 'pitch angle', 'rotor speed', 'active power',
                'nacelle direction', 'wind direction']
        df_temp = df_temp[cols]
        key = f"{i+1}/" + os.path.basename(t)
        df_dict[key] = df_temp
        print(f"Data from turbine {key} loaded.")

    granularity = '1min'
    for t, df in df_dict.items():
        # Resample with 60second granularity
        df=change_granularity(df,granularity=granularity)

        # calculate dynamic yaw misalignment and take a 60-min rolling mean
        df["theta_d"] = (df['wind direction'] - df['nacelle direction']) % 360
        df["theta_d"][df["theta_d"] > +180] -= 360
        df["theta_d"][df["theta_d"] < -180] += 360
        df["theta_d"] = df["theta_d"].rolling(60).mean()
        df = df.dropna()

        df_initial = df.copy()

        # Perform IQR outlier removal
        df = outliers_IQR(df)

        # drop values of wind speed under 5m/s
        df=filter_col(df, 0, less_than=5, bigger_than=11)

        # drop values with pitch angle higher than 2 degrees or lower than -2 degrees
        df=filter_col(df, 1, less_than=-2, bigger_than=2)

        # drop values of rotor speed under 8rpm
        df=filter_col(df, 2, less_than=8, bigger_than=15)

        # drop values of power near the power_limit and near or below zero

        power_limit = 2050
        df=filter_col(df, 4, less_than=1e-4, bigger_than= 1 * power_limit)


        # Keep only dates with lidar measurements
        start = start_date
        end = end_date
        df = filter_dates(df, start, end)

        # Remove Outliers using LoF
        # df = outliers_LoF(df).copy()

        # Add resulting dataframe to the dict of all dataframes
        print(df.shape)
        df_dict[t] = df

    df_labels = pd.read_csv('/data/data2/panos/Yaw.csv')
    df_angles_dict = {}
    for i, t in enumerate(turbines):
        key = f"{i+1}/" + os.path.basename(t)
        df_turbine = df_dict[key]
        df_turbine["y"] = np.nan
        df_l = df_labels[df_labels.Turbines == t]
        for start, end, static in zip(df_l.StartDate, df_l.EndDate, df_l.StaticYaw):
            df_turbine.loc[start:end, 'y'] = static
        df_turbine = df_turbine.dropna()
        df_angles_dict[key] = list(df_l.StaticYaw.values)
        df_dict[key] = df_turbine
        print(f"Static Yaw angles: {df_angles_dict[key]}")

    pipe, scaler = load_model_scaler_cql(name='demo_model')
    # Scale the datasets to 0-1 using minmax scaling
    df_scaled_dict = {}
    scaler = scaler # use scaler from training

    test_list = []
    for t, df in df_dict.items():
        df_scaled = df.copy()
        df_scaled_dict[t] = df_scaled
        df_scaled[df_scaled.columns] = scaler.transform(df_scaled)
        df_scaled['y'] = np.abs(df_dict[t]['y']) # use the absolute value of the non-scaled df
        # save the scaling parameters to be used on test data

    df = pd.concat([d for _, d in df_scaled_dict.items()])
    df = df.sort_index()

    features = ['wind speed', 'pitch angle', 'active power', 'rotor speed', 'nacelle direction', 'wind direction', 'theta_d']
    target = 'y'
    df_unscaled = pd.concat([d for _, d in df_dict.items()])
    df_unscaled = df_unscaled.sort_index()

    print(df.shape, df_unscaled.shape)

    X_test = df[features]
    y_test = df[target]
    y_test = np.abs(y_test)
    prediction = pipe.predict(X_test)
    target = y_test
    rmse = np.sqrt(np.nanmean((prediction.values-target.values)**2))
    mae = np.nanmean(np.abs(prediction.values-target.values))
    mape = mape1(target.values, prediction.values)
    print('TEST SET RESULTS')
    print(f"RMSE = {rmse}")
    print(f"MAE = {mae}")
    print(f"MAPE = {mape}")

    d = window
    df_prediction = pd.DataFrame(np.zeros((X_test.shape[0], 1)), columns=['prediction'], index=X_test.index)
    start = X_test.index.min()
    end = X_test.index.max()
    while end >= start:
        w = start + pd.Timedelta(days=d)
        if len(X_test.loc[start:w]) == 0:
            start = w
            continue
        prediction = pipe.predict(X_test.loc[start:w])
        df_prediction.prediction[start:w] = np.mean(prediction)
        start = w
    df_prediction
    return df_prediction
