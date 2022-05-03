import pandas as pd
from modules.preprocessing import *

def extract_rains(path, start_date, end_date):
    filename = path
    df = pd.read_csv(filename, index_col = 'timestamp')
    df = df.dropna()
    df.index = pd.DatetimeIndex(df.index)
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    df = filter_dates(df, start_date, end_date)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

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
    if dates_rain_stop.size < dates_rain_start.size:
        dates_rain_start = dates_rain_start[:-1] # drop last starting date for lists to match in size
    for idx in range(dates_rain_start.size):
        d1 = dates_rain_start[idx]
        d2 = dates_rain_stop[idx]
        if np.max(precipitation.loc[d1:d2]) >= x:
            ids.append(idx)
    dates_rain_start_filtered = dates_rain_start[ids]
    dates_rain_stop_filtered = dates_rain_stop[ids]
    df_res = pd.concat([dates_rain_start_filtered, dates_rain_stop_filtered], axis=1)
    df_res.columns = ["start", "stop"]
    return df_res
