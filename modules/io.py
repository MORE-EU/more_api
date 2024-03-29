import pandas as pd
import numpy as np
import h5py
import os
import pathlib
import warnings
import gc
from matplotlib import pyplot as plt
import more_utils
more_utils.set_logging_level("DEBUG")
from more_utils.persistence import ModelarDB
from more_utils.service import TimeseriesService
from cassandra.cluster import Cluster
import json
import joblib
from io import BytesIO

CASSANDRA_IP = os.environ.get('CASSANDRA_IP', '172.17.0.2')
SOIL_KEYSPACE = os.environ.get('SOIL_KEYSPACE', 'moreapi')
YAW_KEYSPACE = os.environ.get('YAW_KEYSPACE', 'moreapi')


def save_model_scaler_cql(model, scaler, name='demo_model'):

    cluster = Cluster([CASSANDRA_IP]) # cassandra adress
    session = cluster.connect(YAW_KEYSPACE) # yaw keyspace

    with BytesIO() as serialized_model, BytesIO() as serialized_scaler:
        print(model)
        joblib.dump(model, serialized_model)
        joblib.dump(scaler, serialized_scaler)

        stmt = """INSERT INTO model_storage(id, name, model, scaler)
                   VALUES (uuid(), ?, ?, ?)"""

        prepared = session.prepare(stmt)
        serialized_model.seek(0)
        serialized_scaler.seek(0)
        # save model and scaler to cassandra
        session.execute(prepared, (name, serialized_model.read(), serialized_scaler.read()))

def load_model_scaler_cql(name='demo_model'):

    cluster = Cluster([CASSANDRA_IP]) # cassandra adress
    session = cluster.connect(YAW_KEYSPACE) # yaw keyspace

    #joblib.dump(model, serialized_model)
    #joblib.dump(scaler, serialized_scaler)

    stmt = """SELECT * FROM model_storage
              WHERE name = ?"""

    prepared = session.prepare(stmt)

    # save model and scaler to cassandra
    print(name)
    res = session.execute(prepared, (name,))
    #for r in res:
    #    print(r)
    for row in res:
        m = row.model
        s = row.scaler
        m = BytesIO(m)
        s = BytesIO(s)
    pipe = joblib.load(m)
    scaler = joblib.load(s)
    return pipe, scaler


def save_power_index_cql(df, start_date, end_date,
                         dataset, cp_starts, cp_ends,
                         weeks_train, query_modelar):

    cluster = Cluster([CASSANDRA_IP]) # cassandra adress
    session = cluster.connect(SOIL_KEYSPACE) # soiling keyspace

    res = session.execute("SELECT MAX(tid) from power_index_table")
    max_tid = res.one().system_max_tid

    if max_tid is None:
        next_tid = 0
    else:
        next_tid = max_tid + 1

    stmt = """INSERT INTO power_index_table(id, tid, timestamp, pi, epl,
                                   start_date, end_date, dataset,
                                   cp_starts, cp_ends, weeks_train,
                                   query_modelar)
               VALUES (uuid(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    prepared = session.prepare(stmt)

    # save power index to cassandra
    for index, row in df.iterrows():
        pi = row.power_index
        epl = row.estimated_power_lost

        session.execute(prepared, (next_tid, str(index), pi, epl,
                                   start_date, end_date, dataset,
                                   json.dumps(cp_starts), json.dumps(cp_ends),
                                   weeks_train, query_modelar))


def load_power_index_cql(start_date, end_date,
                         dataset, cp_starts, cp_ends,
                         weeks_train, query_modelar):

    cluster = Cluster([CASSANDRA_IP]) # cassandra adress
    session = cluster.connect(SOIL_KEYSPACE) # soiling keyspace

    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)

    session.row_factory = pandas_factory
    session.default_fetch_size = None

    res = session.execute("""SELECT timestamp, pi, epl from power_index_table
                             WHERE start_date=%s AND end_date=%s AND dataset=%s
                                   AND cp_starts=%s AND cp_ends=%s AND weeks_train=%s
                                   AND query_modelar=%s ALLOW FILTERING""",
                          [start_date, end_date, dataset,
                           json.dumps(cp_starts), json.dumps(cp_ends),
                           weeks_train, query_modelar])

    res_df = res._current_rows
    if res_df.empty:
       return None
    #print(res_df)
    return res_df


def load_df_modelar(ts_ids, value_column_labels, hostname='localhost', limit=None):
    """
    Load the variables from ModelarDB based on their id
    Return them as a pandas DataFrame.

    Args:
        ts_ids: List of time series id to fetch from modelar
        value_column_labels: Names that will be assigned to each column of the data frame
        hostname: IP adrress of the system where ModelarDB is running
        limit:  Constrain the number of rows returned by ModelarDB to at most "limit" number of rows
    Return:
        pandas DataFrame.
    """
    conn_obj = ModelarDB.connect(hostname="localhost", interface="arrow")
    ts_service = TimeseriesService(db_conn=conn_obj)
    merged_time_series = ts_service.get_time_series_data_from_ts_ids(ts_ids=ts_ids,
                                                                     value_column_labels=value_column_labels,
                                                                     limit=limit)
    df = merged_time_series.fetch_all(fetch_type="pandas")
    df = df.set_index('timestamp', drop=True)
    return df


def load_df(path):
    """
    Loading a parquet file to a pandas DataFrame. Return this pandas DataFrame.

    Args:
        path: Path of the under loading DataFrame.
    Return:
        pandas DataFrame.
    """

    df = pd.DataFrame()
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df.set_index(df.index, inplace=True)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df.index)
        df.set_index(df.index, inplace=True)
    return df


def load_mp(path):
    """
    Load the Univariate/Multivariate Matrix profile which was saved from Create_mp in a .npz file.

    Args:
      path: Path of the directory where the file is saved.

    Return:
        mp: Matrixprofile Distances
        mpi: Matrixprofile Indices
    """
    mp={}
    mpi={}
    loaded = np.load(path + ".npz", allow_pickle=True)
    mp = loaded['mp']
    mpi = loaded['mpi']
    return mp, mpi


def save_mdmp_as_h5(dir_path, name, mps, idx, k=0):

    """
    Save a multidimensional matrix profile as a pair of hdf5 files. Input is based on the output of (https://stumpy.readthedocs.io/en/latest/api.html#mstump).

    Args:
       dir_path: Path of the directory where the file will be saved.
       name: Name that will be appended to the file after a default prefix. (i.e. mp_multivariate_<name>.h5)
       mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension
                   (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
       idx: The multi-dimensional matrix profile index where each row of the array corresponds to each matrix profile index for a given dimension.
       k: If mps and idx are one-dimensional k can be used to specify the given dimension of the matrix profile. The default value specifies the 1-D matrix profile.
                 If mps and idx are multi-dimensional, k is ignored.

    Return:

    """
    if mps.ndim != idx.ndim:
        err = 'Dimensions of mps and idx should match'
        raise ValueError(f"{err}")
    if mps.ndim == 1:
        mps = mps[None, :]
        idx = idx[None, :]
        h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','w')
        h5f.create_dataset(f'mp{k}', data=mps[0])
        h5f.close()

        h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','w')
        h5f.create_dataset(f'idx{k}', data=idx[0])
        h5f.close()
        return

    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','w')
    for i in range(mps.shape[0]):
        h5f.create_dataset(f'mp{i}', data=mps[i])
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','w')
    for i in range(mps.shape[0]):
        h5f.create_dataset(f'idx{i}', data=idx[i])
    h5f.close()
    return


def load_mdmp_from_h5(dir_path, name, k):

    """
    Load a multidimensional matrix profile that has been saved as a pair of hdf5 files.

    Args:
      dir_path: Path of the directory where the file is located.
     name: Name that follows the default prefix. (i.e. mp_multivariate_<name>.h5)
      k: Specifies which K-dimensional matrix profile to load.
                 (i.e. k=2 loads the 2-D matrix profile

    Return:
        mp: matrixprofile/stumpy distances
        index: matrixprofile/stumpy indexes



    """
    # Load MP from disk

    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','r')
    mp= h5f[f'mp{k}'][:]
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','r')
    index = h5f[f'idx{k}'][:]
    h5f.close()
    return mp, index


def save_results(results_dir, sub_dir_name, p, df_stats, m, radius, ez, k, max_neighbors):
    """
    Save the results of a specific run in the directory specified by the results_dir and sub_dir_name.
    The results contain some figures that are created with an adaptation of the matrix profile foundation visualize() function.
    The adaptation works for multi dimensional timeseries and can be found at
    (https://github.com/MORE-EU/matrixprofile/blob/master/matrixprofile/visualize.py) as visualize_md()

    Args:
        results: Path of the directory where the results will be saved.
        sub_directory: Path of the sub directory where the results will be saved.
        p: A profile object as it is defined in the matrixprofile foundation python library.
        df_stats: DataFrame with the desired statistics that need to be saved.
        m: The subsequence window size.
        ez: The exclusion zone to use.
        radius: The radius to use.
        k: The number of the top motifs that were calculated.
        max_neighbors: The maximum amount of neighbors to find for each of the top k motifs.

    Return:
        None

    """




    path = os.path.join(results_dir, sub_dir_name)

    print(path)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        figs = visualize_md(p)

        for i,f in enumerate(figs):
            f.savefig(path + f'/fig{i}.png' , facecolor='white', transparent=False, bbox_inches="tight")
            f.clf()

    # remove figures from memory
    plt.close('all')
    gc.collect()

    df_stats.to_csv(path + '/stats.csv')

    lines = [f'Window size (m): {m}',
             f'Radius: {radius} (radius * min_dist)',
             f'Exclusion zone: {ez} * window_size',
             f'Top k motifs: {k}',
             f'Max neighbors: {max_neighbors}']

    with open(path+'/info.txt', 'w') as f:
        for ln in lines:
            f.write(ln + '\n')


def output_changepoints(scores, indices, dates_rain_start, dates_rain_stop, errors_br, errors_ar, error_name_br, error_name_ar, precip):
    """
    Given the output of one of the functions calc_changepoints_one_model, calc_changepoints_many_models,
    returns a dataframe containing all relevant information about changepoints.
    Args:
        scores: An array of scores associated to each input segmnt
        indices: Indices of the input segments (pointing to the input list)
        dates_rain_start: Array of starting points of segments under investigation
        dates_rain_stop: Array of ending points of segments under investigation
        errors_br: Array of errors corresponding to periods preceding the segments under investigation
        errors_ar: Array of errors corresponding to periods following the segments under investigation
        error_name_br: Name of the error used for the period preceding segments
        error_name_ar: Name of the error used for the period following segments
        precip: Array of precipitation values
    Returns:
        A pandas dataframe containing relevant information about the detected changepoints/segments. This includes scores and
        precipitation values for the input segments.
    """
    start_dates = []
    end_dates = []
    all_prec = []
    all_errors_br = []
    all_errors_ar = []
    all_scores = []
    prec1 = []
    prec2 = []
    types = []
    ids = []
    for i in indices:
        d1 = dates_rain_start.iloc[i]
        d2 = dates_rain_stop.iloc[i]
        all_errors_br.append(errors_br[i])
        all_errors_ar.append(errors_ar[i])
        precip2 = precip.loc[d1:d2][1:-1].values
        start_dates.append(d1)
        end_dates.append(d2)
        if len(precip2)>0:
            prec1.append(precip2.max())
            prec2.append(precip2.mean())
        else:
            prec1.append(0)
            prec2.append(0)
        ids.append(i)
        all_scores.append(scores[i])
    return pd.DataFrame.from_dict({"Score": all_scores, "id": ids, "Starting date": start_dates, "Ending date": end_dates, "Max precipitation": np.round(prec1,2),  "Mean precipitation": np.round(prec2,2), error_name_br+" before rain (true-pred)": all_errors_br,  error_name_ar+" after rain (true-pred)": all_errors_ar})
