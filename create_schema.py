from cassandra.cluster import Cluster
import json
import os

CASSANDRA_IP = os.environ.get('CASSANDRA_IP', '172.17.0.2')
SOIL_KEYSPACE = os.environ.get('SOIL_KEYSPACE', 'moreapi')
YAW_KEYSPACE = os.environ.get('YAW_KEYSPACE', 'moreapi')
cluster = Cluster([CASSANDRA_IP])
session = cluster.connect()

res = session.execute(f"CREATE KEYSPACE IF NOT EXISTS {SOIL_KEYSPACE} " + \
                       "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}")

res = session.execute(f"CREATE KEYSPACE IF NOT EXISTS {YAW_KEYSPACE} " + \
                       "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}")

res = session.execute(f"USE {SOIL_KEYSPACE}")

res = session.execute("""CREATE TABLE IF NOT EXISTS power_index_table (
                             id uuid,
                             tid int,
                             timestamp varchar,
                             pi float,
                             epl float,
                             start_date varchar,
                             end_date varchar,
                             dataset varchar,
                             cp_starts varchar,
                             cp_ends varchar,
                             weeks_train int,
                             query_modelar boolean,
                             PRIMARY KEY ((id, start_date, end_date, dataset,
                                           cp_starts, cp_ends, weeks_train,
                                           query_modelar))
                         )""")

res = session.execute(f"USE {YAW_KEYSPACE}")

res = session.execute("""CREATE TABLE IF NOT EXISTS model_storage (
                             id uuid,
                             name varchar,
                             model blob,
                             scaler blob,
                             PRIMARY KEY (name)
                         )""")
