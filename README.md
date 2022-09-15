# More RESTful API services

## Installation

- Clone the repository

```shell

git clone https://github.com/MORE-EU/more_api.git
```

- Get into the repository you just cloned

```shell
cd more_api
```

- Install the dependancies

```shell
pip install -r requirements.txt
```

- Install https://github.com/IBM/more-utils and https://github.com/ModelarData/PyModelarDB

- Install Cassandra by following the instructions found here https://cassandra.apache.org/_/quickstart.html 

- Set the CASSANDRA_IP env variable to the adress where cassandra is running:

```shell
export CASSANDRA_IP=<ip>
```

- To run the rest server execute:

```shell
python rest_server.py
```

- Before issuing any queries make sure to initiallize the Cassandra schema by executing:
```shell
python create_schema.py
```
