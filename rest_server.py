from flask import Flask
from flask import request, jsonify
from changepoint_detection import run_cp_detection

app = Flask(__name__)

data_dict = {"1": {"id":1, "name":"Eugene", "washes": True},
             "2": {"id":2, "name":"Cocoa", "washes": False}}

path_dict = {"1": {"data":"./eugene.csv", "washes":"./eugene_washes.csv"},
             "2": {"data":"./cocoa.csv", "washes":""}}

@app.route("/data", methods=['GET'])
def get_data():
    return jsonify(data_dict)

@app.route("/cp_detection/<dataset_id>", methods=['GET'])
def cp_detection(dataset_id):
    path = path_dict[dataset_id]['data']
    wash_path = path_dict[dataset_id]['washes']
    print(data_dict[dataset_id])
    print(path)
    print(wash_path)
    result_df = run_cp_detection(path, wash_path)
    return result_df.to_json()

if __name__ == "__main__":
    app.run(debug=True)

