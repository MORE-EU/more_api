from flask import Flask
from flask import request, jsonify
from changepoint_detection import run_cp_detection, read_wash_csv
from rains import extract_rains
from power_index import calculate_pi

app = Flask(__name__)

data_dict = {"eugene": {"id":1, "name":"Eugene", "washes": True},
             "cocoa": {"id":2, "name":"Cocoa", "washes": False}}

path_dict = {"eugene": {"data":"./data/eugene.csv", "washes":"./data/eugene_washes.csv"},
             "cocoa": {"data":"./data/cocoa.csv", "washes":""}}

@app.route("/data", methods=['GET'])
def get_data():
    return jsonify(data_dict)

@app.route("/washes/<dataset_id>", methods=['POST'])
def has_washes(dataset_id):
    if dataset_id in data_dict.keys():
        #return jsonify(data_dict[dataset_id]["washes"])
        res = read_wash_csv(path_dict[dataset_id]["washes"])
        return res.astype(str).to_json()
    return False

@app.route("/cp_detection/<dataset_id>", methods=['POST'])
def cp_detection(dataset_id):
    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')
    w_train = request.json.get('w_train', 30)
    wa1 = request.json.get('wa1', 10)
    wa2 = request.json.get('wa2', 5)
    wa3 = request.json.get('wa3', 10)
    wb1 = request.json.get('wb1', 10)
    wb2 = request.json.get('wb2', 5)
    thrsh = request.json.get('thrsh', 1)
    custom_cp_starts = request.json.get('cp_starts', [])
    custom_cp_ends = request.json.get('cp_ends', [])
    path = path_dict[dataset_id]['data']
    wash_path = path_dict[dataset_id]['washes']
    result_df = run_cp_detection(path=path, wash_path=wash_path,
                                 start_date=start_date,
                                 end_date=end_date, w_train=w_train,
                                 wa1=wa1, wa2=wa2, wa3=wa3,
                                 wb1=wb1, wb2=wb2, thrsh=thrsh,
                                 custom_cp_starts=custom_cp_starts,
                                 custom_cp_ends=custom_cp_ends)
    return result_df.astype(str).to_json()

@app.route("/rains/<dataset_id>", methods=['POST'])
def rains(dataset_id):
    path = path_dict[dataset_id]['data']
    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')
    result_df = extract_rains(path, start_date, end_date)
    return result_df.astype(str).to_json()


@app.route("/power_index/<dataset_id>", methods=['POST'])
def pi_calculation(dataset_id):
    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')
    weeks_train = request.json.get('weeks_train', 4)
    cp_starts = request.json.get('cp_starts', [])
    cp_ends = request.json.get('cp_ends', [])
    query_modelar = request.json.get('query_modelar', False)
    path = path_dict[dataset_id]['data']
    result_df = calculate_pi(path=path, start_date=start_date,
                             end_date=end_date, weeks_train=weeks_train,
                             cp_starts=cp_starts, cp_ends=cp_ends,
                             query_modelar=query_modelar)
    path_out = f'./outputs/{dataset_id}_power_index.csv'
    result_df.to_csv(path_out)
    return {"path_output": path_out}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8889, debug=True)

