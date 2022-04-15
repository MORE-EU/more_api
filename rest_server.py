from flask import Flask
from flask import request, jsonify
from changepoint_detection import run_cp_detection

app = Flask(__name__)

data_dict = {"1": {"eugene":"path", "washes":"path"},
             "2": {"cocoa":"path", "washes":""}}

@app.route("/data", methods=['GET'])
def get_data():
    return jsonify(data_dict)

@app.route("/cp_detection", methods=['GET'])
def cp_detection():
    result_df = run_cp_detection()
    print(result_df)
    return result_df.to_json()

if __name__ == "__main__":
    app.run(debug=True)

