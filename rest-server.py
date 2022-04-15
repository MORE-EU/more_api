from flask import Flask
from flask import request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello_world():
    return "Hello"

if __name__ == "__main__":
    app.run(debug=True)

