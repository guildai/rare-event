import json
import os

import argparse

import numpy as np

from flask import request, redirect, g
from flask_api import FlaskAPI

app = FlaskAPI(__name__)
args = None # set during init (not sure how to get config to handlers otherwise)

def main():
    _init_args()
    app.run(
        host=args.host,
        port=args.port,
        threaded=False, # TF threading - simplify to one request at a time
        debug=args.debug)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="checkpoints")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--label")
    p.add_argument("--debug", action="store_true")
    globals()["args"] = p.parse_args()

@app.route("/")
def index():
    return redirect("/predict")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return []
    model = _g_model()
    x = np.array(request.data)
    predicted = model.predict(x)
    mse = _compare_predicted(predicted, x, model)
    return [
        (1 if err > args.threshold else 0, err, i)
        for i, err in enumerate(mse)
    ]

def _compare_predicted(predicted, x, model):
    if model.type == "ae":
        return np.mean(np.power(x - predicted, 2), axis=1)
    elif model.type == "lstm":
        return np.mean(np.power(_flatten(x) - _flatten(predicted), 2), axis=1)
    else:
        assert False, model.type

def _flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

def _g_model():
    try:
        return g.model
    except AttributeError:
        g.model = model = _init_model()
        return model

def _init_model():
    from keras.models import model_from_json
    model_json = open(os.path.join(args.model_dir, "model.json"), "r").read()
    model = model_from_json(model_json)
    model.type = _model_type(model)
    model.load_weights(os.path.join(args.model_dir, "weights.h5"))
    return model

def _model_type(m):
    if m.layers[0].name == "lstm_1":
        return "lstm"
    else:
        return "ae"

@app.route("/echo", methods=["POST", "GET"])
def echo():
    return request.data

@app.route("/label", methods=["GET"])
def label():
    return args.label or "unset"

@app.route("/model", methods=["GET"])
def model():
    model = _g_model()
    return json.loads(model.to_json())

if __name__ == "__main__":
    main()
