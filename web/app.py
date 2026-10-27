import os
import json
from flask import Flask, request, jsonify, send_from_directory, render_template

from web.inference import inference_engine

app = Flask(__name__, template_folder='templates', static_folder='static')

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": inference_engine.model is not None,
        "threshold": float(inference_engine.threshold) if inference_engine.threshold is not None else None,
        "seq_len": inference_engine.seq_len
    })


@app.route('/api/metrics')
def metrics():
    try:
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            m = json.load(f)
        meta = {}
        if os.path.exists(os.path.join(MODEL_DIR, "model_metadata.json")):
            with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
                meta = json.load(f)
        return jsonify({"metrics": m, "metadata": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/examples')
def examples():
    try:
        import numpy as np
        X_test = np.load(os.path.join(DATA_DIR, "test_data.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
        items = []
        # Return first 20 as examples
        for i in range(min(20, len(X_test))):
            items.append({"index": i, "true_label": float(y_test[i])})
        return jsonify({"examples": items, "count": len(X_test)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/example/<int:idx>')
def example(idx):
    try:
        import numpy as np
        X_test = np.load(os.path.join(DATA_DIR, "test_data.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
        if idx < 0 or idx >= len(X_test):
            return jsonify({"error": "index out of range"}), 400
        res = inference_engine.predict(X_test[idx])
        res["true_label"] = float(y_test[idx])
        res["index"] = idx
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON"}), 400
        if 'signal' not in data:
            return jsonify({"error": "Missing 'signal' field"}), 400
        signal = data['signal']
        # Validation
        if not isinstance(signal, (list, tuple)):
            return jsonify({"error": "'signal' must be a list"}), 400
        if len(signal) != inference_engine.seq_len:
            return jsonify({"error": f"Signal must have exactly {inference_engine.seq_len} elements"}), 400
        # Check finite numeric
        import math
        for i, v in enumerate(signal):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return jsonify({"error": f"Non-numeric value at index {i}"}), 400
            if not math.isfinite(v):
                return jsonify({"error": f"Non-finite value at index {i}"}), 400
        res = inference_engine.predict(signal)
        if 'true_label' in data:
            res['true_label'] = data['true_label']
        return jsonify(res)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
