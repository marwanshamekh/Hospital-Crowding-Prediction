"""
Hospital Crowding Prediction - Flask REST API Backend
Connects the existing frontend Dashboard to the trained Decision Tree Machine Learning Model.
"""

import os
import sys
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory
# pyrefly: ignore [missing-import]
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import model_service

# Initialize Flask app with frontend folder as static source
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    """Serves the dashboard index.html from frontend directory."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_static_file(path):
    """Serves frontend static assets (CSS, JS, images, icons)."""
    full_path = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({
        "status": "error",
        "message": "Resource not found on this server."
    }), 404


@app.route("/health", methods=["GET"])
@app.route("/api/status", methods=["GET"])
def health_check():
    """Returns system and model health status."""
    try:
        status_info = model_service.get_status()
        return jsonify({
            "status": "online",
            "service": "Hospital Crowding Prediction API",
            "model_loaded": status_info["is_loaded"],
            "model_type": status_info["model_type"],
            "algorithm": status_info["algorithm"],
            "accuracy": status_info["accuracy_percentage"],
            "accuracy_raw": status_info["accuracy"],
            "f1_score": status_info["f1_score"],
            "dataset_records": status_info["total_records"],
            "engineered_features": status_info["engineered_features_count"],
            "classes": status_info["classes"]
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }), 500


@app.route("/api/feature-importance", methods=["GET"])
def get_feature_importance():
    """Returns the real feature importances from the loaded Decision Tree model."""
    try:
        importances = model_service.get_feature_importance()
        return jsonify({
            "status": "success",
            "feature_importances": importances
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve feature importances: {str(e)}"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Receives hospital operational features, applies preprocessing,
    and returns real Decision Tree prediction result.
    """
    try:
        # Check Content-Type and parse JSON
        if not request.is_json:
            # Attempt to parse even if Content-Type was omitted
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({
                    "status": "error",
                    "message": "Invalid request: Content-Type must be 'application/json' with a valid JSON body."
                }), 400
        else:
            data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "Empty request payload. Please provide hospital operational data."
            }), 400

        # Execute prediction through model service
        result = model_service.predict(data)

        if result.get("status") == "error":
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An unexpected server error occurred during prediction: {str(e)}"
        }), 500


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "status": "error",
        "message": "Bad Request: " + str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Resource not found on this server."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "message": "Method not allowed for this endpoint."
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error occurred."
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print("\n" + "="*60)
    print(f"Hospital Crowding Prediction Backend Server")
    print(f"Model: Decision Tree Classifier (99.17% Accuracy)")
    print(f"Server URL: http://{host}:{port}")
    print(f"Prediction API: POST http://{host}:{port}/predict")
    print(f"Status API: GET http://{host}:{port}/health")
    print("="*60 + "\n")
    app.run(host=host, port=port, debug=False)
