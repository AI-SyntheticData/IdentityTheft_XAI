"""
Flask REST API for real-time identity theft detection
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from typing import Dict, Any

from model_manager import ModelManager
from inference import IdentityTheftDetector
from config import HYBRID_WEIGHTS, DECISION_THRESHOLDS


# Initialize Flask app
app = Flask(__name__)

# Global variables for model
detector = None
model_info = None


def initialize_model():
    """Load trained model on startup"""
    global detector, model_info

    try:
        print("Loading trained model...")
        models_dir = "outputs/models"
        manager = ModelManager(models_dir)
        xgb_model, iso_model, feature_names, metadata = manager.load_model()

        detector = IdentityTheftDetector(
            xgb_model=xgb_model,
            iso_model=iso_model,
            feature_names=feature_names,
            w1=HYBRID_WEIGHTS['supervised'],
            w2=HYBRID_WEIGHTS['anomaly'],
            theta_low=DECISION_THRESHOLDS['low'],
            theta_high=DECISION_THRESHOLDS['high']
        )

        model_info = {
            'version': metadata.get('version', 'unknown'),
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'test_auc': metadata.get('hybrid_metrics', {}).get('roc_auc', None),
        }

        print(f"Model loaded successfully! Version: {model_info['version']}")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503

    return jsonify({
        'status': 'healthy',
        'model_version': model_info.get('version', 'unknown'),
        'model_loaded': True
    }), 200


@app.route('/info', methods=['GET'])
def get_info():
    """Get model information"""
    if detector is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    return jsonify({
        'model_info': model_info,
        'detector_config': detector.get_model_info(),
        'endpoints': {
            'health': '/health',
            'info': '/info',
            'predict': '/predict (POST)',
            'predict_batch': '/predict/batch (POST)',
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict identity theft risk for a single application

    Expected JSON body:
    {
        "age": 35,
        "doc_auth_score": 0.75,
        "face_match_score": 0.82,
        "liveness_score": 0.79,
        "ip_geo_match": 1,
        "vpn_or_tor": 0,
        "device_reuse_count": 2,
        "email_risk_score": 0.25,
        "phone_voip_flag": 0,
        "ssn_high_risk_flag": 0
    }
    """
    if detector is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400

        # Validate required features
        missing_features = set(model_info['feature_names']) - set(data.keys())
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': list(missing_features)
            }), 400

        # Make prediction
        result = detector.predict_single(data, explain=True)

        return jsonify({
            'success': True,
            'prediction': result
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict identity theft risk for multiple applications

    Expected JSON body:
    {
        "applications": [
            {
                "age": 35,
                "doc_auth_score": 0.75,
                ...
            },
            {
                "age": 28,
                "doc_auth_score": 0.42,
                ...
            }
        ],
        "explain": true  // optional, default false
    }
    """
    if detector is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    try:
        # Get request data
        data = request.get_json()

        if not data or 'applications' not in data:
            return jsonify({
                'error': 'No applications provided'
            }), 400

        applications = data['applications']
        explain = data.get('explain', False)

        if not isinstance(applications, list) or len(applications) == 0:
            return jsonify({
                'error': 'Applications must be a non-empty list'
            }), 400

        # Convert to DataFrame
        df = pd.DataFrame(applications)

        # Make predictions
        results = detector.predict_batch(df, explain=explain)

        # Convert to list of dicts
        predictions = results.to_dict(orient='records')

        return jsonify({
            'success': True,
            'n_applications': len(predictions),
            'predictions': predictions
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/thresholds', methods=['GET', 'POST'])
def manage_thresholds():
    """Get or update decision thresholds"""
    if detector is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    if request.method == 'GET':
        return jsonify({
            'threshold_low': detector.theta_low,
            'threshold_high': detector.theta_high,
            'decisions': {
                'Verified': f'risk_score < {detector.theta_low}',
                'Review': f'{detector.theta_low} <= risk_score < {detector.theta_high}',
                'Suspicious': f'risk_score >= {detector.theta_high}'
            }
        }), 200

    elif request.method == 'POST':
        try:
            data = request.get_json()

            theta_low = data.get('threshold_low')
            theta_high = data.get('threshold_high')

            if theta_low is not None and not (0 <= theta_low <= 1):
                return jsonify({'error': 'threshold_low must be between 0 and 1'}), 400

            if theta_high is not None and not (0 <= theta_high <= 1):
                return jsonify({'error': 'threshold_high must be between 0 and 1'}), 400

            if theta_low is not None and theta_high is not None and theta_low >= theta_high:
                return jsonify({'error': 'threshold_low must be less than threshold_high'}), 400

            # Update thresholds
            detector.set_thresholds(theta_low=theta_low, theta_high=theta_high)

            return jsonify({
                'success': True,
                'threshold_low': detector.theta_low,
                'threshold_high': detector.theta_high
            }), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health',
            '/info',
            '/predict',
            '/predict/batch',
            '/thresholds'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500


def main():
    """Start the API server"""
    print("="*60)
    print("IDENTITY THEFT DETECTION API")
    print("="*60)

    # Initialize model
    if not initialize_model():
        print("\nFailed to load model. Please run 'python src/train.py' first.")
        return

    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:8000")
    print("\nAvailable endpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /info            - Model information")
    print("  POST /predict         - Single prediction")
    print("  POST /predict/batch   - Batch predictions")
    print("  GET  /thresholds      - Get decision thresholds")
    print("  POST /thresholds      - Update decision thresholds")
    print("="*60)

    # Run server
    app.run(host='0.0.0.0', port=8000, debug=False)


if __name__ == '__main__':
    main()

