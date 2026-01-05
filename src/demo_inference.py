"""
Demo script showing how to use the trained model for inference
"""

import os
import sys
import pandas as pd
import numpy as np

from model_manager import ModelManager
from inference import IdentityTheftDetector
from config import HYBRID_WEIGHTS, DECISION_THRESHOLDS, RANDOM_STATE


def load_trained_model():
    """Load the latest trained model"""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    manager = ModelManager(models_dir)

    print("Loading trained model...")
    xgb_model, iso_model, feature_names, metadata = manager.load_model()

    print(f"Model version: {metadata.get('version', 'unknown')}")
    print(f"Trained on {metadata.get('training_samples', 'N/A')} samples")
    print(f"Test AUC: {metadata.get('hybrid_metrics', {}).get('roc_auc', 'N/A'):.4f}")

    return xgb_model, iso_model, feature_names, metadata


def example_single_prediction():
    """Example: Make a single prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Application Prediction")
    print("="*60)

    # Load model
    xgb_model, iso_model, feature_names, metadata = load_trained_model()

    # Create detector
    detector = IdentityTheftDetector(
        xgb_model=xgb_model,
        iso_model=iso_model,
        feature_names=feature_names,
        w1=HYBRID_WEIGHTS['supervised'],
        w2=HYBRID_WEIGHTS['anomaly'],
        theta_low=DECISION_THRESHOLDS['low'],
        theta_high=DECISION_THRESHOLDS['high']
    )

    # Example application (suspicious case)
    print("\nCase 1: Suspicious Application")
    application_1 = {
        "age": 28,
        "doc_auth_score": 0.35,      # Low document score
        "face_match_score": 0.42,    # Poor face match
        "liveness_score": 0.38,      # Low liveness
        "ip_geo_match": 0,           # IP mismatch
        "vpn_or_tor": 1,             # VPN detected
        "device_reuse_count": 7,     # High device reuse
        "email_risk_score": 0.85,    # High-risk email
        "phone_voip_flag": 1,        # VOIP number
        "ssn_high_risk_flag": 1,     # High-risk SSN
    }

    result = detector.predict_single(application_1, explain=True)

    print(f"\nDecision: {result['decision']}")
    print(f"Risk Score: {result['risk_score']:.4f}")
    print(f"  - Supervised Score: {result['supervised_score']:.4f}")
    print(f"  - Anomaly Score: {result['anomaly_score']:.4f}")

    if 'top_features' in result:
        print("\nTop Contributing Features:")
        for feat_info in result['top_features']:
            print(f"  - {feat_info['feature']}: {feat_info['value']:.3f} "
                  f"(SHAP: {feat_info['shap_contribution']:+.4f})")

    if 'reasons' in result and result['reasons']:
        print("\nRule-Based Reasons:")
        for reason in result['reasons']:
            print(f"  - {reason}")

    # Example application (legitimate case)
    print("\n" + "-"*60)
    print("Case 2: Legitimate Application")
    application_2 = {
        "age": 42,
        "doc_auth_score": 0.95,
        "face_match_score": 0.92,
        "liveness_score": 0.88,
        "ip_geo_match": 1,
        "vpn_or_tor": 0,
        "device_reuse_count": 1,
        "email_risk_score": 0.15,
        "phone_voip_flag": 0,
        "ssn_high_risk_flag": 0,
    }

    result = detector.predict_single(application_2, explain=True)

    print(f"\nDecision: {result['decision']}")
    print(f"Risk Score: {result['risk_score']:.4f}")
    print(f"  - Supervised Score: {result['supervised_score']:.4f}")
    print(f"  - Anomaly Score: {result['anomaly_score']:.4f}")

    if 'reasons' in result and result['reasons']:
        print("\nRule-Based Reasons:")
        for reason in result['reasons']:
            print(f"  - {reason}")
    else:
        print("\nNo risk factors detected - application appears legitimate")


def example_batch_prediction():
    """Example: Make batch predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Predictions")
    print("="*60)

    # Load model
    xgb_model, iso_model, feature_names, metadata = load_trained_model()

    # Create detector
    detector = IdentityTheftDetector(
        xgb_model=xgb_model,
        iso_model=iso_model,
        feature_names=feature_names,
        w1=HYBRID_WEIGHTS['supervised'],
        w2=HYBRID_WEIGHTS['anomaly'],
        theta_low=DECISION_THRESHOLDS['low'],
        theta_high=DECISION_THRESHOLDS['high']
    )

    # Generate sample data
    print("\nGenerating sample batch data...")
    np.random.seed(RANDOM_STATE)
    n_samples = 100

    batch_data = pd.DataFrame({
        "age": np.random.uniform(18, 75, n_samples),
        "doc_auth_score": np.random.uniform(0.3, 1.0, n_samples),
        "face_match_score": np.random.uniform(0.3, 1.0, n_samples),
        "liveness_score": np.random.uniform(0.3, 1.0, n_samples),
        "ip_geo_match": np.random.binomial(1, 0.7, n_samples),
        "vpn_or_tor": np.random.binomial(1, 0.2, n_samples),
        "device_reuse_count": np.random.poisson(2, n_samples),
        "email_risk_score": np.random.uniform(0.0, 0.8, n_samples),
        "phone_voip_flag": np.random.binomial(1, 0.2, n_samples),
        "ssn_high_risk_flag": np.random.binomial(1, 0.1, n_samples),
    })

    # Make predictions
    results = detector.predict_batch(batch_data, explain=True)

    print(f"\nProcessed {len(results)} applications")
    print("\nDecision Distribution:")
    print(results['decision'].value_counts())

    print("\nRisk Score Statistics:")
    print(results['risk_score'].describe())

    # Show top 5 riskiest applications
    print("\nTop 5 Riskiest Applications:")
    print("-"*60)
    top_risk = results.nlargest(5, 'risk_score')
    for idx, row in top_risk.iterrows():
        print(f"\nApplication {idx}:")
        print(f"  Decision: {row['decision']}")
        print(f"  Risk Score: {row['risk_score']:.4f}")
        print(f"  Doc Auth: {row['doc_auth_score']:.3f}, "
              f"Face Match: {row['face_match_score']:.3f}, "
              f"Liveness: {row['liveness_score']:.3f}")
        if row['reasons']:
            print(f"  Reasons: {row['reasons']}")


def example_threshold_adjustment():
    """Example: Adjust decision thresholds"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Threshold Adjustment")
    print("="*60)

    # Load model
    xgb_model, iso_model, feature_names, metadata = load_trained_model()

    # Create detector with original thresholds
    detector = IdentityTheftDetector(
        xgb_model=xgb_model,
        iso_model=iso_model,
        feature_names=feature_names,
        w1=HYBRID_WEIGHTS['supervised'],
        w2=HYBRID_WEIGHTS['anomaly'],
        theta_low=DECISION_THRESHOLDS['low'],
        theta_high=DECISION_THRESHOLDS['high']
    )

    # Test case
    test_app = {
        "age": 35,
        "doc_auth_score": 0.60,
        "face_match_score": 0.65,
        "liveness_score": 0.62,
        "ip_geo_match": 0,
        "vpn_or_tor": 1,
        "device_reuse_count": 4,
        "email_risk_score": 0.70,
        "phone_voip_flag": 1,
        "ssn_high_risk_flag": 0,
    }

    print("\nOriginal Thresholds (low=0.4, high=0.8):")
    result = detector.predict_single(test_app, explain=False)
    print(f"  Risk Score: {result['risk_score']:.4f}")
    print(f"  Decision: {result['decision']}")

    # Adjust to be more conservative (lower thresholds)
    print("\nMore Conservative Thresholds (low=0.3, high=0.6):")
    detector.set_thresholds(theta_low=0.3, theta_high=0.6)
    result = detector.predict_single(test_app, explain=False)
    print(f"  Risk Score: {result['risk_score']:.4f}")
    print(f"  Decision: {result['decision']}")

    # Adjust to be more lenient (higher thresholds)
    print("\nMore Lenient Thresholds (low=0.5, high=0.9):")
    detector.set_thresholds(theta_low=0.5, theta_high=0.9)
    result = detector.predict_single(test_app, explain=False)
    print(f"  Risk Score: {result['risk_score']:.4f}")
    print(f"  Decision: {result['decision']}")


def main():
    """Run all examples"""
    print("\n")
    print("*"*60)
    print("IDENTITY THEFT DETECTION - INFERENCE DEMO")
    print("*"*60)

    try:
        example_single_prediction()
        example_batch_prediction()
        example_threshold_adjustment()

        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("\nThe model is ready for production use!")
        print("You can integrate the IdentityTheftDetector class into your")
        print("application for real-time identity theft detection.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python src/train.py' first to train the model.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

