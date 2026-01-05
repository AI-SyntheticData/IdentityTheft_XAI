"""
Configuration file for Identity Theft Detection Model
"""

# Model hyperparameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'logloss',
    'n_jobs': -1,
}

ISOLATION_FOREST_PARAMS = {
    'n_estimators': 200,
    'contamination': 0.06,
}

# Hybrid risk score weights
HYBRID_WEIGHTS = {
    'supervised': 0.7,
    'anomaly': 0.3,
}

# Decision thresholds
DECISION_THRESHOLDS = {
    'low': 0.4,      # Below this: Verified
    'high': 0.8,     # Above this: Suspicious
}

# Data generation parameters
DATA_PARAMS = {
    'n_samples': 10000,
    'fraud_ratio': 0.06,
    'test_size': 0.2,
}

# Feature names
FEATURE_NAMES = [
    "age",
    "doc_auth_score",
    "face_match_score",
    "liveness_score",
    "ip_geo_match",
    "vpn_or_tor",
    "device_reuse_count",
    "email_risk_score",
    "phone_voip_flag",
    "ssn_high_risk_flag",
]

# Rule-based thresholds
RULE_THRESHOLDS = {
    'doc_auth_score': 0.6,
    'face_match_score': 0.7,
    'liveness_score': 0.7,
    'ip_geo_match': 0.5,
    'email_risk_score': 0.7,
    'device_reuse_count': 3,
}

# Random state for reproducibility
RANDOM_STATE = 42

# Output directories
OUTPUT_CONFIG = {
    'plots_dir': 'outputs/plots',
    'explanations_dir': 'outputs/explanations',
    'models_dir': 'outputs/models',
}

