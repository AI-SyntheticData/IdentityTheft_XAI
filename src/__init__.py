"""
Identity Theft Detection using Explainable AI (XAI)
A hybrid approach combining XGBoost and IsolationForest with SHAP and LIME explanations
"""

__version__ = "1.0.0"
__author__ = "Sachin Murthy"

from .inference import IdentityTheftDetector
from .model_manager import ModelManager, ModelEvaluator
from .data_preprocessing import DataValidator, FeatureEngineer, preprocess_data

__all__ = [
    'IdentityTheftDetector',
    'ModelManager',
    'ModelEvaluator',
    'DataValidator',
    'FeatureEngineer',
    'preprocess_data',
]

