"""
Model management: saving, loading, and versioning trained models
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class ModelManager:
    """Manage model persistence and versioning"""

    def __init__(self, models_dir: str = "outputs/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(self,
                   xgb_model,
                   iso_model,
                   feature_names: list,
                   metadata: Dict[str, Any] = None,
                   version: str = None) -> str:
        """
        Save trained models with metadata

        Args:
            xgb_model: Trained XGBoost model
            iso_model: Trained IsolationForest model
            feature_names: List of feature names
            metadata: Additional metadata (metrics, hyperparameters, etc.)
            version: Model version string (default: timestamp)

        Returns:
            Path to saved model directory
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_dir = os.path.join(self.models_dir, f"model_{version}")
        os.makedirs(model_dir, exist_ok=True)

        # Save models
        with open(os.path.join(model_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)

        with open(os.path.join(model_dir, "iso_model.pkl"), "wb") as f:
            pickle.dump(iso_model, f)

        # Save feature names
        with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)

        # Save metadata
        full_metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "feature_names": feature_names,
            "n_features": len(feature_names),
        }
        if metadata:
            full_metadata.update(metadata)

        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(full_metadata, f, indent=2)

        print(f"Model saved to: {model_dir}")
        return model_dir

    def load_model(self, version: str = None) -> Tuple[Any, Any, list, Dict]:
        """
        Load trained models

        Args:
            version: Model version to load (default: latest)

        Returns:
            xgb_model, iso_model, feature_names, metadata
        """
        if version is None:
            # Load latest version
            versions = [d for d in os.listdir(self.models_dir)
                       if os.path.isdir(os.path.join(self.models_dir, d))
                       and d.startswith("model_")]
            if not versions:
                raise FileNotFoundError("No saved models found")
            version = sorted(versions)[-1].replace("model_", "")

        model_dir = os.path.join(self.models_dir, f"model_{version}")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model version {version} not found")

        # Load models
        with open(os.path.join(model_dir, "xgb_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)

        with open(os.path.join(model_dir, "iso_model.pkl"), "rb") as f:
            iso_model = pickle.load(f)

        # Load feature names
        with open(os.path.join(model_dir, "feature_names.json"), "r") as f:
            feature_names = json.load(f)

        # Load metadata
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        print(f"Model loaded from: {model_dir}")
        return xgb_model, iso_model, feature_names, metadata

    def list_versions(self) -> list:
        """List all saved model versions"""
        versions = [d.replace("model_", "")
                   for d in os.listdir(self.models_dir)
                   if os.path.isdir(os.path.join(self.models_dir, d))
                   and d.startswith("model_")]
        return sorted(versions)

    def get_model_info(self, version: str = None) -> Dict:
        """Get metadata for a specific model version"""
        if version is None:
            versions = self.list_versions()
            if not versions:
                return {}
            version = versions[-1]

        model_dir = os.path.join(self.models_dir, f"model_{version}")
        metadata_path = os.path.join(model_dir, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}


class ModelEvaluator:
    """Evaluate model performance"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray,
                         y_pred_proba: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )

        y_pred = (y_pred_proba >= threshold).astype(int)

        # Basic metrics
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        })

        return metrics

    @staticmethod
    def calculate_decision_metrics(test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics specific to decision categories

        Args:
            test_df: DataFrame with columns: decision, label, R_hybrid

        Returns:
            Dictionary with decision distribution and accuracy
        """
        metrics = {}

        # Decision distribution
        decision_counts = test_df['decision'].value_counts().to_dict()
        metrics['decision_distribution'] = decision_counts

        # Accuracy by decision category
        for decision in ['Verified', 'Review', 'Suspicious']:
            mask = test_df['decision'] == decision
            if mask.sum() > 0:
                accuracy = (test_df[mask]['label'] ==
                           (1 if decision == 'Suspicious' else 0)).mean()
                metrics[f'{decision.lower()}_accuracy'] = accuracy

        # False positives in "Suspicious" category
        suspicious_mask = test_df['decision'] == 'Suspicious'
        if suspicious_mask.sum() > 0:
            fp_rate = (test_df[suspicious_mask]['label'] == 0).mean()
            metrics['suspicious_false_positive_rate'] = fp_rate

        # False negatives in "Verified" category
        verified_mask = test_df['decision'] == 'Verified'
        if verified_mask.sum() > 0:
            fn_rate = (test_df[verified_mask]['label'] == 1).mean()
            metrics['verified_false_negative_rate'] = fn_rate

        return metrics

    @staticmethod
    def compare_models(metrics1: Dict, metrics2: Dict,
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict:
        """Compare two model performance metrics"""
        comparison = {}

        for key in metrics1:
            if key in metrics2 and isinstance(metrics1[key], (int, float)):
                diff = metrics2[key] - metrics1[key]
                pct_change = (diff / metrics1[key] * 100) if metrics1[key] != 0 else 0
                comparison[key] = {
                    model1_name: metrics1[key],
                    model2_name: metrics2[key],
                    'difference': diff,
                    'pct_change': pct_change,
                }

        return comparison

