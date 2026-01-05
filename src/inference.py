"""
Real-time inference API for identity theft detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import warnings


class IdentityTheftDetector:
    """Real-time identity theft detection inference"""

    def __init__(self,
                 xgb_model,
                 iso_model,
                 feature_names: List[str],
                 w1: float = 0.7,
                 w2: float = 0.3,
                 theta_low: float = 0.4,
                 theta_high: float = 0.8):
        """
        Initialize detector

        Args:
            xgb_model: Trained XGBoost model
            iso_model: Trained IsolationForest model
            feature_names: List of feature names
            w1: Weight for supervised component
            w2: Weight for anomaly component
            theta_low: Low threshold for decision
            theta_high: High threshold for decision
        """
        self.xgb_model = xgb_model
        self.iso_model = iso_model
        self.feature_names = feature_names
        self.w1 = w1
        self.w2 = w2
        self.theta_low = theta_low
        self.theta_high = theta_high

    def _calculate_hybrid_risk(self,
                               p_supervised: np.ndarray,
                               s_anomaly: np.ndarray) -> np.ndarray:
        """Calculate hybrid risk score"""
        return self.w1 * p_supervised + self.w2 * s_anomaly

    def _make_decision(self, risk_score: float) -> str:
        """Map risk score to decision"""
        if risk_score >= self.theta_high:
            return "Suspicious"
        elif risk_score >= self.theta_low:
            return "Review"
        else:
            return "Verified"

    def _get_rule_based_reasons(self, row: pd.Series) -> List[str]:
        """Get rule-based reason codes"""
        reasons = []

        if row.get("doc_auth_score", 1) < 0.6:
            reasons.append("Low document authenticity score")
        if row.get("face_match_score", 1) < 0.7:
            reasons.append("Biometric face mismatch / low similarity")
        if row.get("liveness_score", 1) < 0.7:
            reasons.append("Liveness check weak / possible spoof")
        if row.get("ip_geo_match", 1) < 0.5:
            reasons.append("IP geolocation does not match declared address")
        if row.get("vpn_or_tor", 0) == 1:
            reasons.append("VPN/TOR usage detected")
        if row.get("email_risk_score", 0) > 0.7:
            reasons.append("High-risk email domain")
        if row.get("phone_voip_flag", 0) == 1:
            reasons.append("VOIP phone number")
        if row.get("ssn_high_risk_flag", 0) == 1:
            reasons.append("High-risk SSN (e.g., deceased or mismatched)")
        if row.get("device_reuse_count", 0) > 3:
            reasons.append("Device reused across multiple applications")

        return reasons

    def predict_single(self,
                      features: Dict[str, Any],
                      explain: bool = True) -> Dict[str, Any]:
        """
        Make prediction for a single application

        Args:
            features: Dictionary of feature values
            explain: Whether to include explanations

        Returns:
            Dictionary with prediction results and explanations
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Ensure all features are present
        for fname in self.feature_names:
            if fname not in df.columns:
                warnings.warn(f"Missing feature {fname}, using default value 0")
                df[fname] = 0

        # Get feature array in correct order
        X = df[self.feature_names].values

        # Supervised prediction
        p_supervised = self.xgb_model.predict_proba(X)[0, 1]

        # Anomaly score
        raw_score = self.iso_model.score_samples(X)[0]
        # Normalize (higher = more anomalous)
        # Note: This is a simplified version; in production, store normalization params
        s_anomaly = max(0, min(1, (raw_score + 1) / 2))  # Rough normalization

        # Hybrid risk
        R_hybrid = self._calculate_hybrid_risk(
            np.array([p_supervised]),
            np.array([s_anomaly])
        )[0]

        # Decision
        decision = self._make_decision(R_hybrid)

        # Build result
        result = {
            "decision": decision,
            "risk_score": float(R_hybrid),
            "supervised_score": float(p_supervised),
            "anomaly_score": float(s_anomaly),
        }

        # Add explanations if requested
        if explain:
            # Get SHAP values
            try:
                import shap
                explainer = shap.TreeExplainer(self.xgb_model)
                shap_values = explainer.shap_values(X)[0]

                # Top contributing features
                top_indices = np.argsort(np.abs(shap_values))[::-1][:5]
                top_features = [
                    {
                        "feature": self.feature_names[i],
                        "value": float(X[0, i]),
                        "shap_contribution": float(shap_values[i])
                    }
                    for i in top_indices
                ]
                result["top_features"] = top_features
            except Exception as e:
                warnings.warn(f"Could not generate SHAP explanations: {e}")

            # Rule-based reasons
            result["reasons"] = self._get_rule_based_reasons(df.iloc[0])

        return result

    def predict_batch(self,
                     data: pd.DataFrame,
                     explain: bool = False) -> pd.DataFrame:
        """
        Make predictions for multiple applications

        Args:
            data: DataFrame with features
            explain: Whether to include explanations

        Returns:
            DataFrame with predictions and scores
        """
        # Ensure all features are present
        for fname in self.feature_names:
            if fname not in data.columns:
                warnings.warn(f"Missing feature {fname}, using default value 0")
                data[fname] = 0

        X = data[self.feature_names].values

        # Predictions
        p_supervised = self.xgb_model.predict_proba(X)[:, 1]

        # Anomaly scores
        raw_scores = self.iso_model.score_samples(X)
        s_anomaly = (raw_scores.min() - raw_scores) / (raw_scores.min() - raw_scores.max() + 1e-8)

        # Hybrid risk
        R_hybrid = self._calculate_hybrid_risk(p_supervised, s_anomaly)

        # Decisions
        decisions = [self._make_decision(r) for r in R_hybrid]

        # Build result DataFrame
        result_df = data.copy()
        result_df["decision"] = decisions
        result_df["risk_score"] = R_hybrid
        result_df["supervised_score"] = p_supervised
        result_df["anomaly_score"] = s_anomaly

        # Add explanations if requested
        if explain:
            reasons = [self._get_rule_based_reasons(row) for _, row in data.iterrows()]
            result_df["reasons"] = ["; ".join(r) if r else "" for r in reasons]

        return result_df

    def set_thresholds(self, theta_low: float = None, theta_high: float = None):
        """Update decision thresholds"""
        if theta_low is not None:
            self.theta_low = theta_low
        if theta_high is not None:
            self.theta_high = theta_high

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the detector configuration"""
        return {
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "supervised_weight": self.w1,
            "anomaly_weight": self.w2,
            "threshold_low": self.theta_low,
            "threshold_high": self.theta_high,
            "xgb_params": self.xgb_model.get_params(),
        }


class RiskScoreCalibrator:
    """Calibrate risk scores to match empirical fraud rates"""

    def __init__(self):
        self.calibration_params = {}

    def fit(self, risk_scores: np.ndarray, true_labels: np.ndarray,
            method: str = 'isotonic'):
        """
        Fit calibration model

        Args:
            risk_scores: Raw risk scores
            true_labels: True labels
            method: 'isotonic' or 'sigmoid'
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression

        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(risk_scores, true_labels)
        else:
            # For sigmoid calibration, would need the classifier object
            raise NotImplementedError("Sigmoid calibration not yet implemented")

        self.method = method

    def transform(self, risk_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to risk scores"""
        if not hasattr(self, 'calibrator'):
            warnings.warn("Calibrator not fitted, returning original scores")
            return risk_scores

        return self.calibrator.predict(risk_scores)

