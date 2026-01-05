"""
Data validation and preprocessing utilities for identity theft detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import warnings


class DataValidator:
    """Validates input data for identity theft detection"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.required_features = set(feature_names)

    def validate_input(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data structure and values

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check if all required features are present
        missing_features = self.required_features - set(data.columns)
        if missing_features:
            errors.append(f"Missing features: {missing_features}")

        # Check for null values
        null_counts = data[self.feature_names].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        # Check value ranges
        for col in self.feature_names:
            if col not in data.columns:
                continue

            # Score features should be between 0 and 1
            if 'score' in col:
                if data[col].min() < 0 or data[col].max() > 1:
                    errors.append(f"{col} should be between 0 and 1")

            # Binary features should be 0 or 1
            if col in ['ip_geo_match', 'vpn_or_tor', 'phone_voip_flag', 'ssn_high_risk_flag']:
                if not data[col].isin([0, 1]).all():
                    errors.append(f"{col} should be binary (0 or 1)")

            # Age should be reasonable
            if col == 'age':
                if data[col].min() < 18 or data[col].max() > 120:
                    errors.append(f"Age values seem unrealistic")

        return len(errors) == 0, errors

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        df = data.copy()

        # Clip scores to valid range
        score_cols = [col for col in self.feature_names if 'score' in col]
        for col in score_cols:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)

        # Clip age to reasonable range
        if 'age' in df.columns:
            df['age'] = df['age'].clip(18, 120)

        # Ensure binary columns are binary
        binary_cols = ['ip_geo_match', 'vpn_or_tor', 'phone_voip_flag', 'ssn_high_risk_flag']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].round().clip(0, 1).astype(int)

        # Fill any remaining NaN values with median
        for col in self.feature_names:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        return df


class FeatureEngineer:
    """Create additional features for improved detection"""

    @staticmethod
    def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features from existing ones"""
        df = df.copy()

        # Biometric composite score (avg of doc, face, liveness)
        if all(col in df.columns for col in ['doc_auth_score', 'face_match_score', 'liveness_score']):
            df['biometric_composite'] = (
                df['doc_auth_score'] +
                df['face_match_score'] +
                df['liveness_score']
            ) / 3

        # Digital footprint risk (combines email, phone, vpn/tor)
        if all(col in df.columns for col in ['email_risk_score', 'phone_voip_flag', 'vpn_or_tor']):
            df['digital_footprint_risk'] = (
                df['email_risk_score'] +
                df['phone_voip_flag'] * 0.3 +
                df['vpn_or_tor'] * 0.5
            ) / 1.8  # Normalize

        # Identity consistency score (IP match, no VPN, low device reuse)
        if all(col in df.columns for col in ['ip_geo_match', 'vpn_or_tor', 'device_reuse_count']):
            df['identity_consistency'] = (
                df['ip_geo_match'] +
                (1 - df['vpn_or_tor']) +
                (1 - np.minimum(df['device_reuse_count'] / 10, 1))
            ) / 3

        # SSN and phone risk combined
        if all(col in df.columns for col in ['ssn_high_risk_flag', 'phone_voip_flag']):
            df['identity_doc_risk'] = (df['ssn_high_risk_flag'] + df['phone_voip_flag']) / 2

        return df

    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df = df.copy()

        # Low doc auth AND low face match (strong fraud signal)
        if all(col in df.columns for col in ['doc_auth_score', 'face_match_score']):
            df['doc_face_interaction'] = df['doc_auth_score'] * df['face_match_score']

        # VPN usage WITH IP mismatch (stronger fraud signal)
        if all(col in df.columns for col in ['vpn_or_tor', 'ip_geo_match']):
            df['vpn_ip_mismatch'] = df['vpn_or_tor'] * (1 - df['ip_geo_match'])

        # High device reuse WITH high email risk
        if all(col in df.columns for col in ['device_reuse_count', 'email_risk_score']):
            df['device_email_risk'] = (
                np.minimum(df['device_reuse_count'] / 5, 1) * df['email_risk_score']
            )

        return df


class DataAugmentation:
    """Augment training data to handle class imbalance"""

    @staticmethod
    def smote_like_augmentation(X: np.ndarray, y: np.ndarray,
                                target_ratio: float = 0.5,
                                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple SMOTE-like augmentation for minority class

        Args:
            X: Feature matrix
            y: Labels
            target_ratio: Target ratio of minority to majority class
            random_state: Random seed

        Returns:
            Augmented X, y
        """
        np.random.seed(random_state)

        # Find minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_count = np.max(counts)
        minority_count = np.min(counts)

        # Calculate how many synthetic samples needed
        target_minority_count = int(majority_count * target_ratio)
        n_synthetic = max(0, target_minority_count - minority_count)

        if n_synthetic == 0:
            return X, y

        # Get minority class samples
        minority_mask = y == minority_class
        X_minority = X[minority_mask]

        # Generate synthetic samples
        synthetic_X = []
        for _ in range(n_synthetic):
            # Pick two random minority samples
            idx1, idx2 = np.random.choice(len(X_minority), 2, replace=True)
            sample1, sample2 = X_minority[idx1], X_minority[idx2]

            # Create synthetic sample (random point on line between samples)
            alpha = np.random.random()
            synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
            synthetic_X.append(synthetic_sample)

        synthetic_X = np.array(synthetic_X)
        synthetic_y = np.full(n_synthetic, minority_class)

        # Combine with original data
        X_augmented = np.vstack([X, synthetic_X])
        y_augmented = np.concatenate([y, synthetic_y])

        # Shuffle
        shuffle_idx = np.random.permutation(len(X_augmented))
        return X_augmented[shuffle_idx], y_augmented[shuffle_idx]


def preprocess_data(df: pd.DataFrame,
                   feature_names: List[str],
                   fit_scaler: bool = True,
                   scaler: StandardScaler = None,
                   add_features: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Complete preprocessing pipeline

    Args:
        df: Input dataframe
        feature_names: List of base feature names
        fit_scaler: Whether to fit a new scaler
        scaler: Existing scaler to use (if fit_scaler=False)
        add_features: Whether to add engineered features

    Returns:
        Preprocessed dataframe, scaler object
    """
    # Validate data
    validator = DataValidator(feature_names)
    is_valid, errors = validator.validate_input(df)

    if not is_valid:
        warnings.warn(f"Data validation errors: {errors}")

    # Clean data
    df_clean = validator.clean_data(df)

    # Feature engineering
    if add_features:
        engineer = FeatureEngineer()
        df_clean = engineer.create_composite_features(df_clean)
        df_clean = engineer.create_interaction_features(df_clean)

    # Note: We typically don't scale for tree-based models (XGBoost, IsolationForest)
    # but include this for completeness if needed for other models

    return df_clean, scaler

