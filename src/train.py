"""
Train the identity theft detection model with enhanced features
This is an improved version of the training pipeline
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Import custom modules
from config import (
    XGB_PARAMS, ISOLATION_FOREST_PARAMS, HYBRID_WEIGHTS,
    DECISION_THRESHOLDS, DATA_PARAMS, FEATURE_NAMES, RANDOM_STATE
)
from model_manager import ModelManager, ModelEvaluator
from data_preprocessing import DataValidator, preprocess_data


# -----------------------------
# Setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
EXPL_DIR = os.path.join(OUTPUT_DIR, "explanations")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXPL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_synthetic_onboarding_data(n_samples=10000, fraud_ratio=0.06):
    """
    Generate synthetic identity-onboarding dataset

    Features:
        - age
        - doc_auth_score               (0–1)
        - face_match_score             (0–1)
        - liveness_score               (0–1)
        - ip_geo_match                 (0/1)
        - vpn_or_tor                   (0/1)
        - device_reuse_count           (integer)
        - email_risk_score             (0–1)
        - phone_voip_flag              (0/1)
        - ssn_high_risk_flag           (0/1)

    Label:
        - label = 1 → identity fraud
        - label = 0 → legitimate
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate population
    age_legit = np.random.normal(40, 10, n_legit).clip(18, 85)
    doc_auth_legit = np.random.uniform(0.8, 1.0, n_legit)
    face_match_legit = np.random.uniform(0.8, 1.0, n_legit)
    liveness_legit = np.random.uniform(0.8, 1.0, n_legit)
    ip_geo_match_legit = np.ones(n_legit)
    vpn_tor_legit = np.random.binomial(1, 0.02, n_legit)
    device_reuse_legit = np.random.poisson(1.5, n_legit)
    email_risk_legit = np.random.uniform(0.0, 0.3, n_legit)
    phone_voip_legit = np.random.binomial(1, 0.05, n_legit)
    ssn_high_risk_legit = np.random.binomial(1, 0.02, n_legit)

    # Fraud population
    age_fraud = np.random.normal(35, 12, n_fraud).clip(18, 85)
    doc_auth_fraud = np.random.uniform(0.1, 0.7, n_fraud)
    face_match_fraud = np.random.uniform(0.2, 0.8, n_fraud)
    liveness_fraud = np.random.uniform(0.2, 0.8, n_fraud)
    ip_geo_match_fraud = np.random.binomial(1, 0.4, n_fraud)
    vpn_tor_fraud = np.random.binomial(1, 0.4, n_fraud)
    device_reuse_fraud = np.random.poisson(4.0, n_fraud)
    email_risk_fraud = np.random.uniform(0.4, 1.0, n_fraud)
    phone_voip_fraud = np.random.binomial(1, 0.5, n_fraud)
    ssn_high_risk_fraud = np.random.binomial(1, 0.5, n_fraud)

    X_legit = np.column_stack([
        age_legit, doc_auth_legit, face_match_legit, liveness_legit,
        ip_geo_match_legit, vpn_tor_legit, device_reuse_legit,
        email_risk_legit, phone_voip_legit, ssn_high_risk_legit,
    ])

    X_fraud = np.column_stack([
        age_fraud, doc_auth_fraud, face_match_fraud, liveness_fraud,
        ip_geo_match_fraud, vpn_tor_fraud, device_reuse_fraud,
        email_risk_fraud, phone_voip_fraud, ssn_high_risk_fraud,
    ])

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y.astype(int)

    return df


def train_models(df):
    """Train hybrid model (XGBoost + IsolationForest)"""

    print("Validating and preprocessing data...")
    df_clean, _ = preprocess_data(df, FEATURE_NAMES, add_features=False)

    X = df_clean[FEATURE_NAMES].values
    y = df_clean["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DATA_PARAMS['test_size'],
        stratify=y, random_state=RANDOM_STATE
    )

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Training fraud rate: {y_train.mean():.3f}, Test fraud rate: {y_test.mean():.3f}")

    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS, random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)

    # Train IsolationForest on legitimate samples only
    print("Training IsolationForest on legitimate samples...")
    X_legit_train = X_train[y_train == 0]
    iso_model = IsolationForest(**ISOLATION_FOREST_PARAMS, random_state=RANDOM_STATE)
    iso_model.fit(X_legit_train)

    # Predictions
    p_supervised = xgb_model.predict_proba(X_test)[:, 1]
    raw_scores = iso_model.score_samples(X_test)
    s_anomaly = (raw_scores.min() - raw_scores) / (raw_scores.min() - raw_scores.max() + 1e-8)

    # Hybrid risk score
    w1, w2 = HYBRID_WEIGHTS['supervised'], HYBRID_WEIGHTS['anomaly']
    R_hybrid = w1 * p_supervised + w2 * s_anomaly

    # Evaluation
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)

    evaluator = ModelEvaluator()

    # Supervised metrics
    supervised_metrics = evaluator.calculate_metrics(y_test, p_supervised)
    print(f"\nSupervised Model (XGBoost):")
    print(f"  ROC-AUC: {supervised_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {supervised_metrics['pr_auc']:.4f}")
    print(f"  Precision: {supervised_metrics['precision']:.4f}")
    print(f"  Recall: {supervised_metrics['recall']:.4f}")
    print(f"  F1-Score: {supervised_metrics['f1_score']:.4f}")

    # Hybrid metrics
    hybrid_metrics = evaluator.calculate_metrics(y_test, R_hybrid)
    print(f"\nHybrid Model (XGBoost + IsolationForest):")
    print(f"  ROC-AUC: {hybrid_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {hybrid_metrics['pr_auc']:.4f}")
    print(f"  Precision: {hybrid_metrics['precision']:.4f}")
    print(f"  Recall: {hybrid_metrics['recall']:.4f}")
    print(f"  F1-Score: {hybrid_metrics['f1_score']:.4f}")

    improvement = hybrid_metrics['roc_auc'] - supervised_metrics['roc_auc']
    print(f"\n  Improvement: {improvement:+.4f} AUC points")
    print("="*60)

    # ROC curves
    print("\nGenerating ROC curve plot...")
    fpr_s, tpr_s, _ = roc_curve(y_test, p_supervised)
    fpr_h, tpr_h, _ = roc_curve(y_test, R_hybrid)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_s, tpr_s, label=f"Supervised (AUC={supervised_metrics['roc_auc']:.3f})", linewidth=2)
    plt.plot(fpr_h, tpr_h, label=f"Hybrid (AUC={hybrid_metrics['roc_auc']:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve: Supervised vs Hybrid Identity Theft Detection", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_supervised_vs_hybrid.png"), dpi=300)
    plt.close()

    # Build test DataFrame
    test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    test_df["label"] = y_test
    test_df["p_supervised"] = p_supervised
    test_df["s_anomaly"] = s_anomaly
    test_df["R_hybrid"] = R_hybrid

    # Save model
    print("\nSaving trained models...")
    model_manager = ModelManager(MODELS_DIR)
    metadata = {
        'supervised_metrics': supervised_metrics,
        'hybrid_metrics': hybrid_metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'fraud_ratio_train': float(y_train.mean()),
        'fraud_ratio_test': float(y_test.mean()),
        'hyperparameters': {
            'xgboost': XGB_PARAMS,
            'isolation_forest': ISOLATION_FOREST_PARAMS,
            'hybrid_weights': HYBRID_WEIGHTS,
        }
    }
    model_manager.save_model(xgb_model, iso_model, FEATURE_NAMES, metadata)

    return xgb_model, iso_model, X_train, X_test, test_df, supervised_metrics, hybrid_metrics


def apply_decision_logic(test_df):
    """Apply decision thresholds"""
    def decide(r):
        if r >= DECISION_THRESHOLDS['high']:
            return "Suspicious"
        elif r >= DECISION_THRESHOLDS['low']:
            return "Review"
        else:
            return "Verified"

    test_df = test_df.copy()
    test_df["decision"] = test_df["R_hybrid"].apply(decide)

    # Calculate decision metrics
    evaluator = ModelEvaluator()
    decision_metrics = evaluator.calculate_decision_metrics(test_df)

    print("\n" + "="*60)
    print("DECISION DISTRIBUTION")
    print("="*60)
    for decision, count in decision_metrics['decision_distribution'].items():
        pct = count / len(test_df) * 100
        print(f"  {decision}: {count} ({pct:.1f}%)")

    if 'suspicious_false_positive_rate' in decision_metrics:
        print(f"\n  False Positive Rate (Suspicious): {decision_metrics['suspicious_false_positive_rate']:.3f}")
    if 'verified_false_negative_rate' in decision_metrics:
        print(f"  False Negative Rate (Verified): {decision_metrics['verified_false_negative_rate']:.3f}")
    print("="*60)

    return test_df


def generate_shap_explanations(xgb_model, X_train, X_test):
    """Generate SHAP explanations"""
    print("\nBuilding SHAP explainer...")

    explainer = shap.TreeExplainer(xgb_model)

    print("Computing SHAP values for training set (for visualization)...")
    shap_values_train = explainer.shap_values(X_train)

    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_train, X_train, feature_names=FEATURE_NAMES, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), bbox_inches="tight", dpi=300)
    plt.close()

    if "doc_auth_score" in FEATURE_NAMES:
        print("Generating SHAP dependence plot for doc_auth_score...")
        plt.figure(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            shap.dependence_plot(
                "doc_auth_score", shap_values_train, X_train,
                feature_names=FEATURE_NAMES, show=False
            )
        plt.title("SHAP Dependence: Document Authentication Score",
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_dependence_doc_auth.png"),
                   bbox_inches="tight", dpi=300)
        plt.close()

    return explainer


def build_explanation_report(test_df, explainer, max_cases=200):
    """Build detailed explanation report"""
    print("\nBuilding explanation report...")

    mask = test_df["decision"].isin(["Review", "Suspicious"])
    subset = test_df[mask].copy()
    subset = subset.sort_values("R_hybrid", ascending=False).head(max_cases)

    X_subset = subset[FEATURE_NAMES].values
    shap_values_subset = explainer.shap_values(X_subset)

    explanations = []
    for i, (_, row) in enumerate(subset.iterrows()):
        shap_row = shap_values_subset[i]
        top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
        top_feats = [f"{FEATURE_NAMES[j]} ({shap_row[j]:+.3f})" for j in top_idx]

        # Rule-based reasons
        reasons = []
        if row["doc_auth_score"] < 0.6:
            reasons.append("Low document authenticity score")
        if row["face_match_score"] < 0.7:
            reasons.append("Biometric face mismatch / low similarity")
        if row["liveness_score"] < 0.7:
            reasons.append("Liveness check weak / possible spoof")
        if row["ip_geo_match"] < 0.5:
            reasons.append("IP geolocation mismatch")
        if row["vpn_or_tor"] == 1:
            reasons.append("VPN/TOR usage detected")
        if row["email_risk_score"] > 0.7:
            reasons.append("High-risk email domain")
        if row["phone_voip_flag"] == 1:
            reasons.append("VOIP phone number")
        if row["ssn_high_risk_flag"] == 1:
            reasons.append("High-risk SSN")
        if row["device_reuse_count"] > 3:
            reasons.append("Device reused across multiple applications")

        explanations.append({
            "decision": row["decision"],
            "true_label": int(row["label"]),
            "R_hybrid": float(row["R_hybrid"]),
            "p_supervised": float(row["p_supervised"]),
            "s_anomaly": float(row["s_anomaly"]),
            "top_shap_features": "; ".join(top_feats),
            "rule_based_reasons": "; ".join(reasons) if reasons else "",
        })

    expl_df = pd.DataFrame(explanations)
    csv_path = os.path.join(EXPL_DIR, "explanation_report.csv")
    expl_df.to_csv(csv_path, index=False)
    print(f"Explanation report saved to: {csv_path}")


def generate_lime_example(xgb_model, X_train, X_test):
    """Generate LIME explanation for highest risk case"""
    print("\nGenerating LIME explanation...")

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=FEATURE_NAMES,
        class_names=["legit", "fraud"],
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    p_test = xgb_model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(p_test))

    exp = explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=xgb_model.predict_proba,
        num_features=8
    )

    txt_path = os.path.join(EXPL_DIR, "lime_example.txt")
    with open(txt_path, "w") as f:
        f.write("LIME Explanation for Highest Risk Case\n")
        f.write("="*50 + "\n\n")
        f.write(f"Predicted fraud probability: {p_test[idx]:.4f}\n\n")
        f.write("Feature contributions:\n")
        for feat, val in exp.as_list():
            f.write(f"  {feat}: {val:+.4f}\n")
    print(f"LIME text saved to: {txt_path}")

    html_path = os.path.join(EXPL_DIR, "lime_example.html")
    exp.save_to_file(html_path)
    print(f"LIME HTML saved to: {html_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("IDENTITY THEFT DETECTION - TRAINING PIPELINE")
    print("="*60)

    print("\n1. Generating synthetic onboarding data...")
    df = generate_synthetic_onboarding_data(
        n_samples=DATA_PARAMS['n_samples'],
        fraud_ratio=DATA_PARAMS['fraud_ratio']
    )
    print(f"Generated {len(df)} samples with {df['label'].sum()} fraud cases")

    print("\n2. Training models...")
    xgb_model, iso_model, X_train, X_test, test_df, sup_metrics, hyb_metrics = train_models(df)

    print("\n3. Applying decision logic...")
    test_df = apply_decision_logic(test_df)

    print("\n4. Generating SHAP explanations...")
    explainer = generate_shap_explanations(xgb_model, X_train, X_test)

    print("\n5. Building explanation report...")
    build_explanation_report(test_df, explainer)

    print("\n6. Generating LIME example...")
    generate_lime_example(xgb_model, X_train, X_test)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Explanations: {EXPL_DIR}")
    print(f"  - Models: {MODELS_DIR}")
    print("\nYou can now use the trained model for inference!")
    print("="*60)


if __name__ == "__main__":
    main()

