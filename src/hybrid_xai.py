import os
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
from html_report_generator import HTMLReportGenerator

# -----------------------------
# 0. Setup
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
EXPL_DIR = os.path.join(OUTPUT_DIR, "explanations")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXPL_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.default_rng(RANDOM_STATE)


# -----------------------------
# 1. Synthetic data generator
# -----------------------------

def generate_synthetic_onboarding_data(n_samples=10000, fraud_ratio=0.06):
    """
    Synthetic identity-onboarding dataset aligned with the whitepaper.

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

    # Legitimate population: mostly clean signals
    age_legit = np.random.normal(40, 10, n_legit).clip(18, 85)
    doc_auth_legit = np.random.uniform(0.8, 1.0, n_legit)
    face_match_legit = np.random.uniform(0.8, 1.0, n_legit)
    liveness_legit = np.random.uniform(0.8, 1.0, n_legit)
    ip_geo_match_legit = np.ones(n_legit)  # IP matches declared geography
    vpn_tor_legit = np.random.binomial(1, 0.02, n_legit)
    device_reuse_legit = np.random.poisson(1.5, n_legit)
    email_risk_legit = np.random.uniform(0.0, 0.3, n_legit)
    phone_voip_legit = np.random.binomial(1, 0.05, n_legit)
    ssn_high_risk_legit = np.random.binomial(1, 0.02, n_legit)

    # Fraud population: degraded signals
    age_fraud = np.random.normal(35, 12, n_fraud).clip(18, 85)
    doc_auth_fraud = np.random.uniform(0.1, 0.7, n_fraud)
    face_match_fraud = np.random.uniform(0.2, 0.8, n_fraud)
    liveness_fraud = np.random.uniform(0.2, 0.8, n_fraud)
    ip_geo_match_fraud = np.random.binomial(1, 0.4, n_fraud)  # more mismatches
    vpn_tor_fraud = np.random.binomial(1, 0.4, n_fraud)
    device_reuse_fraud = np.random.poisson(4.0, n_fraud)
    email_risk_fraud = np.random.uniform(0.4, 1.0, n_fraud)
    phone_voip_fraud = np.random.binomial(1, 0.5, n_fraud)
    ssn_high_risk_fraud = np.random.binomial(1, 0.5, n_fraud)

    X_legit = np.column_stack([
        age_legit,
        doc_auth_legit,
        face_match_legit,
        liveness_legit,
        ip_geo_match_legit,
        vpn_tor_legit,
        device_reuse_legit,
        email_risk_legit,
        phone_voip_legit,
        ssn_high_risk_legit,
    ])

    X_fraud = np.column_stack([
        age_fraud,
        doc_auth_fraud,
        face_match_fraud,
        liveness_fraud,
        ip_geo_match_fraud,
        vpn_tor_fraud,
        device_reuse_fraud,
        email_risk_fraud,
        phone_voip_fraud,
        ssn_high_risk_fraud,
    ])

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    feature_names = [
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

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y.astype(int)
    return df, feature_names


# -----------------------------
# 2. Train models & hybrid risk
# -----------------------------

def train_models(df, feature_names):
    X = df[feature_names].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Supervised model (XGBoost) – your "fraud probability" component
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)

    # Unsupervised anomaly detector – trained on legit only
    X_legit_train = X_train[y_train == 0]
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=RANDOM_STATE,
    )
    iso_model.fit(X_legit_train)

    # Predictions on test set
    p_supervised = xgb_model.predict_proba(X_test)[:, 1]

    # IsolationForest: higher score = more normal, so invert
    raw_scores = iso_model.score_samples(X_test)
    s_anomaly = (raw_scores.min() - raw_scores) / (raw_scores.min() - raw_scores.max() + 1e-8)

    # Hybrid risk score (matches your paper: R = f(p, s))
    w1, w2 = 0.7, 0.3
    R_hybrid = w1 * p_supervised + w2 * s_anomaly

    # Evaluation
    auc_supervised = roc_auc_score(y_test, p_supervised)
    auc_hybrid = roc_auc_score(y_test, R_hybrid)
    print(f"AUC Supervised only: {auc_supervised:.3f}")
    print(f"AUC Hybrid (Supervised + Anomaly): {auc_hybrid:.3f}")

    # ROC plot
    fpr_s, tpr_s, _ = roc_curve(y_test, p_supervised)
    fpr_h, tpr_h, _ = roc_curve(y_test, R_hybrid)

    plt.figure()
    plt.plot(fpr_s, tpr_s, label=f"Supervised (AUC={auc_supervised:.3f})")
    plt.plot(fpr_h, tpr_h, label=f"Hybrid (AUC={auc_hybrid:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Supervised vs Hybrid")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_supervised_vs_hybrid.png"))
    plt.close()

    # Build a DataFrame for downstream explanation / decisions
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["label"] = y_test
    test_df["p_supervised"] = p_supervised
    test_df["s_anomaly"] = s_anomaly
    test_df["R_hybrid"] = R_hybrid

    return xgb_model, iso_model, X_train, X_test, test_df, feature_names


# -----------------------------
# 3. Thresholds & decisions
# -----------------------------

def apply_decision_logic(test_df, theta_low=0.4, theta_high=0.8):
    """
    Map hybrid risk score to:
        - "Verified"
        - "Review"
        - "Suspicious"

    This mirrors your paper:
    - Verified: low risk
    - Review: borderline / medium
    - Suspicious: high risk
    """
    def decide(r):
        if r >= theta_high:
            return "Suspicious"
        elif r >= theta_low:
            return "Review"
        else:
            return "Verified"

    test_df = test_df.copy()
    test_df["decision"] = test_df["R_hybrid"].apply(decide)
    return test_df


# -----------------------------
# 4. SHAP explainability
# -----------------------------

def build_shap_explainer(xgb_model, X_background):
    """
    Build SHAP TreeExplainer for the XGBoost model.
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_background)
    return explainer, shap_values


def generate_shap_plots(explainer, shap_values, X_background, feature_names):
    """
    SHAP summary plot + dependence plot for doc_auth_score.
    """
    print("Generating SHAP plots...")
    # Summary plot (global feature importance)
    plt.figure()
    shap.summary_plot(shap_values, X_background, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), bbox_inches="tight")
    plt.close()

    if "doc_auth_score" in feature_names:
        print("Generating SHAP dependence plot for doc_auth_score...")
        plt.figure()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            shap.dependence_plot(
                "doc_auth_score",
                shap_values,
                X_background,
                feature_names=feature_names,
                show=False,
            )
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, "shap_dependence_doc_auth.png"),
            bbox_inches="tight",
        )
        plt.close()


# -----------------------------
# 5. Rule-based reason codes
# -----------------------------

def rule_based_reasons(row):
    """
    Simple rule-based reason codes aligned with your examples in the paper.
    Each rule returns a short string if triggered.
    """
    reasons = []

    if row["doc_auth_score"] < 0.6:
        reasons.append("Low document authenticity score")
    if row["face_match_score"] < 0.7:
        reasons.append("Biometric face mismatch / low similarity")
    if row["liveness_score"] < 0.7:
        reasons.append("Liveness check weak / possible spoof")
    if row["ip_geo_match"] < 0.5:
        reasons.append("IP geolocation does not match declared address")
    if row["vpn_or_tor"] == 1:
        reasons.append("VPN/TOR usage detected")
    if row["email_risk_score"] > 0.7:
        reasons.append("High-risk email domain")
    if row["phone_voip_flag"] == 1:
        reasons.append("VOIP phone number")
    if row["ssn_high_risk_flag"] == 1:
        reasons.append("High-risk SSN (e.g., deceased or mismatched)")

    if row["device_reuse_count"] > 3:
        reasons.append("Device reused across multiple applications")

    return reasons


# -----------------------------
# 6. Explanation report (SHAP + rules)
# -----------------------------

def build_explanation_report(
    test_df,
    feature_names,
    explainer,
    xgb_model,
    max_cases=200,
    theta_low=0.4,
):
    """
    Build a CSV explanation report for Review/Suspicious cases.

    For each case:
        - decision, label, R_hybrid
        - top SHAP features (by absolute value)
        - rule-based reason codes

    Returns:
        DataFrame with explanations
    """
    # Filter Review + Suspicious
    mask = test_df["decision"].isin(["Review", "Suspicious"])
    subset = test_df[mask].copy()

    # Limit size to avoid huge SHAP computations
    subset = subset.sort_values("R_hybrid", ascending=False).head(max_cases)

    X_subset = subset[feature_names].values
    shap_values_subset = explainer.shap_values(X_subset)

    explanations = []
    for i, (_, row) in enumerate(subset.iterrows()):
        shap_row = shap_values_subset[i]
        # top 3 absolute SHAP contributions
        top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
        top_feats = [
            f"{feature_names[j]} ({shap_row[j]:+.3f})"
            for j in top_idx
        ]
        rules = rule_based_reasons(row)

        explanations.append(
            {
                "decision": row["decision"],
                "true_label": int(row["label"]),
                "R_hybrid": float(row["R_hybrid"]),
                "p_supervised": float(row["p_supervised"]),
                "s_anomaly": float(row["s_anomaly"]),
                "top_shap_features": "; ".join(top_feats),
                "rule_based_reasons": "; ".join(rules) if rules else "",
            }
        )

    expl_df = pd.DataFrame(explanations)
    csv_path = os.path.join(EXPL_DIR, "explanation_report.csv")
    expl_df.to_csv(csv_path, index=False)
    print(f"✓ Explanation report CSV saved: {csv_path}")

    return expl_df


# -----------------------------
# 7. LIME example
# -----------------------------

def generate_lime_example(xgb_model, X_train, X_test, feature_names):
    """
    Generate a single LIME explanation for the most suspicious case.
    This is mainly to prove alignment with the LIME part of the paper.
    """
    print("Generating LIME explanation...")
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["legit", "fraud"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    p_test = xgb_model.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(p_test))  # highest fraud probability
    exp = explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=xgb_model.predict_proba,
        num_features=8,
    )

    txt_path = os.path.join(EXPL_DIR, "lime_example.txt")
    with open(txt_path, "w") as f:
        f.write("LIME explanation (feature, contribution):\n")
        for feat, val in exp.as_list():
            f.write(f"{feat}: {val}\n")
    print(f"LIME text explanation saved to: {txt_path}")

    html_path = os.path.join(EXPL_DIR, "lime_example.html")
    exp.save_to_file(html_path)
    print(f"LIME HTML explanation saved to: {html_path}")


# -----------------------------
# 8. Main pipeline
# -----------------------------

def main():
    print("="*80)
    print("🚀 Identity Theft Detection - Hybrid XAI System")
    print("="*80)

    print("\n[1/7] Generating synthetic identity theft data...")
    df, feature_names = generate_synthetic_onboarding_data(
        n_samples=10000,
        fraud_ratio=0.06,
    )
    print(f"✓ Generated {len(df)} records ({df['label'].sum()} fraud cases)")
    print(df.head())

    print("\n[2/7] Training models and computing hybrid risk scores...")
    xgb_model, iso_model, X_train, X_test, test_df, feature_names = train_models(
        df, feature_names
    )

    # Store AUC scores for report
    auc_supervised = roc_auc_score(test_df["label"], test_df["p_supervised"])
    auc_hybrid = roc_auc_score(test_df["label"], test_df["R_hybrid"])

    print("\n[3/7] Applying decision logic (Verified / Review / Suspicious)...")
    test_df = apply_decision_logic(test_df, theta_low=0.4, theta_high=0.8)
    print(f"✓ Decisions: {test_df['decision'].value_counts().to_dict()}")

    print("\n[4/7] Building SHAP explainer...")
    explainer, shap_values_train = build_shap_explainer(xgb_model, X_train)

    print("\n[5/7] Generating SHAP plots...")
    generate_shap_plots(explainer, shap_values_train, X_train, feature_names)
    print("✓ SHAP plots saved")

    print("\n[6/7] Building explanation report (SHAP + rules)...")
    explanations_df = build_explanation_report(
        test_df,
        feature_names,
        explainer,
        xgb_model,
        max_cases=200,
        theta_low=0.4,
    )

    print("\n[7/7] Generating comprehensive HTML report...")
    report_gen = HTMLReportGenerator(OUTPUT_DIR)
    report_path = report_gen.generate_full_report(
        test_df=test_df,
        explanations_df=explanations_df,
        auc_supervised=auc_supervised,
        auc_hybrid=auc_hybrid,
        plots_dir=PLOTS_DIR,
        model_name="Identity Theft Detection"
    )

    print("\n" + "="*80)
    print("✅ All tasks completed successfully!")
    print("="*80)
    print(f"\n📊 Model Performance:")
    print(f"   - Supervised AUC: {auc_supervised:.4f}")
    print(f"   - Hybrid AUC: {auc_hybrid:.4f}")
    print(f"   - Improvement: {(auc_hybrid - auc_supervised):.4f}")
    print(f"\n📁 Outputs:")
    print(f"   - HTML Report: {report_path}")
    print(f"   - Plots: {PLOTS_DIR}")
    print(f"   - Explanations: {EXPL_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
