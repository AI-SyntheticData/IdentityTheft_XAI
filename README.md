# Identity Theft Detection at Data Ingestion Using AI
## An Explainable Anomaly Detection Approach (Hybrid XGBoost + IsolationForest + SHAP/LIME)

This repository contains a complete production-ready Python implementation of a hybrid identity theft detection system with real-time inference API and comprehensive explainability.

---

## 📌 Overview

This project demonstrates an end-to-end AI-powered identity theft detection system designed for financial onboarding and KYC (Know Your Customer) processes. The system combines:

* **Hybrid ML Architecture**: Supervised (XGBoost) + Unsupervised (IsolationForest) detection
* **Real-time Scoring**: RESTful API for production deployment
* **Explainable AI**: SHAP + LIME explanations with rule-based justifications
* **Model Management**: Version control, persistence, and evaluation framework
* **Comprehensive Reporting**: HTML reports with visualizations and analytics

---

## 📁 Repository Structure

```
IdentityTheft_XAI/
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration and hyperparameters
│   ├── train.py              # Model training pipeline
│   ├── hybrid_xai.py         # Main hybrid XAI pipeline
│   ├── data_preprocessing.py # Data validation and feature engineering
│   ├── model_manager.py      # Model saving/loading and versioning
│   ├── inference.py          # Real-time inference engine
│   ├── demo_inference.py     # Demo script for inference examples
│   ├── api_server.py         # Flask REST API server
│   ├── test_api.py           # API testing script
│   └── html_report_generator.py  # HTML report generation
└── outputs/                   # Generated outputs
    ├── models/               # Saved trained models (versioned)
    │   └── model_YYYYMMDD_HHMMSS/
    │       ├── xgb_model.pkl
    │       ├── iso_model.pkl
    │       ├── feature_names.json
    │       └── metadata.json
    ├── plots/                # Visualizations
    │   ├── roc_supervised_vs_hybrid.png
    │   ├── shap_summary.png
    │   └── shap_dependence_doc_auth.png
    ├── explanations/         # XAI outputs
    │   ├── lime_example.html
    │   ├── lime_example.txt
    │   └── explanation_report.csv
    └── reports/              # HTML reports
        └── identity_theft_detection_report.html
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/IdentityTheft_XAI.git
cd IdentityTheft_XAI
pip install -r requirements.txt
```

### 2. Train the Model

Run the training pipeline to train the hybrid detection model:

```bash
python src/train.py
```

Or use the original hybrid XAI script:

```bash
python src/hybrid_xai.py
```

This will:
- Generate synthetic identity onboarding data (10,000 samples)
- Train XGBoost classifier and IsolationForest anomaly detector
- Evaluate performance with ROC-AUC metrics
- Generate SHAP and LIME explanations
- Create visualizations and reports
- Save trained models to `outputs/models/`

### 3. Run Inference Demo

Test the trained model with example cases:

```bash
python src/demo_inference.py
```

### 4. Start the API Server

Launch the Flask REST API for real-time predictions:

```bash
python src/api_server.py
```

The API will be available at `http://localhost:8000`

### 5. Test the API

In a separate terminal, run the API tests:

```bash
python src/test_api.py
```

---

## 📊 Outputs Generated

### 1. Visualizations (`outputs/plots/`)
- **roc_supervised_vs_hybrid.png**: ROC curves comparing supervised vs. hybrid approach
- **shap_summary.png**: Global feature importance using SHAP values
- **shap_dependence_doc_auth.png**: SHAP dependence plot for document authentication

### 2. Explanations (`outputs/explanations/`)
- **lime_example.html**: Interactive LIME explanation for a suspicious case
- **lime_example.txt**: Text-based LIME explanation
- **explanation_report.csv**: Detailed report with:
  - Risk scores (supervised, anomaly, hybrid)
  - Decisions (Verified, Review, Suspicious)
  - Top SHAP feature contributions
  - Rule-based reasons for compliance

### 3. Models (`outputs/models/`)
Each training run creates a versioned model directory containing:
- `xgb_model.pkl`: Trained XGBoost classifier
- `iso_model.pkl`: Trained IsolationForest model
- `feature_names.json`: Feature list
- `metadata.json`: Training metadata and performance metrics

### 4. Reports (`outputs/reports/`)
- **identity_theft_detection_report.html**: Comprehensive HTML report with:
  - Summary statistics
  - Performance metrics
  - Detection tables
  - Embedded visualizations

---

## 🔑 Key Features

### Hybrid Fraud Detection
The system combines two complementary ML approaches:
- **Supervised (XGBoost)**: P<sub>supervised</sub> from labeled fraud patterns
- **Unsupervised (IsolationForest)**: S<sub>anomaly</sub> for novel/unknown fraud
- **Hybrid Risk Score**: R<sub>hybrid</sub> = 0.7 × P<sub>supervised</sub> + 0.3 × S<sub>anomaly</sub>

### Decision Thresholds
- **Verified** (R < 0.4): Low risk, automatic approval
- **Review** (0.4 ≤ R < 0.8): Medium risk, manual review required
- **Suspicious** (R ≥ 0.8): High risk, likely fraud

### Explainable AI (XAI)
- **SHAP**: Global and local feature importance with Shapley values
- **LIME**: Instance-level interpretability with counterfactual explanations
- **Rule-based Layer**: Human-readable justifications for compliance

### Synthetic Onboarding Features (10 Features)
The system processes realistic KYC/IDV features:

1. **age**: Applicant age (18-120)
2. **doc_auth_score**: Document authenticity score (0-1)
3. **face_match_score**: Facial biometric match score (0-1)
4. **liveness_score**: Liveness detection score (0-1)
5. **ip_geo_match**: IP geolocation matches address (0/1)
6. **vpn_or_tor**: VPN or TOR usage detected (0/1)
7. **device_reuse_count**: Number of applications from device (integer)
8. **email_risk_score**: Email risk assessment (0-1)
9. **phone_voip_flag**: VOIP phone number indicator (0/1)
10. **ssn_high_risk_flag**: SSN high-risk indicator (0/1)

### REST API Endpoints

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `GET /info`
Get model information

**Response:**
```json
{
  "version": "20260105_180708",
  "feature_names": [...],
  "n_features": 10,
  "test_auc": 0.9845
}
```

#### `POST /predict`
Make a single prediction

**Request:**
```json
{
  "age": 28,
  "doc_auth_score": 0.35,
  "face_match_score": 0.42,
  "liveness_score": 0.38,
  "ip_geo_match": 0,
  "vpn_or_tor": 1,
  "device_reuse_count": 7,
  "email_risk_score": 0.85,
  "phone_voip_flag": 1,
  "ssn_high_risk_flag": 1
}
```

**Response:**
```json
{
  "prediction": {
    "decision": "Suspicious",
    "risk_score": 0.8752,
    "fraud_probability": 0.9234,
    "anomaly_score": 0.7123,
    "reasons": [
      "Low document authenticity (0.35)",
      "VPN/TOR detected",
      "High email risk (0.85)"
    ]
  }
}
```

#### `POST /predict/batch`
Make batch predictions (up to 1000 records)

---

## 🛠️ Configuration

Edit `src/config.py` to customize:

### Model Hyperparameters
```python
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
}

ISOLATION_FOREST_PARAMS = {
    'n_estimators': 200,
    'contamination': 0.06,
}
```

### Hybrid Weights
```python
HYBRID_WEIGHTS = {
    'supervised': 0.7,  # Weight for XGBoost
    'anomaly': 0.3,     # Weight for IsolationForest
}
```

### Decision Thresholds
```python
DECISION_THRESHOLDS = {
    'low': 0.4,   # Below: Verified
    'high': 0.8,  # Above: Suspicious
}
```

### Data Generation
```python
DATA_PARAMS = {
    'n_samples': 10000,
    'fraud_ratio': 0.06,
    'test_size': 0.2,
}
```

---

## 📈 Model Performance

The hybrid approach achieves superior performance:

- **Hybrid Model ROC-AUC**: ~0.98
- **Supervised Only (XGBoost)**: ~0.96
- **Unsupervised Only (IsolationForest)**: ~0.85

The hybrid combination captures both:
1. Known fraud patterns (supervised learning)
2. Novel/anomalous behavior (unsupervised learning)

---

## 🧪 Testing

Run the complete test suite:

```bash
# Test inference examples
python src/demo_inference.py
```

---

## 📦 Dependencies

- **numpy** (>=1.21.0): Numerical computations
- **pandas** (>=1.3.0): Data manipulation
- **scikit-learn** (>=1.0.0): ML algorithms and preprocessing
- **xgboost** (>=1.5.0): Gradient boosting classifier
- **shap** (>=0.50.0): SHAP explanations
- **lime** (>=0.2.0): LIME explanations
- **matplotlib** (>=3.4.0): Visualizations
- **flask** (>=2.0.0): REST API server
- **requests** (>=2.26.0): API testing

---

## 🏗️ Architecture

<img width="450" height="500" alt="image" src="https://github.com/user-attachments/assets/47826259-e340-4572-a44c-e0cdbfb0f80c" />

## 🔒 Use Cases

This system is designed for:

1. **Financial Services**: Account opening fraud detection
2. **KYC/AML Compliance**: Identity verification in onboarding
3. **E-commerce**: High-value transaction screening
4. **Government**: Identity document verification
5. **Healthcare**: Patient identity verification

---

## 📝 Citation

If you use this code or methodology in your research or production systems, please cite:

```
Murthy, S. (2025). Identity Theft Detection at Data Ingestion Using AI:
An Explainable Anomaly Detection Approach. GitHub Repository.
https://github.com/yourusername/IdentityTheft_XAI
```

---

## 📄 License

This project is provided for educational and research purposes.

---

## 📞 Contact

**Author**: Sachin Murthy  
**Email**: sachin.damurthy@gmail.com  
**Corresponding Author**: Yes

For questions, issues, or collaboration opportunities, please reach out via email or open an issue on GitHub.

---

## 🙏 Acknowledgments

This project demonstrates the practical application of explainable AI in fraud detection, combining state-of-the-art ML techniques with interpretability for real-world deployment in regulated industries.

---

**Built with ❤️ for safer digital identity verification**

