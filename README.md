**Identity Fraud Detection at Data Ingestion (Hybrid XGBoost + IsolationForest + SHAP/LIME)**

This repository contains a complete Python implementation of the hybrid identity theft detection architecture described in the whitepaper:

Identity Theft Detection at Data Ingestion Using AI: An Explainable Anomaly Detection Approach
----------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Overview

This project demonstrates:

* Hybrid supervised + unsupervised identity fraud detection
* Real-time onboarding scoring
* End-to-end pipeline including:
  * Synthetic identity onboarding dataset
  * XGBoost supervised classifier
  * IsolationForest anomaly detector
  * Hybrid risk score
  * Explainable AI (SHAP + LIME)
  * Analyst-style explanation report

----------------------------------------------------------------------------------------------------------------------------------------------------------------

📁Repository Structure

identity-fraud-xai/
├─ requirements.txt
├─ src/
│  └─ run_pipeline.py
└─ outputs/
   ├─ plots/
   └─ explanations/

----------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 How to Run
1. Install dependencies
`pip install -r requirements.txt
`
2. Run the pipeline
`python src/hybrid_xai.py`

----------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 What the Pipeline Produces

After running the script, you will find:

Outputs → plots/
* roc_supervised_vs_hybrid.png
* shap_summary.png
* shap_dependence_doc_auth.png

Outputs → explanations/
* lime_example.html
* lime_example.txt
* explanation_report.csv
  1. Includes decisions (“Verified”, “Review”, “Suspicious”)
  2. Hybrid risk scores
  3. Top SHAP drivers
  4. Rule-based reasons

----------------------------------------------------------------------------------------------------------------------------------------------------------------

**📦 Key Features**

**Hybrid Fraud Detection**
  * 𝑝 _supervised_ ​ from XGBoost
  * 𝑠 _anomaly_ ​ from IsolationForest
  * Combined into a hybrid risk score

**Explainable AI**
  * SHAP → Global and local feature explanations
  * LIME → Case-level interpretability
  * Rule-based layer → Compliance-aligned justifications

**Synthetic Onboarding Data**
  * Simulates realistic features used in financial KYC/IDV systems:
  * Document authenticity score
  * Face match score
  * Liveness score
  * IP–address mismatch
  * VPN/TOR usage
  * Email risk flags
  * Device reuse count
  * SSN high-risk indicator
  * VOIP phone indicator

----------------------------------------------------------------------------------------------------------------------------------------------------------------

📝 Citation
If you use this code, please cite:

**Murthy, S. (2025). Identity Theft Detection at Data Ingestion Using AI:
An Explainable Anomaly Detection Approach.**

----------------------------------------------------------------------------------------------------------------------------------------------------------------

📞 Contact
 Author: Sachin Murthy
 Email: sachin.damurthy@gmail.com
 Corresponding Author: Yes
