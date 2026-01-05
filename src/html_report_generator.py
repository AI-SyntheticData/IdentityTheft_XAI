import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for Identity Theft Detection"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.report_dir = os.path.join(output_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)

    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str

    def image_file_to_base64(self, filepath):
        """Convert image file to base64 string"""
        with open(filepath, 'rb') as f:
            img_str = base64.b64encode(f.read()).decode('utf-8')
        return img_str

    def generate_summary_stats(self, test_df):
        """Generate summary statistics section"""
        total = len(test_df)
        verified = len(test_df[test_df['decision'] == 'Verified'])
        review = len(test_df[test_df['decision'] == 'Review'])
        suspicious = len(test_df[test_df['decision'] == 'Suspicious'])

        actual_fraud = test_df['label'].sum()
        actual_legit = total - actual_fraud

        html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Cases</h3>
                <p class="stat-number">{total}</p>
            </div>
            <div class="stat-card verified">
                <h3>Verified</h3>
                <p class="stat-number">{verified}</p>
                <p class="stat-percent">{verified/total*100:.1f}%</p>
            </div>
            <div class="stat-card review">
                <h3>Review</h3>
                <p class="stat-number">{review}</p>
                <p class="stat-percent">{review/total*100:.1f}%</p>
            </div>
            <div class="stat-card suspicious">
                <h3>Suspicious</h3>
                <p class="stat-number">{suspicious}</p>
                <p class="stat-percent">{suspicious/total*100:.1f}%</p>
            </div>
        </div>
        
        <div class="stats-grid" style="margin-top: 20px;">
            <div class="stat-card fraud">
                <h3>Actual Fraud Cases</h3>
                <p class="stat-number">{int(actual_fraud)}</p>
                <p class="stat-percent">{actual_fraud/total*100:.1f}%</p>
            </div>
            <div class="stat-card legit">
                <h3>Actual Legitimate Cases</h3>
                <p class="stat-number">{int(actual_legit)}</p>
                <p class="stat-percent">{actual_legit/total*100:.1f}%</p>
            </div>
        </div>
        """
        return html

    def generate_detection_table(self, explanations_df, max_rows=50):
        """Generate table of detected suspicious cases"""
        if len(explanations_df) == 0:
            return "<p>No suspicious or review cases detected.</p>"

        # Sort by R_hybrid descending
        df = explanations_df.sort_values('R_hybrid', ascending=False).head(max_rows)

        html = """
        <table class="detection-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Decision</th>
                    <th>Risk Score</th>
                    <th>Fraud Prob</th>
                    <th>Anomaly Score</th>
                    <th>Top SHAP Features</th>
                    <th>Rule-Based Reasons</th>
                </tr>
            </thead>
            <tbody>
        """

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            decision_class = row['decision'].lower()
            html += f"""
                <tr>
                    <td>{idx}</td>
                    <td><span class="badge {decision_class}">{row['decision']}</span></td>
                    <td>{row['R_hybrid']:.3f}</td>
                    <td>{row['p_supervised']:.3f}</td>
                    <td>{row['s_anomaly']:.3f}</td>
                    <td class="feature-cell">{row['top_shap_features']}</td>
                    <td class="reason-cell">{row['rule_based_reasons']}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """
        return html

    def generate_distribution_charts(self, test_df):
        """Generate distribution charts for risk scores"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Risk score distribution by decision
        ax = axes[0, 0]
        for decision in ['Verified', 'Review', 'Suspicious']:
            data = test_df[test_df['decision'] == decision]['R_hybrid']
            if len(data) > 0:
                ax.hist(data, alpha=0.6, label=decision, bins=30)
        ax.set_xlabel('Hybrid Risk Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Risk Score Distribution by Decision')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Risk score distribution by actual label
        ax = axes[0, 1]
        fraud_scores = test_df[test_df['label'] == 1]['R_hybrid']
        legit_scores = test_df[test_df['label'] == 0]['R_hybrid']
        ax.hist(legit_scores, alpha=0.6, label='Legitimate', bins=30, color='green')
        ax.hist(fraud_scores, alpha=0.6, label='Fraud', bins=30, color='red')
        ax.set_xlabel('Hybrid Risk Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Risk Score Distribution by Actual Label')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Decision distribution
        ax = axes[1, 0]
        decision_counts = test_df['decision'].value_counts()
        colors = {'Verified': '#28a745', 'Review': '#ffc107', 'Suspicious': '#dc3545'}
        bars = ax.bar(decision_counts.index, decision_counts.values,
                     color=[colors.get(x, '#6c757d') for x in decision_counts.index])
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Decisions')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')

        # Confusion-like matrix
        ax = axes[1, 1]
        decisions = ['Verified', 'Review', 'Suspicious']
        labels = ['Legit', 'Fraud']
        matrix = np.zeros((3, 2))

        for i, dec in enumerate(decisions):
            for j, lbl in enumerate([0, 1]):
                matrix[i, j] = len(test_df[(test_df['decision'] == dec) & (test_df['label'] == lbl)])

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(2))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(decisions)
        ax.set_xlabel('Actual Label')
        ax.set_ylabel('Decision')
        ax.set_title('Decision vs Actual Label Matrix')

        # Add text annotations
        for i in range(3):
            for j in range(2):
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=12)

        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        return self.fig_to_base64(fig)

    def generate_full_report(self, test_df, explanations_df, auc_supervised, auc_hybrid,
                            plots_dir, model_name="Identity Theft Detection"):
        """Generate complete HTML report"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate distribution charts
        dist_chart_b64 = self.generate_distribution_charts(test_df)

        # Load existing plots
        roc_plot = self.image_file_to_base64(os.path.join(plots_dir, "roc_supervised_vs_hybrid.png"))
        shap_summary = self.image_file_to_base64(os.path.join(plots_dir, "shap_summary.png"))
        shap_dependence = self.image_file_to_base64(os.path.join(plots_dir, "shap_dependence_doc_auth.png"))

        # Generate statistics
        stats_html = self.generate_summary_stats(test_df)

        # Generate detection table
        table_html = self.generate_detection_table(explanations_df)

        # Build complete HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Identity Theft Detection Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: #1e3c72;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .stat-card.verified {{
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        }}
        
        .stat-card.review {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        }}
        
        .stat-card.suspicious {{
            background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
            color: white;
        }}
        
        .stat-card.fraud {{
            background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
            color: white;
        }}
        
        .stat-card.legit {{
            background: linear-gradient(135deg, #81ecec 0%, #00b894 100%);
        }}
        
        .stat-card h3 {{
            font-size: 1.1em;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: 700;
            margin: 10px 0;
        }}
        
        .stat-percent {{
            font-size: 1.1em;
            opacity: 0.8;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .metric-box h3 {{
            color: #495057;
            font-size: 1em;
            margin-bottom: 10px;
        }}
        
        .metric-box .value {{
            font-size: 2em;
            font-weight: 700;
            color: #1e3c72;
        }}
        
        .plot-container {{
            margin: 30px 0;
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .plot-container h3 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .detection-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .detection-table thead {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
        }}
        
        .detection-table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .detection-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .detection-table tbody tr {{
            transition: background-color 0.2s ease;
        }}
        
        .detection-table tbody tr:hover {{
            background-color: #f1f3f5;
        }}
        
        .detection-table tbody tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        
        .badge.verified {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge.review {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge.suspicious {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .feature-cell {{
            font-size: 0.85em;
            color: #495057;
            max-width: 300px;
        }}
        
        .reason-cell {{
            font-size: 0.85em;
            color: #dc3545;
            max-width: 350px;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        
        .alert {{
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-color: #0c5460;
            color: #0c5460;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .content {{
                padding: 20px;
            }}
        }}
        
        .scroll-hint {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Identity Theft Detection Report</h1>
            <p>Hybrid XAI Model - Supervised + Unsupervised Anomaly Detection</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated: {timestamp}</p>
        </div>
        
        <div class="content">
            <!-- Model Performance Section -->
            <div class="section">
                <h2>📊 Model Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Supervised Model AUC</h3>
                        <div class="value">{auc_supervised:.4f}</div>
                    </div>
                    <div class="metric-box">
                        <h3>Hybrid Model AUC</h3>
                        <div class="value">{auc_hybrid:.4f}</div>
                    </div>
                    <div class="metric-box">
                        <h3>Improvement</h3>
                        <div class="value">{(auc_hybrid - auc_supervised):.4f}</div>
                    </div>
                </div>
                
                <div class="plot-container">
                    <h3>ROC Curve Comparison</h3>
                    <img src="data:image/png;base64,{roc_plot}" alt="ROC Curve">
                </div>
            </div>
            
            <!-- Summary Statistics Section -->
            <div class="section">
                <h2>📈 Detection Summary</h2>
                {stats_html}
            </div>
            
            <!-- Distribution Charts Section -->
            <div class="section">
                <h2>📉 Risk Score Distributions</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{dist_chart_b64}" alt="Distribution Charts">
                </div>
            </div>
            
            <!-- SHAP Explainability Section -->
            <div class="section">
                <h2>🔍 SHAP Explainability Analysis</h2>
                <div class="alert alert-info">
                    <strong>SHAP (SHapley Additive exPlanations)</strong> provides insights into which features contribute most to fraud predictions.
                    Positive SHAP values push predictions toward fraud, while negative values push toward legitimate.
                </div>
                
                <div class="plot-container">
                    <h3>Global Feature Importance (SHAP Summary)</h3>
                    <img src="data:image/png;base64,{shap_summary}" alt="SHAP Summary">
                </div>
                
                <div class="plot-container">
                    <h3>Document Authentication Score Impact (SHAP Dependence)</h3>
                    <img src="data:image/png;base64,{shap_dependence}" alt="SHAP Dependence">
                </div>
            </div>
            
            <!-- Detected Cases Section -->
            <div class="section">
                <h2>🚨 Detected Suspicious & Review Cases</h2>
                <div class="alert alert-info">
                    <strong>Top {min(len(explanations_df), 50)} cases</strong> requiring attention, sorted by risk score.
                    Each case includes SHAP-based feature contributions and rule-based reason codes.
                </div>
                {table_html}
                <p class="scroll-hint">Scroll horizontally to view all columns</p>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Identity Theft Detection System</strong> - Hybrid XAI Model with SHAP & LIME</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Model combines supervised learning (XGBoost) with unsupervised anomaly detection (Isolation Forest)
            </p>
        </div>
    </div>
</body>
</html>
"""

        # Save report
        report_path = os.path.join(self.report_dir, "identity_theft_detection_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n{'='*80}")
        print(f"✅ HTML Report generated successfully!")
        print(f"📄 Location: {report_path}")
        print(f"{'='*80}\n")

        return report_path

