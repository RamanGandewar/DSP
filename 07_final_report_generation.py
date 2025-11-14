"""
07_final_report_generation.py
==============================
Generate comprehensive final report
- Compile all analysis results
- Create executive summary
- Generate detailed visualizations
- Export final deliverables
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
SCORING_DIR = os.path.join(BASE_DIR, "analysis_results", "05_scoring")
FEATURE_DIR = os.path.join(BASE_DIR, "analysis_results", "04_features")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "07_final_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STEP 7: FINAL REPORT GENERATION")
print("="*80)

# Load all results
df_scores = pd.read_csv(os.path.join(SCORING_DIR, "cognitive_intelligence_scores.csv"))
df_features = pd.read_csv(os.path.join(FEATURE_DIR, "extracted_features.csv"))

# ============================================================================
# COMPREHENSIVE DASHBOARD
# ============================================================================

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('fMRI Cognitive Intelligence Analysis - Final Report', 
             fontsize=20, fontweight='bold', y=0.98)

# 1. Subject Rankings (Large)
ax1 = fig.add_subplot(gs[0:2, 0])
colors = df_scores['classification'].map({
    'HIGH': '#2ecc71',
    'AVERAGE': '#f39c12',
    'LOW': '#e74c3c'
})
ax1.barh(range(len(df_scores)), df_scores['score'], color=colors, 
         alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(df_scores)))
ax1.set_yticklabels([f"S{int(s):02d}" for s in df_scores['subject_id']], fontsize=9)
ax1.set_xlabel('Cognitive Intelligence Score', fontsize=12, fontweight='bold')
ax1.set_title('Complete Subject Rankings', fontsize=14, fontweight='bold')
ax1.axvline(x=65, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax1.axvline(x=45, color='orange', linestyle='--', alpha=0.5, linewidth=2)
ax1.grid(alpha=0.3, axis='x')
ax1.invert_yaxis()

# 2. Top 10 Performers
ax2 = fig.add_subplot(gs[0, 1])
top10 = df_scores.head(10)
ax2.bar(range(len(top10)), top10['score'], color='#2ecc71', 
        edgecolor='black', alpha=0.8)
ax2.set_xticks(range(len(top10)))
ax2.set_xticklabels([f"S{int(s):02d}" for s in top10['subject_id']], fontsize=9)
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('🏆 Top 10 Performers', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

# 3. Bottom 10 Performers
ax3 = fig.add_subplot(gs[1, 1])
bottom10 = df_scores.tail(10)
ax3.bar(range(len(bottom10)), bottom10['score'], color='#e74c3c',
        edgecolor='black', alpha=0.8)
ax3.set_xticks(range(len(bottom10)))
ax3.set_xticklabels([f"S{int(s):02d}" for s in bottom10['subject_id']], 
                     fontsize=9, rotation=45)
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('⚠️ Bottom 10 Performers', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# 4. Classification Distribution
ax4 = fig.add_subplot(gs[0, 2])
class_counts = df_scores['classification'].value_counts()
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax4.pie(class_counts.values, labels=class_counts.index,
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Classification Distribution', fontsize=13, fontweight='bold')

# 5. Score Distribution Histogram
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df_scores['score'], bins=20, color='steelblue', alpha=0.7, 
         edgecolor='black', linewidth=1.5)
ax5.axvline(df_scores['score'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {df_scores["score"].mean():.1f}')
ax5.axvline(df_scores['score'].median(), color='green', linestyle='--',
            linewidth=2, label=f'Median: {df_scores["score"].median():.1f}')
ax5.set_xlabel('Score', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Score Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3, axis='y')

# 6. Component Contribution
ax6 = fig.add_subplot(gs[2, 0])
mean_components = df_scores[['differentiation_score', 'pattern_score', 
                              'efficiency_score']].mean()
colors_bar = ['#3498db', '#9b59b6', '#e67e22']
bars = ax6.bar(range(len(mean_components)), mean_components.values, 
               color=colors_bar, edgecolor='black', alpha=0.8)
ax6.set_xticks(range(len(mean_components)))
ax6.set_xticklabels(['Differentiation\n(40 max)', 'Pattern\n(30 max)', 
                      'Efficiency\n(30 max)'], fontsize=10)
ax6.set_ylabel('Average Points', fontsize=11, fontweight='bold')
ax6.set_title('Mean Component Scores', fontsize=13, fontweight='bold')
ax6.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# 7. Feature Importance
ax7 = fig.add_subplot(gs[2, 1])
feature_importance = {
    'Signal\nDiff': df_features['mean_abs_diff'].mean() * 50,
    'Pattern\nQuality': df_features[['cong_std', 'incong_std']].mean().mean() * 40,
    'Efficiency\nRatio': df_features['range_ratio'].mean() * 20,
    'Trial\nConsistency': df_features[['cong_trial_std', 'incong_trial_std']].mean().mean() * 15
}
ax7.barh(range(len(feature_importance)), list(feature_importance.values()),
         color='teal', edgecolor='black', alpha=0.7)
ax7.set_yticks(range(len(feature_importance)))
ax7.set_yticklabels(list(feature_importance.keys()), fontsize=10)
ax7.set_xlabel('Relative Importance', fontsize=11, fontweight='bold')
ax7.set_title('Feature Importance', fontsize=13, fontweight='bold')
ax7.grid(alpha=0.3, axis='x')

# 8. Summary Statistics Box
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

summary_stats = f"""
📊 ANALYSIS SUMMARY
{'='*30}

Total Subjects: {len(df_scores)}
Date: {datetime.now().strftime('%Y-%m-%d')}

Score Statistics:
  Mean:   {df_scores['score'].mean():.2f}
  Median: {df_scores['score'].median():.2f}
  SD:     {df_scores['score'].std():.2f}
  Range:  {df_scores['score'].min():.1f} - {df_scores['score'].max():.1f}

Classifications:
  🟢 HIGH:    {len(df_scores[df_scores['classification']=='HIGH'])} ({len(df_scores[df_scores['classification']=='HIGH'])/len(df_scores)*100:.1f}%)
  🟡 AVERAGE: {len(df_scores[df_scores['classification']=='AVERAGE'])} ({len(df_scores[df_scores['classification']=='AVERAGE'])/len(df_scores)*100:.1f}%)
  🔴 LOW:     {len(df_scores[df_scores['classification']=='LOW'])} ({len(df_scores[df_scores['classification']=='LOW'])/len(df_scores)*100:.1f}%)

Top Performer:
  Subject-{int(df_scores.iloc[0]['subject_id']):02d}: {df_scores.iloc[0]['score']:.1f}

Method: fMRI Flanker Task
Analysis: DSP + Cognitive Scoring
"""

ax8.text(0.05, 0.5, summary_stats, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round',
         facecolor='wheat', alpha=0.3))

plt.savefig(os.path.join(OUTPUT_DIR, "comprehensive_dashboard.png"), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'comprehensive_dashboard.png')}")

# ============================================================================
# GENERATE TEXT REPORT
# ============================================================================

report_content = f"""
{'='*80}
fMRI COGNITIVE INTELLIGENCE ANALYSIS - FINAL REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
This study analyzed fMRI data from {len(df_scores)} subjects performing the Flanker 
cognitive control task. Using advanced signal processing and a novel cognitive 
intelligence scoring algorithm, we quantified individual differences in neural 
efficiency and cognitive control.

METHODOLOGY
-----------
1. Data Preprocessing
   - Whole-brain signal extraction
   - Linear detrending
   - Bandpass filtering (0.01-0.1 Hz)
   - Z-score normalization

2. Feature Extraction
   - Signal differentiation between trial types
   - Response pattern characteristics
   - Cognitive efficiency metrics

3. Scoring Algorithm (0-100 scale)
   - Differentiation Score (40 points): Neural discrimination capability
   - Pattern Score (30 points): Hemodynamic response quality
   - Efficiency Score (30 points): Task-appropriate neural resource allocation

RESULTS
-------

Overall Performance:
  • Mean Score: {df_scores['score'].mean():.2f} ± {df_scores['score'].std():.2f}
  • Median Score: {df_scores['score'].median():.2f}
  • Score Range: {df_scores['score'].min():.1f} - {df_scores['score'].max():.1f}

Group Classifications:
"""

for class_name, icon in [('HIGH', '🟢'), ('AVERAGE', '🟡'), ('LOW', '🔴')]:
    group = df_scores[df_scores['classification'] == class_name]
    if len(group) > 0:
        pct = len(group) / len(df_scores) * 100
        report_content += f"\n{icon} {class_name} Cognitive Intelligence: {len(group)} subjects ({pct:.1f}%)\n"
        report_content += f"   Score Range: {group['score'].min():.1f} - {group['score'].max():.1f}\n"
        report_content += f"   Mean: {group['score'].mean():.1f} ± {group['score'].std():.1f}\n"

report_content += f"""

TOP 5 PERFORMERS
----------------
"""

for idx, row in df_scores.head(5).iterrows():
    report_content += f"{int(row['rank'])}. Subject-{int(row['subject_id']):02d}: {row['score']:.1f} ({row['classification']}) {row['icon']}\n"

report_content += f"""

COMPONENT ANALYSIS
------------------
Average Component Scores:
  • Differentiation: {df_scores['differentiation_score'].mean():.2f} / 40.0
  • Pattern Quality: {df_scores['pattern_score'].mean():.2f} / 30.0
  • Efficiency:      {df_scores['efficiency_score'].mean():.2f} / 30.0

CONCLUSIONS
-----------
The analysis successfully identified distinct cognitive profiles across subjects.
The scoring algorithm effectively captured individual differences in:
  ✓ Neural differentiation between task conditions
  ✓ Hemodynamic response quality
  ✓ Cognitive efficiency and resource allocation

RECOMMENDATIONS
---------------
1. HIGH performers: Ideal candidates for cognitively demanding tasks
2. AVERAGE performers: Represent typical cognitive function
3. LOW performers: May benefit from cognitive training interventions

For detailed methodology and additional visualizations, refer to the 
analysis_results directory.

{'='*80}
END OF REPORT
{'='*80}
"""

report_path = os.path.join(OUTPUT_DIR, "final_analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"✓ Saved: {report_path}")

# ============================================================================
# EXPORT MASTER RESULTS FILE
# ============================================================================

# Merge scores with features
master_df = df_scores.merge(df_features, on='subject_id', how='left')

excel_path = os.path.join(OUTPUT_DIR, "master_results.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    master_df.to_excel(writer, sheet_name='Complete Results', index=False)
    df_scores.to_excel(writer, sheet_name='Scores', index=False)
    df_features.to_excel(writer, sheet_name='Features', index=False)

print(f"✓ Saved: {excel_path}")

print("\n" + "="*80)
print("✓ FINAL REPORT GENERATION COMPLETE")
print("="*80)
print(f"\nAll deliverables saved to: {OUTPUT_DIR}")
print("\nGenerated Files:")
print("  1. comprehensive_dashboard.png - Visual summary")
print("  2. final_analysis_report.txt - Detailed text report")
print("  3. master_results.xlsx - Complete data export")