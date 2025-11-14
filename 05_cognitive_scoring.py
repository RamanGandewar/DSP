"""
05_cognitive_scoring.py
=======================
Compute cognitive intelligence scores
- Load extracted features
- Apply scoring algorithm
- Classify subjects
- Generate rankings
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
FEATURE_DIR = os.path.join(BASE_DIR, "analysis_results", "04_features")
TRIAL_DIR = os.path.join(BASE_DIR, "analysis_results", "03_trial_extraction")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "05_scoring")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUBJECTS = 26

print("="*80)
print("STEP 5: COGNITIVE INTELLIGENCE SCORING")
print("="*80)

# Load features and trial data
df_features = pd.read_csv(os.path.join(FEATURE_DIR, "extracted_features.csv"))

with open(os.path.join(TRIAL_DIR, "trial_responses.pkl"), 'rb') as f:
    trial_data = pickle.load(f)

def compute_cognitive_score(features):
    """
    Compute cognitive intelligence score from features
    Score range: 0-100
    """
    
    # ========================================================================
    # Component 1: Signal Differentiation (40 points)
    # ========================================================================
    diff_score = min((features['mean_abs_diff'] * 50 + features['max_abs_diff'] * 30), 40)
    
    # ========================================================================
    # Component 2: Response Pattern Quality (30 points)
    # ========================================================================
    avg_std = (features['cong_std'] + features['incong_std']) / 2
    pattern_score = min(avg_std * 50, 30)
    
    # ========================================================================
    # Component 3: Cognitive Efficiency (30 points)
    # ========================================================================
    if features['range_ratio'] > 1.0:
        efficiency_score = min((features['range_ratio'] - 1) * 30, 30)
    else:
        efficiency_score = max(0, 15 - abs(features['range_ratio'] - 1) * 20)
    
    total_score = diff_score + pattern_score + efficiency_score
    total_score = max(0, min(100, total_score))
    
    return {
        'total_score': total_score,
        'differentiation_score': diff_score,
        'pattern_score': pattern_score,
        'efficiency_score': efficiency_score
    }

def classify_subject(score):
    """Classify based on score"""
    if score >= 65:
        return "HIGH", "🟢"
    elif score >= 45:
        return "AVERAGE", "🟡"
    else:
        return "LOW", "🔴"

# Compute scores for all subjects
results = []

print("\nComputing scores:")
for idx, row in df_features.iterrows():
    scores = compute_cognitive_score(row)
    classification, icon = classify_subject(scores['total_score'])
    
    results.append({
        'subject_id': int(row['subject_id']),
        'score': scores['total_score'],
        'differentiation_score': scores['differentiation_score'],
        'pattern_score': scores['pattern_score'],
        'efficiency_score': scores['efficiency_score'],
        'classification': classification,
        'icon': icon
    })
    
    print(f"  Subject {int(row['subject_id']):02d}: {scores['total_score']:.1f} {icon}")

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
df_results['rank'] = range(1, len(df_results) + 1)

# Save results
csv_path = os.path.join(OUTPUT_DIR, "cognitive_intelligence_scores.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n✓ Saved scores: {csv_path}")

# Display rankings
print("\n" + "="*80)
print("COGNITIVE INTELLIGENCE RANKINGS")
print("="*80)
print(f"\n{'Rank':<6} {'Subject':<12} {'Score':<10} {'Classification':<15} {'Status'}")
print("-" * 80)

for _, row in df_results.iterrows():
    print(f"{row['rank']:<6} Subject-{row['subject_id']:02d}   "
          f"{row['score']:>6.1f}     "
          f"{row['classification']:<15} {row['icon']}")

# Group statistics
print("\n" + "="*80)
print("GROUP STATISTICS")
print("="*80)

for class_name, icon in [('HIGH', '🟢'), ('AVERAGE', '🟡'), ('LOW', '🔴')]:
    group = df_results[df_results['classification'] == class_name]
    if len(group) > 0:
        pct = len(group) / len(df_results) * 100
        print(f"\n{icon} {class_name} Cognitive Intelligence: {len(group)} subjects ({pct:.1f}%)")
        print(f"   Score Range: {group['score'].min():.1f} - {group['score'].max():.1f}")
        print(f"   Mean Score: {group['score'].mean():.1f}")
        subjects = ', '.join([f"{int(s):02d}" for s in group['subject_id'].values])
        print(f"   Subjects: {subjects}")

# Score component breakdown visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Overall rankings
colors = df_results['classification'].map({
    'HIGH': '#2ecc71',
    'AVERAGE': '#f39c12',
    'LOW': '#e74c3c'
})

axes[0, 0].barh(range(len(df_results)), df_results['score'], color=colors, 
                alpha=0.8, edgecolor='black')
axes[0, 0].set_yticks(range(len(df_results)))
axes[0, 0].set_yticklabels([f"Sub-{int(s):02d}" for s in df_results['subject_id']])
axes[0, 0].set_xlabel('Cognitive Intelligence Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Subject Rankings', fontsize=14, fontweight='bold')
axes[0, 0].axvline(x=65, color='green', linestyle='--', alpha=0.5, linewidth=2, label='High (≥65)')
axes[0, 0].axvline(x=45, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Average (≥45)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='x')
axes[0, 0].invert_yaxis()

# 2. Score distribution
axes[0, 1].hist(df_results['score'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(df_results['score'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df_results["score"].mean():.1f}')
axes[0, 1].axvline(df_results['score'].median(), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {df_results["score"].median():.1f}')
axes[0, 1].set_xlabel('Cognitive Intelligence Score', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Score Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# 3. Component breakdown (stacked bar)
components = df_results[['differentiation_score', 'pattern_score', 'efficiency_score']].values
subjects = [f"S{int(s):02d}" for s in df_results['subject_id']]

axes[1, 0].bar(range(len(df_results)), components[:, 0], label='Differentiation (40)', 
               color='#3498db', edgecolor='black')
axes[1, 0].bar(range(len(df_results)), components[:, 1], bottom=components[:, 0],
               label='Pattern Quality (30)', color='#9b59b6', edgecolor='black')
axes[1, 0].bar(range(len(df_results)), components[:, 2], 
               bottom=components[:, 0] + components[:, 1],
               label='Efficiency (30)', color='#e67e22', edgecolor='black')
axes[1, 0].set_xticks(range(len(df_results)))
axes[1, 0].set_xticklabels(subjects, rotation=90, fontsize=8)
axes[1, 0].set_ylabel('Score Points', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Score Component Breakdown', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# 4. Classification pie chart
class_counts = df_results['classification'].value_counts()
colors_pie = {'HIGH': '#2ecc71', 'AVERAGE': '#f39c12', 'LOW': '#e74c3c'}
pie_colors = [colors_pie[c] for c in class_counts.index]

axes[1, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=pie_colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1, 1].set_title('Classification Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cognitive_scores_visualization.png"), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(OUTPUT_DIR, 'cognitive_scores_visualization.png')}")

print("\n" + "="*80)
print("✓ COGNITIVE SCORING COMPLETE")
print("="*80)
print(f"\nSuccessfully scored: {len(df_results)}/{N_SUBJECTS} subjects")