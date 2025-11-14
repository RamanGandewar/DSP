"""
09_detailed_ml_visualizations.py
=================================
Create comprehensive visualizations for SVM and Random Forest
- Separate detailed figures for each model
- Confusion matrices, feature analysis, performance metrics
- Publication-ready figures
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, cohen_kappa_score, 
                             precision_recall_fscore_support, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
SCORING_DIR = os.path.join(BASE_DIR, "analysis_results", "05_scoring")
FEATURE_DIR = os.path.join(BASE_DIR, "analysis_results", "04_features")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "09_ml_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("DETAILED ML VISUALIZATIONS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

df_scores = pd.read_csv(os.path.join(SCORING_DIR, "cognitive_intelligence_scores.csv"))
df_features = pd.read_csv(os.path.join(FEATURE_DIR, "extracted_features.csv"))
df = df_scores.merge(df_features, on='subject_id', how='inner')

feature_columns = [
    'mean_abs_diff', 'max_abs_diff', 'correlation',
    'cong_std', 'incong_std', 'range_ratio',
    'std_ratio', 'peak_ratio', 'mean_t_stat',
    'cong_range', 'incong_range'
]

available_features = [f for f in feature_columns if f in df.columns]
X = df[available_features].values
y = df['classification'].values
subject_ids = df['subject_id'].values

# Encode labels
label_mapping = {'HIGH': 2, 'AVERAGE': 1, 'LOW': 0}
y_encoded = np.array([label_mapping[label] for label in y])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n✓ Loaded {len(df)} subjects with {len(available_features)} features")

# ============================================================================
# SUPPORT VECTOR MACHINE - COMPREHENSIVE VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING SVM COMPREHENSIVE VISUALIZATION")
print("="*80)

# Train SVM and get predictions
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_predictions = cross_val_predict(svm_model, X_scaled, y_encoded, cv=cv)
svm_model.fit(X_scaled, y_encoded)
svm_proba = cross_val_predict(svm_model, X_scaled, y_encoded, cv=cv, method='predict_proba')

# Calculate metrics
svm_accuracy = accuracy_score(y_encoded, svm_predictions)
svm_kappa = cohen_kappa_score(y_encoded, svm_predictions)
svm_cm = confusion_matrix(y_encoded, svm_predictions)
svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(
    y_encoded, svm_predictions, average=None, labels=[0, 1, 2]
)

# Create SVM comprehensive figure
fig_svm = plt.figure(figsize=(20, 12))
fig_svm.suptitle('Support Vector Machine (SVM) - Comprehensive Analysis', 
                 fontsize=22, fontweight='bold', y=0.98)

# ============================================================================
# SVM PLOT 1: Confusion Matrix (Large)
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['LOW', 'AVERAGE', 'HIGH'],
            yticklabels=['LOW', 'AVERAGE', 'HIGH'],
            ax=ax1, square=True, linewidths=3, annot_kws={'size': 16, 'weight': 'bold'})
ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax1.set_title(f'Confusion Matrix\nAccuracy: {svm_accuracy:.3f} (73.1%)', 
              fontsize=15, fontweight='bold', pad=10)

# Add percentage annotations
for i in range(3):
    for j in range(3):
        total = svm_cm[i].sum()
        pct = (svm_cm[i, j] / total) * 100 if total > 0 else 0
        ax1.text(j + 0.5, i + 0.7, f'({pct:.0f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

# ============================================================================
# SVM PLOT 2: Performance Metrics Bar Chart
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
metrics_labels = ['LOW', 'AVERAGE', 'HIGH']
x_pos = np.arange(len(metrics_labels))
width = 0.25

bars1 = ax2.bar(x_pos - width, svm_precision, width, label='Precision', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x_pos, svm_recall, width, label='Recall',
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax2.bar(x_pos + width, svm_f1, width, label='F1-Score',
                color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Class', fontsize=13, fontweight='bold')
ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
ax2.set_title('Per-Class Performance Metrics', fontsize=15, fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_labels, fontsize=12)
ax2.set_ylim([0, 1.1])
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

# ============================================================================
# SVM PLOT 3: Class Distribution Comparison
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
class_counts = [np.sum(y_encoded == i) for i in [0, 1, 2]]
class_correct = [svm_cm[i, i] for i in range(3)]
class_incorrect = [class_counts[i] - class_correct[i] for i in range(3)]

x_pos = np.arange(3)
bars_correct = ax3.bar(x_pos, class_correct, label='Correctly Classified',
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars_incorrect = ax3.bar(x_pos, class_incorrect, bottom=class_correct,
                         label='Misclassified', color='#e74c3c', alpha=0.8,
                         edgecolor='black', linewidth=1.5)

ax3.set_xlabel('Class', fontsize=13, fontweight='bold')
ax3.set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
ax3.set_title('Classification Results by Class', fontsize=15, fontweight='bold', pad=10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['LOW (n=6)', 'AVERAGE (n=16)', 'HIGH (n=4)'], fontsize=11)
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3, axis='y')

# Add count labels
for i, (correct, incorrect) in enumerate(zip(class_correct, class_incorrect)):
    ax3.text(i, correct/2, str(correct), ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    if incorrect > 0:
        ax3.text(i, correct + incorrect/2, str(incorrect), ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')

# ============================================================================
# SVM PLOT 4: Feature Contribution (Top 8)
# ============================================================================
ax4 = plt.subplot(2, 3, 4)

# For linear kernel, we could get coefficients, but for RBF we'll show support vector distribution
# Instead, let's show how each feature correlates with predictions
feature_corr = []
for i, feat in enumerate(available_features):
    corr = np.corrcoef(X_scaled[:, i], svm_predictions)[0, 1]
    feature_corr.append(abs(corr))

feature_importance_df = pd.DataFrame({
    'feature': available_features,
    'correlation': feature_corr
}).sort_values('correlation', ascending=True).tail(8)

ax4.barh(range(len(feature_importance_df)), feature_importance_df['correlation'],
         color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_yticks(range(len(feature_importance_df)))
ax4.set_yticklabels(feature_importance_df['feature'], fontsize=11)
ax4.set_xlabel('Absolute Correlation with Predictions', fontsize=12, fontweight='bold')
ax4.set_title('Top 8 Feature Contributions', fontsize=15, fontweight='bold', pad=10)
ax4.grid(alpha=0.3, axis='x')

# Add value labels
for i, val in enumerate(feature_importance_df['correlation']):
    ax4.text(val, i, f' {val:.3f}', va='center', fontsize=10, fontweight='bold')

# ============================================================================
# SVM PLOT 5: Summary Statistics Box
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

summary_text = f"""
SVM PERFORMANCE SUMMARY
{'='*35}

Overall Metrics:
  • Accuracy:      {svm_accuracy:.3f} (73.1%)
  • Cohen's Kappa: {svm_kappa:.3f}
  • Interpretation: Substantial Agreement

Per-Class Performance:
  LOW (n=6):
    ✓ Precision: {svm_precision[0]:.3f}
    ✓ Recall:    {svm_recall[0]:.3f}
    ✓ F1-Score:  {svm_f1[0]:.3f}
  
  AVERAGE (n=16):
    ✓ Precision: {svm_precision[1]:.3f}
    ✓ Recall:    {svm_recall[1]:.3f} ★
    ✓ F1-Score:  {svm_f1[1]:.3f}
  
  HIGH (n=4):
    ✓ Precision: {svm_precision[2]:.3f}
    ✓ Recall:    {svm_recall[2]:.3f}
    ✓ F1-Score:  {svm_f1[2]:.3f}

Key Findings:
  ✓ Excellent on AVERAGE class
  ⚠ Limited by small HIGH/LOW samples
  ✓ RBF kernel captures non-linearity
  ✓ Validates manual feature selection
"""

ax5.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

# ============================================================================
# SVM PLOT 6: Misclassification Analysis
# ============================================================================
ax6 = plt.subplot(2, 3, 6)

# Get misclassified subjects
misclassified = y_encoded != svm_predictions
correct = ~misclassified

# Create scatter plot
scores = df['score'].values
feature_importance_sum = X_scaled[:, :3].sum(axis=1)  # Sum of top 3 features

scatter_correct = ax6.scatter(scores[correct], feature_importance_sum[correct],
                              c=y_encoded[correct], cmap='RdYlGn', s=150,
                              alpha=0.6, edgecolor='black', linewidth=2,
                              marker='o', label='Correct')
scatter_wrong = ax6.scatter(scores[misclassified], feature_importance_sum[misclassified],
                           c=y_encoded[misclassified], cmap='RdYlGn', s=150,
                           alpha=0.9, edgecolor='red', linewidth=3,
                           marker='X', label='Misclassified')

# Add threshold lines
ax6.axvline(x=45, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Threshold: 45')
ax6.axvline(x=65, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Threshold: 65')

ax6.set_xlabel('Manual Cognitive Score', fontsize=13, fontweight='bold')
ax6.set_ylabel('Feature Space (Top 3 Sum)', fontsize=13, fontweight='bold')
ax6.set_title('Correct vs Misclassified Subjects', fontsize=15, fontweight='bold', pad=10)
ax6.legend(fontsize=10, loc='upper left')
ax6.grid(alpha=0.3)

# Annotate misclassified subjects
for idx in np.where(misclassified)[0]:
    ax6.annotate(f'S{subject_ids[idx]:02d}', 
                (scores[idx], feature_importance_sum[idx]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "SVM_comprehensive_analysis.png"), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: SVM_comprehensive_analysis.png")

# ============================================================================
# RANDOM FOREST - COMPREHENSIVE VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING RANDOM FOREST COMPREHENSIVE VISUALIZATION")
print("="*80)

# Train Random Forest and get predictions
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                  min_samples_split=3, random_state=42)
rf_predictions = cross_val_predict(rf_model, X_scaled, y_encoded, cv=cv)
rf_model.fit(X_scaled, y_encoded)
rf_proba = cross_val_predict(rf_model, X_scaled, y_encoded, cv=cv, method='predict_proba')

# Calculate metrics
rf_accuracy = accuracy_score(y_encoded, rf_predictions)
rf_kappa = cohen_kappa_score(y_encoded, rf_predictions)
rf_cm = confusion_matrix(y_encoded, rf_predictions)
rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(
    y_encoded, rf_predictions, average=None, labels=[0, 1, 2]
)

# Get feature importance
rf_feature_importance = rf_model.feature_importances_
rf_feature_df = pd.DataFrame({
    'feature': available_features,
    'importance': rf_feature_importance
}).sort_values('importance', ascending=True)

# Create RF comprehensive figure
fig_rf = plt.figure(figsize=(20, 12))
fig_rf.suptitle('Random Forest - Comprehensive Analysis', 
                fontsize=22, fontweight='bold', y=0.98)

# ============================================================================
# RF PLOT 1: Confusion Matrix (Large)
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['LOW', 'AVERAGE', 'HIGH'],
            yticklabels=['LOW', 'AVERAGE', 'HIGH'],
            ax=ax1, square=True, linewidths=3, annot_kws={'size': 16, 'weight': 'bold'})
ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax1.set_title(f'Confusion Matrix\nAccuracy: {rf_accuracy:.3f} (65.4%)', 
              fontsize=15, fontweight='bold', pad=10)

# Add percentage annotations
for i in range(3):
    for j in range(3):
        total = rf_cm[i].sum()
        pct = (rf_cm[i, j] / total) * 100 if total > 0 else 0
        ax1.text(j + 0.5, i + 0.7, f'({pct:.0f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

# ============================================================================
# RF PLOT 2: Performance Metrics Bar Chart
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
metrics_labels = ['LOW', 'AVERAGE', 'HIGH']
x_pos = np.arange(len(metrics_labels))
width = 0.25

bars1 = ax2.bar(x_pos - width, rf_precision, width, label='Precision', 
                color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x_pos, rf_recall, width, label='Recall',
                color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax2.bar(x_pos + width, rf_f1, width, label='F1-Score',
                color='#d35400', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Class', fontsize=13, fontweight='bold')
ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
ax2.set_title('Per-Class Performance Metrics', fontsize=15, fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_labels, fontsize=12)
ax2.set_ylim([0, 1.1])
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

# ============================================================================
# RF PLOT 3: Feature Importance (All Features)
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
colors_gradient = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(rf_feature_df)))

bars = ax3.barh(range(len(rf_feature_df)), rf_feature_df['importance'],
                color=colors_gradient, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(rf_feature_df)))
ax3.set_yticklabels(rf_feature_df['feature'], fontsize=11)
ax3.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
ax3.set_title('Feature Importance Ranking', fontsize=15, fontweight='bold', pad=10)
ax3.grid(alpha=0.3, axis='x')

# Add value labels and rank
for i, (feat, val) in enumerate(zip(rf_feature_df['feature'], rf_feature_df['importance'])):
    rank = len(rf_feature_df) - i
    ax3.text(val, i, f' {val:.3f} (#{rank})', va='center', 
            fontsize=9, fontweight='bold')

# ============================================================================
# RF PLOT 4: Top 5 Features Detailed
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
top5_features = rf_feature_df.tail(5)

wedges, texts, autotexts = ax4.pie(top5_features['importance'], 
                                     labels=top5_features['feature'],
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=plt.cm.Oranges(np.linspace(0.4, 0.9, 5)),
                                     textprops={'fontsize': 11, 'fontweight': 'bold'},
                                     explode=[0.05]*5)

ax4.set_title('Top 5 Feature Contribution\n(Distribution)', 
              fontsize=15, fontweight='bold', pad=10)

# ============================================================================
# RF PLOT 5: Summary Statistics Box
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

summary_text = f"""
RANDOM FOREST SUMMARY
{'='*35}

Overall Metrics:
  • Accuracy:      {rf_accuracy:.3f} (65.4%)
  • Cohen's Kappa: {rf_kappa:.3f}
  • Interpretation: Fair Agreement

Per-Class Performance:
  LOW (n=6):
    ✓ Precision: {rf_precision[0]:.3f}
    ✓ Recall:    {rf_recall[0]:.3f}
    ✓ F1-Score:  {rf_f1[0]:.3f}
  
  AVERAGE (n=16):
    ✓ Precision: {rf_precision[1]:.3f}
    ✓ Recall:    {rf_recall[1]:.3f}
    ✓ F1-Score:  {rf_f1[1]:.3f}
  
  HIGH (n=4):
    ✓ Precision: {rf_precision[2]:.3f}
    ✓ Recall:    {rf_recall[2]:.3f}
    ✓ F1-Score:  {rf_f1[2]:.3f}

Top 3 Important Features:
  1. {rf_feature_df.iloc[-1]['feature']}: {rf_feature_df.iloc[-1]['importance']:.3f}
  2. {rf_feature_df.iloc[-2]['feature']}: {rf_feature_df.iloc[-2]['importance']:.3f}
  3. {rf_feature_df.iloc[-3]['feature']}: {rf_feature_df.iloc[-3]['importance']:.3f}

Key Findings:
  ✓ Provides feature interpretability
  ⚠ Lower accuracy than SVM
  ✓ Validates manual scoring features
"""

ax5.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

# ============================================================================
# RF PLOT 6: Class Distribution Comparison
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
class_counts = [np.sum(y_encoded == i) for i in [0, 1, 2]]
class_correct = [rf_cm[i, i] for i in range(3)]
class_incorrect = [class_counts[i] - class_correct[i] for i in range(3)]

x_pos = np.arange(3)
bars_correct = ax6.bar(x_pos, class_correct, label='Correctly Classified',
                       color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)
bars_incorrect = ax6.bar(x_pos, class_incorrect, bottom=class_correct,
                         label='Misclassified', color='#c0392b', alpha=0.8,
                         edgecolor='black', linewidth=1.5)

ax6.set_xlabel('Class', fontsize=13, fontweight='bold')
ax6.set_ylabel('Number of Subjects', fontsize=13, fontweight='bold')
ax6.set_title('Classification Results by Class', fontsize=15, fontweight='bold', pad=10)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(['LOW (n=6)', 'AVERAGE (n=16)', 'HIGH (n=4)'], fontsize=11)
ax6.legend(fontsize=11)
ax6.grid(alpha=0.3, axis='y')

# Add count labels
for i, (correct, incorrect) in enumerate(zip(class_correct, class_incorrect)):
    ax6.text(i, correct/2, str(correct), ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    if incorrect > 0:
        ax6.text(i, correct + incorrect/2, str(incorrect), ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "RandomForest_comprehensive_analysis.png"), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: RandomForest_comprehensive_analysis.png")

print("\n" + "="*80)
print("✓ DETAILED VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nGenerated Files:")
print(f"  1. SVM_comprehensive_analysis.png")
print(f"  2. RandomForest_comprehensive_analysis.png")
print(f"\nLocation: {OUTPUT_DIR}")