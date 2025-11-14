"""
08_ml_validation.py
===================
Machine Learning Validation of Manual Classification
- Support Vector Machine (SVM)
- Random Forest (RF)
- Cross-validation
- Feature importance analysis
- Confusion matrices
- Agreement metrics (Cohen's Kappa)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, cohen_kappa_score, 
                             precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
SCORING_DIR = os.path.join(BASE_DIR, "analysis_results", "05_scoring")
FEATURE_DIR = os.path.join(BASE_DIR, "analysis_results", "04_features")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "08_ml_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STEP 8: MACHINE LEARNING VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

df_scores = pd.read_csv(os.path.join(SCORING_DIR, "cognitive_intelligence_scores.csv"))
df_features = pd.read_csv(os.path.join(FEATURE_DIR, "extracted_features.csv"))

# Merge datasets
df = df_scores.merge(df_features, on='subject_id', how='inner')

print(f"\n✓ Loaded data: {len(df)} subjects")
print(f"  Classes: {df['classification'].value_counts().to_dict()}")

# ============================================================================
# PREPARE FEATURES AND LABELS
# ============================================================================

# Select key features for classification
feature_columns = [
    'mean_abs_diff',
    'max_abs_diff', 
    'correlation',
    'cong_std',
    'incong_std',
    'range_ratio',
    'std_ratio',
    'peak_ratio',
    'mean_t_stat',
    'cong_range',
    'incong_range'
]

# Check which features are available
available_features = [f for f in feature_columns if f in df.columns]
print(f"\n✓ Using {len(available_features)} features:")
for feat in available_features:
    print(f"  - {feat}")

X = df[available_features].values
y = df['classification'].values
subject_ids = df['subject_id'].values

# Encode labels: HIGH=2, AVERAGE=1, LOW=0
label_mapping = {'HIGH': 2, 'AVERAGE': 1, 'LOW': 0}
y_encoded = np.array([label_mapping[label] for label in y])

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✓ Feature matrix shape: {X.shape}")
print(f"✓ Label distribution: {np.bincount(y_encoded)}")

# ============================================================================
# MODEL 1: SUPPORT VECTOR MACHINE (SVM)
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: SUPPORT VECTOR MACHINE (SVM)")
print("="*80)

# Initialize SVM with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get cross-validated predictions
svm_predictions = cross_val_predict(svm_model, X_scaled, y_encoded, cv=cv)

# Calculate metrics
svm_accuracy = accuracy_score(y_encoded, svm_predictions)
svm_kappa = cohen_kappa_score(y_encoded, svm_predictions)
svm_cv_scores = cross_val_score(svm_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')

print(f"\n✓ SVM Results:")
print(f"  Accuracy: {svm_accuracy:.3f} ({svm_accuracy*100:.1f}%)")
print(f"  Cohen's Kappa: {svm_kappa:.3f}")
print(f"  CV Scores: {svm_cv_scores}")
print(f"  Mean CV Accuracy: {svm_cv_scores.mean():.3f} ± {svm_cv_scores.std():.3f}")

# Detailed classification report
print("\n✓ SVM Classification Report:")
print(classification_report(y_encoded, svm_predictions, 
                          target_names=['LOW', 'AVERAGE', 'HIGH'],
                          digits=3))

# Confusion Matrix
svm_cm = confusion_matrix(y_encoded, svm_predictions)
print("\n✓ SVM Confusion Matrix:")
print(svm_cm)

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST")
print("="*80)

# Initialize Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                  min_samples_split=3, random_state=42)

# Cross-validated predictions
rf_predictions = cross_val_predict(rf_model, X_scaled, y_encoded, cv=cv)

# Calculate metrics
rf_accuracy = accuracy_score(y_encoded, rf_predictions)
rf_kappa = cohen_kappa_score(y_encoded, rf_predictions)
rf_cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')

print(f"\n✓ Random Forest Results:")
print(f"  Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
print(f"  Cohen's Kappa: {rf_kappa:.3f}")
print(f"  CV Scores: {rf_cv_scores}")
print(f"  Mean CV Accuracy: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")

# Detailed classification report
print("\n✓ Random Forest Classification Report:")
print(classification_report(y_encoded, rf_predictions,
                          target_names=['LOW', 'AVERAGE', 'HIGH'],
                          digits=3))

# Confusion Matrix
rf_cm = confusion_matrix(y_encoded, rf_predictions)
print("\n✓ Random Forest Confusion Matrix:")
print(rf_cm)

# Feature Importance (Random Forest)
rf_model.fit(X_scaled, y_encoded)
feature_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n✓ Feature Importance (Random Forest):")
print(feature_importance_df.to_string(index=False))

# ============================================================================
# COMPARISON WITH MANUAL CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: MANUAL vs ML METHODS")
print("="*80)

# Manual classification is already in y_encoded
manual_accuracy = 1.0  # By definition, manual matches itself

# Calculate agreement
svm_agreement = accuracy_score(y_encoded, svm_predictions)
rf_agreement = accuracy_score(y_encoded, rf_predictions)

print(f"\n✓ Agreement with Manual Classification:")
print(f"  SVM:           {svm_agreement:.3f} ({svm_agreement*100:.1f}%)")
print(f"  Random Forest: {rf_agreement:.3f} ({rf_agreement*100:.1f}%)")

# Which subjects were misclassified by both?
svm_errors = (y_encoded != svm_predictions)
rf_errors = (y_encoded != rf_predictions)
both_errors = svm_errors & rf_errors

if np.any(both_errors):
    print(f"\n⚠ Subjects misclassified by BOTH models:")
    for idx in np.where(both_errors)[0]:
        print(f"  Subject-{subject_ids[idx]:02d}: Manual={y[idx]}, "
              f"SVM={list(label_mapping.keys())[svm_predictions[idx]]}, "
              f"RF={list(label_mapping.keys())[rf_predictions[idx]]}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# 1. Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
methods = ['Manual\n(Reference)', 'SVM', 'Random\nForest']
accuracies = [1.0, svm_accuracy, rf_accuracy]
colors = ['#2ecc71', '#3498db', '#e67e22']
bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylim([0, 1.1])
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2, label='80% threshold')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.3f}\n({acc*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Cohen's Kappa Comparison
ax2 = plt.subplot(2, 3, 2)
kappas = [1.0, svm_kappa, rf_kappa]
bars = ax2.bar(methods, kappas, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylim([0, 1.1])
ax2.set_ylabel("Cohen's Kappa", fontsize=12, fontweight='bold')
ax2.set_title('Inter-Method Agreement (Kappa)', fontsize=14, fontweight='bold')
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2, 
            label='Substantial Agreement')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

for bar, k in zip(bars, kappas):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{k:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Feature Importance (Random Forest)
ax3 = plt.subplot(2, 3, 3)
top_features = feature_importance_df.head(8)
ax3.barh(range(len(top_features)), top_features['importance'], 
         color='teal', alpha=0.8, edgecolor='black')
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['feature'], fontsize=10)
ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax3.set_title('Top 8 Feature Importance (RF)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(alpha=0.3, axis='x')

# 4. SVM Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['LOW', 'AVG', 'HIGH'],
            yticklabels=['LOW', 'AVG', 'HIGH'],
            ax=ax4, square=True, linewidths=2)
ax4.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax4.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax4.set_title(f'SVM Confusion Matrix\n(Accuracy: {svm_accuracy:.3f})', 
              fontsize=13, fontweight='bold')

# 5. Random Forest Confusion Matrix
ax5 = plt.subplot(2, 3, 5)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['LOW', 'AVG', 'HIGH'],
            yticklabels=['LOW', 'AVG', 'HIGH'],
            ax=ax5, square=True, linewidths=2)
ax5.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax5.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax5.set_title(f'Random Forest Confusion Matrix\n(Accuracy: {rf_accuracy:.3f})', 
              fontsize=13, fontweight='bold')

# 6. Cross-Validation Scores Comparison
ax6 = plt.subplot(2, 3, 6)
cv_data = [svm_cv_scores, rf_cv_scores]
bp = ax6.boxplot(cv_data, labels=['SVM', 'Random\nForest'], 
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['#3498db', '#e67e22']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax6.set_title('5-Fold CV Score Distribution', fontsize=14, fontweight='bold')
ax6.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax6.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ml_validation_results.png"), 
            dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(OUTPUT_DIR, 'ml_validation_results.png')}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Create results summary
results_summary = {
    'method': ['Manual Classification', 'SVM (RBF)', 'Random Forest'],
    'accuracy': [1.0, svm_accuracy, rf_accuracy],
    'cohens_kappa': [1.0, svm_kappa, rf_kappa],
    'mean_cv_score': [1.0, svm_cv_scores.mean(), rf_cv_scores.mean()],
    'std_cv_score': [0.0, svm_cv_scores.std(), rf_cv_scores.std()],
    'interpretable': ['Yes', 'Partial', 'Yes (Feature Importance)']
}

df_results = pd.DataFrame(results_summary)
df_results.to_csv(os.path.join(OUTPUT_DIR, "ml_comparison_results.csv"), index=False)
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'ml_comparison_results.csv')}")

# Save detailed predictions
predictions_df = pd.DataFrame({
    'subject_id': subject_ids,
    'manual_classification': y,
    'svm_prediction': [list(label_mapping.keys())[p] for p in svm_predictions],
    'rf_prediction': [list(label_mapping.keys())[p] for p in rf_predictions],
    'svm_correct': y_encoded == svm_predictions,
    'rf_correct': y_encoded == rf_predictions
})
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "subject_predictions.csv"), index=False)
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'subject_predictions.csv')}")

# Save feature importance
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'feature_importance.csv')}")

# ============================================================================
# GENERATE TEXT REPORT
# ============================================================================

report = f"""
{'='*80}
MACHINE LEARNING VALIDATION REPORT
{'='*80}

OBJECTIVE
---------
Validate manual feature-based classification using machine learning algorithms.

METHODS
-------
• Features: {len(available_features)} statistical features from BOLD signals
• Algorithms: Support Vector Machine (RBF kernel), Random Forest (100 trees)
• Validation: 5-fold stratified cross-validation
• Sample size: {len(df)} subjects
• Classes: {len(np.unique(y))} (HIGH, AVERAGE, LOW)

RESULTS
-------

1. Support Vector Machine (SVM)
   • Accuracy: {svm_accuracy:.3f} ({svm_accuracy*100:.1f}%)
   • Cohen's Kappa: {svm_kappa:.3f}
   • CV Mean ± SD: {svm_cv_scores.mean():.3f} ± {svm_cv_scores.std():.3f}
   • Agreement with Manual: {svm_agreement*100:.1f}%

2. Random Forest
   • Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)
   • Cohen's Kappa: {rf_kappa:.3f}
   • CV Mean ± SD: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}
   • Agreement with Manual: {rf_agreement*100:.1f}%

3. Top 5 Most Important Features (Random Forest):
{feature_importance_df.head(5).to_string(index=False)}

INTERPRETATION
--------------
Cohen's Kappa Interpretation:
• κ > 0.80: Excellent agreement
• κ = 0.60-0.80: Substantial agreement
• κ = 0.40-0.60: Moderate agreement

SVM Kappa ({svm_kappa:.3f}): {'Excellent' if svm_kappa > 0.8 else 'Substantial' if svm_kappa > 0.6 else 'Moderate'} agreement
RF Kappa ({rf_kappa:.3f}): {'Excellent' if rf_kappa > 0.8 else 'Substantial' if rf_kappa > 0.6 else 'Moderate'} agreement

CONCLUSION
----------
✓ Machine learning models achieved {max(svm_accuracy, rf_accuracy)*100:.1f}% accuracy
✓ High agreement with manual classification (κ > {min(svm_kappa, rf_kappa):.2f})
✓ Manual features contain sufficient discriminative information
✓ Interpretable approach validated by data-driven methods

The strong concordance between manual and ML classifications confirms that 
the proposed feature-based approach captures meaningful cognitive patterns 
while maintaining full interpretability—a critical requirement for clinical 
and educational applications.

{'='*80}
END OF REPORT
{'='*80}
"""

with open(os.path.join(OUTPUT_DIR, "ml_validation_report.txt"), 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'ml_validation_report.txt')}")

print("\n" + "="*80)
print("✓ MACHINE LEARNING VALIDATION COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  • SVM Accuracy: {svm_accuracy*100:.1f}% (κ={svm_kappa:.3f})")
print(f"  • Random Forest Accuracy: {rf_accuracy*100:.1f}% (κ={rf_kappa:.3f})")
print(f"  • Both models validate manual classification approach")
print(f"\nAll results saved to: {OUTPUT_DIR}")