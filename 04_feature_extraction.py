"""
04_feature_extraction.py
========================
Extract features from trial responses
- Signal differentiation metrics
- Response pattern characteristics
- Cognitive efficiency measures
- Statistical features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
TRIAL_DIR = os.path.join(BASE_DIR, "analysis_results", "03_trial_extraction")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "04_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUBJECTS = 26

print("="*80)
print("STEP 4: FEATURE EXTRACTION")
print("="*80)

# Load trial responses
with open(os.path.join(TRIAL_DIR, "trial_responses.pkl"), 'rb') as f:
    trial_data = pickle.load(f)

def extract_features(congruent_responses, incongruent_responses):
    """
    Extract comprehensive features from trial responses
    """
    if len(congruent_responses) == 0 or len(incongruent_responses) == 0:
        return None
    
    # Average responses
    cong_avg = np.mean(congruent_responses, axis=0)
    incong_avg = np.mean(incongruent_responses, axis=0)
    
    features = {}
    
    # ========================================================================
    # DIFFERENTIATION FEATURES
    # ========================================================================
    features['mean_abs_diff'] = np.mean(np.abs(cong_avg - incong_avg))
    features['max_abs_diff'] = np.max(np.abs(cong_avg - incong_avg))
    features['diff_variance'] = np.var(cong_avg - incong_avg)
    features['correlation'] = np.corrcoef(cong_avg, incong_avg)[0, 1]
    
    # ========================================================================
    # SIGNAL CHARACTERISTICS
    # ========================================================================
    features['cong_mean'] = np.mean(cong_avg)
    features['cong_std'] = np.std(cong_avg)
    features['cong_range'] = np.max(cong_avg) - np.min(cong_avg)
    features['cong_peak'] = np.max(np.abs(cong_avg))
    
    features['incong_mean'] = np.mean(incong_avg)
    features['incong_std'] = np.std(incong_avg)
    features['incong_range'] = np.max(incong_avg) - np.min(incong_avg)
    features['incong_peak'] = np.max(np.abs(incong_avg))
    
    # ========================================================================
    # EFFICIENCY METRICS
    # ========================================================================
    features['range_ratio'] = features['incong_range'] / (features['cong_range'] + 0.01)
    features['std_ratio'] = features['incong_std'] / (features['cong_std'] + 0.01)
    features['peak_ratio'] = features['incong_peak'] / (features['cong_peak'] + 0.01)
    
    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================
    features['cong_peak_time'] = np.argmax(np.abs(cong_avg))
    features['incong_peak_time'] = np.argmax(np.abs(incong_avg))
    features['peak_time_diff'] = abs(features['incong_peak_time'] - features['cong_peak_time'])
    
    # ========================================================================
    # TRIAL CONSISTENCY
    # ========================================================================
    features['cong_trial_std'] = np.mean(np.std(congruent_responses, axis=0))
    features['incong_trial_std'] = np.mean(np.std(incongruent_responses, axis=0))
    
    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================
    # T-test at each timepoint
    t_stats = []
    for tp in range(min(len(cong_avg), len(incong_avg))):
        t_stat, _ = stats.ttest_ind(congruent_responses[:, tp], incongruent_responses[:, tp])
        t_stats.append(abs(t_stat))
    features['mean_t_stat'] = np.mean(t_stats)
    features['max_t_stat'] = np.max(t_stats)
    
    return features

# Extract features for all subjects
all_features = []

for subject_id in range(1, N_SUBJECTS + 1):
    subject_key = f'sub-{subject_id:02d}'
    print(f"\nSubject {subject_id:02d}:", end=" ")
    
    run_features = []
    
    for run in ['run1', 'run2']:
        trials = trial_data[subject_key][run]
        
        if trials and 'congruent' in trials and 'incongruent' in trials:
            features = extract_features(trials['congruent'], trials['incongruent'])
            
            if features:
                features['subject_id'] = subject_id
                features['run'] = run
                run_features.append(features)
                print(f"[{run} ✓]", end=" ")
    
    if run_features:
        # Average features across runs
        avg_features = {'subject_id': subject_id}
        feature_keys = [k for k in run_features[0].keys() if k not in ['subject_id', 'run']]
        
        for key in feature_keys:
            values = [f[key] for f in run_features]
            avg_features[key] = np.mean(values)
        
        all_features.append(avg_features)
        print(f"→ {len(feature_keys)} features extracted")

# Create feature dataframe
df_features = pd.DataFrame(all_features)

# Save features
csv_path = os.path.join(OUTPUT_DIR, "extracted_features.csv")
df_features.to_csv(csv_path, index=False)
print(f"\n✓ Saved features: {csv_path}")

# Visualize feature distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

key_features = [
    'mean_abs_diff', 'max_abs_diff', 'correlation',
    'cong_range', 'incong_range', 'range_ratio',
    'cong_std', 'incong_std', 'mean_t_stat'
]

for idx, feature in enumerate(key_features):
    if idx < len(axes):
        axes[idx].hist(df_features[feature], bins=15, color='steelblue', 
                      edgecolor='black', alpha=0.7)
        axes[idx].axvline(df_features[feature].mean(), color='red', 
                         linestyle='--', linewidth=2, label=f'Mean: {df_features[feature].mean():.3f}')
        axes[idx].set_xlabel(feature.replace('_', ' ').title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'feature_distributions.png')}")

# Feature correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
corr_matrix = df_features[key_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlations.png"), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'feature_correlations.png')}")

# Feature summary statistics
summary_stats = df_features[key_features].describe()
summary_stats.to_csv(os.path.join(OUTPUT_DIR, "feature_statistics.csv"))
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'feature_statistics.csv')}")

print("\n" + "="*80)
print("✓ FEATURE EXTRACTION COMPLETE")
print("="*80)
print(f"\nExtracted {len(key_features)} key features from {len(df_features)} subjects")