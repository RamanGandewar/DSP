"""
01_dataset_exploration.py
========================
Explore and visualize the fMRI dataset structure
- Dataset overview
- Subject distribution
- Data availability check
- Initial statistics
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from pathlib import Path

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "01_exploration")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STEP 1: DATASET EXPLORATION")
print("="*80)

# Check dataset structure
N_SUBJECTS = 26
missing_data = []
subject_info = []

for sub_id in range(1, N_SUBJECTS + 1):
    sub_dir = os.path.join(BASE_DIR, f"sub-{sub_id:02d}")
    
    info = {
        'subject_id': sub_id,
        'has_anat': False,
        'has_func_run1': False,
        'has_func_run2': False,
        'has_events_run1': False,
        'has_events_run2': False
    }
    
    # Check anatomical
    anat_file = os.path.join(sub_dir, "anat", f"sub-{sub_id:02d}_T1w.nii.gz")
    info['has_anat'] = os.path.exists(anat_file)
    
    # Check functional runs
    for run in [1, 2]:
        func_file = os.path.join(sub_dir, "func", f"sub-{sub_id:02d}_task-flanker_run-{run}_bold.nii.gz")
        events_file = os.path.join(sub_dir, "func", f"sub-{sub_id:02d}_task-flanker_run-{run}_events.tsv")
        
        info[f'has_func_run{run}'] = os.path.exists(func_file)
        info[f'has_events_run{run}'] = os.path.exists(events_file)
    
    subject_info.append(info)

df_subjects = pd.DataFrame(subject_info)

# Print summary
print(f"\n✓ Total Subjects: {N_SUBJECTS}")
print(f"✓ Complete Anatomical Scans: {df_subjects['has_anat'].sum()}")
print(f"✓ Complete Functional Run 1: {df_subjects['has_func_run1'].sum()}")
print(f"✓ Complete Functional Run 2: {df_subjects['has_func_run2'].sum()}")

# Sample a subject's data dimensions
sample_bold = nib.load(os.path.join(BASE_DIR, "sub-01/func/sub-01_task-flanker_run-1_bold.nii.gz"))
bold_shape = sample_bold.shape
print(f"\n✓ BOLD Data Dimensions: {bold_shape}")
print(f"  - Spatial: {bold_shape[0]} x {bold_shape[1]} x {bold_shape[2]}")
print(f"  - Temporal: {bold_shape[3]} timepoints")

# Sample events file
sample_events = pd.read_csv(os.path.join(BASE_DIR, "sub-01/func/sub-01_task-flanker_run-1_events.tsv"), sep='\t')
print(f"\n✓ Trial Types Found: {sample_events['trial_type'].unique()}")
print(f"✓ Total Trials per Run: {len(sample_events)}")

# Visualize dataset completeness
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data Availability Heatmap
data_matrix = df_subjects.iloc[:, 1:].values.astype(int)
sns.heatmap(data_matrix.T, annot=False, cmap='RdYlGn', cbar_kws={'label': 'Available'},
            xticklabels=df_subjects['subject_id'], 
            yticklabels=['Anatomical', 'Func Run1', 'Func Run2', 'Events Run1', 'Events Run2'],
            ax=axes[0, 0])
axes[0, 0].set_title('Data Availability Across Subjects', fontweight='bold')
axes[0, 0].set_xlabel('Subject ID')

# Plot 2: Completeness Summary
completeness = df_subjects.iloc[:, 1:].sum()
axes[0, 1].bar(range(len(completeness)), completeness.values, color='steelblue', edgecolor='black')
axes[0, 1].set_xticks(range(len(completeness)))
axes[0, 1].set_xticklabels(completeness.index, rotation=45, ha='right')
axes[0, 1].set_ylabel('Number of Subjects')
axes[0, 1].set_title('Data Type Completeness', fontweight='bold')
axes[0, 1].axhline(y=N_SUBJECTS, color='green', linestyle='--', alpha=0.5)
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Trial Type Distribution
trial_counts = sample_events['trial_type'].value_counts()
axes[1, 0].bar(range(len(trial_counts)), trial_counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[1, 0].set_xticks(range(len(trial_counts)))
axes[1, 0].set_xticklabels(trial_counts.index, rotation=45, ha='right')
axes[1, 0].set_ylabel('Number of Trials')
axes[1, 0].set_title('Trial Types (Sample Subject)', fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Dataset Overview Text
axes[1, 1].axis('off')
overview_text = f"""
DATASET OVERVIEW
================

Total Subjects: {N_SUBJECTS}
Data Format: BIDS-compliant

Per Subject:
• 1 Anatomical Scan (T1w)
• 2 Functional Runs (BOLD)
• 2 Event Files (TSV)

BOLD Dimensions:
• Voxels: {bold_shape[0]}×{bold_shape[1]}×{bold_shape[2]}
• Timepoints: {bold_shape[3]}
• TR: 2.0 seconds

Task: Flanker Task
• Congruent Trials
• Incongruent Trials
• Tests cognitive control

Quality Control:
✓ MRIQC reports available
"""
axes[1, 1].text(0.1, 0.5, overview_text, fontsize=11, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dataset_overview.png"), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(OUTPUT_DIR, 'dataset_overview.png')}")

# Save subject info
df_subjects.to_csv(os.path.join(OUTPUT_DIR, "subject_data_availability.csv"), index=False)
print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'subject_data_availability.csv')}")

print("\n" + "="*80)
print("✓ DATASET EXPLORATION COMPLETE")
print("="*80)