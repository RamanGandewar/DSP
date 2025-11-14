"""
02_signal_preprocessing.py
==========================
Extract and preprocess fMRI signals
- Load BOLD data
- Extract brain timeseries
- Apply detrending
- Bandpass filtering
- Normalization
- Save preprocessed signals
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal, stats
import matplotlib.pyplot as plt
import pickle

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "02_preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUBJECTS = 26
TR = 2.0  # Repetition Time in seconds

print("="*80)
print("STEP 2: SIGNAL PREPROCESSING")
print("="*80)

def preprocess_bold_signal(bold_data):
    """
    Complete preprocessing pipeline for BOLD signal
    """
    # Extract whole-brain average
    brain_mask = bold_data.mean(axis=3) > (bold_data.mean() * 0.1)
    n_timepoints = bold_data.shape[3]
    timeseries = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        masked_data = bold_data[:, :, :, t][brain_mask]
        timeseries[t] = np.mean(masked_data)
    
    # Step 1: Detrending (remove linear drift)
    detrended = signal.detrend(timeseries)
    
    # Step 2: Bandpass filtering (0.01 - 0.1 Hz)
    fs = 1.0 / TR
    nyquist = fs / 2.0
    low_freq = 0.01 / nyquist
    high_freq = 0.1 / nyquist
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    filtered = signal.filtfilt(b, a, detrended)
    
    # Step 3: Z-score normalization
    normalized = stats.zscore(filtered)
    
    return {
        'raw': timeseries,
        'detrended': detrended,
        'filtered': filtered,
        'normalized': normalized
    }

# Process all subjects
preprocessed_data = {}

for subject_id in range(1, N_SUBJECTS + 1):
    print(f"\nProcessing Subject {subject_id:02d}:")
    
    subject_data = {'run1': None, 'run2': None}
    
    for run in [1, 2]:
        try:
            bold_path = os.path.join(
                BASE_DIR,
                f"sub-{subject_id:02d}/func/sub-{subject_id:02d}_task-flanker_run-{run}_bold.nii.gz"
            )
            
            img = nib.load(bold_path)
            bold_data = img.get_fdata()
            
            processed = preprocess_bold_signal(bold_data)
            subject_data[f'run{run}'] = processed
            
            print(f"  ✓ Run {run}: {bold_data.shape[3]} timepoints processed")
            
        except Exception as e:
            print(f"  ✗ Run {run}: Failed - {str(e)}")
    
    preprocessed_data[f'sub-{subject_id:02d}'] = subject_data

# Save preprocessed data
pickle_path = os.path.join(OUTPUT_DIR, "preprocessed_signals.pkl")
with open(pickle_path, 'wb') as f:
    pickle.dump(preprocessed_data, f)
print(f"\n✓ Saved preprocessed signals: {pickle_path}")

# Visualize preprocessing steps for sample subject
sample_subject = 'sub-01'
sample_run = 'run1'
sample_data = preprocessed_data[sample_subject][sample_run]

if sample_data:
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    time = np.arange(len(sample_data['raw'])) * TR
    
    # Raw signal
    axes[0, 0].plot(time, sample_data['raw'], 'b-', linewidth=0.8)
    axes[0, 0].set_title('1. Raw BOLD Signal', fontweight='bold')
    axes[0, 0].set_ylabel('Intensity (a.u.)')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(sample_data['raw'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Raw Signal Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Frequency')
    
    # Detrended signal
    axes[1, 0].plot(time, sample_data['detrended'], 'g-', linewidth=0.8)
    axes[1, 0].set_title('2. After Detrending', fontweight='bold')
    axes[1, 0].set_ylabel('Intensity (a.u.)')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(sample_data['detrended'], bins=50, color='green', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Detrended Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')
    
    # Filtered signal
    axes[2, 0].plot(time, sample_data['filtered'], 'orange', linewidth=0.8)
    axes[2, 0].set_title('3. After Bandpass Filtering (0.01-0.1 Hz)', fontweight='bold')
    axes[2, 0].set_ylabel('Intensity (a.u.)')
    axes[2, 0].grid(alpha=0.3)
    
    axes[2, 1].hist(sample_data['filtered'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    axes[2, 1].set_title('Filtered Distribution', fontweight='bold')
    axes[2, 1].set_xlabel('Intensity')
    axes[2, 1].set_ylabel('Frequency')
    
    # Normalized signal
    axes[3, 0].plot(time, sample_data['normalized'], 'r-', linewidth=0.8)
    axes[3, 0].set_title('4. After Z-score Normalization (Final)', fontweight='bold')
    axes[3, 0].set_ylabel('Z-score')
    axes[3, 0].set_xlabel('Time (seconds)')
    axes[3, 0].grid(alpha=0.3)
    axes[3, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    axes[3, 1].hist(sample_data['normalized'], bins=50, color='red', edgecolor='black', alpha=0.7)
    axes[3, 1].set_title('Normalized Distribution (μ=0, σ=1)', fontweight='bold')
    axes[3, 1].set_xlabel('Z-score')
    axes[3, 1].set_ylabel('Frequency')
    axes[3, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "preprocessing_steps.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'preprocessing_steps.png')}")

# Generate preprocessing summary
summary = {
    'total_subjects': N_SUBJECTS,
    'successful_subjects': len([s for s in preprocessed_data.values() if s['run1'] or s['run2']]),
    'preprocessing_steps': [
        '1. Brain mask extraction',
        '2. Whole-brain average timeseries',
        '3. Linear detrending',
        '4. Bandpass filter (0.01-0.1 Hz)',
        '5. Z-score normalization'
    ],
    'sampling_rate': f'{1/TR} Hz',
    'TR': f'{TR} seconds'
}

with open(os.path.join(OUTPUT_DIR, "preprocessing_summary.txt"), 'w') as f:
    f.write("PREPROCESSING SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print("\n" + "="*80)
print("✓ SIGNAL PREPROCESSING COMPLETE")
print("="*80)