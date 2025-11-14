"""
fMRI Signal Viewer
==================
Interactive visualization of fMRI signals for individual subjects
- View raw BOLD signal
- See preprocessed signals
- Explore trial-locked responses
- Compare congruent vs incongruent trials
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import nibabel as nib
from scipy import signal, stats
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
PREPROCESS_DIR = os.path.join(BASE_DIR, "analysis_results", "02_preprocessing")
TRIAL_DIR = os.path.join(BASE_DIR, "analysis_results", "03_trial_extraction")
OUTPUT_DIR = os.path.join(BASE_DIR, "signal_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subject to visualize (CHANGE THIS TO VIEW DIFFERENT SUBJECTS)
SUBJECT_ID = 1  # Change from 1 to 26
RUN = 1         # Change to 1 or 2

TR = 2.0  # Repetition time in seconds

print("="*80)
print(f"fMRI SIGNAL VISUALIZATION - SUBJECT {SUBJECT_ID:02d}, RUN {RUN}")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

# Load preprocessed signals
try:
    with open(os.path.join(PREPROCESS_DIR, "preprocessed_signals.pkl"), 'rb') as f:
        preprocessed_data = pickle.load(f)
    print("✓ Loaded preprocessed signals")
except:
    print("✗ Could not load preprocessed signals - run 02_signal_preprocessing.py first")
    preprocessed_data = None

# Load trial responses
try:
    with open(os.path.join(TRIAL_DIR, "trial_responses.pkl"), 'rb') as f:
        trial_data = pickle.load(f)
    print("✓ Loaded trial responses")
except:
    print("✗ Could not load trial responses - run 03_trial_extraction.py first")
    trial_data = None

# Load event file
events_path = os.path.join(
    BASE_DIR, 
    f"sub-{SUBJECT_ID:02d}/func/sub-{SUBJECT_ID:02d}_task-flanker_run-{RUN}_events.tsv"
)
events = pd.read_csv(events_path, sep='\t')
print(f"✓ Loaded {len(events)} events")

# Load raw BOLD data
bold_path = os.path.join(
    BASE_DIR,
    f"sub-{SUBJECT_ID:02d}/func/sub-{SUBJECT_ID:02d}_task-flanker_run-{RUN}_bold.nii.gz"
)
img = nib.load(bold_path)
bold_data = img.get_fdata()
print(f"✓ Loaded BOLD data: {bold_data.shape}")

# ============================================================================
# EXTRACT SIGNALS
# ============================================================================

# Extract whole-brain average (raw)
brain_mask = bold_data.mean(axis=3) > (bold_data.mean() * 0.1)
n_timepoints = bold_data.shape[3]
raw_signal = np.zeros(n_timepoints)

for t in range(n_timepoints):
    masked_data = bold_data[:, :, :, t][brain_mask]
    raw_signal[t] = np.mean(masked_data)

print(f"✓ Extracted raw signal: {len(raw_signal)} timepoints")

# Get preprocessed signals
subject_key = f'sub-{SUBJECT_ID:02d}'
if preprocessed_data:
    signals = preprocessed_data[subject_key][f'run{RUN}']
    detrended_signal = signals['detrended']
    filtered_signal = signals['filtered']
    normalized_signal = signals['normalized']
    print("✓ Retrieved preprocessed signals")
else:
    detrended_signal = signal.detrend(raw_signal)
    filtered_signal = detrended_signal  # Skip filtering for quick view
    normalized_signal = stats.zscore(filtered_signal)
    print("⚠ Using quick preprocessing (no filtering)")

# Get trial responses
if trial_data:
    trial_responses = trial_data[subject_key][f'run{RUN}']
    congruent_trials = trial_responses['congruent']
    incongruent_trials = trial_responses['incongruent']
    print(f"✓ Retrieved {len(congruent_trials)} congruent, {len(incongruent_trials)} incongruent trials")
else:
    congruent_trials = None
    incongruent_trials = None

# ============================================================================
# COMPREHENSIVE VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

fig.suptitle(f'fMRI Signal Analysis: Subject {SUBJECT_ID:02d}, Run {RUN}', 
             fontsize=18, fontweight='bold')

time_axis = np.arange(len(raw_signal)) * TR

# ============================================================================
# ROW 1: RAW AND PREPROCESSED SIGNALS
# ============================================================================

# Plot 1: Raw BOLD Signal
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_axis, raw_signal, 'b-', linewidth=1.2, alpha=0.7)
ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('BOLD Intensity (a.u.)', fontsize=11, fontweight='bold')
ax1.set_title('Raw BOLD Signal (Whole-Brain Average)', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Add event markers
for idx, row in events.iterrows():
    onset = row['onset']
    trial_type = str(row['trial_type']).lower()
    if 'congruent' in trial_type and 'incongruent' not in trial_type:
        color = 'green'
        marker = '▼'
    else:
        color = 'red'
        marker = '▲'
    ax1.axvline(x=onset, color=color, alpha=0.3, linestyle='--', linewidth=1)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linestyle='--', alpha=0.5, label='Congruent Trial'),
    Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Incongruent Trial')
]
ax1.legend(handles=legend_elements, loc='upper right')

# Plot 2: Detrended Signal
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_axis, detrended_signal, 'g-', linewidth=1)
ax2.set_xlabel('Time (seconds)', fontsize=10)
ax2.set_ylabel('Intensity', fontsize=10)
ax2.set_title('After Detrending', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 3: Filtered Signal
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_axis, filtered_signal, 'orange', linewidth=1)
ax3.set_xlabel('Time (seconds)', fontsize=10)
ax3.set_ylabel('Intensity', fontsize=10)
ax3.set_title('After Bandpass Filter (0.01-0.1 Hz)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 4: Normalized Signal
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_axis, normalized_signal, 'r-', linewidth=1)
ax4.set_xlabel('Time (seconds)', fontsize=10)
ax4.set_ylabel('Z-score', fontsize=10)
ax4.set_title('Final Normalized Signal', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
ax4.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)

# ============================================================================
# ROW 2: FREQUENCY ANALYSIS
# ============================================================================

# Plot 5: Power Spectrum (Raw)
ax5 = fig.add_subplot(gs[2, 0])
freqs_raw = np.fft.rfftfreq(len(raw_signal), TR)
fft_raw = np.abs(np.fft.rfft(raw_signal))
ax5.semilogy(freqs_raw, fft_raw, 'b-', linewidth=2)
ax5.set_xlabel('Frequency (Hz)', fontsize=10)
ax5.set_ylabel('Power', fontsize=10)
ax5.set_title('Raw Signal - Power Spectrum', fontsize=12, fontweight='bold')
ax5.axvspan(0.01, 0.1, alpha=0.2, color='green', label='Neural Band')
ax5.grid(alpha=0.3)
ax5.legend()
ax5.set_xlim([0, 0.25])

# Plot 6: Power Spectrum (Filtered)
ax6 = fig.add_subplot(gs[2, 1])
freqs_filt = np.fft.rfftfreq(len(filtered_signal), TR)
fft_filt = np.abs(np.fft.rfft(filtered_signal))
ax6.semilogy(freqs_filt, fft_filt, 'orange', linewidth=2)
ax6.set_xlabel('Frequency (Hz)', fontsize=10)
ax6.set_ylabel('Power', fontsize=10)
ax6.set_title('Filtered Signal - Power Spectrum', fontsize=12, fontweight='bold')
ax6.axvspan(0.01, 0.1, alpha=0.2, color='green', label='Neural Band')
ax6.grid(alpha=0.3)
ax6.legend()
ax6.set_xlim([0, 0.25])

# Plot 7: Signal Statistics
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

stats_text = f"""
SIGNAL STATISTICS
==================

Timepoints: {len(raw_signal)}
Duration: {len(raw_signal) * TR:.1f} sec
TR: {TR} sec

Raw Signal:
  Mean: {np.mean(raw_signal):.2f}
  Std:  {np.std(raw_signal):.2f}
  Range: {np.ptp(raw_signal):.2f}

Normalized Signal:
  Mean: {np.mean(normalized_signal):.3f}
  Std:  {np.std(normalized_signal):.3f}
  Range: {np.ptp(normalized_signal):.2f}

Trials:
  Congruent: {len(events[events['trial_type'].str.contains('congruent', case=False) & ~events['trial_type'].str.contains('incongruent', case=False)])}
  Incongruent: {len(events[events['trial_type'].str.contains('incongruent', case=False)])}
  Total: {len(events)}
"""

ax7.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round',
         facecolor='lightblue', alpha=0.3))

# ============================================================================
# ROW 3: TRIAL-LOCKED RESPONSES
# ============================================================================

if congruent_trials is not None and incongruent_trials is not None:
    trial_time = np.arange(6) * TR  # 6 TRs = 12 seconds
    
    # Plot 8: Congruent Trials
    ax8 = fig.add_subplot(gs[3, 0])
    for trial in congruent_trials:
        ax8.plot(trial_time, trial, 'g-', alpha=0.15, linewidth=0.8)
    ax8.plot(trial_time, np.mean(congruent_trials, axis=0), 'darkgreen', 
             linewidth=3, label='Mean', marker='o', markersize=6)
    ax8.fill_between(trial_time,
                      np.mean(congruent_trials, axis=0) - np.std(congruent_trials, axis=0),
                      np.mean(congruent_trials, axis=0) + np.std(congruent_trials, axis=0),
                      alpha=0.3, color='green')
    ax8.set_xlabel('Time from trial onset (sec)', fontsize=10)
    ax8.set_ylabel('Z-score', fontsize=10)
    ax8.set_title(f'Congruent Trials (n={len(congruent_trials)})', 
                  fontsize=12, fontweight='bold', color='darkgreen')
    ax8.legend()
    ax8.grid(alpha=0.3)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 9: Incongruent Trials
    ax9 = fig.add_subplot(gs[3, 1])
    for trial in incongruent_trials:
        ax9.plot(trial_time, trial, 'r-', alpha=0.15, linewidth=0.8)
    ax9.plot(trial_time, np.mean(incongruent_trials, axis=0), 'darkred',
             linewidth=3, label='Mean', marker='o', markersize=6)
    ax9.fill_between(trial_time,
                      np.mean(incongruent_trials, axis=0) - np.std(incongruent_trials, axis=0),
                      np.mean(incongruent_trials, axis=0) + np.std(incongruent_trials, axis=0),
                      alpha=0.3, color='red')
    ax9.set_xlabel('Time from trial onset (sec)', fontsize=10)
    ax9.set_ylabel('Z-score', fontsize=10)
    ax9.set_title(f'Incongruent Trials (n={len(incongruent_trials)})',
                  fontsize=12, fontweight='bold', color='darkred')
    ax9.legend()
    ax9.grid(alpha=0.3)
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 10: Direct Comparison
    ax10 = fig.add_subplot(gs[3, 2])
    cong_mean = np.mean(congruent_trials, axis=0)
    incong_mean = np.mean(incongruent_trials, axis=0)
    
    ax10.plot(trial_time, cong_mean, 'g-', linewidth=3, 
              label='Congruent', marker='o', markersize=7)
    ax10.plot(trial_time, incong_mean, 'r-', linewidth=3,
              label='Incongruent', marker='s', markersize=7)
    ax10.fill_between(trial_time, cong_mean, incong_mean, alpha=0.2, color='gray')
    ax10.set_xlabel('Time from trial onset (sec)', fontsize=10)
    ax10.set_ylabel('Z-score', fontsize=10)
    ax10.set_title('Direct Comparison', fontsize=12, fontweight='bold')
    ax10.legend()
    ax10.grid(alpha=0.3)
    ax10.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Calculate differentiation
    diff = np.mean(np.abs(cong_mean - incong_mean))
    ax10.text(0.5, 0.95, f'Mean |Difference|: {diff:.3f}',
              transform=ax10.transAxes, fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
              verticalalignment='top')

else:
    # Placeholder if trial data not available
    ax8 = fig.add_subplot(gs[3, :])
    ax8.text(0.5, 0.5, 'Trial-locked responses not available\nRun 03_trial_extraction.py first',
             ha='center', va='center', fontsize=14, color='red',
             transform=ax8.transAxes)
    ax8.axis('off')

# ============================================================================
# SAVE FIGURE
# ============================================================================

output_path = os.path.join(OUTPUT_DIR, f"subject_{SUBJECT_ID:02d}_run_{RUN}_complete_view.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comprehensive visualization: {output_path}")

plt.show()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nTo view different subjects, change:")
print(f"  SUBJECT_ID = {SUBJECT_ID}  (change to 1-26)")
print(f"  RUN = {RUN}  (change to 1 or 2)")