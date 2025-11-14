"""
fMRI Signal Viewer (Batch Mode)
===============================
Processes fMRI BOLD signals for 26 subjects × 2 runs each:
- Loads raw BOLD + event files
- Computes raw, detrended, filtered, and normalized signals
- Generates full 10-panel visualization per subject/run
- Saves plots and .pkl signal files
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
OUTPUT_DIR = os.path.join(BASE_DIR, "signal_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TR = 2.0  # Repetition time (seconds)

# Load preprocessed data (optional)
preprocessed_path = os.path.join(BASE_DIR, "preprocessed_fmri_data.pkl")
preprocessed_data = None
if os.path.exists(preprocessed_path):
    with open(preprocessed_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    print("✓ Loaded preprocessed data")
else:
    print("⚠ No preprocessed data found. Using raw processing.")

# ============================================================================
# MAIN PROCESSING LOOP (All 26 subjects × 2 runs)
# ============================================================================
for SUBJECT_ID in range(1, 27):
    for RUN in [1, 2]:
        try:
            print("=" * 80)
            print(f"Processing Subject {SUBJECT_ID:02d}, Run {RUN}")
            print("=" * 80)

            # ---------------- LOAD DATA ----------------
            events_path = os.path.join(
                BASE_DIR,
                f"sub-{SUBJECT_ID:02d}/func/sub-{SUBJECT_ID:02d}_task-flanker_run-{RUN}_events.tsv"
            )
            if not os.path.exists(events_path):
                print(f"✗ Skipping (no events file found): {events_path}")
                continue
            events = pd.read_csv(events_path, sep='\t')

            bold_path = os.path.join(
                BASE_DIR,
                f"sub-{SUBJECT_ID:02d}/func/sub-{SUBJECT_ID:02d}_task-flanker_run-{RUN}_bold.nii.gz"
            )
            if not os.path.exists(bold_path):
                print(f"✗ Skipping (no BOLD file found): {bold_path}")
                continue

            img = nib.load(bold_path)
            bold_data = img.get_fdata()

            # ---------------- EXTRACT SIGNALS ----------------
            brain_mask = bold_data.mean(axis=3) > (bold_data.mean() * 0.1)
            n_timepoints = bold_data.shape[3]
            raw_signal = np.zeros(n_timepoints)
            for t in range(n_timepoints):
                raw_signal[t] = np.mean(bold_data[:, :, :, t][brain_mask])

            subject_key = f'sub-{SUBJECT_ID:02d}'
            if preprocessed_data and subject_key in preprocessed_data:
                signals = preprocessed_data[subject_key].get(f'run{RUN}', {})
                detrended_signal = signals.get('detrended', signal.detrend(raw_signal))
                filtered_signal = signals.get('filtered', detrended_signal)
                normalized_signal = signals.get('normalized', stats.zscore(filtered_signal))
            else:
                detrended_signal = signal.detrend(raw_signal)
                filtered_signal = detrended_signal
                normalized_signal = stats.zscore(filtered_signal)

            # ---------------- SAVE SIGNALS ----------------
            subject_signal_dir = os.path.join(OUTPUT_DIR, "individual_signals")
            os.makedirs(subject_signal_dir, exist_ok=True)
            signal_dict = {
                'subject_id': SUBJECT_ID,
                'run': RUN,
                'TR': TR,
                'raw_signal': raw_signal,
                'detrended_signal': detrended_signal,
                'filtered_signal': filtered_signal,
                'normalized_signal': normalized_signal
            }
            signal_save_path = os.path.join(subject_signal_dir, f"subject_{SUBJECT_ID:02d}_run_{RUN}_signals.pkl")
            with open(signal_save_path, 'wb') as f:
                pickle.dump(signal_dict, f)

            # ---------------- PLOTTING ----------------
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(5, 2, figure=fig)
            time_axis = np.arange(len(raw_signal)) * TR

            # Raw signal
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(time_axis, raw_signal, color='gray', linewidth=1.2)
            ax1.set_title("Raw BOLD Signal")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Signal Intensity")
            ax1.grid(alpha=0.3)

            # Detrended
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(time_axis, detrended_signal, color='blue', linewidth=1.2)
            ax2.set_title("Detrended Signal")
            ax2.grid(alpha=0.3)

            # Filtered
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(time_axis, filtered_signal, color='green', linewidth=1.2)
            ax3.set_title("Filtered Signal")
            ax3.grid(alpha=0.3)

            # Normalized
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(time_axis, normalized_signal, color='red', linewidth=1.2)
            ax4.set_title("Normalized Signal (Z-score)")
            ax4.grid(alpha=0.3)

            # Events overlay
            ax5 = fig.add_subplot(gs[2, :])
            ax5.plot(time_axis, normalized_signal, color='black', linewidth=1.0)
            for _, event in events.iterrows():
                onset, duration = event['onset'], event['duration']
                ax5.axvspan(onset, onset + duration, alpha=0.3, color='orange')
            ax5.set_title("Event Overlay (Trial Windows)")
            ax5.set_xlabel("Time (s)")
            ax5.grid(alpha=0.3)

            # Frequency domain
            ax6 = fig.add_subplot(gs[3, 0])
            freqs, psd = signal.welch(normalized_signal, fs=1/TR)
            ax6.semilogy(freqs, psd, color='purple')
            ax6.set_title("Frequency Spectrum")
            ax6.set_xlabel("Frequency (Hz)")
            ax6.set_ylabel("Power Spectral Density")
            ax6.grid(alpha=0.3)

            # Trial responses
            trial_responses = []
            for _, event in events.iterrows():
                onset_idx = int(event['onset'] / TR)
                duration_idx = int(event['duration'] / TR)
                trial_signal = normalized_signal[onset_idx:onset_idx + duration_idx]
                if len(trial_signal) > 0:
                    trial_responses.append(trial_signal)
            if len(trial_responses) > 0:
                mean_trial = np.mean(np.vstack([np.pad(trial, (0, max(map(len, trial_responses)) - len(trial)), constant_values=np.nan) for trial in trial_responses]), axis=0)
                ax7 = fig.add_subplot(gs[3, 1])
                ax7.plot(mean_trial, color='teal')
                ax7.set_title("Mean Trial-Locked Response")
                ax7.grid(alpha=0.3)

            # Summary comparison placeholder
            ax8 = fig.add_subplot(gs[4, :])
            ax8.plot(time_axis, normalized_signal, 'k-', alpha=0.7)
            ax8.set_title(f"Subject {SUBJECT_ID:02d} - Run {RUN}: Overall Signal")
            ax8.set_xlabel("Time (s)")
            ax8.grid(alpha=0.3)

            fig.suptitle(f"fMRI Signal Visualization – Subject {SUBJECT_ID:02d}, Run {RUN}", fontsize=16, weight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # ---------------- SAVE FIGURE ----------------
            output_path = os.path.join(OUTPUT_DIR, f"subject_{SUBJECT_ID:02d}_run_{RUN}_complete_view.png")
            plt.savefig(output_path, dpi=250, bbox_inches='tight')
            plt.close(fig)

            print(f"✓ Saved figure and signals for Subject {SUBJECT_ID:02d}, Run {RUN}")

        except Exception as e:
            print(f"✗ Error processing Subject {SUBJECT_ID:02d}, Run {RUN}: {e}")
            continue

print("\n✅ Processing completed for all subjects and runs!")
