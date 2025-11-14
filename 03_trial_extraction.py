"""
03_trial_extraction.py
======================
Extract trial-locked responses
- Load preprocessed signals
- Load event timing
- Extract response windows
- Separate congruent vs incongruent
- Visualize trial responses
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

BASE_DIR = r"C:\Users\HP\Desktop\SEM 5\DSP\CP\Dataset"
PREPROCESS_DIR = os.path.join(BASE_DIR, "analysis_results", "02_preprocessing")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results", "03_trial_extraction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUBJECTS = 26
TR = 2.0
WINDOW_TRS = 6  # 12 seconds window

print("="*80)
print("STEP 3: TRIAL-LOCKED RESPONSE EXTRACTION")
print("="*80)

# Load preprocessed signals
with open(os.path.join(PREPROCESS_DIR, "preprocessed_signals.pkl"), 'rb') as f:
    preprocessed_data = pickle.load(f)

def extract_trial_responses(normalized_signal, events_df):
    """
    Extract trial-locked responses
    """
    congruent_responses = []
    incongruent_responses = []
    
    for idx, row in events_df.iterrows():
        onset_tr = int(row['onset'] / TR)
        
        if onset_tr + WINDOW_TRS <= len(normalized_signal):
            response = normalized_signal[onset_tr:onset_tr + WINDOW_TRS]
            trial_type = str(row['trial_type']).lower()
            
            if 'congruent' in trial_type and 'incongruent' not in trial_type:
                congruent_responses.append(response)
            elif 'incongruent' in trial_type:
                incongruent_responses.append(response)
    
    return np.array(congruent_responses), np.array(incongruent_responses)

# Extract trials for all subjects
trial_data = {}

for subject_id in range(1, N_SUBJECTS + 1):
    print(f"\nSubject {subject_id:02d}:")
    
    subject_key = f'sub-{subject_id:02d}'
    trial_data[subject_key] = {'run1': {}, 'run2': {}}
    
    for run in [1, 2]:
        try:
            # Load events
            events_path = os.path.join(
                BASE_DIR,
                f"sub-{subject_id:02d}/func/sub-{subject_id:02d}_task-flanker_run-{run}_events.tsv"
            )
            events = pd.read_csv(events_path, sep='\t')
            
            # Get preprocessed signal
            signal_data = preprocessed_data[subject_key][f'run{run}']
            
            if signal_data:
                normalized_signal = signal_data['normalized']
                cong, incong = extract_trial_responses(normalized_signal, events)
                
                trial_data[subject_key][f'run{run}'] = {
                    'congruent': cong,
                    'incongruent': incong,
                    'n_congruent': len(cong),
                    'n_incongruent': len(incong)
                }
                
                print(f"  Run {run}: {len(cong)} congruent, {len(incong)} incongruent trials")
            else:
                print(f"  Run {run}: No preprocessed data")
                
        except Exception as e:
            print(f"  Run {run}: Failed - {str(e)}")

# Save trial data
pickle_path = os.path.join(OUTPUT_DIR, "trial_responses.pkl")
with open(pickle_path, 'wb') as f:
    pickle.dump(trial_data, f)
print(f"\n✓ Saved trial responses: {pickle_path}")

# Visualize trial responses for sample subject
sample_subject = 'sub-01'
sample_run = 'run1'
sample_trials = trial_data[sample_subject][sample_run]

if sample_trials:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    time_axis = np.arange(WINDOW_TRS) * TR
    
    # Congruent trials
    cong_data = sample_trials['congruent']
    axes[0, 0].plot(time_axis, cong_data.T, 'g-', alpha=0.3, linewidth=0.5)
    axes[0, 0].plot(time_axis, cong_data.mean(axis=0), 'darkgreen', linewidth=3, label='Mean')
    axes[0, 0].set_title('Congruent Trials - All Responses', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Z-score')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    axes[0, 1].plot(time_axis, cong_data.mean(axis=0), 'darkgreen', linewidth=3, marker='o')
    axes[0, 1].fill_between(time_axis, 
                            cong_data.mean(axis=0) - cong_data.std(axis=0),
                            cong_data.mean(axis=0) + cong_data.std(axis=0),
                            alpha=0.3, color='green')
    axes[0, 1].set_title('Congruent - Mean ± SD', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Z-score')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    axes[0, 2].imshow(cong_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[0, 2].set_title('Congruent - Trial Heatmap', fontweight='bold', fontsize=12)
    axes[0, 2].set_xlabel('Time (TRs)')
    axes[0, 2].set_ylabel('Trial Number')
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2], label='Z-score')
    
    # Incongruent trials
    incong_data = sample_trials['incongruent']
    axes[1, 0].plot(time_axis, incong_data.T, 'r-', alpha=0.3, linewidth=0.5)
    axes[1, 0].plot(time_axis, incong_data.mean(axis=0), 'darkred', linewidth=3, label='Mean')
    axes[1, 0].set_title('Incongruent Trials - All Responses', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Z-score')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    axes[1, 1].plot(time_axis, incong_data.mean(axis=0), 'darkred', linewidth=3, marker='o')
    axes[1, 1].fill_between(time_axis,
                            incong_data.mean(axis=0) - incong_data.std(axis=0),
                            incong_data.mean(axis=0) + incong_data.std(axis=0),
                            alpha=0.3, color='red')
    axes[1, 1].set_title('Incongruent - Mean ± SD', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Z-score')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    axes[1, 2].imshow(incong_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    axes[1, 2].set_title('Incongruent - Trial Heatmap', fontweight='bold', fontsize=12)
    axes[1, 2].set_xlabel('Time (TRs)')
    axes[1, 2].set_ylabel('Trial Number')
    plt.colorbar(axes[1, 2].images[0], ax=axes[1, 2], label='Z-score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trial_responses_visualization.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'trial_responses_visualization.png')}")

# Generate extraction summary
total_trials = 0
for subj in trial_data.values():
    for run in ['run1', 'run2']:
        if subj[run]:
            total_trials += subj[run].get('n_congruent', 0)
            total_trials += subj[run].get('n_incongruent', 0)

summary_text = f"""
TRIAL EXTRACTION SUMMARY
========================

Total Subjects Processed: {N_SUBJECTS}
Extraction Window: {WINDOW_TRS} TRs ({WINDOW_TRS * TR} seconds)
Total Trials Extracted: {total_trials}

Trial Types:
- Congruent: Easy trials (→→→→→)
- Incongruent: Hard trials (→→←→→)

Next Step:
Compute cognitive intelligence scores based on:
1. Signal differentiation between trial types
2. Response pattern quality
3. Cognitive efficiency
"""

with open(os.path.join(OUTPUT_DIR, "extraction_summary.txt"), 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\n" + "="*80)
print("✓ TRIAL EXTRACTION COMPLETE")
print("="*80)