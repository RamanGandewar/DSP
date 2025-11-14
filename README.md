# fMRI Cognitive Intelligence Analysis Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **A comprehensive Digital Signal Processing (DSP) framework for quantifying cognitive intelligence from fMRI BOLD signals using the Flanker cognitive control task.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output Files](#output-files)
- [Results & Validation](#results--validation)
- [Algorithm Details](#algorithm-details)
- [Visualization Gallery](#visualization-gallery)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a **novel multi-component cognitive intelligence scoring framework** that analyzes functional Magnetic Resonance Imaging (fMRI) data to quantify individual differences in neural efficiency and cognitive control. Unlike traditional behavioral measures or complex deep learning approaches, our framework provides:

- **Interpretable 0-100 scale cognitive scores** with clinical applicability
- **Three-component assessment**: Neural differentiation (40 pts), Pattern quality (30 pts), Cognitive efficiency (30 pts)
- **Validated classification**: HIGH (≥65), AVERAGE (45-65), LOW (<45) intelligence categories
- **Robust reliability**: Cross-run correlation r=0.78 (p<0.001)
- **Strong discrimination**: Cohen's d=2.89 between HIGH and LOW performers

### Research Context

**Published in**: IEEE Conference Format  
**Authors**: Raman Gandewar, Prathamesh Ghalsasi, Abha Marathe, Medha Wyawahare  
**Institution**: Vishwakarma Institute of Technology, Pune, India  
**Department**: Electronics and Telecommunication Engineering

---

## ✨ Key Features

### 🔬 Scientific Capabilities

- **Standardized Preprocessing Pipeline**
  - Linear detrending to remove scanner drift
  - Butterworth bandpass filtering (0.01-0.1 Hz) for neural signal isolation
  - Z-score normalization for inter-subject comparability
  - Brain mask extraction for whole-brain averaging

- **Advanced Feature Engineering**
  - 18 neurophysiologically-motivated features
  - Signal differentiation metrics (mean/max absolute difference, correlation)
  - Hemodynamic response characteristics (amplitude, range, peak timing)
  - Cognitive efficiency ratios (resource allocation patterns)
  - Trial consistency measures (within-condition variability)
  - Statistical significance tests (t-statistics across timepoints)

- **Multi-Component Scoring Algorithm**
  - Differentiation Score (40 points): Neural discrimination capability
  - Pattern Quality Score (30 points): Hemodynamic response robustness
  - Efficiency Score (30 points): Appropriate resource allocation
  - Composite integration with empirically-derived thresholds

- **Comprehensive Validation**
  - Cross-run consistency analysis
  - Inter-group statistical comparisons (t-tests, effect sizes)
  - Outlier detection using IQR method
  - Trial-type discrimination performance metrics (83.7% accuracy)
  - Feature correlation analysis

### 💻 Technical Features

- **Modular Architecture**: Seven independent pipeline stages
- **BIDS Compatibility**: Works with Brain Imaging Data Structure format
- **Extensive Visualization**: 15+ plot types across all analysis stages
- **Reproducible Results**: Deterministic processing with saved intermediate outputs
- **Batch Processing**: Handles multiple subjects and runs automatically
- **Quality Control**: Automated data completeness checks

---

## 🏗️ System Architecture

```
fMRI-Cognitive-Intelligence-Framework/
│
├── 01_dataset_exploration.py         # Stage 1: Data completeness & overview
├── 02_signal_preprocessing.py        # Stage 2: BOLD signal preprocessing
├── 03_trial_extraction.py            # Stage 3: Trial-locked response extraction
├── 04_feature_extraction.py          # Stage 4: Feature engineering (18 features)
├── 05_cognitive_scoring.py           # Stage 5: Multi-component scoring
├── 06_results_validation.py          # Stage 6: Statistical validation
├── 07_final_report_generation.py    # Stage 7: Comprehensive reporting
├── fmri_signal_viewer.py             # Interactive single-subject visualization
├── fmri_signal_viewer_all.py         # Batch visualization (all subjects)
│
├── Dataset/                          # BIDS-formatted fMRI data
│   ├── sub-01/
│   │   ├── anat/
│   │   │   └── sub-01_T1w.nii.gz
│   │   └── func/
│   │       ├── sub-01_task-flanker_run-1_bold.nii.gz
│   │       ├── sub-01_task-flanker_run-1_events.tsv
│   │       ├── sub-01_task-flanker_run-2_bold.nii.gz
│   │       └── sub-01_task-flanker_run-2_events.tsv
│   ├── sub-02/
│   ├── ...
│   └── sub-26/
│
└── analysis_results/                 # Generated outputs
    ├── 01_exploration/
    ├── 02_preprocessing/
    ├── 03_trial_extraction/
    ├── 04_features/
    ├── 05_scoring/
    ├── 06_validation/
    └── 07_final_report/
```

---

## 🔧 Installation

### Prerequisites

- **Python 3.8+** (recommended: 3.9 or 3.10)
- **Operating System**: Windows, Linux, or macOS
- **RAM**: Minimum 8GB (16GB recommended for batch processing)
- **Storage**: ~5GB for dataset + outputs

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/fmri-cognitive-intelligence.git
cd fmri-cognitive-intelligence
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv fmri_env
source fmri_env/bin/activate  # On Windows: fmri_env\Scripts\activate

# Or using conda
conda create -n fmri_env python=3.9
conda activate fmri_env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
nibabel>=3.2.0
scikit-learn>=0.24.0
openpyxl>=3.0.0
```

### Step 4: Verify Installation

```bash
python -c "import nibabel; import scipy; import pandas; print('All packages installed successfully!')"
```

---

## 📊 Dataset Requirements

### BIDS Format Structure

The framework expects data in **Brain Imaging Data Structure (BIDS)** format:

```
Dataset/
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz              # T1-weighted anatomical scan
│   └── func/
│       ├── sub-01_task-flanker_run-1_bold.nii.gz    # Functional BOLD (Run 1)
│       ├── sub-01_task-flanker_run-1_events.tsv     # Trial timing (Run 1)
│       ├── sub-01_task-flanker_run-2_bold.nii.gz    # Functional BOLD (Run 2)
│       └── sub-01_task-flanker_run-2_events.tsv     # Trial timing (Run 2)
```

### Event File Format

**Example events.tsv**:
```
onset	duration	trial_type
2.5	2.0	congruent
6.0	2.0	incongruent
10.5	2.0	congruent
```

---

## 🚀 Usage

### Quick Start (Complete Pipeline)

Run all seven stages sequentially:

```bash
python 01_dataset_exploration.py
python 02_signal_preprocessing.py
python 03_trial_extraction.py
python 04_feature_extraction.py
python 05_cognitive_scoring.py
python 06_results_validation.py
python 07_final_report_generation.py
```

### Automated Batch Script

```bash
#!/bin/bash
# run_complete_analysis.sh

stages=(
    "01_dataset_exploration.py"
    "02_signal_preprocessing.py"
    "03_trial_extraction.py"
    "04_feature_extraction.py"
    "05_cognitive_scoring.py"
    "06_results_validation.py"
    "07_final_report_generation.py"
)

for stage in "${stages[@]}"; do
    echo "Running $stage..."
    python "$stage"
    if [ $? -ne 0 ]; then
        echo "Error in $stage. Aborting."
        exit 1
    fi
done

echo "Pipeline completed successfully!"
```

---

## 📈 Pipeline Stages

### Scoring Algorithm

**Component 1: Differentiation (40 points)**:
```python
S_diff = min(50 × mean_abs_diff + 30 × max_abs_diff, 40)
```

**Component 2: Pattern Quality (30 points)**:
```python
S_pattern = min(50 × avg_std, 30)
```

**Component 3: Efficiency (30 points)**:
```python
if range_ratio > 1.0:
    S_eff = min(30 × (range_ratio - 1), 30)
else:
    S_eff = max(0, 15 - 20 × |range_ratio - 1|)
```

**Total Score**: 0-100 scale
```python
Score_total = S_diff + S_pattern + S_eff
```

---

## 📊 Results & Validation

### Population Results (N=26)

**Score Distribution**:
```
Mean Score:      57.23 ± 12.84
Median Score:    57.13
Range:           35.44 - 79.24
IQR:             [48.39, 63.21]
```

**Classification Breakdown**:
```
HIGH (≥65):      6 subjects (23.1%)
AVERAGE (45-65): 17 subjects (65.4%)
LOW (<45):       3 subjects (11.5%)
```

**Component Contributions**:
```
Differentiation: 31.48 ± 8.92 / 40 (78.7%)
Pattern Quality: 10.34 ± 4.67 / 30 (34.5%)
Efficiency:      15.41 ± 7.83 / 30 (51.4%)
```

### Validation Metrics

**Test-Retest Reliability**:
```
Pearson r:           0.78 (p < 0.001)
Mean absolute diff:  4.67 ± 3.21 points
ICC:                 0.76 (good reliability)
```

**Group Comparisons**:
```
HIGH vs LOW:     t(7) = 11.23, p < 0.001, d = 2.89 (very large effect)
```

**Trial Discrimination**:
```
Overall Accuracy:  83.7%
Sensitivity:       86.2%
Specificity:       81.5%
F1-Score:          85.2%
```

### Top 5 Performers

| Rank | Subject | Score | Differentiation | Pattern | Efficiency | Classification |
|------|---------|-------|----------------|---------|------------|----------------|
| 1    | 21      | 79.24 | 40.00          | 13.92   | 25.32      | HIGH          |
| 2    | 16      | 78.64 | 40.00          | 8.64    | 30.00      | HIGH          |
| 3    | 11      | 71.10 | 40.00          | 13.39   | 17.71      | HIGH          |
| 4    | 6       | 70.05 | 40.00          | 6.96    | 23.09      | HIGH          |
| 5    | 18      | 65.02 | 25.89          | 8.00    | 30.00      | HIGH          |

---

## 🧮 Algorithm Details

### Preprocessing Equations

**Brain Mask**:
```
M(x,y,z) = 1  if  Ī(x,y,z) > 0.1 × μ_global
           0  otherwise
```

**Whole-Brain Signal**:
```
S(t) = (1/|M|) × Σ I(x,y,z,t)  for all (x,y,z) ∈ M
```

**Detrending**:
```
S_detrend(t) = S(t) - (α̂t + β̂)
```

**Bandpass Filter** (Butterworth, 0.01-0.1 Hz):
```
S_filt = ButterworthBandpass(S_detrend, order=4, f_low=0.01, f_high=0.1)
```

**Z-score Normalization**:
```
S_norm(t) = (S_filt(t) - μ) / σ
```

### Feature Extraction

**18 Core Features**:
1. **Differentiation (3)**: mean_abs_diff, max_abs_diff, correlation
2. **Signal Characteristics (8)**: mean, std, range, peak (for both conditions)
3. **Efficiency (3)**: range_ratio, std_ratio, peak_ratio
4. **Temporal (3)**: cong_peak_time, incong_peak_time, peak_time_diff
5. **Consistency (2)**: cong_trial_std, incong_trial_std

---

## 🎨 Visualization Gallery

### Sample Outputs

#### 1. Dataset Overview
![Dataset Overview](docs/images/dataset_overview.png)
*Heatmap showing data completeness across 26 subjects with anatomical, functional, and event files.*

#### 2. Preprocessing Pipeline
![Preprocessing](docs/images/preprocessing_steps.png)
*Sequential transformation: Raw → Detrended → Filtered → Normalized signals with distributions.*

#### 3. Trial-Locked Responses
![Trial Responses](docs/images/trial_responses_visualization.png)
*Comparison of congruent (green) vs incongruent (red) hemodynamic responses showing mean ± SD.*

#### 4. Feature Distributions
![Features](docs/images/feature_distributions.png)
*Histograms of 9 key features with mean markers showing population variability.*

#### 5. Cognitive Scores
![Scores](docs/images/cognitive_scores_visualization.png)
*Complete rankings with color-coded classifications: HIGH (green), AVERAGE (yellow), LOW (red).*

#### 6. Validation Analysis
![Validation](docs/images/validation_analysis.png)
*Cross-run consistency (r=0.78), group comparisons, and outlier detection.*

#### 7. Comprehensive Dashboard
![Dashboard](docs/images/comprehensive_dashboard.png)
*Executive summary with rankings, distributions, components, and statistics.*

#### 8. Individual Subject Analysis
![Subject Analysis](docs/images/subject_21_run_1_complete_view.png)
*Complete signal analysis for HIGH performer showing preprocessing, frequency spectrum, and trial responses.*

---

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{gandewar2025fmri,
  title={Cognitive Intelligence Assessment Using fMRI Signal Analysis: A Novel Multi-Component Scoring Framework},
  author={Gandewar, Raman and Ghalsasi, Prathamesh and Marathe, Abha and Wyawahare, Medha},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  organization={Vishwakarma Institute of Technology}
}
```

**Research Paper**: [Link to IEEE Xplore / ArXiv]

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow PEP 8 style guidelines
   - Add docstrings to new functions
   - Update documentation as needed
4. **Test your changes**:
   ```bash
   python -m pytest tests/
   ```
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

### Contribution Areas

- **New Features**: Additional cognitive tasks, regional analysis, network metrics
- **Improvements**: Algorithm optimization, better visualizations
- **Bug Fixes**: Report issues or submit fixes
- **Documentation**: Tutorials, examples, translations
- **Testing**: Unit tests, integration tests, edge cases

### Code Style

```python
# Follow this style for consistency
def compute_feature(signal: np.ndarray, parameter: float = 1.0) -> float:
    """
    Brief description of function.
    
    Args:
        signal: Input BOLD timeseries (1D array)
        parameter: Scaling factor (default: 1.0)
    
    Returns:
        Computed feature value
    
    Example:
        >>> signal = np.random.randn(100)
        >>> feature = compute_feature(signal, parameter=2.0)
    """
    result = np.mean(signal) * parameter
    return result
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Raman Gandewar, Prathamesh Ghalsasi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact

### Authors

**Raman Gandewar**  
📧 Email: raman.gandewar231@vit.edu  
🎓 Institution: Vishwakarma Institute of Technology, Pune  
💼 Department: Electronics and Telecommunication Engineering

**Prathamesh Ghalsasi**  
📧 Email: prathamesh.ghalsasi23@vit.edu  
🎓 Institution: Vishwakarma Institute of Technology, Pune  
💼 Department: Electronics and Telecommunication Engineering

### Supervisors

**Dr. Abha Marathe**  
📧 Email: abha.marathe@vit.edu  
🎓 Professor, Electronics and Telecommunication Department

**Dr. Medha Wyawahare**  
📧 Email: medha.wyawahare@vit.edu  
🎓 Professor, Electronics and Telecommunication Department

### Repository

🔗 **GitHub**: [https://github.com/your-username/fmri-cognitive-intelligence](https://github.com/your-username/fmri-cognitive-intelligence)  
📝 **Issues**: [Report bugs or request features](https://github.com/your-username/fmri-cognitive-intelligence/issues)  
💬 **Discussions**: [Join the community discussion](https://github.com/your-username/fmri-cognitive-intelligence/discussions)

---

## 🙏 Acknowledgments

- **Vishwakarma Institute of Technology** for providing computational resources and research support
- **OpenNeuro** community for making neuroimaging datasets publicly available
- **BIDS** (Brain Imaging Data Structure) initiative for standardization
- **NiBabel**, **SciPy**, **NumPy**, **Pandas**, **Matplotlib**, and **Seaborn** development teams
- All contributors and users of this framework

---

## 📚 Additional Resources

### Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)
- [FAQ](docs/faq.md)

### Related Papers

1. Thomas et al. (2025). "An fMRI dataset for investigating language control and cognitive control in bilinguals." *Nature Scientific Data*.
2. Wang et al. (2023). "Explainable fMRI-based brain decoding via spatial temporal-pyramid graph convolutional network." *Human Brain Mapping*.
3. Wen et al. (2019). "Interpretable, highly accurate brain decoding using intrinsic functional networks and LSTM RNNs." *NeuroImage*.
4. Chatterjee et al. (2022). "Identifying depression using resting-state fMRI and machine learning." *Brain Sciences*.

### External Links

- [BIDS Specification](https://bids-specification.readthedocs.io/)
- [OpenNeuro](https://openneuro.org/)
- [NiBabel Documentation](https://nipy.org/nibabel/)
- [Human Connectome Project](https://www.humanconnectome.org/)
- [fMRIPrep](https://fmriprep.org/)

---

## 🔄 Version History

### Version 1.0.0 (2025-01-15)
- ✨ Initial release
- 🎯 Complete 7-stage analysis pipeline
- 📊 Multi-component scoring algorithm
- ✅ Validation with N=26 subjects
- 📈 Comprehensive visualization suite
- 📖 Full documentation and examples

### Roadmap

#### Version 1.1.0 (Planned)
- [ ] Regional analysis (prefrontal cortex, anterior cingulate, parietal)
- [ ] Network connectivity metrics
- [ ] Multi-task support (n-back, Stroop)
- [ ] Real-time processing module
- [ ] GUI application

#### Version 2.0.0 (Future)
- [ ] Deep learning integration (Graph Neural Networks)
- [ ] Multimodal fusion (fMRI + EEG)
- [ ] Clinical validation in patient populations
- [ ] Longitudinal tracking capabilities
- [ ] Cloud-based processing

---

## ⚠️ Disclaimer

This software is provided for **research and educational purposes only**. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

**Important Notes**:
- Ensure proper ethical approval and informed consent for human subject research
- Follow data protection regulations (HIPAA, GDPR) when handling medical data
- Validate results thoroughly before publication or clinical application
- The scoring thresholds are empirically derived and may require adjustment for different populations

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/fmri-cognitive-intelligence&type=Date)](https://star-history.com/#your-username/fmri-cognitive-intelligence&Date)

---

## 📈 Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/fmri-cognitive-intelligence)
![GitHub contributors](https://img.shields.io/github/contributors/your-username/fmri-cognitive-intelligence)
![GitHub stars](https://img.shields.io/github/stars/your-username/fmri-cognitive-intelligence?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/fmri-cognitive-intelligence?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/fmri-cognitive-intelligence)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-username/fmri-cognitive-intelligence)

---

