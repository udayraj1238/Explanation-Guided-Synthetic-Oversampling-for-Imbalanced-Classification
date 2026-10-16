# Attribution-Constrained SMOTE (AC-SMOTE)
**A Controlled Evaluation of Explanation-Guided Oversampling for Imbalanced Classification**

This repository contains the complete experimental code, raw benchmark data, statistical testing suites, and supplementary tables for the paper:
> *Attribution-Constrained SMOTE: A Controlled Evaluation of Explanation-Guided Oversampling for Imbalanced Classification* (CICT 2026, Paper ID: 758).

## Repository Structure
- `src/`: Core implementation of AC-SMOTE (`src/methods/ac_smote.py`), baselines, data loaders, evaluation pipelines, and statistical analysis scripts.
- `results/`: Complete 3,600-row 15-fold cross-validation results (`raw_results.csv`), multi-seed evaluations, ablation sweeps, calibration results, boundary fidelity logs, and statistical test outputs (`statistical_tests.json`).
- `supplement/`: Supplementary tables, paired Wilcoxon test results across all 5 metrics, rank-biserial effect sizes, Holm/BH-adjusted p-values, and equivalence testing outputs (`supplementary_tables.pdf`).
- `overleaf_project/`: LaTeX camera-ready source code, figures, and tables.

## Reproducibility & Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/udayraj1238/Explanation-Guided-Synthetic-Oversampling-for-Imbalanced-Classification.git
cd Explanation-Guided-Synthetic-Oversampling-for-Imbalanced-Classification
pip install -r requirements.txt
```

Run experiments or statistical tests:
```bash
# Verify environment and pipeline integrity
python sanity_check.py

# Run confirmatory statistical tests and corrections
python src/statistical_tests.py
```

## Note on Runtime Timings
In `results/raw_results.csv`, the timing columns (`fit_time_seconds`) for the six main methods reflect the total wall-clock duration of the resample-and-fit stage (resampling execution plus classifier training). In contrast, the released `src/experiment.py` script isolates and measures only the resampler invocation.
