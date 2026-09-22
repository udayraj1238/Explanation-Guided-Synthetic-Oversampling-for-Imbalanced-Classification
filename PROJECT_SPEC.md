AC-SMOTE Execution Guide
Explanation-Guided Synthetic Oversampling for Imbalanced Classification
Target: CICT-2026 (10th IEEE International Conference on ICT), Track 1 — AI, ML & Data Analytics Window: Sept 22 – Sept 30, 2026 (8 working days + 1 submission-buffer day)

One honest sentence before you start: following this guide exactly will give you a technically sound, well-benchmarked, honestly-reported paper with a real (if modest) novelty claim. That maximizes your odds. It does not guarantee acceptance — no plan can promise that for any peer-reviewed venue. Focus on making every number in the paper true and every claim defensible, and you'll have done everything within your control.

0. How To Use This Document With Antigravity
This document is the single source of truth for the project. It is written so that every design decision — file names, function signatures, algorithm formulas, dataset choices, hyperparameters, evaluation protocol — is already made. Antigravity's job is to implement exactly what is specified below, not to design the study.

Practical workflow:

Open this whole file in your project folder as PROJECT_SPEC.md (Step 1.3 tells you where).
At the start of each work session, tell Antigravity: "Read PROJECT_SPEC.md, specifically Section [N]. Implement exactly what it specifies. Do not change the algorithm, file structure, dataset list, or evaluation protocol without asking me first. If anything is ambiguous, stop and ask me rather than assuming."
Section 11 (Day-by-Day Schedule) contains ready-to-paste prompts for each phase — use them verbatim as your message to Antigravity.
Never let Antigravity (or yourself) write a results number, a citation, or a claim into the paper draft that hasn't actually been produced by running the code. This is the single most important rule in this document. Everything else is negotiable; this is not.
Table of Contents
Project Overview & Novelty Statement
Conference Formatting Requirements (verified)
Environment Setup (exact commands, Windows)
Project Directory Structure
Datasets — exact list, sources, download steps
The Method: Attribution-Constrained SMOTE (AC-SMOTE) — full formal spec
Baselines
Experimental Protocol (CV, metrics, leakage rules)
Statistical Testing Plan
Code Architecture — file-by-file, function-by-function
Day-by-Day / Hour-by-Hour Schedule (Sept 22–30)
Paper Writing Plan (IEEE 6-page structure)
Results → Figures/Tables Mapping
Pre-Submission Checklist
Risk & Troubleshooting Appendix
Copy-Paste Prompts for Antigravity, by phase
1. Project Overview & Novelty Statement
Working title: "Attribution-Constrained SMOTE: Guiding Synthetic Oversampling with Feature Importance for Imbalanced Classification"

One-paragraph pitch: Standard SMOTE-family oversamplers (SMOTE, Borderline-SMOTE, ADASYN) generate synthetic minority samples by interpolating between a real minority point and its neighbor, treating every feature dimension identically — they have no notion of which features actually drive the classifier's decision. This can produce synthetic points that blur the decision boundary along the features that matter most. We propose AC-SMOTE, which uses cheap SHAP-based feature attributions from a lightweight surrogate model to (a) bias neighbor search toward points that are close on decision-relevant dimensions, and (b) shrink the interpolation step size along high-importance dimensions while allowing full-range interpolation along low-importance ones — preserving the class-defining signal while still adding useful diversity. We benchmark this against four standard baselines across ten public imbalanced datasets and three classifiers, with full statistical testing.

Why this is a real (not oversold) contribution:

It is a specific, precisely-defined mechanism (not "we added SHAP to SMOTE" hand-waving) — Section 6 gives the exact math.
It sits in a genuine gap: explainability methods and resampling methods are usually studied separately; using attribution to steer generation (not just to explain results afterward) is the differentiating angle.
The evaluation design (10 datasets, 3 classifiers, proper nested resampling, Friedman/Nemenyi/Wilcoxon testing, an ablation that isolates which part of the mechanism helps) is itself a contribution — many student papers in this space use 1–2 datasets and no significance testing, which is an easy and legitimate thing for a reviewer to flag. You will not have that weakness.
Whichever way the numbers land — AC-SMOTE wins on most datasets, or wins only under specific conditions (e.g., only for tree-based classifiers, or only above a certain imbalance ratio) — you have a genuine, reportable, defensible finding. Do not be tempted to hide or explain away results that don't flatter the method; a paper that honestly reports "this helps under conditions X and Y, not Z" is more publishable than one that overclaims and gets caught by a reviewer re-running the numbers.
Track fit: Track 1 — AI, ML & Data Analytics. Explicitly touches "Explainable and Interpretable AI," "Pattern Recognition," and "Data Science" from the track's own topic list.

2. Conference Formatting Requirements
Verified from the live CICT-2026 site (Sept 22, 2026):

Submission portal: https://cmt3.research.microsoft.com/CICT2026
Full paper deadline: September 30, 2026 (extended from Aug 30)
Notification of acceptance: October 30, 2026
Official guidelines page: https://www.cict2026.iiitk.ac.in/Guidelines.html — this returned a temporary error when I tried to fetch it just now. You must open this link yourself on Day 1 and re-confirm the numbers below before you format anything.
Inferred from CICT-2021 (same organizer, same conference series — treat as "very likely correct, but verify"):

IEEE two-column format, A4, 10-point font
Maximum 6 printed pages, including figures, tables, and references
Abstract: maximum 250 words, minimum 3 keywords
Double-blind review — do not put author names, affiliations, or emails anywhere in the submitted PDF body
Plagiarism-checked against IEEE policy
You must select one track at submission time — select Track 1: AI, ML & Data Analytics
Template: official IEEE conference template (LaTeX "IEEEtran" or MS Word), downloadable from https://www.ieee.org/conferences/publishing/templates.html and/or linked directly from the CICT-2026 guidelines page
Action item for Day 1, Hour 1: open the CICT-2026 guidelines page yourself, confirm page limit / abstract length / template link, and update this section if anything differs. Do not skip this — formatting violations are grounds for desk rejection before anyone even reads your novelty claim.

3. Environment Setup (Windows, your laptop, CPU-only — no GPU needed)
Important reassurance: this entire project is intentionally designed to need zero GPU. Everything runs on tabular data with classical/lightweight models (Logistic Regression, Random Forest, XGBoost) and CPU-based SHAP (TreeExplainer, which is fast and exact for tree models — not the slow KernelExplainer). Your RTX 3050 will sit idle for this whole project, and that's fine — it removes an entire category of risk (driver issues, CUDA versions, OOM errors) from an 8-day plan.

Open PowerShell and run these commands in order:

# 1. Check Python version (need 3.10 or 3.11). If missing, install from python.org.
python --version

# 2. Create the project folder
mkdir C:\Users\uday\ac-smote-paper
cd C:\Users\uday\ac-smote-paper

# 3. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# If PowerShell blocks the activation script with an execution-policy error, run this once:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# then re-run the Activate.ps1 line above.

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install every package this project needs, pinned loosely (exact patch version doesn't matter)
pip install numpy pandas scikit-learn imbalanced-learn xgboost shap matplotlib seaborn scipy statsmodels scikit-posthocs tqdm jupyter
Verify the install worked:

python -c "import numpy, pandas, sklearn, imblearn, xgboost, shap, matplotlib, seaborn, scipy, statsmodels, scikit_posthocs; print('ALL IMPORTS OK')"
If that prints ALL IMPORTS OK, environment setup is done. If any import fails, the error will name the missing package — pip install <that package> and re-run the check.

Note for Antigravity: replace C:\Users\uday\ac-smote-paper above with the actual path it creates — the exact path doesn't matter, but once chosen, use it consistently everywhere else in this guide (all later paths in this document are given relative to the project root, i.e. relative to wherever this folder ends up).

4. Project Directory Structure
Create this exact structure before writing any code. Antigravity should create every folder and empty file below on Day 1, Hour 2, before implementing anything.

ac-smote-paper/
├── PROJECT_SPEC.md              <- this document, copied in verbatim
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                     <- untouched downloaded files (csv)
│   └── dataset_manifest.json    <- config: name -> {source, path, expected_n, expected_d, expected_ratio}
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── ac_smote.py
│   │   └── baselines.py
│   ├── experiment.py
│   ├── run_all_experiments.py
│   ├── run_ablation.py
│   ├── statistical_tests.py
│   └── make_figures.py
├── results/
│   ├── raw_results.csv
│   ├── ablation_results.csv
│   ├── summary_table.csv
│   ├── statistical_tests.json
│   └── figures/
└── paper/
    ├── draft.md                 <- working draft, plain text/markdown, converted to the IEEE template only at the end
    └── figures/                 <- final copies of exactly the figures used in the paper
requirements.txt content:

numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
statsmodels>=0.14
scikit-posthocs>=0.7
tqdm
