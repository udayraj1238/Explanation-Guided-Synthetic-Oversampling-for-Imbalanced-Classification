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

5. Datasets
Use 10 datasets total: 8 from the standard imbalanced-learn benchmark collection (zero manual download needed — one Python call fetches all of them) plus 2 real-world datasets from Kaggle (manual download, no API key needed).

5.1 — Eight datasets via imbalanced-learn
from imblearn.datasets import fetch_datasets
all_data = fetch_datasets()   # downloads once, caches locally afterwards
Use exactly these 8 keys from the returned dictionary (chosen to span a wide range of imbalance ratios and dimensionalities — mild to extreme):

Key (use exactly this string)	Approx. imbalance ratio*	Approx. n samples	Approx. n features
ecoli	~8.6:1	336	7
satimage	~9.3:1	6,435	36
spectrometer	~11:1	531	93
car_eval_34	~12:1	1,728	21
oil	~22:1	937	49
wine_quality	~26:1	4,898	11
mammography	~42:1	11,183	6
abalone_19	~130:1	4,177	10
*These ratios are from memory of the standard benchmark documentation and may be slightly off. Do not trust this column — Day 1's first coding task is to load each dataset and print its actual n_samples, n_features, and minority:majority ratio, and use those real, verified numbers everywhere from that point on (including in the paper's dataset table).

5.2 — Two datasets via Kaggle (manual download)
Credit Card Fraud Detection — extreme real-world imbalance (~0.17% positive, ratio ≈ 577:1), 284,807 rows, 30 numeric features (V1–V28 + Time + Amount).
URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Click "Download," extract, place creditcard.csv into data/raw/.
Telco Customer Churn — moderate real-world imbalance (~26.6% positive, ratio ≈ 2.8:1), 7,043 rows, mixed numeric/categorical features (one-hot encode the categoricals in data_loader.py).
URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Click "Download," extract, place WA_Fn-UseC_-Telco-Customer-Churn.csv into data/raw/.
You need a free Kaggle account to click "Download" on either page — no API token, no CLI setup required.

5.3 — dataset_manifest.json
Antigravity should generate this file automatically on Day 1 after loading every dataset once — it is the verified record of what you actually have, and every later script reads dataset names from this file rather than hard-coding them twice:

{
  "ecoli":          {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "satimage":       {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "spectrometer":   {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "car_eval_34":    {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "oil":            {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "wine_quality":   {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "mammography":    {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "abalone_19":     {"source": "imblearn", "n_samples": null, "n_features": null, "ratio": null},
  "creditcard":     {"source": "kaggle:mlg-ulb/creditcardfraud", "path": "data/raw/creditcard.csv", "n_samples": null, "n_features": null, "ratio": null},
  "telco_churn":    {"source": "kaggle:blastchar/telco-customer-churn", "path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "n_samples": null, "n_features": null, "ratio": null}
}
Fill in the null fields with real measured values once each dataset is loaded — this becomes Table 1 in the paper almost verbatim.

6. The Method: Attribution-Constrained SMOTE (AC-SMOTE)
This is the core contribution. Follow this specification exactly — do not let Antigravity "improve" or simplify it without your explicit sign-off, since any deviation changes what the paper is claiming.

6.1 — Intuition (for the paper's writing, not for the code)
Standard SMOTE picks a minority point, finds a nearby minority neighbor, and interpolates between them with the same random step size λ applied identically to every feature. It has no notion of which features actually separate the classes. This can push synthetic points toward the majority class specifically along the dimensions that matter most for classification, blurring the boundary. AC-SMOTE fixes this by (1) choosing neighbors that are close specifically on decision-relevant dimensions, and (2) shrinking the interpolation step on those same dimensions, so synthetic points stay faithful to the real class signal while still gaining diversity along dimensions that don't matter.

6.2 — Formal definition
Inputs: training features X_train (n × d), training labels y_train (n,), minority label value, hyperparameters k (neighbors, default 5), alpha (neighbor-weighting strength, default 2.0), beta (interpolation constraint strength, default 0.7), surrogate_n_estimators (default 50), random_state.

Step 1 — Surrogate model and attribution vector

Fit surrogate = RandomForestClassifier(n_estimators=surrogate_n_estimators, max_depth=6, random_state=random_state) on the full (X_train, y_train).
Compute explainer = shap.TreeExplainer(surrogate), then shap_values = explainer.shap_values(X_min) where X_min is the subset of X_train with the minority label. Use the SHAP values corresponding to the minority class.
Compute raw importance per feature: imp[j] = mean_i( abs(shap_values[i, j]) ) for j = 1..d.
Min-max normalize to get w_hat[j] = (imp[j] - min(imp)) / (max(imp) - min(imp) + 1e-12), giving values in [0, 1]. If max(imp) == min(imp) (degenerate case, e.g. very few minority samples), set every w_hat[j] = 0.5.
Step 2 — Attribution-weighted neighbor search (within the minority class only) For any two minority points x_i, x_k (each a length-d vector), define the weighted distance:

d_w(x_i, x_k) = sqrt( sum_j ( (1 + alpha * w_hat[j]) * (x_i[j] - x_k[j])**2 ) )
For every minority point, find its k nearest neighbors (also minority points) under d_w. Implement this as an explicit n_min x n_min weighted distance matrix (function weighted_distance_matrix(X_min, w_hat, alpha) in utils.py) rather than trying to force it through a library's built-in metric argument — this keeps the computation transparent and easy to unit-test. This is cheap: minority classes are small by definition, so an O(n_min² · d) matrix is fast even for the largest dataset here.

Step 3 — Attribution-constrained interpolation To generate one synthetic sample:

Pick a minority point x_i at random (with replacement, cycling through all minority points to reach the target count — this matches how standard SMOTE distributes generation).
Pick one neighbor x_nn uniformly at random from x_i's k-nearest set (from Step 2).
Draw a single global lambda ~ Uniform(0, 1).
For each feature j, compute the per-feature step size:
lambda_j = lambda * (1 - beta * w_hat[j])
lambda_j = clip(lambda_j, 0, 1)
Generate: x_new[j] = x_i[j] + lambda_j * (x_nn[j] - x_i[j]) for every feature j.
Repeat until you've generated enough synthetic points to fully balance the classes (same default target as imblearn's SMOTE with sampling_strategy='auto'), so that every method being compared oversamples to the same final class balance — this is essential for a fair comparison.

Sanity check to implement as an automated test: when alpha = 0 and beta = 0, Step 2's distance collapses to plain Euclidean distance and Step 3's lambda_j collapses to the same lambda for every feature — i.e., AC-SMOTE(0, 0) should behave like standard SMOTE. Write a test that runs both on the same dataset with the same random seed and confirms the synthetic samples have statistically indistinguishable means and covariances (they won't be bit-identical because tie-breaking among equal-distance neighbors can differ, but the distributions should match closely). This test existing and passing is itself evidence of correctness you can mention in the paper if asked.

6.3 — Required class interface (must match imblearn's API so it's a drop-in replacement)
class ACSMOTE:
    def __init__(self, sampling_strategy='auto', k_neighbors=5, alpha=2.0,
                 beta=0.7, surrogate_n_estimators=50, random_state=42):
        ...

    def fit_resample(self, X, y):
        # returns X_resampled, y_resampled
        ...
6.4 — Hyperparameters to sweep in the ablation (Section 6, Day 4)
Hyperparameter	Default (main experiments)	Ablation sweep values
beta (interpolation constraint)	0.7	0, 0.3, 0.5, 0.7, 1.0
alpha (neighbor weighting)	2.0	0, 1, 2, 4
k (neighbors)	5	3, 5, 7
The beta sweep is the most important one for the paper's story: beta=0 isolates "does attribution-weighted neighbor selection alone help?" and increasing beta shows whether the constrained-interpolation piece adds anything on top. This ablation directly supports the novelty claim, so don't skip it even under time pressure.

7. Baselines
Implement all of these behind one factory function so experiment.py never has special-case logic per method:

# src/methods/baselines.py
def get_resampler(name: str, random_state: int):
    """
    name in {'none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote'}
    Returns an object with .fit_resample(X, y) -> (X_res, y_res),
    or None for 'none' (meaning: use the raw imbalanced training fold as-is).
    """
name	Implementation
none	No resampling — train directly on the imbalanced fold. This is the "why bother" baseline.
ros	imblearn.over_sampling.RandomOverSampler(random_state=random_state)
smote	imblearn.over_sampling.SMOTE(random_state=random_state)
bsmote	imblearn.over_sampling.BorderlineSMOTE(random_state=random_state)
adasyn	imblearn.over_sampling.ADASYN(random_state=random_state)
acsmote	src.methods.ac_smote.ACSMOTE(random_state=random_state) (defaults from Section 6.4)
Classifiers (also behind a factory, get_classifier(name, random_state)):

name	Implementation
logreg	sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=random_state)
rf	sklearn.ensemble.RandomForestClassifier(n_estimators=200, random_state=random_state)
xgb	xgboost.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
8. Experimental Protocol
The single most important correctness rule in this entire project: resampling must be fit only on the training fold, never on the test fold, and never on the full dataset before splitting. If you resample before splitting, or resample the test fold, synthetic points derived from test-fold information leak into training and every number in your results table becomes invalid — this is the single most common mistake in imbalance-learning papers and the fastest way to get desk-rejected if a reviewer notices, or worse, to publish a paper with wrong numbers. Antigravity must implement the CV loop so that for every fold: split first, fit the resampler on the train portion only, transform the train portion, fit the classifier on the resampled train portion, and evaluate on the original, untouched, un-resampled test portion.

Cross-validation: sklearn.model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) — 15 total train/test splits per (dataset, classifier, resampler) combination. Stratified so class balance is preserved in every fold's test portion; repeated so the mean metric estimate per cell is stable.

Metrics (compute all five for every fold, store all of them):

F1-score of the minority class: sklearn.metrics.f1_score(y_true, y_pred, pos_label=minority_label)
Average precision / AUC-PR: sklearn.metrics.average_precision_score(y_true, y_score) (use predicted probability, not hard label)
Matthews Correlation Coefficient: sklearn.metrics.matthews_corrcoef(y_true, y_pred)
G-mean: imblearn.metrics.geometric_mean_score(y_true, y_pred)
Balanced accuracy: sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
Full experiment grid: 10 datasets × 3 classifiers × 6 resamplers × 15 CV splits = 2,700 total model fits. Every dataset here is small-to-medium (largest is the 284,807-row Credit Card Fraud set) and every model is lightweight — on a laptop CPU this should complete in at most a few hours, likely less. Treat this as a background batch job you can start and walk away from (Section 11, Day 3).

Output schema — results/raw_results.csv, one row per fold:

dataset, classifier, resampler, fold_id, f1, auc_pr, mcc, gmean, bal_acc, fit_time_seconds
Checkpointing (mandatory, given the multi-hour unattended run): run_all_experiments.py must append each completed row to the CSV immediately, and on startup must check which (dataset, classifier, resampler, fold_id) combinations already exist in the CSV and skip them. This means the script is safely restartable if your laptop sleeps, a package throws an error partway through, or you need to stop and resume across two days.

9. Statistical Testing Plan
Do not report only mean scores — a reviewer in this space will expect significance testing, and it strengthens the paper regardless. Implement in src/statistical_tests.py:

Per classifier, per metric: run a Friedman test (scipy.stats.friedmanchisquare) across the 6 resampling methods, using each dataset's mean score (averaged over the 15 folds) as one "block." This tests whether the 6 methods differ significantly at all, across the 10 datasets.
If the Friedman test is significant (p < 0.05): run a Nemenyi post-hoc test (scikit_posthocs.posthoc_nemenyi_friedman) to see which specific pairs of methods differ, and produce a critical-difference diagram (standard in ML benchmarking papers, and a good, reviewer-recognizable figure — scikit_posthocs can generate this directly, or use sign_plot).
Pairwise, AC-SMOTE vs. each baseline individually: run a Wilcoxon signed-rank test (scipy.stats.wilcoxon) on the 10 paired per-dataset mean scores (AC-SMOTE's score vs. that baseline's score, one pair per dataset). Report the test statistic, p-value, and how many datasets favored each side.
Save everything to results/statistical_tests.json, structured as:
{
  "friedman": {"logreg": {"f1": {"stat": ..., "p": ...}, "auc_pr": {...}, ...}, "rf": {...}, "xgb": {...}},
  "wilcoxon_acsmote_vs_baseline": {"logreg": {"smote": {"stat": ..., "p": ..., "wins": ..., "losses": ..., "ties": ...}, ...}, ...}
}
Honesty rule: if AC-SMOTE does not significantly beat a baseline on some metric/classifier combination, report that plainly in the results table rather than omitting that combination. A results table with some non-significant cells, honestly labeled, is normal and expected in this literature — it is not a flaw, and hiding it would be.

10. Code Architecture — File by File
This is the exact blueprint. Antigravity should implement these files in this order (matches Section 11's schedule). Every function signature below is fixed; internal implementation details are Antigravity's to fill in.

src/data_loader.py

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    name: one of the 10 keys in dataset_manifest.json.
    Returns X (n_samples, n_features) float array, y (n_samples,) int array with
    1 = minority class, 0 = majority class. Handles imblearn.fetch_datasets()
    lookups for the 8 imblearn keys, and CSV loading + cleaning (including
    one-hot encoding of categoricals and dropping ID columns) for 'creditcard'
    and 'telco_churn'.
    """

def load_all_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convenience wrapper: calls load_dataset for all 10 manifest entries."""

def verify_and_update_manifest() -> None:
    """
    Loads every dataset once, computes actual n_samples, n_features, and
    minority:majority ratio, and overwrites the null fields in
    data/dataset_manifest.json with the real measured values. Run this once,
    on Day 1, before anything else touches the data.
    """
src/utils.py

def weighted_distance_matrix(X_min: np.ndarray, w_hat: np.ndarray, alpha: float) -> np.ndarray:
    """Returns the n_min x n_min matrix of d_w distances defined in Section 6.2 Step 2."""

def compute_all_metrics(y_true, y_pred, y_score) -> dict:
    """Returns {'f1':..., 'auc_pr':..., 'mcc':..., 'gmean':..., 'bal_acc':...} per Section 8."""

def set_all_seeds(seed: int) -> None:
    """Seeds numpy, python's random module, and anything else with its own RNG."""
src/methods/ac_smote.py — implements the ACSMOTE class exactly per Section 6.3, plus the Step-1/2/3 helper functions it calls internally, plus the alpha=0, beta=0 sanity-check test described in Section 6.2.

src/methods/baselines.py — implements get_resampler(name, random_state) and get_classifier(name, random_state) exactly per Section 7's tables.

src/experiment.py

def run_single_cell(X, y, dataset_name, classifier_name, resampler_name,
                     cv_splitter, random_state) -> list[dict]:
    """
    Runs the full CV loop for one (dataset, classifier, resampler) combination.
    For each of the 15 splits: split first, fit resampler on train fold only
    (skip this step entirely if resampler_name == 'none'), fit classifier on
    the resampled train fold, evaluate on the untouched original test fold.
    Returns one dict per fold matching the raw_results.csv schema (Section 8),
    each also tagged with dataset/classifier/resampler/fold_id.
    """
src/run_all_experiments.py — the main driver. Loops over all 10 × 3 × 6 combinations, calls run_single_cell for each, and appends results to results/raw_results.csv immediately after each cell completes (checkpointing per Section 8). Prints a running progress bar (tqdm) and an ETA.

src/run_ablation.py — separate driver for the beta/alpha/k sweep (Section 6.4), run on the XGBoost classifier only, on a representative subset of 5 datasets (pick a spread: 1 mild-imbalance, 1 moderate, 1 severe, plus the two Kaggle sets) to keep runtime down. Writes to results/ablation_results.csv.

src/statistical_tests.py — implements exactly the four steps in Section 9, reading from results/raw_results.csv and writing results/statistical_tests.json and results/summary_table.csv (the mean ± std per dataset/classifier/resampler/metric — this table is close to a direct copy into the paper).

src/make_figures.py — generates, and saves as separate files into results/figures/:

critical_difference_diagram.png — from the Nemenyi results (Section 9)
boxplot_<metric>.png — one per metric, boxplot of scores across the 6 methods (pooled across datasets, one figure per classifier or faceted)
pca_synthetic_comparison.png — 2D PCA scatter (fit PCA on real data, project both real and synthetic points) comparing standard SMOTE's synthetic points vs. AC-SMOTE's synthetic points, for 2 illustrative low-dimensional datasets (suggest ecoli and wine_quality)
ablation_beta_sweep.png — line plot of F1/AUC-PR vs. beta, from ablation_results.csv
runtime_comparison.png — bar chart of mean fit_time_seconds per resampler, pooled across datasets — this is the figure that honestly shows AC-SMOTE's SHAP-computation overhead vs. baselines; report it plainly, don't hide it.

11. Day-by-Day / Hour-by-Hour Schedule
Assumes an ~8-hour working day. If your actual available hours differ, keep the order and content of tasks identical and compress/stretch the hour numbers to fit your real schedule — do not skip steps to save time; skip breadth (e.g., fewer ablation datasets) instead, per Section 15.

DAY 1 — Tuesday, Sept 22 (today): Setup + Verified Data
Hour 1: Complete Section 3 (environment setup) end-to-end. Confirm ALL IMPORTS OK prints.
Hour 2: Create the exact folder structure from Section 4. Copy this entire document into ac-smote-paper/PROJECT_SPEC.md. Open https://www.cict2026.iiitk.ac.in/Guidelines.html yourself and confirm/correct Section 2's page-limit and abstract-length numbers.
Hour 3: Manually download the 2 Kaggle datasets (Section 5.2) into data/raw/.
Hour 4–5: Implement data_loader.py (Section 10). Implement and run verify_and_update_manifest(). Confirm data/dataset_manifest.json now has real, non-null numbers for all 10 datasets.
Hour 6: Implement utils.py's compute_all_metrics and set_all_seeds. Write a throwaway script that loads one dataset, does a single train/test split, fits LogisticRegression with no resampling, and prints all 5 metrics — this is your smoke test that the basic pipeline works before anything novel is built on top of it.
Hour 7: Implement baselines.py (Section 7) — this is thin wrapping of imblearn classes, should be quick.
Hour 8: Re-run the smoke test, this time looping over none, ros, smote, bsmote, adasyn on one dataset with one classifier, printing all 5 metrics for each. Confirm all 5 run without errors and produce sane-looking numbers (F1/AUC-PR/MCC/G-mean/balanced-accuracy all between 0 and 1, none of them exactly 0 or exactly 1 across the board — that would indicate a bug).
✅ Checkpoint before moving to Day 2: dataset_manifest.json fully populated with real numbers; smoke test runs 5 baseline resamplers end-to-end on real data with sane metrics.

DAY 2 — Wednesday, Sept 23: Implement AC-SMOTE
Hour 1–2: Implement Section 6.2 Step 1 (surrogate model + SHAP importance vector w_hat) as its own testable function. Verify w_hat is length-d, all values in [0, 1].
Hour 3–4: Implement Section 6.2 Step 2 (weighted_distance_matrix in utils.py, then k-NN lookup on top of it, restricted to the minority class).
Hour 5–6: Implement Section 6.2 Step 3 (constrained interpolation) and wire all three steps together into the ACSMOTE class (Section 6.3).
Hour 7: Implement and run the alpha=0, beta=0 sanity check described in Section 6.2 — confirm AC-SMOTE's output distribution matches plain SMOTE's under those settings.
Hour 8: Run ACSMOTE with default hyperparameters (Section 6.4) on all 10 datasets in a quick loop, confirming no errors/NaNs/exploding values, and eyeball a quick PCA scatter of one dataset's synthetic points to sanity-check they look reasonable (roughly within the real data's range, not degenerate/collapsed to a point).
✅ Checkpoint: ACSMOTE.fit_resample() runs cleanly on every dataset; sanity check passes; output looks visually reasonable on at least one dataset.
DAY 3 � Thursday, Sept 24: Full Experiment Runner + Kick Off the Big Run
Hour 1�2: Implement experiment.py's run_single_cell (Section 10), obeying the leakage rule in Section 8 exactly (split ? resample train only ? fit ? evaluate on original test).
Hour 3�4: Implement run_all_experiments.py with checkpointing (Section 8). Do a tiny dry run: 2 datasets � 1 classifier � all 6 resamplers � 15 folds. Time it, and extrapolate to estimate the full grid's total runtime.
Hour 5: Fix anything the dry run surfaced. Then launch the full grid (10 � 3 � 6 � 15 = 2,700 fits) as a background process you can leave running.
Hour 6�8 (while the run is going in the background): Start the paper draft (paper/draft.md, plain markdown for now). Write a first pass of the Introduction and begin Related Work (see Section 12 for what belongs in each � note the Related Work section requires real literature search, done properly on Day 6; for now just sketch the structure and the categories of prior work you'll compare against).
? Checkpoint: full experiment grid is running (or finished, if your laptop is fast) with raw_results.csv growing; Introduction has a solid first draft.
