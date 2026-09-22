import numpy as np
import scipy.stats as stats
from src.data_loader import load_dataset
from src.methods.ac_smote import ACSMOTE
from imblearn.over_sampling import SMOTE
from src.utils import set_all_seeds

def run_sanity_check():
    """
    Hour 7: Implement and run the alpha=0, beta=0 sanity check.
    Confirm AC-SMOTE's output distribution matches plain SMOTE's.
    """
    print("--- Running alpha=0, beta=0 Sanity Check ---")
    dataset_name = 'ecoli'
    X, y = load_dataset(dataset_name)
    
    # 1. Plain SMOTE
    set_all_seeds(42)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # 2. AC-SMOTE with alpha=0, beta=0
    set_all_seeds(42)
    acsmote = ACSMOTE(alpha=0.0, beta=0.0, random_state=42)
    X_acsmote, y_acsmote = acsmote.fit_resample(X, y)
    
    # We only care about the synthetic points
    n_original = len(X)
    X_synth_smote = X_smote[n_original:]
    X_synth_acsmote = X_acsmote[n_original:]
    
    print(f"Generated {len(X_synth_smote)} SMOTE samples and {len(X_synth_acsmote)} AC-SMOTE samples.")
    assert len(X_synth_smote) == len(X_synth_acsmote), "Sample counts differ!"
    
    # Compare means
    mean_smote = np.mean(X_synth_smote, axis=0)
    mean_acsmote = np.mean(X_synth_acsmote, axis=0)
    
    # Compare covariance
    cov_smote = np.cov(X_synth_smote, rowvar=False)
    cov_acsmote = np.cov(X_synth_acsmote, rowvar=False)
    
    mean_diff = np.linalg.norm(mean_smote - mean_acsmote)
    cov_diff = np.linalg.norm(cov_smote - cov_acsmote)
    
    print(f"L2 Norm of Mean Difference: {mean_diff:.6f}")
    print(f"Frobenius Norm of Covariance Difference: {cov_diff:.6f}")
    
    if mean_diff < 0.1 and cov_diff < 0.1:
        print("[PASS] Sanity Check PASSED: AC-SMOTE(0,0) distribution closely matches plain SMOTE.")
    else:
        print("[FAIL] Sanity Check FAILED: Distributions are significantly different.")

if __name__ == '__main__':
    run_sanity_check()
