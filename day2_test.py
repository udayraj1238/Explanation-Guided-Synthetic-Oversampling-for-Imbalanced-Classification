import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.data_loader import load_all_datasets
from src.methods.ac_smote import ACSMOTE
from src.utils import set_all_seeds

def run_day2_test():
    print("--- Running AC-SMOTE on all datasets (Hour 8) ---")
    datasets = load_all_datasets()
    
    os.makedirs('results/figures', exist_ok=True)
    
    set_all_seeds(42)
    acsmote = ACSMOTE(random_state=42)
    
    for name, (X, y) in datasets.items():
        print(f"Testing {name:15} (shape: {X.shape}, pos_ratio: {np.mean(y):.3f})...", end=" ")
        try:
            X_res, y_res = acsmote.fit_resample(X, y)
            
            # Checks
            assert not np.isnan(X_res).any(), "NaNs detected in synthetic features!"
            assert not np.isinf(X_res).any(), "Infs detected in synthetic features!"
            
            n_synth = len(X_res) - len(X)
            print(f"OK (generated {n_synth} samples, new balance: {np.mean(y_res):.3f})")
            
            # PCA for 'ecoli'
            if name == 'ecoli':
                print("Generating PCA scatter plot for ecoli...")
                pca = PCA(n_components=2, random_state=42)
                
                # Fit PCA on real data only
                pca.fit(X)
                
                X_real_2d = pca.transform(X)
                X_synth_2d = pca.transform(X_res[len(X):])
                
                y_real = y
                
                plt.figure(figsize=(8, 6))
                
                # Plot majority
                plt.scatter(X_real_2d[y_real == 0, 0], X_real_2d[y_real == 0, 1], 
                            c='blue', label='Majority (Real)', alpha=0.5, marker='o')
                
                # Plot minority real
                plt.scatter(X_real_2d[y_real == 1, 0], X_real_2d[y_real == 1, 1], 
                            c='red', label='Minority (Real)', alpha=0.8, marker='s')
                
                # Plot minority synthetic
                plt.scatter(X_synth_2d[:, 0], X_synth_2d[:, 1], 
                            c='orange', label='Minority (AC-SMOTE)', alpha=0.5, marker='x')
                
                plt.title("PCA Projection: AC-SMOTE Synthetic Samples (ecoli)")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.legend()
                
                fig_path = "results/figures/pca_sanity_check.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {fig_path}")
                
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == '__main__':
    run_day2_test()
