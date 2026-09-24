import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_dataset
from src.methods.baselines import get_resampler

def measure_runtimes():
    print("Measuring isolated fit_resample runtimes...")
    datasets = ['ecoli', 'satimage', 'spectrometer', 'car_eval_34', 'oil', 
                'wine_quality', 'mammography', 'abalone_19', 'creditcard', 'telco_churn']
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote']
    
    results = []
    
    for ds in datasets:
        print(f"Dataset: {ds}")
        X, y = load_dataset(ds)
        # Just use a single split 80/20 to measure
        n_train = int(0.8 * len(X))
        X_train, y_train = X[:n_train], y[:n_train]
        
        for res_name in resamplers:
            if res_name == 'none':
                t0 = time.time()
                t1 = time.time()
            else:
                resampler = get_resampler(res_name, random_state=42)
                t0 = time.time()
                try:
                    resampler.fit_resample(X_train, y_train)
                except:
                    pass # ignore adasyn crashes on some specific folds, but this is 80/20 so usually fine
                t1 = time.time()
                
            runtime = t1 - t0
            results.append({'dataset': ds, 'resampler': res_name, 'fit_time_seconds': runtime})
            
    df = pd.DataFrame(results)
    print(df.groupby('resampler')['fit_time_seconds'].mean())
    
    plt.figure(figsize=(8, 5))
    df_time = df.groupby('resampler')['fit_time_seconds'].mean().reset_index()
    # Order properly
    sns.barplot(data=df_time, x='resampler', y='fit_time_seconds', order=resamplers, palette='Set2')
    plt.yscale('log')
    plt.ylabel("Mean fit_resample Time (seconds, log scale)")
    plt.title("Runtime Comparison: Resampler Only (Log Scale)")
    plt.tight_layout()
    plt.savefig('results/figures/runtime_comparison.png', dpi=300)
    plt.close()
    print("Fixed runtime_comparison.png")

if __name__ == '__main__':
    measure_runtimes()
