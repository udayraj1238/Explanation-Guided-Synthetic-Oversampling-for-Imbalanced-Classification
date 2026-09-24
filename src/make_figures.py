import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
import scikit_posthocs as sp
from sklearn.decomposition import PCA
from src.data_loader import load_dataset
from src.methods.baselines import get_resampler

def make_figures():
    os.makedirs('results/figures', exist_ok=True)
    df = pd.read_csv('results/raw_results.csv')
    metrics = ['f1', 'auc_pr', 'mcc', 'gmean', 'bal_acc']
    resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote']
    classifiers = df['classifier'].unique()
    
    # 1. Critical Difference Diagram (using Nemenyi sign_plot on Random Forest G-mean as representative)
    print("Generating Nemenyi sign plot...")
    plt.figure(figsize=(8, 6))
    rf_data = df[df['classifier'] == 'rf'].groupby(['dataset', 'resampler'])['gmean'].mean().unstack()[resamplers]
    # Nemenyi Friedman posthoc test
    p_vals = sp.posthoc_nemenyi_friedman(rf_data.values)
    p_vals = pd.DataFrame(p_vals, index=resamplers, columns=resamplers)
    sp.sign_plot(p_vals)
    plt.title("Nemenyi Significance (Random Forest, G-mean)")
    plt.tight_layout()
    plt.savefig('results/figures/critical_difference_diagram.png', dpi=300)
    plt.close()

    # 2. Boxplots per metric
    print("Generating per-metric boxplots...")
    # Calculate dataset-level means first to avoid dominating the boxplot with 15 CV points per cell
    df_mean = df.groupby(['dataset', 'classifier', 'resampler'])[metrics].mean().reset_index()
    for metric in metrics:
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=df_mean, x='classifier', y=metric, hue='resampler', hue_order=resamplers, palette='Set2')
        plt.title(f"{metric.upper()} across classifiers")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results/figures/boxplot_{metric}.png', dpi=300)
        plt.close()

    # 3. PCA Synthetic Comparison
    print("Generating PCA synthetic comparison...")
    pca_datasets = ['ecoli', 'wine_quality']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, ds_name in enumerate(pca_datasets):
        X, y = load_dataset(ds_name)
        # We only want to compare SMOTE and ACSMOTE
        smote = get_resampler('smote', random_state=42)
        acsmote = get_resampler('acsmote', random_state=42)
        
        # Fit on real data
        pca = PCA(n_components=2, random_state=42)
        X_pca_real = pca.fit_transform(X)
        
        X_smote, y_smote = smote.fit_resample(X, y)
        X_ac, y_ac = acsmote.fit_resample(X, y)
        
        # Identify purely synthetic points (the ones added to the end)
        n_real = len(X)
        X_synth_smote = X_smote[n_real:]
        X_synth_ac = X_ac[n_real:]
        
        X_pca_smote = pca.transform(X_synth_smote)
        X_pca_ac = pca.transform(X_synth_ac)
        
        axes[i].scatter(X_pca_real[y==0, 0], X_pca_real[y==0, 1], alpha=0.3, label='Majority (Real)', color='gray', s=15)
        axes[i].scatter(X_pca_real[y==1, 0], X_pca_real[y==1, 1], alpha=0.9, label='Minority (Real)', color='blue', s=30, marker='x')
        axes[i].scatter(X_pca_smote[:, 0], X_pca_smote[:, 1], alpha=0.5, label='SMOTE (Synth)', color='red', s=10)
        axes[i].scatter(X_pca_ac[:, 0], X_pca_ac[:, 1], alpha=0.5, label='AC-SMOTE (Synth)', color='green', s=10)
        axes[i].set_title(ds_name)
        if i == 1:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('results/figures/pca_synthetic_comparison.png', dpi=300)
    plt.close()

    # 4. Ablation Beta Sweep
    print("Generating Ablation Beta Sweep...")
    df_ab = pd.read_csv('results/ablation_results.csv')
    # Filter only beta sweep values (alpha=2.0, k=5)
    df_beta = df_ab[(df_ab['alpha'] == 2.0) & (df_ab['k'] == 5)]
    df_beta_mean = df_beta.groupby(['dataset', 'beta'])[['f1', 'auc_pr']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(data=df_beta_mean, x='beta', y='f1', hue='dataset', marker='o', ax=axes[0])
    axes[0].set_title("F1-score vs Beta")
    axes[0].set_ylabel("F1")
    
    sns.lineplot(data=df_beta_mean, x='beta', y='auc_pr', hue='dataset', marker='o', ax=axes[1])
    axes[1].set_title("AUC-PR vs Beta")
    axes[1].set_ylabel("AUC-PR")
    
    plt.tight_layout()
    plt.savefig('results/figures/ablation_beta_sweep.png', dpi=300)
    plt.close()

    # 5. Runtime Comparison
    print("Generating Runtime Comparison...")
    plt.figure(figsize=(8, 5))
    df_time = df.groupby(['classifier', 'resampler'])['fit_time_seconds'].mean().reset_index()
    sns.barplot(data=df_time, x='classifier', y='fit_time_seconds', hue='resampler', hue_order=resamplers, palette='Set2')
    plt.yscale('log')
    plt.ylabel("Mean Fit Time (seconds, log scale)")
    plt.title("Runtime Comparison (Log Scale)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/runtime_comparison.png', dpi=300)
    plt.close()
    
    print("All figures generated successfully.")

if __name__ == '__main__':
    make_figures()
