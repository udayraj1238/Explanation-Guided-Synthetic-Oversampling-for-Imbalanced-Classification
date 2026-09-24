import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikit_posthocs as sp

df = pd.read_csv('results/raw_results.csv')
resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote']
rf_data = df[df['classifier'] == 'rf'].groupby(['dataset', 'resampler'])['gmean'].mean().unstack()[resamplers]
p_vals = sp.posthoc_nemenyi_friedman(rf_data.values)
p_vals = pd.DataFrame(p_vals, index=resamplers, columns=resamplers)

plt.figure(figsize=(8, 6))
sns.heatmap(p_vals, annot=True, cmap='RdYlGn', vmin=0, vmax=1, cbar=True)
plt.title("Nemenyi Significance p-values (RF, G-mean)")
plt.tight_layout()
plt.savefig('results/figures/critical_difference_diagram.png', dpi=300)
plt.close()
