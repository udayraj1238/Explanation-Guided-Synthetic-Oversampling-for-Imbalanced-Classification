import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/raw_results.csv')
resamplers = ['none', 'ros', 'smote', 'bsmote', 'adasyn', 'acsmote']
rf_data = df[df['classifier'] == 'rf'].groupby(['dataset', 'resampler'])['gmean'].mean().unstack()[resamplers]
p_vals = sp.posthoc_nemenyi_friedman(rf_data.values)
print(p_vals)
