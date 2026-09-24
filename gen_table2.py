# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv('results/summary_table.csv')
lines = []
for ds in df['dataset'].unique()[:3]: # just a subset to check how to build it
    for clf in df['classifier'].unique()[:1]:
        sub = df[(df['dataset']==ds) & (df['classifier']==clf)]
        for _, row in sub.iterrows():
            res = row['resampler']
            # safely handle unicode 
            gmean = str(row['gmean']).replace('A', ' \\pm ').replace('±', ' \\pm ').replace('', ' \\pm ')
            lines.append(f"{ds} & {clf} & {res} & $ \\\\")
print("\n".join(lines))
