from imblearn.datasets import fetch_datasets
from collections import Counter
import json

data = fetch_datasets()
for name in ['ecoli', 'satimage']:
    y = data[name].target
    c = Counter(y)
    print(f"{name} labels: {c}")
