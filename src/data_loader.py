import pandas as pd
import numpy as np
import json
from collections import Counter
from imblearn.datasets import fetch_datasets
import os

_IMBLEARN_CACHE = None

def get_imblearn_datasets():
    global _IMBLEARN_CACHE
    if _IMBLEARN_CACHE is None:
        _IMBLEARN_CACHE = fetch_datasets()
    return _IMBLEARN_CACHE

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name in ["creditcard", "telco_churn"]:
        if name == "creditcard":
            df = pd.read_csv("data/raw/creditcard.csv")
            # Dropping 'Time' or ID? Spec says: "30 numeric features (V1-V28 + Time + Amount)". So keep Time.
            # But wait, credit card openml uses 'Class' as target. Let's make it robust to 'Class' or 'class'.
            target_col = 'Class' if 'Class' in df.columns else 'class'
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
        else:
            df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
            # Dropping ID column
            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])
                
            # Target is 'Churn'
            target_col = 'Churn' if 'Churn' in df.columns else 'class'
            y_raw = df.pop(target_col)
            # Depending on if it's string 'Yes' or 1/0
            if y_raw.dtype == object:
                y = (y_raw.astype(str) == 'Yes').astype(int).values
            else:
                y = y_raw.values
                
            # Clean TotalCharges which may have whitespace
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'] = df['TotalCharges'].fillna(0)
                
            # One-hot encode categoricals
            df = pd.get_dummies(df, drop_first=True)
            X = df.values.astype(np.float64)
            
        # Ensure 1=minority, 0=majority
        counts = Counter(y)
        minority_class = min(counts, key=counts.get)
        y_mapped = np.where(y == minority_class, 1, 0)
        return X.astype(np.float64), y_mapped
        
    else:
        # Imblearn datasets
        data = get_imblearn_datasets()
        dataset = data[name]
        X = dataset.data
        y_raw = dataset.target
        
        counts = Counter(y_raw)
        minority_class = min(counts, key=counts.get)
        y = np.where(y_raw == minority_class, 1, 0)
        return X.astype(np.float64), y

def load_all_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)
    return {name: load_dataset(name) for name in manifest.keys()}

def verify_and_update_manifest() -> None:
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    for name in manifest.keys():
        X, y = load_dataset(name)
        n_samples, n_features = X.shape
        counts = Counter(y)
        min_count = counts[1]
        maj_count = counts[0]
        ratio = f"{maj_count/min_count:.2f}:1"
        
        manifest[name]["n_samples"] = n_samples
        manifest[name]["n_features"] = n_features
        manifest[name]["ratio"] = ratio
        
    with open('data/dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    verify_and_update_manifest()
    print("Manifest verified and updated.")
