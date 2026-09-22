import pandas as pd
from sklearn.datasets import fetch_openml
import os

os.makedirs('data/raw', exist_ok=True)

print("Downloading Telco Customer Churn from OpenML...")
# ID 42728 was wrong? Let's use name instead
telco = fetch_openml(name='telco-customer-churn', version=1, as_frame=True, parser='auto')
telco.frame.to_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
print("Telco saved. Rows:", len(telco.frame))

print("Downloading Credit Card Fraud from OpenML...")
cc = fetch_openml(name='creditcard', version=1, as_frame=True, parser='auto')
cc.frame.to_csv('data/raw/creditcard.csv', index=False)
print("Credit Card saved. Rows:", len(cc.frame))
