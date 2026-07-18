"""
Data loading and SMOTE balancing utilities.
"""

import pandas as pd
from imblearn.over_sampling import SVMSMOTE

RANDOM_STATE = 42

def load_data(path="data/final_dataset.xlsx"):
    data = pd.read_excel(path).dropna()
    print('Data distribution:')
    for label, count in data['突破性专利'].value_counts().items():
        print(f"  Label {label}: {count} samples")
    X = data.drop(['突破性专利'], axis=1)
    y = data['突破性专利']
    return X, y

def balance_data(X_train, y_train):
    """Apply SVM-SMOTE to balance training data."""
    neg_ratio = sum(y_train == 0) / len(y_train)
    target_ratio = min(0.2, neg_ratio * 1.5)
    smote = SVMSMOTE(sampling_strategy=target_ratio, random_state=RANDOM_STATE, k_neighbors=5)
    return smote.fit_resample(X_train, y_train)