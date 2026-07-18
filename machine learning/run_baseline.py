"""
Baseline classifier comparison (Table 4).
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data_utils import load_data, balance_data
from classifiers import get_classifier

DATA_PATH = "data/final_dataset.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def run_baseline():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = ['XGBoost', 'LightGBM', 'RandomForest', 'SVM', 'MLP', 'CatBoost']
    results = {}

    for name in models:
        clf = get_classifier(name)
        if name != 'CatBoost':
            X_train_bal, y_train_bal = balance_data(X_train, y_train)
            clf.fit(X_train_bal, y_train_bal)
        else:
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[name] = f1
        print(f"{name}: F1 = {f1:.4f}")

    print("\n" + "=" * 50)
    print("Table 4 Results")
    print("=" * 50)
    for name, f1 in results.items():
        print(f"{name}: {f1:.4f}")

if __name__ == "__main__":
    run_baseline()