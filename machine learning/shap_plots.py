"""
SHAP analysis for the optimized CatBoost model (Figures 6 & 7).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
from catboost import CatBoostClassifier
import joblib
from sklearn.model_selection import train_test_split
from data_utils import load_data

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_shap_analysis(model, X_data, y_data, data_name="Test Set"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    if isinstance(shap_values, list) and len(shap_values) >= 2:
        shap_values_pos = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.shape[1] == 2:
        shap_values_pos = shap_values[:, 1]
    else:
        shap_values_pos = shap_values

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {data_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_bar_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: shap_bar_importance.png")

    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        shap.summary_plot(shap_values[1], X_data, show=False)
    else:
        shap.summary_plot(shap_values, X_data, show=False)
    plt.title(f"SHAP Value Distribution - {data_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: shap_summary_plot.png")

    shap_df = pd.DataFrame(shap_values_pos, columns=X_data.columns)
    importance = shap_df.abs().mean().sort_values(ascending=False)
    print("\nTop 20 features by mean |SHAP|:")
    for i, (f, v) in enumerate(importance.head(20).items(), 1):
        print(f"{i:<5} {f:<30} {v:.4f}")

    return importance

def generate_shap_plots():
    print("=" * 60)
    print("SHAP Analysis")
    print("=" * 60)

    print("\nLoading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nLoading model...")
    try:
        model_data = joblib.load('final_optimized_catboost_model.pkl')
        model = model_data['model']
        print("  Loaded from final_optimized_catboost_model.pkl")
    except:
        print("  No saved model found. Training a default model...")
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.1, depth=10,
            l2_leaf_reg=3, border_count=254,
            loss_function='Logloss', eval_metric='F1',
            auto_class_weights='Balanced', random_state=42, verbose=False
        )
        model.fit(X_train, y_train)
        print("  Training complete")

    importance = plot_shap_analysis(model, X_test, y_test, "Test Set")
    if importance is not None:
        importance.to_csv('shap_feature_importance.csv')
        print("\nSaved: shap_feature_importance.csv")

    print("\n" + "=" * 60)
    print("SHAP analysis complete")
    print("=" * 60)
    print("Generated files:")
    print("  - shap_bar_importance.png")
    print("  - shap_summary_plot.png")
    print("  - shap_feature_importance.csv")

if __name__ == "__main__":
    generate_shap_plots()