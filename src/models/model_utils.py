# src/models/train_model.py

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as ss

from src.utils.io_utils import get_project_root
from src.eda.eda_utils import _maybe_save
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DEFAULT_FIGSIZE = (10, 6)
model_results = []

def plot_target_cramers(df: pd.DataFrame, target: str = 'is_profitable', categorical_cols: Optional[list] = None, save_path: Optional[str] = None) -> plt.Axes:
    """
    Barplot: Cramér's V antara setiap fitur kategorikal dengan target.
    """
    if categorical_cols is None:
        categorical_cols = ['ship_mode', 'segment', 'market', 'region', 'category', 'sub_category', 'order_priority', 'country']

    if target not in df.columns or any(col not in df.columns for col in categorical_cols):
        raise ValueError("Kolom target atau fitur kategorikal tidak ditemukan di DataFrame.")

    correlations = {}
    for col in categorical_cols:
        score = cramers_v(df[col], df[target])
        correlations[col] = score

    corr_series = pd.Series(correlations).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=corr_series.values, y=corr_series.index, hue=corr_series.values, palette='viridis', ax=ax)
    ax.set_title("Cramér's V Correlation with Target (is_profitable)")
    ax.set_xlabel("Cramér's V Score")
    ax.set_xlim(0, 1)
    
    for i, v in enumerate(corr_series.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def confusion_matrix_plot(model: str, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[list] = None, save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot confusion matrix sebagai heatmap.
    """
    display_labels = labels if labels is not None else 'auto'

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=display_labels, yticklabels=display_labels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model}')

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def feature_importance_plot(model, feature_names: list, top_n: int = 10, save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot feature importance dari model tree-based.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]  # Top N Features

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance Score')
    ax.set_title(f'Top {top_n} Feature Importances')

    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def coefficient_plot(model, feature_names: list, top_n: int = 10, save_path: Optional[str] = None) -> plt.Axes:
    """
    Plots the top N coefficients from a linear model (Logistic Regression).
    Positive coefficients = Green, Negative coefficients = Red.
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model does not have 'coef_' attribute.")

    # coefs = model.coef_.flatten()
    coefs = model.coef_[0]
    
    df_coef = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })
    
    df_top = df_coef.sort_values(by='Abs_Coefficient', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in df_top['Coefficient']]
    bars = ax.barh(df_top['Feature'], df_top['Coefficient'], color=colors)
    
    ax.set_xlabel('Coefficient Value (Impact on Profitability)')
    ax.set_title(f'Top {top_n} Feature Coefficients')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--') # Zero line for reference

    max_val = df_top['Abs_Coefficient'].max()
    offset = max_val * 0.02 
    
    for bar, value in zip(bars, df_top['Coefficient']):
        text_x = bar.get_width() + offset if value >= 0 else bar.get_width() - offset
        ha = 'left' if value >= 0 else 'right'
        
        ax.text(text_x, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', 
                va='center', ha=ha, fontweight='bold', fontsize=10)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def get_model_metrics(model, model_name: str, X_test, y_test) -> Dict[str, Any]:
    """
    Runs predictions and extracts key metrics (Accuracy, F1, Recall, Precision)
    specifically focusing on the Unprofitable class (0).
    """
    # 1. Predict
    y_pred = model.predict(X_test)
    
    # 2. Get detailed report as a Dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 3. Extract specific numbers
    # We focus on '0' (Unprofitable) because it's the minority class of interest
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 (Unprofitable)": report['0']['f1-score'],
        "Recall (Unprofitable)": report['0']['recall'],
        "Precision (Unprofitable)": report['0']['precision'],
        "F1 (Profitable)": report['1']['f1-score']
    }
    
    return metrics