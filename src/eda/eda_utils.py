# src/eda/eda_utils.py

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as ss

from src.utils.io_utils import get_project_root

DEFAULT_FIGSIZE = (10, 6)


def _maybe_save(fig: plt.Figure, save_path: Optional[str]) -> None:
    """
    Simpan figure jika save_path tidak None.
    """
    if save_path is None:
        return

    path = Path(get_project_root() / save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"[INFO] Figure disimpan ke: {path}")


def plot_sales_by_category(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Axes:
    """
    Barplot: total sales per category.
    """
    if "category" not in df.columns or "sales" not in df.columns:
        raise ValueError("Kolom 'category' atau 'sales' tidak ditemukan di DataFrame.")

    data = (
        df.groupby("category")["sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.barplot(data=data, x="category", y="sales", ax=ax)
    ax.set_title("Total Sales per Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def plot_sales_trend_by_month(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Axes:
    """
    Line plot: total sales per bulan berdasarkan order_date.
    """
    if "order_date" not in df.columns or "sales" not in df.columns:
        raise ValueError("Butuh kolom 'order_date' dan 'sales' untuk plot tren.")

    data = (
        df.set_index("order_date")
        .resample("M")["sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(data["order_date"], data["sales"])
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Order Date")
    ax.set_ylabel("Total Sales")

    fig.autofmt_xdate()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def plot_discount_vs_profit(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Axes:
    """
    Scatter + trend line: Discount vs Profit.
    Untuk visual saja, profit di-clip ke quantile 1%–99% agar outlier ekstrem tidak mendominasi plot.
    """
    if "discount" not in df.columns or "profit" not in df.columns:
        raise ValueError("Butuh kolom 'discount' dan 'profit' untuk scatter plot.")

    plot_df = df[["discount", "profit"]].dropna().copy()

    # Clip outlier profit
    q_low, q_high = plot_df["profit"].quantile([0.01, 0.99])
    plot_df = plot_df[(plot_df["profit"] >= q_low) & (plot_df["profit"] <= q_high)]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.regplot(
        data=plot_df,
        x="discount",
        y="profit",
        scatter_kws={"alpha": 0.2, "s": 10},
        line_kws={"color": "red"},
        ax=ax,
    )

    ax.set_title("Discount vs Profit (with Trend Line)")
    ax.set_xlabel("Discount")
    ax.set_ylabel("Profit")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Axes:
    """
    Heatmap korelasi untuk kolom numerik (dengan nilai korelasi).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap (Numeric Features)")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


def plot_correlation_target(df: pd.DataFrame, target_col: str, save_path: Optional[str] = None) -> plt.Axes:
    """
    Heatmap korelasi untuk kolom numerik (dengan nilai korelasi) dengan fitur target.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr = df[numeric_cols].corr()

    sorted_corr = corr[[target_col]].sort_values(by=target_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sorted_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Correlation Heatmap (Numeric Features) to Target: {target_col}")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return ax


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


def cramers_v(x, y):
    """
    Fungsi Cramér's V untuk hitung korelasi kolom kategorikal.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    denominator = min((kcorr-1), (rcorr-1))
    if denominator == 0:
        return 0
    return np.sqrt(phi2corr / denominator)