# src/eda/eda_utils.py

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
        raise ValueError(
            "Kolom 'category' atau 'sales' tidak ditemukan di DataFrame.")

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
        raise ValueError(
            "Butuh kolom 'order_date' dan 'sales' untuk plot tren.")

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
    Hanya untuk keperluan visual, profit di-clip ke quantile 1%–99%
    supaya outlier ekstrem tidak mendominasi plot.
    """
    if "discount" not in df.columns or "profit" not in df.columns:
        raise ValueError(
            "Butuh kolom 'discount' dan 'profit' untuk scatter plot.")

    # Ambil hanya kolom yang diperlukan & buang NA
    plot_df = df[["discount", "profit"]].dropna().copy()

    # Clip outlier profit di luar 1%–99% quantile (untuk visual saja)
    q_low, q_high = plot_df["profit"].quantile([0.01, 0.99])
    plot_df = plot_df[(plot_df["profit"] >= q_low) &
                      (plot_df["profit"] <= q_high)]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Scatter + garis tren
    sns.regplot(
        data=plot_df,
        x="discount",
        y="profit",
        scatter_kws={"alpha": 0.2, "s": 10},   # titik lebih kecil & transparan
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
        annot=True,        # tampilkan angka
        fmt=".2f",         # dua angka di belakang koma
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
