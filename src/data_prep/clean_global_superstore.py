# src/data_prep/clean_global_superstore.py

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import load_csv, save_csv


RAW_FILE = "data/raw/Global_Superstore2.csv"
CLEAN_FILE = "data/processed/global_superstore_clean.csv"


def load_raw_data(path: str = RAW_FILE) -> pd.DataFrame:
    """
    Load dataset Global Superstore mentah dari data/raw.
    Pakai encoding 'latin1' karena file bukan UTF-8.
    """
    df = load_csv(path, encoding="latin1")
    return df


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarkan nama kolom: lowercase, spasi & '-' jadi underscore.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse kolom tanggal jika ada (order_date, ship_date).
    """
    df = df.copy()
    for col in ["order_date", "ship_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def drop_duplicates_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplikasi berdasarkan key yang tersedia.
    Fallback: drop duplicates full row.
    """
    df = df.copy()
    candidate_keys = [
        ["row_id"],
        ["order_id", "product_id"],
        ["order_id", "product_name"],
    ]

    used_key = None
    for keys in candidate_keys:
        if all(k in df.columns for k in keys):
            used_key = keys
            break

    before = len(df)
    if used_key is not None:
        df = df.drop_duplicates(subset=used_key)
        print(
            f"[INFO] Drop duplicates by {used_key}: {before - len(df)} baris dihapus.")
    else:
        df = df.drop_duplicates()
        print(
            f"[INFO] Drop full-row duplicates: {before - len(df)} baris dihapus.")

    return df


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tangani missing values di kolom numerik esensial:
    - Drop baris yang missing di kolom: sales, quantity, profit, discount (kalau ada).
    Return:
        df_bersih, df_dibuang
    """
    df = df.copy()

    essential_numeric = [c for c in ["sales", "quantity",
                                     "profit", "discount"] if c in df.columns]

    if not essential_numeric:
        print("[WARN] Tidak menemukan kolom numerik esensial untuk cek missing.")
        return df, pd.DataFrame(columns=df.columns)

    mask_missing = df[essential_numeric].isna().any(axis=1)
    dropped = df[mask_missing].copy()
    kept = df[~mask_missing].copy()

    print(
        f"[INFO] Missing di kolom {essential_numeric}: {mask_missing.sum()} baris di-drop.")
    return kept, dropped


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah fitur turunan:
    - order_year, order_month, order_quarter
    - shipping_days (ship_date - order_date)
    - profit_margin = profit / sales (jika sales > 0)
    - sales_per_quantity = sales / quantity (jika quantity > 0)
    """
    df = df.copy()

    if "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_quarter"] = df["order_date"].dt.quarter

    if {"order_date", "ship_date"}.issubset(df.columns):
        df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days

    if {"sales", "profit"}.issubset(df.columns):
        df["profit_margin"] = np.where(
            df["sales"] > 0,
            df["profit"] / df["sales"],
            np.nan,
        )

    if {"sales", "quantity"}.issubset(df.columns):
        df["sales_per_quantity"] = np.where(
            df["quantity"] > 0,
            df["sales"] / df["quantity"],
            np.nan,
        )

    return df


def basic_type_casting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi kolom kategorikal ke dtype 'category'.
    """
    df = df.copy()
    cat_candidates = [
        "ship_mode", "segment", "country", "city", "state",
        "region", "market", "category", "sub_category", "order_priority",
    ]

    for col in cat_candidates:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def clean_global_superstore(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline utama cleaning:
    1) Standardise nama kolom
    2) Parse tanggal
    3) Drop duplicates
    4) Handle missing values
    5) Tambah fitur turunan
    6) Type casting kategori
    """
    print("[STEP] Standardise column names...")
    df = standardise_column_names(df_raw)

    print("[STEP] Parse date columns...")
    df = parse_dates(df)

    print("[STEP] Drop duplicates...")
    df = drop_duplicates_safe(df)

    print("[STEP] Handle missing values (numeric essentials)...")
    df, dropped = handle_missing_values(df)
    if len(dropped) > 0:
        save_csv(
            dropped, "data/interim/global_superstore_dropped_missing.csv", index=False)
        print("[INFO] Baris yang dibuang karena missing disimpan di data/interim/")

    print("[STEP] Add derived features...")
    df = add_derived_features(df)

    print("[STEP] Basic type casting...")
    df = basic_type_casting(df)

    print("[INFO] Cleaning selesai. Shape akhir:", df.shape)
    return df


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_FILE) -> str:
    """
    Simpan dataset yang sudah dibersihkan ke data/processed.
    """
    save_csv(df, path, index=False)
    print(f"[INFO] Data bersih disimpan ke: {path}")
    return path
