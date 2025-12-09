# src/data_prep/clean_global_superstore.py

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from src.utils.io_utils import load_csv, save_csv

PathLike = Union[str, Path]

RAW_FILE = "data/raw/Global_Superstore2.csv"
CLEAN_FILE = "data/processed/global_superstore_clean.csv"
MODEL_READY_FILE = "data/processed/global_superstore_model_ready.csv"


# ---------- Load data ----------

def load_raw_data(path: PathLike = RAW_FILE) -> pd.DataFrame:
    """
    Load dataset Global Superstore mentah dari data/raw.
    Encoding akan otomatis fallback ke 'latin1' jika 'utf-8' gagal.
    """
    df = load_csv(path)
    return df


# ---------- Step-step cleaning ----------

def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarkan nama kolom: lowercase, hapus spasi, ganti spasi & '-' dengan underscore.
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
    - Drop baris yang missing di: sales, quantity, profit, discount (kalau ada).
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


def handle_postal_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tangani kolom postal_code yang punya banyak missing.
    Di sini kita drop kolom tersebut karena:
    - persentase missing sangat tinggi
    - informasi lokasi sudah tercakup di city/state/region/market.
    """
    df = df.copy()
    if "postal_code" in df.columns:
        missing_pct = df["postal_code"].isna().mean() * 100
        print(f"[INFO] postal_code missing: {missing_pct:.2f}%")
        print("[INFO] Drop kolom postal_code (banyak missing & redundant dengan city/state/region/market).")
        df = df.drop(columns=["postal_code"])
    return df


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
        df["order_year"] = df["order_date"].dt.year.astype("Int32")
        df["order_month"] = df["order_date"].dt.month.astype("Int32")
        df["order_quarter"] = df["order_date"].dt.quarter.astype("Int32")

    if {"order_date", "ship_date"}.issubset(df.columns):
        df["shipping_days"] = (
            df["ship_date"] - df["order_date"]).dt.days.astype("int64")

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


def create_profit_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat target biner is_profitable:
    - 1 jika profit > 0
    - 0 jika profit <= 0
    """
    df = df.copy()
    if "profit" not in df.columns:
        raise ValueError(
            "Kolom 'profit' tidak ditemukan, tidak bisa membuat is_profitable.")

    df["is_profitable"] = (df["profit"] > 0).astype("int32")
    print("[INFO] Kolom target 'is_profitable' dibuat (1 = profit > 0, 0 = profit <= 0).")
    return df


# ---------- Encoding untuk modelling ----------

def encode_categoricals_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Siapkan dataset untuk modelling klasifikasi is_profitable:

    - Drop kolom ID & kolom yang menyebabkan leakage:
      ['row_id', 'order_id', 'customer_id', 'customer_name',
       'product_id', 'product_name', 'profit', 'profit_margin']

    - Encode ordinal untuk 'order_priority':
        Low < Medium < High < Critical

    - One-hot encoding (get_dummies) untuk semua kolom object/category
      lainnya (drop_first=True) sehingga hasilnya full numerik.

    Kolom target 'is_profitable' tetap dipertahankan.
    """
    df_model = df.copy()

    # --- Drop kolom non-fitur / leaky ---
    drop_cols = [
        "row_id",
        "order_id",
        "customer_id",
        "customer_name",
        "product_id",
        "product_name",
        "profit",          # leakage
        "profit_margin",   # leakage
    ]
    drop_existing = [c for c in drop_cols if c in df_model.columns]
    if drop_existing:
        print(f"[INFO] Drop kolom non-fitur / leaky: {drop_existing}")
        df_model = df_model.drop(columns=drop_existing)

    # --- Identifikasi fitur numerik & kategorikal SEBELUM encoding ---
    all_numeric = df_model.select_dtypes(include=["number"]).columns.tolist()
    # target sebaiknya tidak dianggap fitur
    numeric_feature_cols = [c for c in all_numeric if c != "is_profitable"]

    cat_feature_cols = df_model.select_dtypes(
        include=["object", "category"]).columns.tolist()

    print("\n[INFO] Ringkasan fitur sebelum encoding:")
    print(f"  - Target        : 'is_profitable'")
    print(f"  - Fitur numerik : {numeric_feature_cols}")
    print(f"  - Fitur kategorikal: {cat_feature_cols}")

    # --- Ordinal encoding untuk order_priority ---
    if "order_priority" in df_model.columns:
        print(
            "\n[INFO] Ordinal encoding untuk 'order_priority' (Low<Medium<High<Critical)")
        priority_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

        raw = df_model["order_priority"].astype(str).str.strip().str.title()
        df_model["order_priority"] = raw.map(priority_map)

        median_rank = int(df_model["order_priority"].median())
        df_model["order_priority"] = df_model["order_priority"].fillna(
            median_rank).astype("int32")

    # --- One-hot encoding untuk kategorikal lain ---
    cat_cols_for_dummies = df_model.select_dtypes(
        include=["object", "category"]).columns.tolist()
    if cat_cols_for_dummies:
        print(
            f"\n[INFO] One-hot encoding (get_dummies, drop_first=True) untuk: {cat_cols_for_dummies}")
        df_model = pd.get_dummies(
            df_model, columns=cat_cols_for_dummies, drop_first=True)
    else:
        print(
            "\n[INFO] Tidak ada kolom kategorikal lain untuk di-encode dengan one-hot.")

    return df_model


# ---------- Pipeline utama & saving ----------

def clean_global_superstore(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline utama cleaning:
    1) Standardise nama kolom
    2) Parse tanggal
    3) Drop duplicates
    4) Handle missing numerik esensial
    5) Handle postal_code (drop)
    6) Fitur turunan
    7) Type casting kategori
    8) Buat target is_profitable
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
        print("[INFO] Baris yang dibuang karena missing disimpan di data/interim/global_superstore_dropped_missing.csv")

    print("[STEP] Handle postal_code (missing tinggi & redundant)...")
    df = handle_postal_code(df)

    print("[STEP] Add derived features...")
    df = add_derived_features(df)

    print("[STEP] Basic type casting (categorical)...")
    df = basic_type_casting(df)

    print("[STEP] Create profit flag (is_profitable)...")
    df = create_profit_flag(df)

    print("[INFO] Cleaning selesai. Shape akhir:", df.shape)
    return df


def save_clean_data(df: pd.DataFrame, path: PathLike = CLEAN_FILE) -> str:
    """
    Simpan dataset yang sudah dibersihkan ke data/processed.
    """
    save_csv(df, path, index=False)
    print(f"[INFO] Data bersih disimpan ke: {path}")
    return str(path)


def save_model_ready_data(df_model: pd.DataFrame, path: PathLike = MODEL_READY_FILE) -> str:
    """
    Simpan dataset yang sudah di-encode (siap modelling) ke data/processed.
    """
    save_csv(df_model, path, index=False)
    print(f"[INFO] Data model-ready disimpan ke: {path}")
    return str(path)
