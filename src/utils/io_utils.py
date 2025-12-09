# src/utils/io_utils.py

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

# Path proyek (folder yang berisi README.md, data/, src/, dll)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def get_project_root() -> Path:
    """
    Return folder root project.
    """
    return PROJECT_ROOT


def get_data_path(kind: str = "raw") -> Path:
    """
    Return path ke folder data.
    kind: 'raw', 'processed', atau 'interim'
    """
    kind = kind.lower()
    if kind not in {"raw", "processed", "interim"}:
        raise ValueError(
            f"kind harus 'raw', 'processed', atau 'interim', bukan '{kind}'")

    return DATA_DIR / kind


def load_csv(path: PathLike, encoding: str = "utf-8", **read_kwargs) -> pd.DataFrame:
    """
    Load CSV dengan encoding default 'utf-8'.
    Jika gagal (UnicodeDecodeError), otomatis coba 'latin1'.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    default_kwargs = {"encoding": encoding, "low_memory": False}
    default_kwargs.update(read_kwargs)

    try:
        df = pd.read_csv(path, **default_kwargs)
    except UnicodeDecodeError:
        # fallback ke latin1
        print(
            f"[WARN] Gagal decode dengan {encoding}, mencoba ulang dengan 'latin1'...")
        default_kwargs["encoding"] = "latin1"
        df = pd.read_csv(path, **default_kwargs)

    return df


def save_csv(df: pd.DataFrame, path: PathLike, index: bool = False, **to_csv_kwargs) -> Path:
    """
    Simpan DataFrame ke CSV dengan path relatif ke project root.
    Otomatis buat folder jika belum ada.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    default_kwargs = {"encoding": "utf-8"}
    default_kwargs.update(to_csv_kwargs)

    df.to_csv(path, index=index, **default_kwargs)
    return path
