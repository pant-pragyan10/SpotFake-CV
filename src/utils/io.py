from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(dataframe: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        dataframe.to_parquet(output_path, index=False)
    elif suffix == ".json":
        output_path.write_text(dataframe.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported dataframe format: {output_path.suffix}")
    return output_path


def save_text(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_json(payload: dict, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path
