"""Validate dataframes against schemas/tables.yaml before writing.

Cheap, but it is the difference between a pipeline and a pile of scripts:
a stage cannot emit a table that a later stage cannot read.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .config import REPO_ROOT

_SCHEMA_PATH = REPO_ROOT / "schemas" / "tables.yaml"


def load_schemas() -> dict:
    with open(_SCHEMA_PATH) as fh:
        return yaml.safe_load(fh)


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, table: str, strict: bool = True) -> pd.DataFrame:
    schemas = load_schemas()
    if table not in schemas:
        raise SchemaError(f"unknown table '{table}'")
    spec = schemas[table]
    cols = spec.get("columns")
    if not cols:
        return df

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"[{table}] missing columns: {missing}")

    extra = [c for c in df.columns if c not in cols]
    if extra and strict:
        raise SchemaError(f"[{table}] unexpected columns: {extra}")

    pk = spec.get("primary_key")
    if pk and df.duplicated(subset=pk).any():
        n = int(df.duplicated(subset=pk).sum())
        raise SchemaError(f"[{table}] {n} duplicate rows on primary key {pk}")

    for name, meta in cols.items():
        rng = meta.get("range")
        if rng is not None and len(df):
            bad = df[(df[name] < rng[0]) | (df[name] > rng[1])]
            if len(bad):
                raise SchemaError(f"[{table}] {len(bad)} values of '{name}' outside {rng}")
        enum = meta.get("enum")
        if enum is not None and len(df):
            bad = set(df[name].dropna().unique()) - set(enum)
            if bad:
                raise SchemaError(f"[{table}] illegal values in '{name}': {sorted(bad)}")
    return df[list(cols)]


def write_table(df: pd.DataFrame, table: str, path: Path, strict: bool = True) -> Path:
    df = validate(df, table, strict=strict)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_table(table: str, path: Path, strict: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return validate(df, table, strict=strict)
