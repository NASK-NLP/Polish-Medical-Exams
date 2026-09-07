#!/usr/bin/env python3
"""Upload pandas pickle files to Hugging Face Datasets.

Examples:
    python scripts/run_scripts/upload_pickles_to_hf.py \
        --data-dir data \
        --repo-id your-hf-username/polish-medical-exams

    python scripts/run_scripts/upload_pickles_to_hf.py \
        --data-dir data \
        --repo-id your-hf-username/polish-medical-exams \
        --one-repo-per-file
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload pandas pickle files to Hugging Face Datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing pickle files (default: data).",
    )
    parser.add_argument(
        "--pattern",
        default="*_pandas.pickle",
        help="Glob pattern for pickle files (default: *_pandas.pickle).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF dataset repo id, e.g. username/polish-medical-exams.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the dataset repo as private.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses cached login or HF_TOKEN env var.",
    )
    parser.add_argument(
        "--one-repo-per-file",
        action="store_true",
        help=(
            "If set, upload each pickle as a separate dataset repo. "
            "Repo name becomes <repo-id>-<file-stem>."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Max shard size used by push_to_hub (default: 500MB).",
    )
    return parser.parse_args()


def to_hf_safe_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-._")
    return safe or "dataset"


def normalize_split_name(file_stem: str) -> str:
    # Keep viewer-friendly split names in HF by removing legacy suffix.
    return file_stem.removesuffix("_pandas")


def load_pickle_as_dataset(path: Path) -> Dataset:
    obj = pd.read_pickle(path)

    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name or "value")
    elif isinstance(obj, dict):
        df = pd.DataFrame(obj)
    elif isinstance(obj, list):
        df = pd.DataFrame({"value": obj})
    else:
        raise TypeError(
            f"Unsupported object type in {path.name}: {type(obj).__name__}. "
            "Expected DataFrame/Series/dict/list."
        )

    # Convert columns with types that datasets/Arrow can't handle:
    #   - pandas ArrowDtype-backed columns (e.g. extension<arrow.uuid>)
    #   - object columns containing uuid.UUID instances
    import math
    for col in df.columns:
        dtype = df[col].dtype
        # ArrowDtype (pandas 2.0+ with pyarrow backend) – convert to standard numpy dtype.
        if isinstance(dtype, pd.ArrowDtype):
            import pyarrow as pa
            pa_type = dtype.pyarrow_dtype
            if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
                df[col] = df[col].astype(object)  # plain object/str column
            else:
                # For UUID and other extension types, stringify.
                df[col] = df[col].apply(
                    lambda v: str(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
                )
        # Object columns may contain uuid.UUID instances.
        elif dtype == object:
            sample = df[col].dropna()
            if not sample.empty and hasattr(sample.iloc[0], "hex"):
                df[col] = df[col].apply(lambda v: str(v) if v is not None else None)

    # Avoid persisting pandas index as a synthetic column unless user wants it explicitly.
    return Dataset.from_pandas(df, preserve_index=False)


def find_pickles(data_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(p for p in data_dir.glob(pattern) if p.is_file())


def align_datasets_to_common_schema(datasets: dict[str, "Dataset"]) -> dict[str, "Dataset"]:
    """Add missing columns and unify feature types so all datasets share the same schema."""
    from datasets import Value, Dataset as _Dataset

    # Collect superset of column names (first-seen order).
    all_columns: list[str] = []
    seen: set[str] = set()
    for ds in datasets.values():
        for col in ds.column_names:
            if col not in seen:
                all_columns.append(col)
                seen.add(col)

    # For each column, pick the "best" feature type: prefer any non-null type over null.
    best_features: dict[str, object] = {}
    for col in all_columns:
        for ds in datasets.values():
            if col in ds.features:
                feat = ds.features[col]
                # Value('null') loses to any concrete type.
                if col not in best_features or (
                    isinstance(best_features[col], Value)
                    and best_features[col].dtype == "null"
                    and not (isinstance(feat, Value) and feat.dtype == "null")
                ):
                    best_features[col] = feat

    # Build each dataset: add missing columns then cast to the unified feature type.
    aligned: dict[str, "Dataset"] = {}
    for name, ds in datasets.items():
        missing = [c for c in all_columns if c not in ds.column_names]
        if missing:
            df = ds.to_pandas()
            for col in missing:
                df[col] = None
            ds = _Dataset.from_pandas(df[all_columns], preserve_index=False)
        elif ds.column_names != all_columns:
            ds = ds.select_columns(all_columns)

        # Cast every column to the unified best type.
        from datasets import Features
        target_features = Features({col: best_features[col] for col in all_columns})
        if ds.features != target_features:
            ds = ds.cast(target_features)

        aligned[name] = ds
    return aligned


def upload_as_single_repo(
    repo_id: str,
    files: Iterable[Path],
    private: bool,
    token: str | None,
    max_shard_size: str,
) -> Tuple[int, Dict[str, int]]:
    uploaded = 0
    row_counts: Dict[str, int] = {}

    for path in files:
        subset_name = to_hf_safe_name(normalize_split_name(path.stem))
        ds = load_pickle_as_dataset(path)
        # Upload each pickle under its own subset (config), using a single split "all".
        ds.push_to_hub(
            repo_id=repo_id,
            config_name=subset_name,
            split="all",
            private=private,
            token=token,
            max_shard_size=max_shard_size,
        )
        uploaded += 1
        row_counts[subset_name] = ds.num_rows

    return uploaded, row_counts


def upload_each_to_separate_repos(
    base_repo_id: str,
    files: Iterable[Path],
    private: bool,
    token: str | None,
    max_shard_size: str,
) -> Tuple[int, Dict[str, int]]:
    uploaded = 0
    row_counts: Dict[str, int] = {}

    for path in files:
        suffix = to_hf_safe_name(normalize_split_name(path.stem))
        repo_id = f"{base_repo_id}-{suffix}"
        ds = load_pickle_as_dataset(path)
        ds.push_to_hub(
            repo_id=repo_id,
            split="all",
            private=private,
            token=token,
            max_shard_size=max_shard_size,
        )
        uploaded += 1
        row_counts[repo_id] = ds.num_rows

    return uploaded, row_counts


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = list(find_pickles(data_dir, args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No pickle files found in {data_dir} with pattern '{args.pattern}'."
        )

    print(f"Found {len(files)} pickle files in {data_dir}:")
    for p in files:
        print(f"  - {p.name}")

    if args.one_repo_per_file:
        uploaded, counts = upload_each_to_separate_repos(
            base_repo_id=args.repo_id,
            files=files,
            private=args.private,
            token=args.token,
            max_shard_size=args.max_shard_size,
        )
        print(f"\nUploaded {uploaded} dataset repos.")
        for repo, n_rows in counts.items():
            print(f"  - {repo}: {n_rows} rows")
    else:
        uploaded, counts = upload_as_single_repo(
            repo_id=args.repo_id,
            files=files,
            private=args.private,
            token=args.token,
            max_shard_size=args.max_shard_size,
        )
        print(f"\nUploaded dataset repo: {args.repo_id}")
        print(f"Created/updated {uploaded} subsets with split 'all':")
        for subset, n_rows in counts.items():
            print(f"  - {subset}/all: {n_rows} rows")


if __name__ == "__main__":
    main()
 