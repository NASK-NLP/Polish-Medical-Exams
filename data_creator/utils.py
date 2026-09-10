from __future__ import annotations

from datasets import load_dataset

from config.paths import HF_REPO


def resolve_source_split_ids(dataset_str: str) -> tuple[str, str | None]:
    """Map user-provided dataset identifier to PL/EN source split ids."""
    key = dataset_str.lower()

    if "ldek" in key:
        return "r_ldek_pandas", "r_ldek_en_pandas"
    if "lek" in key:
        return "r_lek_pandas", "r_lek_en_pandas"
    if "pes" in key:
        return "r_pes_latest_pandas", None
    if "diagnostics" in key:
        return "r_diagnostics_pandas", None
    if "pharmacy" in key:
        return "r_pharmacy_pandas", None

    raise ValueError(f"Unsupported dataset identifier: {dataset_str}")


def load_hf_split_compatible(split_name: str):
    """Load a dataset split from HF, preferring subset/config + split='all'."""
    base = split_name.removesuffix("_pandas")
    candidates = [split_name, base, f"{base}_pandas"]
    tried = set()
    last_error = None

    for candidate in candidates:
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            return load_dataset(HF_REPO, candidate, split="all").to_pandas()
        except Exception as err:
            last_error = err
            continue

    # Fallback for folder-based parquet layout (e.g. <subset>/all-*.parquet).
    for candidate in candidates:
        try:
            return load_dataset(
                "parquet",
                data_files={"train": f"hf://datasets/{HF_REPO}/{candidate}/*.parquet"},
                split="train",
            ).to_pandas()
        except Exception as err:
            last_error = err
            continue

    for candidate in candidates:
        try:
            return load_dataset(HF_REPO, split=candidate).to_pandas()
        except Exception as err:
            last_error = err
            continue

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not resolve split: {split_name}")