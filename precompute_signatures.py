"""
Offline signature computation and caching.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from sit_clean.data_provider.data_factory import load_prices_csv, prepare_datasets


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _build_cache_root(output_dir: Path, window_size: int, horizon: int, pool: int) -> Path:
    return output_dir / f"signature_cache_W{window_size}H{horizon}" / f"pool_{pool}"


def _save_memmaps(root: Path, prefix: str, x: np.ndarray, cross: np.ndarray, ret: np.ndarray, dates: np.ndarray):
    np.save(root / f"{prefix}_x.npy", x)
    np.save(root / f"{prefix}_cross.npy", cross)
    np.save(root / f"{prefix}_ret.npy", ret)
    np.save(root / f"{prefix}_dates.npy", dates)


def _ds_to_arrays(ds) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert Dataset_Sig to arrays."""
    x_list, cross_list, ret_list, dates = [], [], [], []
    for sample in tqdm(ds, desc="Converting dataset"):
        x_list.append(sample["x_sigs"])
        cross_list.append(sample["cross_sigs"])
        ret_list.append(sample["future_return_unscaled"])
        dates.append(sample["date"])
    return (
        np.stack(x_list, axis=0),
        np.stack(cross_list, axis=0),
        np.stack(ret_list, axis=0),
        np.array(dates),
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute signatures for SIT.")
    parser.add_argument("--data_path", type=str, default="./data/full_dataset.csv")
    parser.add_argument("--data_pool", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./signature_cache")

    args = parser.parse_args()

    prices_df = load_prices_csv(args.data_path)
    tickers = [c for c in prices_df.columns if c != "Date"][: args.data_pool]
    prices_df = prices_df[["Date"] + tickers]

    print("Building datasets...")
    train_ds, val_ds, test_ds, scaler = prepare_datasets(
        prices_df, window_size=args.window_size, horizon=args.horizon, tickers=tickers
    )

    cache_root = _build_cache_root(Path(args.output_dir), args.window_size, args.horizon, args.data_pool)
    _ensure_dir(cache_root)

    print("Converting train dataset...")
    train_x, train_cross, train_ret, train_dates = _ds_to_arrays(train_ds)
    print("Converting val dataset...")
    val_x, val_cross, val_ret, val_dates = _ds_to_arrays(val_ds)
    print("Converting test dataset...")
    test_x, test_cross, test_ret, test_dates = _ds_to_arrays(test_ds)

    print("Saving memmaps...")
    _save_memmaps(cache_root, "train", train_x, train_cross, train_ret, train_dates)
    _save_memmaps(cache_root, "val", val_x, val_cross, val_ret, val_dates)
    _save_memmaps(cache_root, "test", test_x, test_cross, test_ret, test_dates)

    print("Saving scaler and metadata...")
    with open(cache_root / "signature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata: Dict[str, Dict] = {
        "train": {"x": train_x.shape, "cross": train_cross.shape, "ret": train_ret.shape},
        "val": {"x": val_x.shape, "cross": val_cross.shape, "ret": val_ret.shape},
        "test": {"x": test_x.shape, "cross": test_cross.shape, "ret": test_ret.shape},
        "params": {
            "window_size": args.window_size,
            "horizon": args.horizon,
            "data_pool": args.data_pool,
        },
    }
    with open(cache_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Signature cache saved to {cache_root}")


if __name__ == "__main__":
    main()
