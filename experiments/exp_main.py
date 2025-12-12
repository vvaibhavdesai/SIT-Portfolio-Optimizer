"""
Experiment runner for the SIT model.

Provides training, validation, testing, and backtesting loops with CVaR loss.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from losses.cvar_loss import CVaRLoss
from models.sit import SIT


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _to_device(batch_item: Any, device: torch.device) -> Any:
    if isinstance(batch_item, torch.Tensor):
        return batch_item.to(device)
    if isinstance(batch_item, np.ndarray):
        return torch.from_numpy(batch_item).to(device)
    return batch_item


def _extract_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[pd.Timestamp]]:
    """
    Accept either a dict or a tuple and move tensors to device.
    Expected keys/ordering: x_sigs, cross_sigs, future_return_unscaled, date_feats, date.
    """
    if isinstance(batch, dict):
        x_sigs = _to_device(batch["x_sigs"], device)
        cross_sigs = _to_device(batch["cross_sigs"], device)
        future_returns = _to_device(batch["future_return_unscaled"], device)
        date_feats = _to_device(batch["date_feats"], device)
        dates = batch.get("date", batch.get("dates"))
    else:
        x_sigs, cross_sigs, date_feats, future_returns, dates = batch  # type: ignore
        x_sigs = _to_device(x_sigs, device)
        cross_sigs = _to_device(cross_sigs, device)
        date_feats = _to_device(date_feats, device)
        future_returns = _to_device(future_returns, device)
    return x_sigs, cross_sigs, date_feats, future_returns, dates


def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int):
    lr = base_lr * (0.5 ** ((epoch - 1) // 1))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _generate_rebalance_dates(start: str = "2020-01-01", end: str = "2024-12-31") -> set:
    """Monthly (BMS) rebalance dates within the period."""
    dates = pd.bdate_range(start=start, end=end, freq="BMS")
    return set(pd.to_datetime(dates))


def _compute_basic_metrics(returns: np.ndarray, equity_curve: np.ndarray) -> Dict[str, float]:
    """Fallback metrics computation (annualized) if external metrics are unavailable."""
    daily_ret = returns
    ann_factor = 252.0
    mean_ret = float(np.mean(daily_ret))
    std_ret = float(np.std(daily_ret))
    sharpe = mean_ret / std_ret * np.sqrt(ann_factor) if std_ret > 0 else np.nan

    downside = daily_ret[daily_ret < 0]
    down_std = float(np.std(downside)) if downside.size > 0 else 0.0
    sortino = mean_ret / down_std * np.sqrt(ann_factor) if down_std > 0 else np.nan

    cumulative = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / cumulative - 1.0
    max_dd = float(drawdown.min()) if drawdown.size > 0 else np.nan

    ann_ret = ((1 + mean_ret) ** ann_factor) - 1
    ann_vol = std_ret * np.sqrt(ann_factor)
    win_rate = float((daily_ret > 0).mean()) if daily_ret.size > 0 else np.nan

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "final_wealth": float(equity_curve[-1]) if len(equity_curve) else np.nan,
        "win_rate": win_rate,
    }


class Args:
    """Container for experiment arguments (defaults applied when missing)."""

    def __init__(
        self,
        data_path: str = "./data/full_dataset.csv",
        data_pool: int = 30,
        window_size: int = 60,
        horizon: int = 20,
        batch_size: int = 64,
        train_epochs: int = 30,
        learning_rate: float = 1e-3,
        patience: int = 3,
        d_model: int = 8,
        n_heads: int = 4,
        num_layers: int = 1,
        ff_dim: int = 32,
        hidden_c: int = 64,
        dropout: float = 0.1,
        max_position: float = 0.1,
        temperature: float = 1.3,
        cvar_alpha: float = 0.95,
        trade_cost_bps: float = 0.0,
        use_gpu: bool = True,
        use_multi_gpu: bool = False,
        use_precomputed: bool = False,
        precomp_root: str = "./signature_cache",
        num_workers: int = 0,
        seed: int = 42,
        num_assets: Optional[int] = None,
        bias_embed_dim: int = 16,
        device: str = "cpu",
        output_dir: str = "./results",
        checkpoint_dir: str = "./checkpoints",
        train_loader: Optional[Iterable] = None,
        val_loader: Optional[Iterable] = None,
        test_loader: Optional[Iterable] = None,
    ):
        self.data_path = data_path
        self.data_pool = data_pool
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.hidden_c = hidden_c
        self.dropout = dropout
        self.max_position = max_position
        self.temperature = temperature
        self.cvar_alpha = cvar_alpha
        self.trade_cost_bps = trade_cost_bps
        self.use_gpu = use_gpu
        self.use_multi_gpu = use_multi_gpu
        self.use_precomputed = use_precomputed
        self.precomp_root = precomp_root
        self.num_workers = num_workers
        self.seed = seed
        self.num_assets = num_assets if num_assets is not None else data_pool
        self.bias_embed_dim = bias_embed_dim
        self.device = device
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


class Exp_Main:
    """Main experiment class with training, validation, and backtesting."""

    def __init__(self, args: Args):
        self.args = args
        self.device = torch.device(args.device)

        self.model = SIT(
            num_assets=args.num_assets,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            hidden_c=args.hidden_c,
            bias_embed_dim=args.bias_embed_dim,
            dropout=args.dropout,
            max_position=args.max_position,
            temperature=args.temperature,
        ).to(self.device)

        self.criterion = CVaRLoss(alpha=args.cvar_alpha)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.base_lr = args.learning_rate
        self.patience = args.patience
        self.epochs = args.train_epochs
        self.temperature = args.temperature

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader
        self.test_loader = args.test_loader

        self.checkpoint_dir = Path(args.checkpoint_dir)
        _ensure_dir(self.checkpoint_dir)
        self.output_dir = Path(args.output_dir)
        _ensure_dir(self.output_dir)

        self.best_ckpt = self.checkpoint_dir / "checkpoint.pth"

    def train(self, setting: str = "default"):
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("train_loader and val_loader must be provided.")

        best_val = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            _set_lr(self.optimizer, self.base_lr, epoch)
            self.model.train()

            epoch_loss = 0.0
            start_time = time.time()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}")

            for step, batch in pbar:
                x_sigs, cross_sigs, date_feats, future_returns, _ = _extract_batch(batch, self.device)

                model_out = self.model(x_sigs, cross_sigs, date_feats, future_returns)
                if isinstance(model_out, tuple) and len(model_out) == 2:
                    _, mu_hat = model_out
                else:
                    mu_hat = model_out
                weights = F.softmax(mu_hat / self.temperature, dim=-1)

                loss, logs = self.criterion(weights, future_returns)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if (step + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    samples_processed = (step + 1) * x_sigs.size(0)
                    speed = samples_processed / max(elapsed, 1e-6)
                    eta = (len(self.train_loader) - step - 1) * (elapsed / (step + 1))
                    pbar.set_postfix(
                        loss=loss.item(),
                        cvar=logs["mean_cvar"].item(),
                        speed=f"{speed:.1f}/s",
                        eta=f"{eta/60:.1f}m",
                    )

            avg_train_loss = epoch_loss / len(self.train_loader)
            val_cvar = self.validate(self.val_loader)
            print(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} val_cvar={val_cvar:.6f}")

            if val_cvar < best_val:
                best_val = val_cvar
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_ckpt)
                print(f"  Saved new best checkpoint to {self.best_ckpt}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        return best_val

    def validate(self, val_loader) -> float:
        self.model.eval()
        total_cvar = 0.0
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                x_sigs, cross_sigs, date_feats, future_returns, _ = _extract_batch(batch, self.device)
                model_out = self.model(x_sigs, cross_sigs, date_feats, future_returns)
                if isinstance(model_out, tuple) and len(model_out) == 2:
                    _, mu_hat = model_out
                else:
                    mu_hat = model_out
                weights = F.softmax(mu_hat / self.temperature, dim=-1)
                loss, logs = self.criterion(weights, future_returns)
                total_cvar += logs["mean_cvar"].item()
                steps += 1
        return total_cvar / max(steps, 1)

    def test(self, setting: str = "default"):
        if self.test_loader is None:
            raise ValueError("test_loader must be provided.")
        if self.best_ckpt.exists():
            self.model.load_state_dict(torch.load(self.best_ckpt, map_location=self.device))
            print(f"Loaded best checkpoint from {self.best_ckpt}")
        self.model.eval()

        # Collect first occurrence per date
        date_map: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Predicting (test)"):
                x_sigs, cross_sigs, date_feats, future_returns, dates = _extract_batch(batch, self.device)
                model_out = self.model(x_sigs, cross_sigs, date_feats, future_returns)
                if isinstance(model_out, tuple) and len(model_out) == 2:
                    _, mu_hat = model_out
                else:
                    mu_hat = model_out
                weights = F.softmax(mu_hat / self.temperature, dim=-1)

                weights_np = weights.detach().cpu().numpy()
                returns_np = future_returns.detach().cpu().numpy()

                # Expect dates as iterable length B
                for i, dt in enumerate(dates):
                    if dt in date_map:
                        continue
                    date_map[pd.to_datetime(dt)] = {
                        "weights": weights_np[i, 0],      # use first horizon step
                        "returns": returns_np[i, 0],      # use first horizon step
                    }

        if not date_map:
            raise ValueError("No test predictions were generated.")

        # Backtest
        rebalance_dates = _generate_rebalance_dates()
        sorted_dates = sorted(date_map.keys())

        capital = 1.0
        trade_cost_bps = self.args.trade_cost_bps
        current_weights = np.zeros(self.args.num_assets, dtype=np.float32)

        equity_curve = []
        daily_returns = []
        weight_records = []

        for dt in sorted_dates:
            data = date_map[dt]
            new_weights = data["weights"]
            asset_returns = data["returns"]

            rebalance = (dt in rebalance_dates) or np.allclose(current_weights.sum(), 0.0)
            if rebalance:
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost = turnover * (trade_cost_bps * 1e-4) * capital
                current_weights = new_weights
            else:
                cost = 0.0

            port_ret = float(np.dot(current_weights, asset_returns))
            daily_pnl = capital * port_ret - cost
            capital += daily_pnl

            daily_returns.append(port_ret)
            equity_curve.append(capital)
            weight_records.append((dt, current_weights.copy()))

        equity_curve = np.array(equity_curve, dtype=float)
        daily_returns = np.array(daily_returns, dtype=float)

        try:
            from utils.metrics import compute_metrics

            metrics = compute_metrics(daily_returns, equity_curve)
        except Exception:
            metrics = _compute_basic_metrics(daily_returns, equity_curve)

        # Save results
        results_path = self.output_dir / f"results_{setting}.csv"
        positions_path = self.output_dir / f"positions_{setting}.csv"
        plot_path = self.output_dir / f"equity_curve_{setting}.png"

        pd.DataFrame([metrics]).to_csv(results_path, index=False)

        pos_df = pd.DataFrame(
            [
                {"Date": dt, **{f"w_{i}": w for i, w in enumerate(weights)}}
                for dt, weights in weight_records
            ]
        )
        pos_df.to_csv(positions_path, index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(sorted_dates, equity_curve, label="Equity Curve")
        plt.title("Backtest Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Wealth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Backtest complete. Metrics saved to {results_path}")
        return metrics
