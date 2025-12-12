"""
CLI entry point for the clean SIT implementation.
"""
from __future__ import annotations
import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_factory import load_prices_csv, prepare_datasets
from experiments.exp_main import Args, Exp_Main


def custom_collate(batch):
    """Custom collate function that handles dates separately"""
    x_sigs = torch.stack([torch.from_numpy(b['x_sigs']) for b in batch])
    cross_sigs = torch.stack([torch.from_numpy(b['cross_sigs']) for b in batch])
    date_feats = torch.stack([torch.from_numpy(b['date_feats']) for b in batch])
    future_returns = torch.stack([torch.from_numpy(b['future_return_unscaled']) for b in batch])
    dates = [b['date'] for b in batch]
    
    return {
        'x_sigs': x_sigs,
        'cross_sigs': cross_sigs,
        'date_feats': date_feats,
        'future_return_unscaled': future_returns,
        'dates': dates
    }


def fix_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare train/val/test dataloaders from CSV."""
    prices_df = load_prices_csv(args.data_path)
    all_tickers = [c for c in prices_df.columns if c != "Date"]
    tickers = all_tickers[: args.data_pool] if args.data_pool > 0 else all_tickers
    
    train_ds, val_ds, test_ds, scaler = prepare_datasets(
        prices_df, window_size=args.window_size, horizon=args.horizon, tickers=tickers
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=custom_collate)
    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="SIT - Signature-based Portfolio Optimization")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to price CSV")
    parser.add_argument("--data_pool", type=int, default=30, help="Number of assets to use")
    parser.add_argument("--window_size", type=int, default=60, help="Historical window W")
    parser.add_argument("--horizon", type=int, default=20, help="Future horizon H")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--ff_dim", type=int, default=32, help="Feedforward dimension")
    parser.add_argument("--hidden_c", type=int, default=64, help="Hidden size for cross-sig MLP")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_position", type=float, default=0.1, help="Max position size (unused)")
    parser.add_argument("--temperature", type=float, default=1.3, help="Softmax temperature")
    
    # Training parameters
    parser.add_argument("--train_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--cvar_alpha", type=float, default=0.95, help="CVaR confidence level")
    
    # Backtest parameters
    parser.add_argument("--trade_cost_bps", type=float, default=10.0, help="Trading cost in bps")
    
    # Other
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./results", help="Results directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    fix_seed(args.seed)
    
    # Setup device
    device_str = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    
    # Build loaders
    train_loader, val_loader, test_loader = build_loaders(args)
    
    # Create experiment args - pass all parameters that Args expects
    exp_args = Args(
        data_path=args.data_path,
        data_pool=args.data_pool,
        window_size=args.window_size,
        horizon=args.horizon,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        hidden_c=args.hidden_c,
        dropout=args.dropout,
        max_position=args.max_position,
        temperature=args.temperature,
        cvar_alpha=args.cvar_alpha,
        trade_cost_bps=args.trade_cost_bps,
        device=device_str,
        num_assets=args.data_pool,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    # Create experiment
    exp = Exp_Main(exp_args)
    
    # Setting string for naming
    setting = f"SIT_pool{args.data_pool}_w{args.window_size}_h{args.horizon}_dm{args.d_model}_nh{args.n_heads}_nl{args.num_layers}"
    print(f">>> Training : {setting}")
    exp.train(setting=setting)

    print(f">>> Testing : {setting}")
    exp.test(setting=setting)

if __name__ == "__main__":
    main()
