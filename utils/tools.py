"""
Training utilities for SIT.
"""

from __future__ import annotations

import torch


class EarlyStopping:
    """
    Early stops training if validation score (e.g., CVaR) doesn't improve.
    """

    def __init__(self, patience: int = 3, verbose: bool = False, delta: float = 0.0, path: str = "checkpoint.pth"):
        """
        Args:
            patience: epochs to wait after last improvement.
            verbose: print messages on improvement.
            delta: minimum change to qualify as improvement (higher is better).
            path: where to save the best model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = None

    def step(self, val_score: float, model: torch.nn.Module) -> bool:
        """
        Update early stopping state.

        Returns:
            True if training should stop, False otherwise.
        """
        score = -val_score  # lower val_score (e.g., CVaR) is better
        if self.best_score is None:
            self.best_score = score
            self.val_best = val_score
            self._save_checkpoint(model, val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_best = val_score
            self._save_checkpoint(model, val_score)
            self.counter = 0
        return self.early_stop

    def _save_checkpoint(self, model: torch.nn.Module, val_score: float):
        """Save model checkpoint when validation improves."""
        if self.verbose:
            print(f"Validation improved ({self.val_best} --> {val_score}). Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)


class LRScheduler:
    """
    Learning rate scheduler utilities.
    """

    @staticmethod
    def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, mode: str = "type1"):
        """
        Adjust learning rate in-place.

        type1: lr = base_lr * 0.5 ** ((epoch-1)//1)
        type2: placeholder for discrete schedules if needed.
        """
        if mode == "type1":
            lr = base_lr * (0.5 ** ((epoch - 1) // 1))
        elif mode == "type2":
            lr = base_lr  # extend as needed
        else:
            raise ValueError(f"Unknown LR scheduler mode: {mode}")

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
