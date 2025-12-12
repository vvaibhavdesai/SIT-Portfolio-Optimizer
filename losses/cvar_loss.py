"""
CVaR (Conditional Value at Risk) loss for portfolio optimization.

This decision-focused loss computes CVaR across the horizon dimension for each
sample, then averages across the batch.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class CVaRLoss(nn.Module):
    """Compute CVaR loss over horizon steps for a batch of portfolios."""

    def __init__(self, alpha: float = 0.95):
        """
        Args:
            alpha: CVaR confidence level (e.g., 0.95 for 95% CVaR).
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self, weights: torch.Tensor, future_returns: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            weights: (B, H, D) portfolio weights (e.g., softmax outputs)
            future_returns: (B, H, D) unscaled asset returns

        Returns:
            loss: scalar CVaR loss
            logs: dict with mean_port_ret, mean_cvar, mean_var
        """
        if weights.shape != future_returns.shape:
            raise ValueError("weights and future_returns must have the same shape (B, H, D).")

        # Portfolio return per horizon step
        port_ret = torch.sum(weights * future_returns, dim=-1)  # (B, H)

        # Loss sign flip: we minimize negative returns
        port_loss = -port_ret  # (B, H)

        # VaR across horizon (per sample), then CVaR
        VaR = torch.quantile(port_loss, self.alpha, dim=1)  # (B,)
        CVaR = VaR + torch.relu(port_loss - VaR.unsqueeze(1)).mean(dim=1) / (1 - self.alpha)  # (B,)

        loss = CVaR.mean()

        logs = {
            "mean_port_ret": port_ret.mean(),
            "mean_cvar": CVaR.mean(),
            "mean_var": VaR.mean(),
        }
        return loss, logs
