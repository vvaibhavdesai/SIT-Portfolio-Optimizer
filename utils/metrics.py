"""
Backtest and evaluation metrics for SIT portfolios.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio assuming 252 trading days.
    """
    if returns.size == 0:
        return float("nan")
    excess = returns - risk_free_rate / 252.0
    std = np.std(excess)
    if std == 0:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(252.0))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sortino ratio using downside deviation.
    """
    if returns.size == 0:
        return float("nan")
    excess = returns - risk_free_rate / 252.0
    downside = excess[excess < 0]
    down_std = np.std(downside) if downside.size > 0 else 0.0
    if down_std == 0:
        return float("nan")
    return float(np.mean(excess) / down_std * np.sqrt(252.0))


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown from peak.
    """
    if equity_curve.size == 0:
        return float("nan")
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / cumulative_max - 1.0
    return float(drawdown.min())


def calculate_annual_return(equity_curve: np.ndarray, years: float) -> float:
    """
    Annualized return given total horizon in years.
    """
    if equity_curve.size == 0 or years <= 0:
        return float("nan")
    final = equity_curve[-1]
    if final <= 0:
        return float("nan")
    return float(final ** (1.0 / years) - 1.0)


def calculate_annual_volatility(returns: np.ndarray) -> float:
    """
    Annualized volatility assuming 252 trading days.
    """
    if returns.size == 0:
        return float("nan")
    return float(np.std(returns) * np.sqrt(252.0))


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Fraction of positive return days.
    """
    if returns.size == 0:
        return float("nan")
    return float((returns > 0).mean())


def calculate_final_wealth_factor(equity_curve: np.ndarray) -> float:
    """
    Final wealth divided by initial wealth.
    """
    if equity_curve.size == 0:
        return float("nan")
    return float(equity_curve[-1] / equity_curve[0])


def compute_metrics(returns: np.ndarray, equity_curve: np.ndarray, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Aggregate backtest metrics into a dictionary.
    """
    days = len(returns)
    years = days / 252.0 if days > 0 else 0.0

    metrics = {
        "sharpe": calculate_sharpe_ratio(returns, risk_free_rate),
        "sortino": calculate_sortino_ratio(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "annual_return": calculate_annual_return(equity_curve, years),
        "annual_vol": calculate_annual_volatility(returns),
        "final_wealth": float(equity_curve[-1]) if equity_curve.size else float("nan"),
        "win_rate": calculate_win_rate(returns),
        "wealth_factor": calculate_final_wealth_factor(equity_curve),
    }
    return metrics
