"""
Streamlit interface for the SIT (Signature-based Transformer) portfolio optimizer.

Features:
- Dashboard with latest metrics and equity curve.
- Training UI with live progress streamed from the CLI trainer.
- Results analysis with interactive charts and downloads.
- Backtest view with date filtering and turnover costs.
- Data explorer with Plotly visualizations.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import threading
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "full_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

sys.path.append(str(BASE_DIR))

from data_provider.data_factory import load_prices_csv  # noqa: E402
from models.sit import SIT  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402
from run import custom_collate  # noqa: E402


st.set_page_config(
    page_title="SIT Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(data_path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load price data."""
    df = load_prices_csv(data_path)
    return df


@st.cache_data(show_spinner=False)
def list_available_models() -> List[Dict[str, object]]:
    """Scan the results directory for saved runs."""
    if not RESULTS_DIR.exists():
        return []
    items: List[Dict[str, object]] = []
    for csv_path in RESULTS_DIR.glob("results_*.csv"):
        name = csv_path.stem.replace("results_", "")
        equity_path = RESULTS_DIR / f"equity_curve_{name}.png"
        positions_path = RESULTS_DIR / f"positions_{name}.csv"
        items.append(
            {
                "name": name,
                "results": csv_path,
                "equity": equity_path if equity_path.exists() else None,
                "positions": positions_path if positions_path.exists() else None,
                "mtime": csv_path.stat().st_mtime,
            }
        )
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


@st.cache_data(show_spinner=False)
def load_results(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_positions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource(show_spinner=False)
def load_trained_model(
    checkpoint_path: Path,
    num_assets: int,
    model_kwargs: Dict[str, object],
    device: str,
) -> SIT:
    """Load a SIT model from checkpoint."""
    model = SIT(
        num_assets=num_assets,
        d_model=int(model_kwargs.get("d_model", 8)),
        n_heads=int(model_kwargs.get("n_heads", 4)),
        num_layers=int(model_kwargs.get("num_layers", 2)),
        ff_dim=int(model_kwargs.get("ff_dim", 32)),
        hidden_c=int(model_kwargs.get("hidden_c", 64)),
        bias_embed_dim=int(model_kwargs.get("bias_embed_dim", 16)),
        dropout=float(model_kwargs.get("dropout", 0.1)),
        max_position=float(model_kwargs.get("max_position", 0.1)),
        temperature=float(model_kwargs.get("temperature", 1.3)),
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_trained_model_from_name(model_name: str):
    """Load trained model checkpoint by name and return model plus parsed config."""
    import re

    config: Dict[str, int] = {}
    match = re.search(r"pool(\d+)", model_name)
    config["data_pool"] = int(match.group(1)) if match else 30
    match = re.search(r"_w(\d+)", model_name)
    config["window_size"] = int(match.group(1)) if match else 60
    match = re.search(r"_h(\d+)", model_name)
    config["horizon"] = int(match.group(1)) if match else 20
    match = re.search(r"_dm(\d+)", model_name)
    config["d_model"] = int(match.group(1)) if match else 8
    match = re.search(r"_nh(\d+)", model_name)
    config["n_heads"] = int(match.group(1)) if match else 4
    match = re.search(r"_nl(\d+)", model_name)
    config["num_layers"] = int(match.group(1)) if match else 2

    device = torch.device("cpu")
    model = SIT(
        num_assets=config["data_pool"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        ff_dim=32,
        hidden_c=64,
        dropout=0.1,
        max_position=0.1,
        temperature=1.3,
    ).to(device)

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pth"
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINT_DIR / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        st.error(
            f"""
Model architecture mismatch while loading {model_name}

Error: {e}

Detected config from filename:
- Assets: {config['data_pool']}
- Layers: {config['num_layers']}
- d_model: {config['d_model']}
- Heads: {config['n_heads']}

Possible issues:
- Filename does not match actual saved model
- Checkpoint corrupted
- Architecture changed since training

Try retraining or selecting a different checkpoint.
"""
        )
        raise

    model.eval()
    return model, config


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load uploaded CSV content from bytes."""
    return pd.read_csv(io.BytesIO(file_bytes))


def get_available_model_names() -> List[str]:
    """Find trained model names from checkpoints/results files."""
    models: List[str] = []
    if CHECKPOINT_DIR.exists():
        for file in CHECKPOINT_DIR.glob("*.pth"):
            if file.name == "checkpoint.pth":
                continue
            models.append(file.stem)
    if RESULTS_DIR.exists():
        for file in RESULTS_DIR.glob("results_SIT_*.csv"):
            name = file.stem.replace("results_", "")
            if name not in models:
                models.append(name)
    return sorted(models)

def get_available_models_with_checkpoints() -> List[str]:
    """Return model names that have a checkpoint matching basic config expectations."""
    valid: List[str] = []
    # Check results-derived names first
    if RESULTS_DIR.exists():
        for file in RESULTS_DIR.glob("results_SIT_*.csv"):
            model_name = file.stem.replace("results_", "")
            checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pth"
            if not checkpoint_path.exists():
                checkpoint_path = CHECKPOINT_DIR / "checkpoint.pth"
            if not checkpoint_path.exists():
                continue
            try:
                expected = parse_model_config(model_name)
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                else:
                    state = checkpoint
                has_layer_1 = any("temporal_layers.1." in k for k in state.keys())
                actual_layers = 2 if has_layer_1 else 1
                actual_assets = state["asset_embed.weight"].shape[0] if "asset_embed.weight" in state else None
                if expected.get("num_layers") and expected["num_layers"] != actual_layers:
                    continue
                if expected.get("data_pool") and actual_assets and expected["data_pool"] != actual_assets:
                    continue
                valid.append(model_name)
            except Exception:
                continue
    # Also include standalone checkpoints
    if CHECKPOINT_DIR.exists():
        for file in CHECKPOINT_DIR.glob("*.pth"):
            if file.stem == "checkpoint":
                continue
            try:
                checkpoint = torch.load(file, map_location="cpu")
                if isinstance(checkpoint, dict):
                    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                else:
                    state = checkpoint
                has_layer_1 = any("temporal_layers.1." in k for k in state.keys())
                num_layers = 2 if has_layer_1 else 1
                num_assets = state["asset_embed.weight"].shape[0] if "asset_embed.weight" in state else None
                inferred_name = file.stem
                if inferred_name not in valid:
                    valid.append(inferred_name)
            except Exception:
                continue
    return sorted(set(valid))


def parse_model_config(model_name: str) -> Dict[str, Optional[int]]:
    """Extract model hyperparameters from filename."""
    cfg: Dict[str, Optional[int]] = {
        "data_pool": None,
        "window_size": None,
        "horizon": None,
        "d_model": None,
        "n_heads": None,
        "num_layers": None,
    }
    patterns = {
        "data_pool": r"pool(\d+)",
        "window_size": r"_w(\d+)",
        "horizon": r"_h(\d+)",
        "d_model": r"_dm(\d+)",
        "n_heads": r"_nh(\d+)",
        "num_layers": r"_nl(\d+)",
    }
    for key, pat in patterns.items():
        match = re.search(pat, model_name)
        if match:
            cfg[key] = int(match.group(1))
    return cfg


def get_trained_assets_from_pool(data_pool: int) -> List[str]:
    """Return the ordered list of tickers used for training (first N from full dataset)."""
    try:
        base_df = load_data()
        tickers = [c for c in base_df.columns if c != "Date"]
        return tickers[: data_pool]
    except Exception:
        return []


def generate_rebalance_dates(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> List[pd.Timestamp]:
    """Generate rebalance dates by frequency."""
    if freq == "Daily":
        dr = pd.bdate_range(start, end, freq="B")
    elif freq == "Weekly":
        dr = pd.bdate_range(start, end, freq="W-MON")
    else:
        dr = pd.bdate_range(start, end, freq="BMS")
    return list(pd.to_datetime(dr))


# --------------------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------------------
def _default_train_state() -> Dict[str, object]:
    return {
        "status": "idle",
        "progress": 0.0,
        "logs": [],
        "latest_metrics": None,
        "return_code": None,
        "thread": None,
        "error": None,
    }


if "train_state" not in st.session_state:
    st.session_state.train_state = _default_train_state()


def _parse_epoch_line(line: str) -> Optional[Tuple[int, float, float]]:
    """Parse epoch logs like: [Epoch 1] train_loss=... val_cvar=..."""
    match = re.search(r"\[Epoch (\d+)\].*train_loss=([\d.eE+-]+).*val_cvar=([\d.eE+-]+)", line)
    if not match:
        return None
    epoch = int(match.group(1))
    train_loss = float(match.group(2))
    val_cvar = float(match.group(3))
    return epoch, train_loss, val_cvar


def _run_training_subprocess(cmd: List[str], total_epochs: int):
    """Run the CLI trainer in a background thread and stream progress."""
    train_state = st.session_state.train_state
    train_state.update(status="running", progress=0.0, logs=[], latest_metrics=None, error=None)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        train_state.update(status="error", error=str(exc))
        return

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            train_state["logs"].append(line)
            if len(train_state["logs"]) > 300:
                train_state["logs"] = train_state["logs"][-300:]
            parsed = _parse_epoch_line(line)
            if parsed:
                epoch, tr_loss, val_cvar = parsed
                train_state["latest_metrics"] = {"epoch": epoch, "train_loss": tr_loss, "val_cvar": val_cvar}
                train_state["progress"] = min(epoch / max(total_epochs, 1), 1.0)
        process.wait()
        train_state["return_code"] = process.returncode
        train_state["status"] = "completed" if process.returncode == 0 else "error"
        if process.returncode != 0 and train_state.get("error") is None:
            train_state["error"] = "Training process returned a non-zero exit code."
    finally:
        if process.stdout:
            process.stdout.close()


def start_training(params: Dict[str, object], data_path: str):
    """Kick off background training via the CLI script."""
    if st.session_state.train_state.get("status") == "running":
        st.warning("Training is already running.")
        return

    cmd = [
        sys.executable,
        str(BASE_DIR / "run.py"),
        "--data_path",
        data_path,
        "--data_pool",
        str(params["data_pool"]),
        "--window_size",
        str(params["window_size"]),
        "--horizon",
        str(params["horizon"]),
        "--d_model",
        str(params["d_model"]),
        "--n_heads",
        str(params["n_heads"]),
        "--num_layers",
        str(params["num_layers"]),
        "--temperature",
        str(params["temperature"]),
        "--train_epochs",
        str(params["train_epochs"]),
        "--batch_size",
        str(params["batch_size"]),
        "--learning_rate",
        str(params["learning_rate"]),
    ]

    cmd.extend(
        [
            "--ff_dim",
            str(params.get("ff_dim", 32)),
            "--hidden_c",
            str(params.get("hidden_c", 64)),
            "--dropout",
            str(params.get("dropout", 0.1)),
            "--patience",
            str(params.get("patience", 3)),
            "--cvar_alpha",
            str(params.get("cvar_alpha", 0.95)),
            "--trade_cost_bps",
            str(params.get("trade_cost_bps", 10.0)),
        ]
    )

    if params.get("use_gpu") and torch.cuda.is_available():
        cmd.append("--use_gpu")

    thread = threading.Thread(
        target=_run_training_subprocess,
        args=(cmd, int(params["train_epochs"])),
        daemon=True,
    )
    st.session_state.train_state["thread"] = thread
    thread.start()


# --------------------------------------------------------------------------------------
# Backtest and analytics helpers
# --------------------------------------------------------------------------------------
def compute_backtest_from_positions(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    trade_cost_bps: float,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]], np.ndarray]:
    """Recompute equity curve from saved positions and raw prices."""
    pos = positions_df.copy()
    pos["Date"] = pd.to_datetime(pos["Date"])
    if date_range:
        start, end = date_range
        pos = pos[(pos["Date"] >= start) & (pos["Date"] <= end)]
    if pos.empty:
        raise ValueError("No position data in the selected date range.")

    weights_cols = [c for c in pos.columns if c.startswith("w_")]
    num_assets = len(weights_cols)
    tickers = [c for c in prices_df.columns if c != "Date"][:num_assets]

    price_df = prices_df[["Date"] + tickers].copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df = price_df.set_index("Date").sort_index()
    returns_df = price_df.pct_change().dropna()

    merged = pos.set_index("Date").join(returns_df, how="inner")
    if merged.empty:
        raise ValueError("Positions and price data do not overlap.")

    portfolio_returns = []
    equity_curve = []
    turnover_records: List[Dict[str, object]] = []
    wealth = 1.0
    prev_weights = np.zeros(num_assets, dtype=float)

    for date, row in merged.iterrows():
        weights = row[weights_cols].to_numpy(dtype=float)
        asset_returns = row[tickers].to_numpy(dtype=float)

        turnover = float(np.sum(np.abs(weights - prev_weights)))
        cost = turnover * (trade_cost_bps * 1e-4)

        gross = float(np.dot(weights, asset_returns))
        net = gross - cost

        wealth *= (1.0 + net)
        portfolio_returns.append(net)
        equity_curve.append(wealth)
        turnover_records.append({"Date": date, "turnover": turnover, "cost": cost})

        prev_weights = weights

    dates = merged.index.to_pydatetime()
    return np.array(portfolio_returns), np.array(equity_curve), turnover_records, np.array(dates)


def render_metrics_row(metrics: Dict[str, float]):
    """Display key performance numbers."""
    cols = st.columns(6)
    cols[0].metric("Sharpe Ratio", f"{metrics.get('sharpe', np.nan):.3f}")
    cols[1].metric("Sortino Ratio", f"{metrics.get('sortino', np.nan):.3f}")
    cols[2].metric("Annual Return", f"{metrics.get('annual_return', np.nan)*100:.2f}%")
    cols[3].metric("Annual Vol", f"{metrics.get('annual_vol', np.nan)*100:.2f}%")
    cols[4].metric("Max Drawdown", f"{metrics.get('max_drawdown', np.nan)*100:.2f}%")
    cols[5].metric("Final Wealth", f"{metrics.get('final_wealth', np.nan):.2f}x")


def _choose_latest_model() -> Optional[Dict[str, object]]:
    models = list_available_models()
    return models[0] if models else None


# --------------------------------------------------------------------------------------
# UI: Sidebar navigation
# --------------------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [

        "Dashboard",
        "Documentation",
        "Data Explorer",
        "Results Analysis",
        "Backtest",
        "Live Testing",
        # "Training",
    ],
)

st.sidebar.markdown("Model files are read from `results/` and `checkpoints/`.")
dark_mode = st.sidebar.toggle("Dark mode for plots", value=False)

# --------------------------------------------------------------------------------------
# Page: Dashboard
# --------------------------------------------------------------------------------------
if page == "Dashboard":
    st.title("SIT Portfolio Optimizer")
    st.markdown(
        "Interactive dashboard for the Signature-based Transformer that builds long-only portfolios "
        "and optimizes tail risk via CVaR."
    )

    latest = _choose_latest_model()
    if latest:
        metrics_df = load_results(latest["results"])
        metrics = metrics_df.iloc[0].to_dict()
        st.subheader(f"Latest model: {latest['name']}")
        render_metrics_row(metrics)
        if latest["equity"]:
            st.image(str(latest["equity"]), caption="Equity curve", use_container_width=True)
    else:
        st.info(
            "No saved runs found in `results/`. Default stats: Sharpe 0.51, Annual Return 8.27%, "
            "Max Drawdown -33.9%, Final Wealth 1.48x."
        )

    st.markdown("---")
    st.subheader("Data snapshot")
    try:
        df = load_data()
        st.write(df.head())
        st.caption(f"{len(df)} rows, {len(df.columns) - 1} assets")
    except Exception as exc:  # pragma: no cover - UI guard
        st.error(f"Failed to load data: {exc}")


# --------------------------------------------------------------------------------------
# Page: Documentation
# --------------------------------------------------------------------------------------
elif page == "Documentation":
    st.title("SIT Portfolio Optimizer - Documentation")
    st.markdown("*Complete guide to understanding and using the SIT model*")
    st.markdown(
        "- Jump to a topic with the tabs below, or skim the quick checklist at the end for a fast start."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Introduction",
            "How SIT Works",
            "Using the App",
            "Parameters Guide",
            "Interpreting Results",
            "Architecture",
            "Best Practices",
            "Glossary",
        ]
    )

    # Tab 1: Introduction
    with tab1:
        st.header("Introduction")
        st.markdown(
            "Traditional mean-variance optimizers struggle with non-linear dynamics, regime shifts, and tail risk. "
            "SIT combines **path signatures**, **transformer attention**, and **CVaR optimization** to build "
            "long-only, diversified portfolios that react to complex price paths."
        )
        st.subheader("Key features")
        st.markdown(
            "- Path signatures capture higher-order path information.\n"
            "- Dual attention: temporal dynamics + cross-asset relationships.\n"
            "- CVaR-focused loss to control downside risk.\n"
            "- Long-only, softmax-normalized weights; rebalancing with turnover costs."
        )
        st.subheader("Performance highlights (2020-2024)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", "0.51")
        c2.metric("Annual Return", "8.27%")
        c3.metric("Max Drawdown", "-33.9%")
        c4.metric("Period", "2020-2024")
        st.subheader("Why SIT vs. traditional methods?")
        st.markdown(
            "- Handles non-linear patterns that simple momentum/mean-reversion miss.\n"
            "- Learns cross-asset structure via attention instead of fixed covariance estimates.\n"
            "- Directly penalizes tail risk through CVaR instead of variance-only objectives.\n"
            "- Produces stable, long-only weights with temperature-controlled diversification."
        )

    # Tab 2: How SIT Works
    with tab2:
        st.header("How SIT Works")
        st.info("Step 1: Compute path signatures per asset window to summarize price trajectories.")
        st.info("Step 2: Add cross-signatures to capture pairwise co-movements.")
        st.info("Step 3: Encode time features (day-of-week/month/year) for seasonality context.")
        st.info("Step 4: Apply temporal attention, then asset attention with dynamic bias.")
        st.info("Step 5: Project to returns, softmax to weights, optimize CVaR with turnover costs.")

        st.subheader("Signature formulas")
        st.code(
            "Level-1: p[-1] - p[0]\n"
            "Level-2: sum(p[:-1] * diff(p))\n"
            "Cross-sig: sum(cumsum(diff(p_j))[:-1] * diff(p_k)[1:])",
            language="python",
        )
        st.latex(r"\text{Level-1}(p) = p_{T} - p_{0}")
        st.latex(r"\text{Level-2}(p) = \sum_{t=0}^{T-1} p_t \cdot (p_{t+1}-p_t)")
        st.latex(r"\text{Cross}(j,k) = \sum_t \Big(\sum_{\tau \le t} \Delta p_{j,\tau}\Big) \cdot \Delta p_{k,t+1}")

        st.subheader("Transformer flow")
        with st.expander("See processing steps"):
            st.markdown(
                "1. Embed signatures + dates + asset IDs into tokens.\n"
                "2. Temporal self-attention per asset (causal mask over horizon).\n"
                "3. Asset self-attention with dynamic bias from cross-signatures.\n"
                "4. Project tokens to return scores; softmax with temperature to weights."
            )

        st.subheader("CVaR optimization")
        st.markdown("SIT minimizes Conditional Value at Risk (CVaR) to curb tail losses:")
        st.latex(r"\text{CVaR}_\alpha = \mathbb{E}[ L \mid L \ge \text{VaR}_\alpha ]")
        st.markdown("Loss encourages lower downside while respecting turnover and long-only constraints.")

    # Tab 3: Using the App
    with tab3:
        st.header("Using the App")
        st.markdown("A quick guide to each tab and how to train.")
        st.subheader("Dashboard")
        st.markdown("- View the latest metrics, equity curve, and data snapshot.")

        st.subheader("Training")
        st.markdown(
            "- Configure data/model/training settings.\n"
            "- Start training without blocking the UI; progress and logs stream live.\n"
            "- Upload a custom CSV (Date + prices) if desired."
        )
        st.warning("Training time: ~20-40 minutes for standard config (30 assets, W=60, H=20).")
        st.info("GPU acceleration is used if available when the checkbox is selected.")

        st.subheader("Results Analysis")
        st.markdown(
            "- Inspect metrics, equity curve images, and portfolio weights.\n"
            "- Download metrics/positions/checkpoints; compare multiple models side-by-side."
        )

        st.subheader("Backtest")
        st.markdown(
            "- Recompute equity/turnover over custom date ranges with configurable trade costs.\n"
            "- Export transaction details."
        )

        st.subheader("Data Explorer")
        st.markdown(
            "- Plot prices, correlations, and return distributions.\n"
            "- Filter by date and assets for quick sanity checks."
        )

        st.subheader("Step-by-step training guide")
        st.markdown(
            "1) Pick assets (start with 10 for a smoke test, then 30 for standard runs).\n"
            "2) Set window/horizon (e.g., W=60, H=20).\n"
            "3) Choose model size (d_model=8, n_heads=4, layers=2) and temperature=1.3.\n"
            "4) Set epochs=30, batch=64, lr=1e-3, patience=3, trade_cost=10 bps.\n"
            "5) Click **Start training** and monitor logs.\n"
            "6) Review outputs in Results Analysis and Backtest."
        )

    # Tab 4: Parameters Guide
    with tab4:
        st.header("Parameters Guide")
        st.markdown("Ranges, defaults, and tuning guidance for common parameters.")

        with st.expander("Data parameters"):
            st.markdown(
                "- **Number of assets**: 5-76 (default 30). More assets increase diversification but training cost.\n"
                "- **window_size (W)**: 20-120 (default 60). Lookback length; larger captures more context but slows training.\n"
                "- **horizon (H)**: 5-30 (default 20). Prediction window; larger H increases temporal modeling burden."
            )

        with st.expander("Model parameters"):
            st.markdown(
                "- **d_model**: 4-32 (default 8). Core width; higher boosts capacity but risks overfit.\n"
                "- **n_heads**: 2-8 (default 4). Must divide d_model; more heads capture diverse patterns.\n"
                "- **num_layers**: 1-4 (default 2). Stacking improves expressiveness, increases training time.\n"
                "- **temperature**: 0.5-3.0 (default 1.3). Lower -> concentrated weights; higher -> more diversification.\n"
                "- **dropout**: 0-0.5 (default 0.1). Regularizes attention/FFN layers."
            )

        with st.expander("Training parameters"):
            st.markdown(
                "- **epochs**: 5-50 (default 30). Early stopping (patience=3) prevents overrun.\n"
                "- **batch_size**: 16-128 (default 64). Larger batches stabilize but may limit GPU memory.\n"
                "- **learning_rate**: default 0.001. Lower if loss is unstable; higher for faster warm-up.\n"
                "- **patience**: default 3. Stops when val CVaR stops improving."
            )

        with st.expander("Backtest parameters"):
            st.markdown(
                "- **trade_cost_bps**: 0-50 (default 10). Applied to turnover each rebalance; higher costs discourage churn."
            )

        st.info("Recommendations: start with d_model=8, n_heads=4, layers=2, W=60, H=20, temp=1.3, trade_cost=10 bps.")

    # Tab 5: Interpreting Results
    with tab5:
        st.header("Interpreting Results")
        st.markdown("How to read the metrics and plots.")

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Sharpe Ratio**\n- < 0.5: weak\n- 0.5-1.0: good\n- > 1.0: excellent")
        c2.markdown("**Max Drawdown**\n- -20% to -40%: moderate risk\n- < -40%: high risk\n- Closer to 0% is better")
        c3.markdown("**Win Rate**\n- >50% shows positive edge\n- Consider with drawdowns and vol")

        c4, c5 = st.columns(2)
        c4.markdown("**Sortino** focuses on downside volatility; prefer > Sharpe when negatives matter.")
        c5.markdown("**Annual Return/Vol** compare CAGR vs. risk; check if reward/vol is improving run-over-run.")

        st.subheader("Equity curve analysis")
        st.markdown(
            "- Smooth upward slope with controlled drawdowns is ideal.\n"
            "- Plateaus suggest low signal; steep drops highlight regimes where the model struggles.\n"
            "- Compare curves across models to see stability vs. upside."
        )

        st.subheader("Portfolio weights")
        st.markdown(
            "- Look for sensible diversification; extreme concentration may indicate overfit.\n"
            "- Temperature -> spreads weights; lower concentrates bets.\n"
            "- Monitor turnover: high turnover + high costs can erode returns."
        )

    # Tab 6: Architecture
    with tab6:
        st.header("Architecture")
        st.markdown("High-level flow from data to portfolio weights.")
        mermaid_diagram = """
graph LR
    A[Prices + Dates] --> B[Signature computation]
    B --> C[Temporal attention per asset]
    C --> D[Asset attention + cross-sig bias]
    D --> E[Return projection]
    E --> F[Softmax weights]
    F --> G[CVaR loss + turnover]
"""
        if hasattr(st, "mermaid"):
            st.mermaid(mermaid_diagram)
        else:
            st.markdown(f"```mermaid\n{mermaid_diagram}\n```")

        st.subheader("Training process")
        st.markdown(
            "1) Build rolling signature datasets (train/val/test).\n"
            "2) Train with CVaR loss and early stopping on val CVaR.\n"
            "3) Save best checkpoint and test on held-out dates."
        )

        st.subheader("Backtesting")
        st.markdown(
            "- Use predicted weights per horizon step; rebalance monthly (or first weight) with turnover costs.\n"
            "- Track equity, drawdown, and metrics over 2020-2024."
        )

        st.subheader("CVaR math")
        st.latex(r"\text{VaR}_\alpha = \inf \{ x \mid P(L \le x) \ge \alpha \}")
        st.latex(r"\text{CVaR}_\alpha = \mathbb{E}[ L \mid L \ge \text{VaR}_\alpha ]")
        st.markdown("Loss minimizes CVaR across horizon returns with softmax-constrained weights.")

    # Tab 7: Best Practices
    with tab7:
        st.header("Best Practices")
        st.subheader("3-phase starter plan")
        st.markdown(
            "1) **Smoke test**: 10 assets, W=30, H=10, epochs=5 to confirm pipeline.\n"
            "2) **Standard run**: 30 assets, W=60, H=20, epochs=30, patience=3, temp=1.3.\n"
            "3) **Refinement**: Tune d_model (8->16), dropout (0.1->0.2), temp (1.1->1.5) based on drawdowns."
        )

        st.subheader("Tuning tips")
        st.markdown(
            "- Increase **d_model / layers** for richer dynamics; watch for overfit.\n"
            "- Raise **temperature** to spread weights if concentration is high.\n"
            "- Lower **learning rate** if loss is noisy; raise slightly if slow to improve.\n"
            "- Adjust **dropout** upward if overfitting; downward if underfitting."
        )

        st.subheader("Common issues")
        st.warning("Training loss not decreasing -> try lower learning rate or larger d_model.")
        st.warning("Overfitting -> increase dropout or reduce model size/layers.")
        st.warning("Concentrated weights -> increase temperature.")

        st.subheader("Dos and don'ts")
        st.success(
            "Do: use early stopping (patience=3), account for 10 bps costs, compare multiple seeds/models, "
            "inspect weight stability."
        )
        st.error(
            "Don't: ignore drawdowns, train with too few assets for long horizons, or deploy without cost assumptions."
        )

        st.subheader("Deployment considerations")
        st.markdown(
            "- Refresh models periodically to track new regimes.\n"
            "- Monitor live turnover vs. assumed costs.\n"
            "- Validate on out-of-sample periods before production."
        )

    # Tab 8: Glossary
    with tab8:
        st.header("Glossary")
        glossary = {
            "Path signature": "Series summary using iterated integrals capturing path shape.",
            "Cross-signature": "Pairwise signature capturing co-movement between two assets.",
            "Temporal attention": "Attention over horizon steps for each asset.",
            "Asset attention": "Attention over assets with dynamic bias from cross-signatures.",
            "CVaR": "Conditional Value at Risk; expected loss beyond VaR threshold.",
            "VaR": "Value at Risk; loss level not exceeded with probability alpha.",
            "Temperature": "Softmax scaling controlling diversification vs. concentration.",
            "Horizon (H)": "Forward prediction window length.",
            "Window size (W)": "Historical lookback window length.",
            "Turnover": "Sum of absolute weight changes at rebalance; drives costs.",
            "Rebalance date": "Time when weights are reset/applied.",
        }
        query = st.text_input("Search terms", "")
        filtered = {k: v for k, v in glossary.items() if query.lower() in k.lower() or query.lower() in v.lower()}
        for term, desc in filtered.items():
            st.markdown(f"**{term}** - {desc}")
        if not filtered:
            st.info("No matching terms. Try another keyword.")

    st.markdown("---")
    st.subheader("Quick start checklist")
    st.checkbox("Use standard config: 30 assets, W=60, H=20, temp=1.3", key="qs1", value=False, disabled=True)
    st.checkbox("Set epochs=30, batch=64, lr=1e-3, patience=3", key="qs2", value=False, disabled=True)
    st.checkbox("Enable GPU if available", key="qs3", value=False, disabled=True)
    st.checkbox("Account for trade costs (10 bps)", key="qs4", value=False, disabled=True)
    st.checkbox("Review equity curve and drawdown after training", key="qs5", value=False, disabled=True)


# --------------------------------------------------------------------------------------
# Page: Training
# --------------------------------------------------------------------------------------
elif page == "Training":
    st.title("Train a new SIT model")
    st.markdown("Configure parameters and launch training without blocking the UI.")

    with st.form("train_form"):
        st.subheader("Data configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            data_pool = st.slider("Number of assets", 5, 76, 30)
            window_size = st.slider("Window size", 20, 120, 60)
        with c2:
            horizon = st.slider("Horizon", 5, 30, 20)
            batch_size = st.slider("Batch size", 16, 128, 64, step=8)
        with c3:
            upload = st.file_uploader("Upload custom dataset (CSV with Date + prices)", type=["csv"])

        st.subheader("Model configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            d_model = st.slider("d_model", 4, 32, 8)
            n_heads = st.slider("n_heads", 2, 8, 4)
        with c2:
            num_layers = st.slider("num_layers", 1, 4, 2)
            ff_dim = st.slider("ff_dim", 16, 128, 32, step=8)
        with c3:
            temperature = st.slider("Temperature", 0.5, 3.0, 1.3, step=0.1)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1, step=0.05)

        st.subheader("Training configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            train_epochs = st.slider("Epochs", 5, 50, 30)
            learning_rate = st.number_input("Learning rate", value=0.001, format="%.4f")
        with c2:
            patience = st.slider("Patience", 1, 10, 3)
            cvar_alpha = st.slider("CVaR alpha", 0.90, 0.99, 0.95, step=0.01)
        with c3:
            trade_cost_bps = st.slider("Trade cost (bps)", 0.0, 50.0, 10.0, step=0.5)
            use_gpu = st.checkbox("Use GPU if available", value=torch.cuda.is_available())

        submitted = st.form_submit_button("Start training", type="primary")

    # Handle uploaded dataset persistence
    data_path = str(DATA_PATH)
    if upload is not None:
        try:
            tmp_dir = Path(tempfile.gettempdir())
            tmp_path = tmp_dir / "sit_uploaded_dataset.csv"
            tmp_path.write_bytes(upload.getvalue())
            data_path = str(tmp_path)
            st.success(f"Using uploaded dataset stored at {tmp_path}")
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(f"Could not save uploaded file, falling back to default data: {exc}")

    params = {
        "data_pool": data_pool,
        "window_size": window_size,
        "horizon": horizon,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
        "ff_dim": ff_dim,
        "hidden_c": 64,
        "dropout": dropout,
        "temperature": temperature,
        "train_epochs": train_epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "cvar_alpha": cvar_alpha,
        "trade_cost_bps": trade_cost_bps,
        "use_gpu": use_gpu,
    }

    config_json = json.dumps(params, indent=2)
    st.download_button("Download config JSON", data=config_json, file_name="sit_config.json", mime="application/json")

    if submitted:
        start_training(params, data_path)

    train_state = st.session_state.train_state
    st.markdown("### Training status")
    status_col, prog_col = st.columns([1, 2])
    status_col.write(f"Status: **{train_state['status']}**")
    prog_col.progress(train_state.get("progress", 0.0))

    if train_state.get("latest_metrics"):
        lm = train_state["latest_metrics"]
        st.info(f"Latest epoch {lm['epoch']}: loss={lm['train_loss']:.4f}, val_cvar={lm['val_cvar']:.4f}")

    st.markdown("#### Logs")
    st.text_area("Trainer output", value="\n".join(train_state.get("logs", [])), height=200)

    if train_state.get("status") == "completed":
        st.success("Training finished. Check `results/` and `checkpoints/` for outputs.")
    if train_state.get("status") == "error":
        st.error(f"Training failed: {train_state.get('error', 'Unknown error')}")


# --------------------------------------------------------------------------------------
# Page: Results Analysis
# --------------------------------------------------------------------------------------
elif page == "Results Analysis":
    st.title("Results analysis")

    models = list_available_models()
    if not models:
        st.warning("No results found in `results/`. Train a model first.")
    else:
        model_names = [m["name"] for m in models]
        selected_name = st.selectbox("Select model", model_names)
        compare_names = st.multiselect("Compare models", model_names, default=[model_names[0]])

        selected = next((m for m in models if m["name"] == selected_name), models[0])
        metrics_df = load_results(selected["results"])
        metrics = metrics_df.iloc[0].to_dict()
        render_metrics_row(metrics)

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            if selected["equity"]:
                st.image(str(selected["equity"]), caption="Equity curve", use_container_width=True)
            else:
                st.info("No equity curve image found.")
        with c2:
            st.dataframe(metrics_df, use_container_width=True)
            st.download_button(
                "Download metrics CSV",
                data=selected["results"].read_bytes(),
                file_name=selected["results"].name,
                mime="text/csv",
            )
            if CHECKPOINT_DIR.joinpath("checkpoint.pth").exists():
                st.download_button(
                    "Download checkpoint",
                    data=CHECKPOINT_DIR.joinpath("checkpoint.pth").read_bytes(),
                    file_name="checkpoint.pth",
                )

        if selected["positions"]:
            st.markdown("---")
            st.subheader("Portfolio weights")
            positions_df = load_positions(selected["positions"])
            weight_cols = [c for c in positions_df.columns if c.startswith("w_")]
            avg_weights = positions_df[weight_cols].mean().sort_values(ascending=False)
            max_assets = max(1, len(weight_cols))
            default_top = min(5, max_assets)
            top_n = st.slider("Assets to plot", 1, min(10, max_assets), default_top)
            top_assets = avg_weights.head(top_n).index.tolist()

            line_fig = px.line(
                positions_df,
                x="Date",
                y=top_assets,
                labels={"value": "Weight", "Date": "Date", "variable": "Asset"},
                title="Weights over time (top assets)",
            )
            line_fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white", hovermode="x unified")
            st.plotly_chart(line_fig, use_container_width=True)

            area_fig = go.Figure()
            for col in top_assets:
                area_fig.add_trace(
                    go.Scatter(
                        x=positions_df["Date"],
                        y=positions_df[col],
                        mode="lines",
                        name=col,
                        stackgroup="one",
                    )
                )
            area_fig.update_layout(
                title="Stacked weights",
                yaxis_title="Weight",
                template="plotly_dark" if dark_mode else "plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(area_fig, use_container_width=True)

            st.download_button(
                "Download positions CSV",
                data=selected["positions"].read_bytes(),
                file_name=selected["positions"].name,
                mime="text/csv",
            )

        if compare_names:
            st.markdown("---")
            st.subheader("Comparison table")
            compare_rows = []
            for name in compare_names:
                m = next((x for x in models if x["name"] == name), None)
                if not m:
                    continue
                df = load_results(m["results"])
                row = df.iloc[0].to_dict()
                row["model"] = name
                compare_rows.append(row)
            if compare_rows:
                cmp_df = pd.DataFrame(compare_rows).set_index("model")
                st.dataframe(cmp_df, use_container_width=True)


# --------------------------------------------------------------------------------------
# Page: Backtest
# --------------------------------------------------------------------------------------
elif page == "Backtest":
    st.title("Backtest viewer")
    models = list_available_models()
    if not models:
        st.warning("No saved positions found.")
    else:
        model_names = [m["name"] for m in models]
        selected_name = st.selectbox("Model", model_names)
        selected = next((m for m in models if m["name"] == selected_name), models[0])
        if not selected["positions"]:
            st.error("Selected model has no positions CSV.")
        else:
            positions_df = load_positions(selected["positions"])
            try:
                data_df = load_data()
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Could not load price data: {exc}")
                data_df = None

            min_date = positions_df["Date"].min()
            max_date = positions_df["Date"].max()
            date_range = st.slider(
                "Date range",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            )
            trade_cost_bps = st.slider("Trade cost (bps)", 0.0, 50.0, 10.0, step=0.5)
            run_bt = st.button("Run backtest")

            if run_bt and data_df is not None:
                try:
                    returns, equity, turnover_records, bt_dates = compute_backtest_from_positions(
                        positions_df, data_df, trade_cost_bps, date_range
                    )
                    metrics = compute_metrics(returns, equity)
                    render_metrics_row(metrics)

                    curve_fig = px.line(
                        x=pd.to_datetime(bt_dates),
                        y=equity,
                        labels={"x": "Date", "y": "Equity"},
                        title="Cumulative wealth",
                    )
                    curve_fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white")
                    st.plotly_chart(curve_fig, use_container_width=True)

                    turnover_df = pd.DataFrame(turnover_records)
                    st.markdown("Transaction details")
                    st.dataframe(turnover_df, use_container_width=True)
                    st.download_button(
                        "Download transactions CSV",
                        data=turnover_df.to_csv(index=False),
                        file_name="turnover.csv",
                        mime="text/csv",
                    )
                except Exception as exc:  # pragma: no cover - UI guard
                    st.error(f"Backtest failed: {exc}")


# --------------------------------------------------------------------------------------
# Page: Live Testing
# --------------------------------------------------------------------------------------
elif page == "Live Testing":
    st.title("Live Testing")
    st.markdown("Test your trained models on custom CSV data")
    st.markdown("---")

    with st.expander("How to Use Live Testing - Quick Guide", expanded=False):
        st.markdown(
            """
## Live Testing Guide

### Required CSV Format
- First column: `Date` (YYYY-MM-DD format)
- Remaining columns: Daily closing prices for each asset
- No missing values, no gaps in dates
- All prices must be positive

### Example CSV Structure
```
Date,AAPL,MSFT,GOOGL,AMZN
2024-01-02,185.64,376.04,140.93,151.94
2024-01-03,184.25,375.37,139.69,151.03
2024-01-04,181.91,367.93,138.45,149.61
```

### Where to Get Data
**Yahoo Finance (Recommended):**
1. Go to https://finance.yahoo.com
2. Search for a stock ticker (e.g., AAPL)
3. Click "Historical Data" tab
4. Select date range (Max = all available)
5. Click "Download" button
6. Repeat for each stock
7. Combine into one CSV file
8. Use "Adj Close" prices (accounts for splits/dividends)

**Other Sources:**
- Alpha Vantage: https://www.alphavantage.co
- Investing.com: https://www.investing.com
- Quandl: https://data.nasdaq.com

### Workflow Steps
1. Select a trained model
2. Upload your CSV file
3. Configure test dates and parameters
4. Map your assets to model's expected assets
5. Run backtest
6. View results and download reports

### Important Notes
- Need at least `window_size` days of history before test start
- Test on out-of-sample data (not training period)
- Use realistic transaction costs (10 bps = 0.1%)
- More matched assets = better results
            """
        )

    # Session defaults
    for key in [
        "selected_model",
        "model_config",
        "trained_assets",
        "uploaded_df",
        "uploaded_assets",
        "test_config",
        "asset_mapping",
        "live_test_results",
    ]:
        st.session_state.setdefault(key, None)

    # Step 1: model selection
    st.header("Step 1: Select Your Trained Model")
    available_models = get_available_models_with_checkpoints()
    if not available_models:
        st.error("No trained models with checkpoints found. Train a model first.")
        st.stop()
    selected_model = st.selectbox(
        "Choose a trained model",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models
        else 0,
    )
    st.session_state.selected_model = selected_model
    try:
        model, loaded_config = load_trained_model_from_name(selected_model)
        st.session_state.model_config = loaded_config
        trained_assets = get_trained_assets_from_pool(loaded_config["data_pool"])
        st.session_state.trained_assets = trained_assets
        st.success(
            f"Model loaded: assets={loaded_config['data_pool']}, window={loaded_config['window_size']}, "
            f"horizon={loaded_config['horizon']}, d_model={loaded_config['d_model']}, "
            f"heads={loaded_config['n_heads']}, layers={loaded_config['num_layers']}"
        )
        
        with st.expander(f"View {len(trained_assets)} training assets"):
            cols = st.columns(5)
            for i, asset in enumerate(trained_assets):
                cols[i % 5].code(asset)
                
    except Exception as exc:
        st.error(f"Failed to load model {selected_model}: {exc}")
        st.stop()

    st.markdown("---")

    # Step 2: upload CSV
    st.header("Step 2: Upload Your CSV Data")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Upload CSV with Date column and daily price columns. No missing or non-positive values.")
    with col2:
        st.code("""Example:
Date,AAPL,MSFT
2024-01-02,185.64,376.04
2024-01-03,184.25,375.37
2024-01-04,181.91,367.93
""")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"], help="Must include Date column")
    if not uploaded_file:
        st.warning("Upload a CSV to continue.")
        st.stop()
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if "Date" not in uploaded_df.columns:
            st.error(f"Missing Date column. Columns found: {', '.join(uploaded_df.columns)}")
            st.stop()
        uploaded_df["Date"] = pd.to_datetime(uploaded_df["Date"])
        uploaded_df = uploaded_df.sort_values("Date").reset_index(drop=True)
        asset_cols = [c for c in uploaded_df.columns if c != "Date"]
        if not asset_cols:
            st.error("No asset columns found.")
            st.stop()
        missing = uploaded_df[asset_cols].isnull().sum().sum()
        if missing > 0:
            st.error(f"Found {missing} missing values. Clean the file and retry.")
            st.stop()
        invalid = (uploaded_df[asset_cols] <= 0).sum().sum()
        if invalid > 0:
            st.error("Prices must be positive. Fix data and retry.")
            st.stop()
        st.success("CSV validated.")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date Range", f"{uploaded_df['Date'].min().date()} to {uploaded_df['Date'].max().date()}")
        col2.metric("Assets", len(asset_cols))
        col3.metric("Data Points", len(uploaded_df))
        col4.metric("Quality", "Valid")
        
        # Preview
        with st.expander("Preview Data"):
            tab1, tab2 = st.tabs(["First 10 rows", "Last 10 rows"])
            with tab1:
                st.dataframe(uploaded_df.head(10), use_container_width=True)
            with tab2:
                st.dataframe(uploaded_df.tail(10), use_container_width=True)
        
        st.session_state.uploaded_df = uploaded_df
        st.session_state.uploaded_assets = asset_cols
    except Exception as exc:
        st.error(f"Error reading CSV: {exc}")
        st.stop()
    st.markdown("---")

    # Step 3: configuration
    st.header("Step 3: Testing Configuration")
    data_min = uploaded_df["Date"].min()
    data_max = uploaded_df["Date"].max()
    required_history = st.session_state.model_config["window_size"]
    min_valid_start = data_min + pd.Timedelta(days=required_history)
    
    st.info(f"Your data: {data_min.date()} to {data_max.date()}. Model requires {required_history} days history. "
            f"Earliest valid test start: {min_valid_start.date()}")
    
    col1, col2 = st.columns(2)
    with col1:
        test_start = st.date_input(
            "Test Start Date",
            value=min_valid_start.date() if min_valid_start.date() <= data_max.date() else data_max.date(),
            min_value=data_min.date(),
            max_value=data_max.date(),
        )
    with col2:
        test_end = st.date_input("Test End Date", value=data_max.date(), min_value=test_start, max_value=data_max.date())
    available_history = (pd.Timestamp(test_start) - data_min).days
    if available_history < required_history:
        st.error(
            f"Insufficient history: have {available_history} days, need {required_history}. "
            f"Move start to {min_valid_start.date()} or later."
        )
        st.stop()
    else:
        st.success(f"Sufficient history: {available_history} days (need {required_history}).")
    col3, col4 = st.columns(2)
    with col3:
        trade_cost_bps = st.slider("Transaction Cost (bps)", 0.0, 50.0, 10.0, 1.0)
    with col4:
        rebalance_freq = st.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly"], index=2)
    st.session_state.test_config = {
        "start": test_start,
        "end": test_end,
        "cost_bps": trade_cost_bps,
        "freq": rebalance_freq,
    }
    st.markdown("---")

    # Step 4: asset mapping
    st.header("Step 4: Asset Mapping")
    trained_assets = st.session_state.trained_assets or []
    uploaded_assets = st.session_state.uploaded_assets or []
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model expects:**")
        st.code("\n".join(trained_assets[:10]) + ("\n..." if len(trained_assets) > 10 else ""))
    with col2:
        st.write("**Your data has:**")
        st.code("\n".join(uploaded_assets[:10]) + ("\n..." if len(uploaded_assets) > 10 else ""))
    
    auto_map = {a: a for a in trained_assets if a in uploaded_assets}
    use_auto = st.checkbox("Use auto-detected mappings", value=True)
    mapping = {}
    if use_auto:
        mapping = auto_map.copy()
        st.success(f"Auto-mapped {len(mapping)} assets.")
        unmapped = [a for a in trained_assets if a not in mapping]
        if unmapped:
            st.warning(f"{len(unmapped)} assets not mapped: {', '.join(unmapped[:5])}" + ("..." if len(unmapped) > 5 else ""))
    else:
        st.write("Manual mapping (showing first 20 assets):")
        for asset in trained_assets[:20]:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{asset}**")
            with col2:
                mapped = st.selectbox(
                    f"Map to",
                    options=["[Skip]"] + uploaded_assets,
                    index=(uploaded_assets.index(asset) + 1) if asset in uploaded_assets else 0,
                    key=f"map_{asset}",
                    label_visibility="collapsed",
                )
                if mapped != "[Skip]":
                    mapping[asset] = mapped
    
    coverage = len(mapping) / max(len(trained_assets), 1) * 100
    col1, col2 = st.columns(2)
    col1.metric("Mapped assets", len(mapping))
    col2.metric("Coverage", f"{coverage:.1f}%")
    
    if coverage < 50:
        st.error("At least 50% coverage recommended for meaningful results.")
        st.stop()
    st.session_state.asset_mapping = mapping
    st.markdown("---")

    # Step 5: run test
    st.header("Step 5: Run Test")
    if st.button("Run backtest on uploaded data", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.text("Step 1/6: Preparing mapped data...")
                progress.progress(10)
                df = st.session_state.uploaded_df.copy()
                mapping = st.session_state.asset_mapping
                model_cfg = st.session_state.model_config
                mapped_df = pd.DataFrame()
                mapped_df["Date"] = df["Date"]
                for m_asset, u_asset in mapping.items():
                    mapped_df[m_asset] = df[u_asset]
                unmapped = [a for a in trained_assets if a not in mapped_df.columns]
                if unmapped:
                    st.info(f"Filling {len(unmapped)} unmapped assets with constant prices (zero returns).")
                for a in unmapped:
                    mapped_df[a] = 100.0
                test_cfg = st.session_state.test_config
                test_start_ts = pd.Timestamp(test_cfg["start"])
                test_end_ts = pd.Timestamp(test_cfg["end"])
                data_start_ts = test_start_ts - pd.Timedelta(days=model_cfg["window_size"] + model_cfg["horizon"])
                mapped_df = mapped_df[(mapped_df["Date"] >= data_start_ts) & (mapped_df["Date"] <= test_end_ts)]

                status.text("Step 2/6: Creating signature dataset...")
                progress.progress(30)
                from data_provider.data_loader import Dataset_Sig
                dataset = Dataset_Sig(
                    prices_df=mapped_df,
                    window_size=model_cfg["window_size"],
                    horizon=model_cfg["horizon"],
                    tickers=trained_assets,
                    scaler=None,
                )
                from torch.utils.data import DataLoader
                loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

                status.text("Step 3/6: Generating predictions...")
                progress.progress(50)
                device = torch.device("cpu")
                model.eval()
                preds = []
                pred_dates = []
                with torch.no_grad():
                    for batch in loader:
                        x_sigs = batch["x_sigs"].to(device)
                        cross_sigs = batch["cross_sigs"].to(device)
                        date_feats = batch["date_feats"].to(device)
                        fut = batch["future_return_unscaled"].to(device)
                        weights, _ = model(x_sigs, cross_sigs, date_feats, fut)
                        preds.append(weights[0, 0, :].cpu().numpy())
                        pred_dates.append(batch["dates"][0])

                status.text("Step 4/6: Running backtest with transaction costs...")
                progress.progress(70)
                preds_df = pd.DataFrame(preds, columns=trained_assets)
                preds_df["Date"] = pd.to_datetime(pred_dates)
                preds_df = preds_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)
                returns_df = mapped_df.set_index("Date")[trained_assets].pct_change().shift(-1)
                rebalance_dates = set(generate_rebalance_dates(test_start_ts, test_end_ts, test_cfg["freq"]))
                capital = 1.0
                equity_curve = [capital]
                dates_list = [test_start_ts]
                positions_history = []
                current_weights = preds_df.iloc[0][trained_assets].values if len(preds_df) > 0 else np.zeros(len(trained_assets))
                
                for _, row in preds_df.iterrows():
                    date = row["Date"]
                    if date < test_start_ts or date > test_end_ts:
                        continue
                    if date not in returns_df.index:
                        continue
                    day_ret = returns_df.loc[date].values
                    is_rebalance = date in rebalance_dates
                    if is_rebalance:
                        new_w = row[trained_assets].values
                        turnover = float(np.sum(np.abs(new_w - current_weights)))
                        cost = turnover * (test_cfg["cost_bps"] * 1e-4) * capital
                        current_weights = new_w
                    else:
                        cost = 0.0
                    port_ret = float(np.dot(current_weights, day_ret))
                    capital *= 1.0 + port_ret
                    capital -= cost
                    equity_curve.append(capital)
                    dates_list.append(date)
                    positions_history.append({
                        "Date": date, 
                        **{a: w for a, w in zip(trained_assets, current_weights)}, 
                        "Cost": cost,
                        "Rebalanced": is_rebalance
                    })

                status.text("Step 5/6: Computing performance metrics...")
                progress.progress(90)
                equity_curve = np.array(equity_curve)
                ret = np.diff(equity_curve) / equity_curve[:-1]
                if len(ret) == 0:
                    st.error("No returns computed. Check date range and data quality.")
                    st.stop()
                    
                metrics = compute_metrics(ret)
                metrics["final_wealth"] = float(equity_curve[-1])
                metrics["win_rate"] = float(np.mean(ret > 0)) if len(ret) else 0.0
                metrics["num_trades"] = sum(1 for p in positions_history if p.get("Rebalanced", False))
                metrics["total_costs"] = float(sum(p.get("Cost", 0.0) for p in positions_history))
                metrics["avg_trade_cost"] = metrics["total_costs"] / metrics["num_trades"] if metrics["num_trades"] > 0 else 0.0

                st.session_state.live_test_results = {
                    "metrics": metrics,
                    "equity_curve": equity_curve,
                    "dates": dates_list,
                    "positions": pd.DataFrame(positions_history),
                }
                
                status.text("Step 6/6: Complete!")
                progress.progress(100)
                time.sleep(0.3)
                status.empty()
                progress.empty()
                st.success("Backtest completed successfully!")
                
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                import traceback
                with st.expander("Full error traceback"):
                    st.code(traceback.format_exc())
                st.stop()

    # Step 6: results
    if st.session_state.live_test_results:
        st.header("Step 6: Results")
        res = st.session_state.live_test_results
        metrics = res["metrics"]
        
        # Metrics
        cols = st.columns(4)
        cols[0].metric("Sharpe", f"{metrics.get('sharpe', np.nan):.3f}")
        cols[1].metric("Annual Return", f"{metrics.get('annual_return', np.nan)*100:.2f}%")
        cols[2].metric("Max Drawdown", f"{metrics.get('max_drawdown', np.nan)*100:.2f}%")
        cols[3].metric("Final Wealth", f"{metrics.get('final_wealth', np.nan):.2f}x")
        
        cols2 = st.columns(4)
        cols2[0].metric("Sortino", f"{metrics.get('sortino', np.nan):.3f}")
        cols2[1].metric("Annual Vol", f"{metrics.get('annual_vol', np.nan)*100:.2f}%")
        cols2[2].metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
        cols2[3].metric("Total Costs", f"${metrics.get('total_costs', 0):.4f}")

        # Equity curve
        st.subheader("Equity Curve")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=res["dates"], 
            y=res["equity_curve"][1:], 
            mode="lines", 
            name="Portfolio Value",
            line=dict(color='#00D9FF', width=2)
        ))
        fig_eq.update_layout(
            xaxis_title="Date",
            yaxis_title="Wealth",
            template="plotly_dark" if dark_mode else "plotly_white",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Detailed stats table
        st.subheader("Detailed Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Annual Return', 'Annual Volatility',
                      'Max Drawdown', 'Final Wealth', 'Win Rate', 'Number of Rebalances',
                      'Total Transaction Costs', 'Avg Cost per Rebalance'],
            'Value': [
                f"{metrics.get('sharpe', np.nan):.3f}",
                f"{metrics.get('sortino', np.nan):.3f}",
                f"{metrics.get('annual_return', np.nan)*100:.2f}%",
                f"{metrics.get('annual_vol', np.nan)*100:.2f}%",
                f"{metrics.get('max_drawdown', np.nan)*100:.2f}%",
                f"{metrics.get('final_wealth', np.nan):.3f}x",
                f"{metrics.get('win_rate', 0)*100:.1f}%",
                f"{metrics.get('num_trades', 0)}",
                f"${metrics.get('total_costs', 0):.4f}",
                f"${metrics.get('avg_trade_cost', 0):.6f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Positions preview
        st.subheader("Positions (first 20 days)")
        st.dataframe(res["positions"].head(20), use_container_width=True)

        # Downloads
        st.subheader("Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pos_csv = res["positions"].to_csv(index=False)
            st.download_button(
                "Download Positions CSV", 
                pos_csv, 
                file_name=f"live_test_positions_{selected_model}.csv", 
                mime="text/csv"
            )
        
        with col2:
            metrics_csv = stats_df.to_csv(index=False)
            st.download_button(
                "Download Metrics CSV",
                metrics_csv,
                file_name=f"live_test_metrics_{selected_model}.csv",
                mime="text/csv"
            )
        
        with col3:
            equity_data = pd.DataFrame({
                'Date': res["dates"],
                'Wealth': res["equity_curve"][1:]
            })
            equity_csv = equity_data.to_csv(index=False)
            st.download_button(
                "Download Equity Curve CSV",
                equity_csv,
                file_name=f"live_test_equity_{selected_model}.csv",
                mime="text/csv"
            )


# --------------------------------------------------------------------------------------
# Page: Data Explorer
# --------------------------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("Data explorer")
    try:
        df = load_data()
        st.caption(f"{len(df)} rows, {len(df.columns) - 1} assets")
        st.dataframe(df.head(1000), use_container_width=True)
    except Exception as exc:  # pragma: no cover - UI guard
        st.error(f"Could not load data: {exc}")
        df = None

    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
        tickers = [c for c in df.columns if c != "Date"]
        default_assets = tickers[:3]
        assets = st.multiselect("Assets", tickers, default=default_assets)
        start_default = max(df["Date"].min(), df["Date"].max() - pd.Timedelta(days=365 * 2))
        date_range = st.slider(
            "Date range",
            min_value=df["Date"].min().to_pydatetime(),
            max_value=df["Date"].max().to_pydatetime(),
            value=(
                start_default.to_pydatetime(),
                df["Date"].max().to_pydatetime(),
            ),
        )
        mask = (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
        sub_df = df.loc[mask]

        if assets:
            price_fig = px.line(
                sub_df,
                x="Date",
                y=assets,
                title="Asset prices",
                labels={"value": "Price", "variable": "Asset"},
            )
            price_fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white")
            st.plotly_chart(price_fig, use_container_width=True)

            returns = sub_df.set_index("Date")[assets].pct_change().dropna()
            corr = returns.corr()
            heat_fig = px.imshow(
                corr,
                text_auto=False,
                color_continuous_scale="RdBu_r",
                title="Correlation matrix",
            )
            heat_fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white")
            st.plotly_chart(heat_fig, use_container_width=True)

            dist_fig = px.histogram(
                returns.melt(var_name="Asset", value_name="Return"),
                x="Return",
                color="Asset",
                nbins=50,
                title="Return distribution",
            )
            dist_fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white")
            st.plotly_chart(dist_fig, use_container_width=True)


# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Status: ready")