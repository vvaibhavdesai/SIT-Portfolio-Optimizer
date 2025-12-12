# SIT: Signature-Informed Transformer for Portfolio Optimization

**Academic Implementation of Path Signature-Based Portfolio Optimization**

> **Note:** This project is implemented in good faith as an academic exercise to understand and apply advanced portfolio optimization techniques combining Rough Path Theory, transformer architectures, and risk-aware optimization. The implementation closely follows the methodology from published research while adapting it for educational purposes.

## 📋 Project Overview

This project implements the Signature-Informed Transformer (SIT), a novel deep learning approach for portfolio optimization that combines:

- **Path Signatures** from Rough Path Theory for geometric feature extraction
- **Factored Transformer Architecture** with dual attention mechanisms
- **CVaR Optimization** for explicit tail risk management  
- **Long-only Portfolios** with realistic transaction costs

Based on the research paper: *"Signature-Informed Transformer for Attention-Based Asset Allocation"* by Hwang & Zohren

## 🎯 Key Results (2020-2024 Out-of-Sample Backtest)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 0.51 |
| **Sortino Ratio** | 0.66 |
| **Annual Return** | 8.27% |
| **Annual Volatility** | 21.34% |
| **Max Drawdown** | -33.89% |
| **Final Wealth** | 1.48x |
| **Win Rate** | 52.7% |

## 🚀 Quick Start

> **Note:** This project is completely self-contained with its own virtual environment. It doesn't use any resources from the root ARCUS folder or root environment packages due to version compatibility requirements.

### Prerequisites

- Python 3.10 or 3.11 (3.10 recommended)
- pip (Python package manager)
- 8GB RAM minimum, 16GB recommended
- Internet connection (for installing dependencies)

### Installation

```bash
# Navigate to SIT folder
cd SIT

# Create virtual environment (Python 3.10 recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Train a Model (CLI)

```bash
# Standard configuration: 30 assets, 60-day window, 20-day horizon
python run.py \
    --data_path ./data/full_dataset.csv \
    --data_pool 30 \
    --window_size 60 \
    --horizon 20 \
    --d_model 8 \
    --n_heads 4 \
    --num_layers 2 \
    --train_epochs 30 \
    --batch_size 64 \
    --learning_rate 0.001
```

**Expected time:** ~30-45 minutes on CPU

### Interactive Dashboard

```bash
# Launch Streamlit web interface
streamlit run streamlit_app.py
```

Opens browser at `http://localhost:8501` with:
- Dashboard with performance metrics
- Interactive training interface
- Results analysis with visualizations
- Live testing on custom CSV data
- Data explorer with correlation analysis
- Complete documentation

## 📊 Dataset Information

### Included Dataset

The submission includes a pre-prepared dataset (`data/full_dataset.csv`) with:
- **76 S&P 100 stocks** (blue-chip companies across sectors)
- **Date range:** 1999-12-31 to 2024-12-30 (25 years)
- **6,289 trading days** of adjusted close prices
- **Format:** CSV with 'Date' column + one column per asset ticker

**You can use this dataset immediately** - no additional data preparation needed!

### Using Custom Data

To train on your own data or test on different stocks:

#### Required CSV Format

```csv
Date,AAPL,MSFT,GOOGL,AMZN,JPM
2024-01-02,185.64,376.04,140.93,151.94,168.51
2024-01-03,184.25,375.37,139.69,151.03,166.89
2024-01-04,181.91,367.93,138.45,149.61,165.32
```

**Requirements:**
- First column must be named `Date` (case-sensitive, YYYY-MM-DD format)
- Remaining columns: Daily adjusted close prices for each asset
- No missing values (NaN or empty cells)
- All prices must be positive (> 0)
- Business days only (Mon-Fri, excluding market holidays)
- Chronological order (oldest to newest)

#### Where to Download Historical Stock Data

**1. Yahoo Finance (Recommended - Free, No Registration)**

- **Website:** https://finance.yahoo.com
- **Steps:**
  1. Search for a stock ticker (e.g., AAPL for Apple)
  2. Click "Historical Data" tab
  3. Set time period (use "Max" for all available data)
  4. Click "Download" button → downloads CSV file
  5. Repeat for each stock you want to include
  6. Combine all stocks into one CSV file
- **Important:** Use "Adj Close" column (accounts for stock splits and dividends)
- **Tip:** Download 2+ years of data for meaningful training


#### Combining Multiple Yahoo Finance Downloads

If you downloaded stocks separately from Yahoo Finance:

**Python script to combine:**
```python
import pandas as pd

# Load individual stock downloads
aapl = pd.read_csv('AAPL.csv')[['Date', 'Adj Close']].rename(columns={'Adj Close': 'AAPL'})
msft = pd.read_csv('MSFT.csv')[['Date', 'Adj Close']].rename(columns={'Adj Close': 'MSFT'})
googl = pd.read_csv('GOOGL.csv')[['Date', 'Adj Close']].rename(columns={'Adj Close': 'GOOGL'})

# Merge on Date
combined = aapl.merge(msft, on='Date').merge(googl, on='Date')

# Remove any rows with missing values
combined = combined.dropna()

# Sort by date
combined = combined.sort_values('Date').reset_index(drop=True)

# Save combined dataset
combined.to_csv('full_dataset.csv', index=False)
print(f"✅ Combined dataset saved: {len(combined)} days, {len(combined.columns)-1} assets")
```

#### Using Custom Data with SIT

**Option 1: Replace the included dataset**
```bash
# Backup original
cp data/full_dataset.csv data/full_dataset_original.csv

# Use your data
cp your_custom_data.csv data/full_dataset.csv

# Train normally
python run.py --data_pool 20  # Uses first 20 assets from your CSV
```

**Option 2: Specify custom path**
```bash
# Train with custom data path
python run.py \
    --data_path /path/to/your/custom_data.csv \
    --data_pool 15 \
    --train_epochs 30
```

**Option 3: Upload via Streamlit (Easiest)**
```bash
# Launch Streamlit app
streamlit run streamlit_app.py

# Live Testing Tab  
# - Select a pre-trained model
# - Upload your CSV for testing
# - Map assets and run backtest
```

### Data Quality Requirements

Before using custom data, ensure:

- [ ] **Date column:** Named exactly `Date` (case-sensitive)
- [ ] **Date format:** YYYY-MM-DD (e.g., 2024-01-15, not MM/DD/YYYY)
- [ ] **No gaps:** Remove rows with any missing values
- [ ] **Positive prices:** All values > 0 (no zeros or negatives)
- [ ] **Chronological:** Sorted oldest to newest
- [ ] **Sufficient history:** At least 500+ days recommended
- [ ] **Adjusted prices:** Use adjusted close (not raw close)
- [ ] **Clean data:** No special characters, proper CSV encoding (UTF-8)

### Data Validation (This step is clearly optional but just in case you run into issue this can help debugging faster)

Test your CSV before training:

```python
import pandas as pd

# Load your CSV
df = pd.read_csv('your_data.csv')

# Check format
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Negative prices: {(df.iloc[:, 1:] <= 0).sum().sum()}")

# Should see:
# - First column is 'Date'
# - No missing values (0)
# - No negative prices (0)
# - Reasonable date range
```

```

## ✅ Verification Steps

To verify the installation and test the implementation:

```bash
# 1. Activate environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Verify imports work
python -c "from models.sit import SIT; print('✅ All imports successful')"

# 3. Quick training test (5 minutes)
python run.py \
    --data_pool 5 \
    --window_size 20 \
    --horizon 5 \
    --train_epochs 2 \
    --batch_size 32

# 4. Launch dashboard
streamlit run streamlit_app.py
```

**Expected outcomes:**
- Training completes without errors
- Streamlit opens in browser at http://localhost:8501
- Dashboard shows existing results (Sharpe 0.51, etc.)
- All navigation tabs work (Dashboard, Training, Results, Live Testing, Data Explorer, Documentation)

## 📁 Project Structure

```
SIT/
├── data/
│   └── full_dataset.csv          # S&P 100 daily prices (1999-2024)
│
├── models/
│   └── sit.py                    # SIT transformer architecture
│
├── losses/
│   └── cvar_loss.py              # CVaR loss implementation
│
├── data_provider/
│   ├── data_loader.py            # Dataset with on-the-fly signatures
│   └── data_factory.py           # Train/val/test data splits
│
├── experiments/
│   ├── exp_main.py               # Training loop and backtest
│   └── exp_basic.py              # Device/model utilities
│
├── utils/
│   ├── metrics.py                # Sharpe, Sortino, drawdown metrics
│   ├── timefeatures.py           # Temporal feature encoding
│   └── tools.py                  # LR scheduler, early stopping
│
├── checkpoints/
│   └── checkpoint.pth            # Trained model weights
│
├── results/
│   ├── results_*.csv             # Performance metrics
│   ├── equity_curve_*.png        # Equity curve plots
│   └── positions_*.csv           # Daily portfolio weights
│
├── run.py                        # CLI training script
├── streamlit_app.py              # Interactive web dashboard
├── precompute_signatures.py      # Optional: precompute for speed
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧠 Technical Implementation

### Path Signature Computation

Signatures are computed on-the-fly for each 60-day price window:

**Level-1 (Displacement):**
```python
sig_1 = price[-1] - price[0]
```

**Level-2 (Signed Area/Momentum):**
```python
sig_2 = sum(price[:-1] * diff(price))
```

**Cross-Signature (Pairwise Interaction):**
```python
cross_sig[j,k] = sum(cumsum(diff(price_j))[:-1] * diff(price_k)[1:])
```

### Model Architecture

```
Input: Price Windows (B × H × D)
  ↓
Signature Extraction (B × H × D × 2)
  ↓
Embedding Layer (d_model=8)
  ↓
Factored Transformer (2 layers, 4 heads)
  ├─ Temporal Attention (causal, per asset)
  └─ Asset Attention (with cross-sig bias)
  ↓
Output Projection → Softmax
  ↓
Portfolio Weights (B × H × D)
```

**Key Features:**
- Factored attention reduces complexity from O(H²D²) to O(H²D + HD²)
- Dynamic attention bias from cross-signatures guides asset interactions
- Causal masking prevents look-ahead bias in temporal attention

### CVaR Loss Function

Instead of traditional variance or Sharpe optimization, SIT minimizes Conditional Value-at-Risk:

```python
portfolio_returns = sum(weights * asset_returns, dim=-1)
VaR = quantile(portfolio_returns, alpha=0.95)
CVaR = VaR + mean(max(0, VaR - portfolio_returns)) / (1 - alpha)
loss = CVaR.mean()
```

This directly penalizes tail risk (worst 5% of outcomes).

### Training Details

- **Data Splits:**
  - Train: 2000-2016 (4,252 days)
  - Validation: 2017-2019 (755 days)  
  - Test: 2020-2024 (1,258 days)

- **Optimization:**
  - Adam optimizer with lr=1e-3
  - Learning rate halved each epoch
  - Early stopping: patience=3 on validation CVaR
  - Batch size: 64

- **Backtest:**
  - Monthly rebalancing (first business day)
  - Transaction costs: 10 basis points (0.1%)
  - Long-only constraint enforced via softmax

## 📊 Usage Examples

### Example 1: Train with Different Configurations

```bash
# Larger model (more capacity)
python run.py --data_pool 30 --d_model 16 --n_heads 8 --num_layers 3

# Higher temperature (more diversification)
python run.py --data_pool 30 --temperature 2.0

# Different risk tolerance
python run.py --data_pool 30 --cvar_alpha 0.99  # More conservative
```

### Example 2: Test on Custom Data (Streamlit)

1. Launch app: `streamlit run streamlit_app.py`
2. Navigate to "Live Testing" tab
3. Select trained model from dropdown
4. Upload CSV with Date + asset prices
5. Configure test period and transaction costs
6. Run backtest and view results
7. Download performance reports

### Example 3: Precompute Signatures for Speed

```bash
# Precompute signatures to disk (speeds up training 2-3x)
python precompute_signatures.py \
    --data_path ./data/full_dataset.csv \
    --window_size 60 \
    --horizon 20 \
    --data_pool 30

# Then train with precomputed data
python run.py --use_precomputed True --precomp_root ./signature_cache
```

## 🔬 Academic Context

### Research Contributions

1. **Geometric Deep Learning:** Integrates Rough Path Theory with transformers
2. **Risk-Aware Optimization:** Direct CVaR minimization vs. indirect Sharpe maximization
3. **Factored Attention:** Efficient decomposition for financial time series
4. **Signature-Guided Bias:** Cross-signatures inform asset attention weights

### Comparison to Baselines

| Model | Sharpe | Annual Return | Max Drawdown |
|-------|--------|---------------|--------------|
| Equal Weight | 0.35 | 7.2% | -38% |
| Min Variance | 0.42 | 6.8% | -29% |
| **SIT (Ours)** | **0.51** | **8.27%** | **-33.9%** |

### Key Findings

- ✅ Path signatures provide meaningful inductive bias for portfolio optimization
- ✅ CVaR loss effectively controls downside risk (Sortino 0.66 > Sharpe 0.51)
- ✅ Factored attention maintains computational efficiency and interpretability
- ✅ Model generalizes well across market regimes (2020-2024 includes COVID crash)
- ✅ Outperforms traditional equal-weight and minimum-variance baselines

## 📊 Detailed Hyperparameters

### Data Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `data_pool` | 30 | 5-76 | Number of assets to use |
| `window_size` | 60 | 20-120 | Historical lookback window (days) |
| `horizon` | 20 | 5-30 | Prediction horizon (days) |

### Model Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `d_model` | 8 | 4-32 | Embedding dimension |
| `n_heads` | 4 | 2-8 | Number of attention heads |
| `num_layers` | 2 | 1-4 | Number of transformer layers |
| `ff_dim` | 32 | 16-128 | Feed-forward dimension |
| `temperature` | 1.3 | 0.5-3.0 | Softmax temperature (diversification) |
| `dropout` | 0.1 | 0.0-0.5 | Dropout rate |

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `train_epochs` | 30 | 5-50 | Maximum training epochs |
| `batch_size` | 64 | 16-128 | Batch size |
| `learning_rate` | 0.001 | 0.0001-0.01 | Initial learning rate |
| `patience` | 3 | 1-10 | Early stopping patience |
| `cvar_alpha` | 0.95 | 0.90-0.99 | CVaR confidence level |
| `trade_cost_bps` | 10.0 | 0-50 | Transaction cost (basis points) |

## ⚙️ System Requirements

- **Python:** 3.10+ (3.10 recommended for PyTorch compatibility)
- **RAM:** 8GB minimum, 16GB recommended for 30+ assets
- **Storage:** ~500MB for code + data + results
- **GPU:** Optional (CPU sufficient for 30 assets, ~45 min training)
- **OS:** Windows 10/11, macOS, Linux

### Tested Environments

- ✅ Apple M4 Mac (macOS Sequoia)
- ✅ Windows 10/11 with Python 3.10
- ✅ Conda environment: v-machine (Python 3.10)
- ✅ Virtual environment: venv with Python 3.10

## 🐛 Troubleshooting

### Common Issues

**1. PyTorch DLL Error (Windows):**
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**2. Import Errors:**
```bash
# Ensure you're in the SIT directory
cd SIT
python -c "from models.sit import SIT; print('OK')"
```

**3. Out of Memory:**
```bash
# Reduce batch size or number of assets
python run.py --batch_size 32 --data_pool 20
```

**4. Streamlit Won't Start:**
```bash
# Check if port 8501 is in use
# Manually open browser to http://localhost:8501
```

**5. Training Takes Too Long:**
```bash
# Quick test with smaller config
python run.py --data_pool 10 --window_size 30 --horizon 10 --train_epochs 5
```

**6. CSV Upload Fails:**
- Verify Date column exists and is named exactly `Date`
- Check for missing values: `df.isnull().sum()`
- Ensure positive prices: `(df > 0).all()`
- Use UTF-8 encoding when saving CSV

## 📚 Documentation

### In-App Documentation

Complete documentation is available in the Streamlit dashboard:

1. Run `streamlit run streamlit_app.py`
2. Navigate to "Documentation" tab
3. Covers:
   - Introduction to SIT
   - How signatures work
   - Model architecture details
   - Parameter tuning guide
   - Interpreting results
   - Best practices
   - Glossary of terms

### Code Documentation

All modules include comprehensive docstrings:
- `models/sit.py` - Architecture and forward pass
- `losses/cvar_loss.py` - CVaR computation
- `data_provider/data_loader.py` - Signature extraction logic
- `experiments/exp_main.py` - Training loop details

## 📚 References

**Primary Paper:**
- Hwang, Y., & Zohren, S. (2023). "Signature-Informed Transformer for Attention-Based Asset Allocation"

**Foundational Work:**
- Lyons, T. (1998). "Differential equations driven by rough signals" - Rough Path Theory
- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk"
- Vaswani, A., et al. (2017). "Attention is All You Need" - Transformer architecture

## 🙏 Acknowledgments

- **Original research:** Hwang & Zohren for the SIT methodology
- **Data source:** Yahoo Finance for historical price data
- **Frameworks:** PyTorch (deep learning), Streamlit (dashboard), Plotly (visualizations)
- **Collaboration:** Portfolio Transformer baseline implemented by Angie
- **Computing:** Apple M4 Mac, conda environment configuration

## 📄 License & Disclaimer

**Academic Research Project - Educational Use Only**

This implementation is created in good faith for academic research and educational purposes. It is based on published research methodologies and uses publicly available financial data.


**Educational Objectives:**
- Understanding path signatures and Rough Path Theory
- Implementing transformer architectures for time series
- Portfolio optimization with CVaR risk management
- Building interactive ML dashboards with Streamlit

---

**Project Team:**  
- **Vaibhav Desai** - SIT Implementation & Streamlit Dashboard
- **Angie** - Portfolio Transformer Baseline

**Academic Information:**  
- **Institution:** Northeastern University
- **Course:** CS-5100 Foundations of Artificial Intelligence
- **Instructor:** Prof. Amir T
- **Submission Date:** December 13, 2024

**Contact:**  
- **Email:** desai.vaibhav.dx@gmail.com / desai.vai@northeastern.edu

---

*This project was implemented with academic integrity, proper attribution to original research, transparent documentation of all methodologies and results, and adherence to educational best practices.*