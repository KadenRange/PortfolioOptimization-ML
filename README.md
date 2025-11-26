# ML-Enhanced Portfolio Construction (PIT S&P 500 + ML Alpha + MVO)

This project implements a reproducible portfolio backtesting engine that combines:
- Point-in-time S&P 500 membership (no look-ahead bias)
- Successor ticker mapping for delisted/merged companies
- ETF universe expansion for more stable covariance estimation
- Machine learning alpha (Ridge, XGBoost, LightGBM)
- Mean-variance optimization with Ledoit–Wolf shrinkage
- Monte Carlo risk simulation at each rebalance
- Full portfolio/value/weights visualization

All outputs (models, data, results) are saved to disk for reproducibility.

---

## 1. Requirements

**Python:** 3.10+

**Install dependencies:**
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm cvxpy yfinance joblib ecos scs

---

## 2. Required Data

Download the point-in-time S&P 500 membership (CSV) from: https://github.com/fja05680/sp500

Required file: s&p500_history.csv

Place it in the same directory as `main.py`.

---

## 3. Running the Project
All results are written to:

data/

models/

results/

Key outputs:
- `results/portfolio_equity_curves.pdf`  
- `results/strategy_stats.csv`  
- `results/*_final_weights.pdf`  
- `results/*_holdings_area.pdf`  
- `results/mc_*_summary.csv`

These include equity curves, statistical performance, top holdings evolution, and Monte Carlo risk metrics.

---

## 4. Overview of Method

**Feature Engineering**
- 1d/5d log returns  
- 10d mean reversion  
- 10d realized volatility  
- 20d momentum  
- 21-day forward return (cross-sectional z-score)

**Models**
- Ridge  
- XGBoost  
- LightGBM (best validation MSE)

**Strategies**
1. Equal Weight  
2. Mean-Variance (historical μ/Σ with Ledoit–Wolf)  
3. ML-Enhanced Mean-Variance (ML-scored universe → pseudo-μ → MVO)

**Risk**
- 10,000-path Monte Carlo at each rebalance  
- VaR(5%), CVaR(5%), expected return, volatility, Sharpe

---

## 5. Reproducibility

Saved files allow re-running without retraining:

```python
import joblib, pandas as pd

model = joblib.load("models/best_ml_model.joblib")
prices = pd.read_pickle("data/prices.pkl")
features = pd.read_pickle("data/features_all.pkl")
```

You can call the backtest functions directly with these.
