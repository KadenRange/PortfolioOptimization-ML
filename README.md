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

Place it in the resources folder.

---

## 3. Running the Project

## Running the Project (From Submitted ZIP)

If you are running this project from the submitted ZIP file, use the following steps:

### Google Colab Instructions

1. Upload the ZIP file to Colab.

2. Run the following to upload it programmatically:

```
from google.colab import files
uploaded = files.upload()  # Upload PortfolioOptimization-ML.zip
```

3. Unzip and enter the project directory:

```
!unzip PortfolioOptimization-ML.zip -d project
%cd project
```

4. Install dependencies and run the full pipeline:

```
!pip install -r requirements.txt
!python Port_Optimization.py
```

All results will be written to:

```
resources/results/
```

---

## Optional: Running the Latest Version from GitHub Instead of the ZIP

If you'd prefer to run the most up-to-date version of the project:

```
!git clone https://github.com/KadenRange/PortfolioOptimization-ML.git
%cd PortfolioOptimization-ML

!pip install -r requirements.txt
!python Port_Optimization.py
```


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

model = joblib.load("resources/models/best_ml_model.joblib")
prices = pd.read_pickle("resources/data/prices.pkl")
features = pd.read_pickle("resources/data/features_all.pkl")
```

You can call the backtest functions directly with these.
