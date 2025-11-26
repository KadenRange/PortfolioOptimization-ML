#!/usr/bin/env python
# coding: utf-8

# **1. Imports & Configuration** This sets up our environment. We chose specific dates (2012-2020 for training) to ensure our model learns from different market regimes before being tested on the post-COVID era (2021-2024).

# In[ ]:


import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import joblib
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.covariance import LedoitWolf
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# We suppress specific LightGBM warnings to keep our backtest logs clean
warnings.filterwarnings("ignore", message="X does not have valid feature names")
sns.set_style("whitegrid")


# We separate Training from Backtesting to prevent data leakage.
PRICE_START    = "2012-01-01"
TRAIN_END_DATE = "2020-12-31"   # ML Model sees data up to this point
BACKTEST_START = "2021-01-01"   # Simulation runs on unseen data from here
BACKTEST_END   = "2024-12-31"

LOOKBACK_DAYS  = 252            # 1-Year rolling window for Covariance estimation
REBALANCE_DAYS = 63             # Quarterly rebalancing to manage turnover costs
MAX_WEIGHT     = 0.05           # 5% cap per asset to force diversification
TOP_K          = 75             # Subset size for the optimizer to improve solver speed

RANDOM_SEED    = 42
SP500_HISTORY_CSV = "resources/s&p500_history.csv"   # PIT membership file

# Create output directories if they don't exist
OUT_DATA_DIR     = "resources/data"
OUT_MODELS_DIR   = "resources/models"
OUT_RESULTS_DIR  = "resources/results"

for d in [OUT_DATA_DIR, OUT_MODELS_DIR, OUT_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(RANDOM_SEED)
print("Environment configured.")


# **2. ETF Universe Definition:** We augmented the S&P 500 universe with liquid ETFs. This was a strategic decision to ensure the optimizer always has "safe harbor" assets (like Bonds or Gold) available during high-volatility periods.

# In[ ]:


# We include these liquid ETFs to ensure the optimizer has broad
# asset classes available for hedging, not just single stocks.
ETF_TICKERS = [
    # Sector ETFs
    'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
    'XLRE', 'XLU', 'XLY', 'XLC', 'XLV',
    # Factor / Style ETFs
    'SPLV', 'MTUM', 'QUAL', 'VLUE', 'IWM', 'VTV',
    # Broad Market
    'SPY', 'QQQ', 'DIA', 'VT', 'VTI',
    # International / EM
    'EFA', 'VWO', 'EWJ', 'EWU', 'EWC',
    # Bonds (Critical for risk-off regimes)
    'TLT', 'IEF', 'SHY', 'HYG', 'LQD',
    # Commodities / Volatility
    'GLD', 'SLV', 'DBC', 'USO', 'UUP', 'VIXY'
]


# **3. Successor Mapping (Data Cleaning):** Data quality is paramount. We implemented this mapping to handle corporate actions (like mergers) that would otherwise look like "delistings" to the model. This prevents the backtester from selling a stock just because it changed its name.

# In[ ]:


# We identified these specific corporate actions that broke our
# data pipeline and mapped them to their modern tickers.
SUCCESSOR_MAP = {
    "WFM":  ["AMZN"],     "TWC":  ["CMCSA"],    "COV":  ["MDT"],
    "LVLT": ["LUMN"],     "CELG": ["BMY"],      "STJ":  ["ABT"],
    "BXLT": ["TAK"],      "KRFT": ["KHC"],      "HNZ":  ["KHC"],
    "PCP":  ["BRK-B"],    "DTV":  ["T"],        "SYMC": ["GEN"],
    "RAI":  ["BTI"],      "BRCM": ["AVGO"],     "CVC":  ["CHTR"],
    "GAS":  ["SO"],       "LLTC": ["ADI"],      "ARG":  ["APD"],
    "SIAL": ["DHR"],      "NYX":  ["ICE"],      "FDO":  ["DLTR"],
    "JOY":  ["CAT"],      "CVH":  ["AET"],      "PGN":  ["DUK"],
    "APC":  ["OXY"],      "PXD":  ["XOM"],      "TWTR": ["X"],
    "FB":   ["META"],     "MXIM": ["ADI"],      "ALTR": ["INTC"],
    "BCR":  ["BDX"],      "CFN":  ["BDX"],      "GGP":  ["BAM"],
    "WLTW": ["WTW"],      "HCBK": ["MTB"],      "PBCT": ["MTB"],
    "WCG":  ["CNC"],      "MON":  ["BAYRY"],    "DISCK":["WBD"],
    "DISCA":["WBD"],      "VIAB": ["PARA"],     "CBS":  ["PARA"],
    "PARA": ["PARA"],     "HRS":  ["LHX"],      "RTN":  ["RTX"],
    "UTX":  ["RTX"],      "ABMD": ["MDT"],      "ALXN": ["AZN"],
    "DWDP": ["DOW"],      "XLNX": ["AMD"],      "AGN":  ["ABBV"],
    "TIF":  ["MC.PA"],    "FRX":  ["AGN"],      "LO":   ["RAI"],
    "GMCR": ["KDP"],

    # We explicitly drop these due to bankruptcy or privatization
    "APOL": [], "SPLS": [], "JCP":  [], "DF":   [],
    "CHK":  [], "MNK":  [], "DNR":  [],
}


# **4. Point-in-Time History Loading:** We load the historical S&P 500 composition. We normalize it using the map above to ensure that at any given date in the past, we are trading the correct universe of stocks.

# In[ ]:


def load_sp500_history(path_csv):
    """Parses the raw CSV containing historical index membership."""
    if not os.path.exists(path_csv):
        # We handle the missing file case for the demo
        print("Warning: history file not found. Using dummy data for testing.")
        dates = pd.date_range(start=PRICE_START, end=BACKTEST_END, freq='M')
        return pd.DataFrame({'date': dates, 'tickers': 'AAPL,MSFT,GOOG,AMZN'})

    sp = pd.read_csv(path_csv)
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp.sort_values("date").reset_index(drop=True)
    return sp

def normalize_sp500_with_successors(sp_df, mapping):
    """Applies our successor map to the raw history."""
    sp = sp_df.copy()

    def map_row(ticker_str):
        tickers = [t.strip() for t in ticker_str.split(",") if t.strip()]
        mapped = []
        for t in tickers:
            if t in mapping and mapping[t]:
                mapped.append(mapping[t][0]) # Use mapped successor
            elif t in mapping and not mapping[t]:
                continue # Skip delisted/bankrupt
            else:
                mapped.append(t)
        mapped = sorted(set(mapped))
        return ",".join(mapped)

    sp["tickers"] = sp["tickers"].apply(map_row)
    return sp

def get_sp500_members_on(date, sp_df):
    """Returns the valid investment universe for a specific date."""
    rows = sp_df[sp_df["date"] <= date]
    if rows.empty:
        return []
    last_row = rows.iloc[-1]
    tickers = [t.strip() for t in last_row["tickers"].split(",") if t.strip()]
    return tickers

# Load and Normalize
sp500_df_raw = load_sp500_history(SP500_HISTORY_CSV)
sp500_df = normalize_sp500_with_successors(sp500_df_raw, SUCCESSOR_MAP)
sp500_df.to_csv(os.path.join(OUT_DATA_DIR, "sp500_history_normalized.csv"), index=False)
print("S&P 500 History loaded and normalized.")


# **5. Price Downloads:** We download the price data for all assets. We included a specific fix for BRK.B because Yahoo Finance often rejects the dot notation, requiring a dash BRK-B.

# In[ ]:


def build_universe_from_sp500(sp_df, start_date, end_date, extra_tickers=None):
    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)
    sub = sp_df[(sp_df["date"] >= start_date) & (sp_df["date"] <= end_date)]
    all_tickers = set()
    for row in sub["tickers"]:
        all_tickers |= set(t.strip() for t in row.split(",") if t.strip())
    if extra_tickers:
        all_tickers |= set(extra_tickers)
    return sorted(all_tickers)

def download_prices_yf(tickers, start_date, end_date):
    print(f"Downloading prices for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.sort_index()

# Build Universe & Download
universe_all = build_universe_from_sp500(sp500_df, PRICE_START, BACKTEST_END, ETF_TICKERS)
prices_raw = download_prices_yf(universe_all, PRICE_START, BACKTEST_END)

# Fix Ticker Formatting (e.g., BRK.B -> BRK-B)
def fix_ticker_formatting(df):
    fixed = {}
    for t in df.columns:
        if df[t].isna().all() and "." in t:
            alt = t.replace(".", "-")
            try:
                # Quick probe to see if the dashed version exists
                test = yf.download(alt, start=PRICE_START, end=BACKTEST_END, progress=False)["Close"]
                if not test.isna().all():
                    fixed[t] = alt
            except Exception: pass
    return fixed

fmt_map = fix_ticker_formatting(prices_raw)
for old, new in fmt_map.items():
    print(f"[FORMAT FIX] {old} → {new}")
    prices_raw[new] = prices_raw[old]
    del prices_raw[old]

prices = prices_raw.dropna(axis=1, how="all")
prices.to_pickle(os.path.join(OUT_DATA_DIR, "prices.pkl"))
print("Final Price Matrix Shape:", prices.shape)


# **6. Feature Engineering:** Here we calculate our Alpha signals. We chose to target the Cross-Sectional Z-Score rather than raw returns. This neutralizes market beta—meaning our model learns to pick "winners vs losers" rather than just predicting "everything goes up."

# In[ ]:


def engineer_features(prices, horizon=21):
    """
    We calculate technical factors including Momentum (20d), Volatility (10d),
    and Mean Reversion (MA deviation).
    Target: Cross-Sectional Z-Score of forward returns.
    """
    prices_sorted = prices.copy().sort_index()
    log_ret = np.log(prices_sorted / prices_sorted.shift(1))

    records = []
    for t in prices_sorted.columns:
        s_price = prices_sorted[t]
        s_log   = log_ret[t]

        df_t = pd.DataFrame(index=prices_sorted.index)
        df_t["ticker"]       = t
        df_t["log_ret_1d"]   = s_log
        df_t["log_ret_5d"]   = s_log.rolling(5).sum()
        df_t["ma_10d_dev"]   = s_price.rolling(10).mean() / s_price - 1
        df_t["vol_10d"]      = s_log.rolling(10).std()
        df_t["mom_20d"]      = s_price.pct_change(20, fill_method=None)

        # Forward target (next month's return)
        df_t["target_raw"] = s_log.shift(-horizon).rolling(horizon).sum()
        records.append(df_t)

    feats = pd.concat(records)
    feats = feats.dropna()

    # We use Z-scores to normalize across time
    def cs_z(x):
        return (x - x.mean()) / (x.std(ddof=0) + 1e-8)

    feats["target"] = feats.groupby("date")["target_raw"].transform(cs_z)
    feats = feats.drop(columns=["target_raw"])

    return feats

features_all = engineer_features(prices, horizon=21)
features_all.to_pickle(os.path.join(OUT_DATA_DIR, "features_all.pkl"))

# Chronological Split
features_train = features_all[features_all.index <= TRAIN_END_DATE]
features_test  = features_all[features_all.index > TRAIN_END_DATE]

X_train = features_train.drop(columns=["target"])
y_train = features_train["target"]
print(f"Features Generated. Training shape: {X_train.shape}")


# **7. Model Training & Selection:** We define three candidate models (Ridge, XGBoost, LightGBM) and evaluate them using a chronological validation set. We automatically select the best model to use for the backtest.

# In[ ]:


def build_ml_models():
    """Defines our candidate model architectures."""
    categorical = ["ticker"]
    numeric = ["log_ret_1d", "log_ret_5d", "ma_10d_dev", "vol_10d", "mom_20d"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    models = {
        "ridge": Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))]),
        "xgb": Pipeline([("prep", preprocessor), ("model", XGBRegressor(n_estimators=300, max_depth=6, n_jobs=-1, random_state=RANDOM_SEED))]),
        "lgbm": Pipeline([("prep", preprocessor), ("model", LGBMRegressor(n_estimators=300, num_leaves=31, random_state=RANDOM_SEED))])
    }
    return models

def train_and_select_model(X_train, y_train, features_all, train_end_date):
    # Validation Split (Last 20% of training data)
    cutoff = int(len(X_train) * 0.8)
    X_tr_sub, y_tr_sub = X_train.iloc[:cutoff], y_train.iloc[:cutoff]
    X_val_sub, y_val_sub = X_train.iloc[cutoff:], y_train.iloc[cutoff:]

    models = build_ml_models()
    val_mse = {}

    print("Evaluating models on validation set...")
    for name, model in models.items():
        model.fit(X_tr_sub.drop(columns=["target"], errors='ignore'), y_tr_sub)
        preds = model.predict(X_val_sub.drop(columns=["target"], errors='ignore'))
        mse = mean_squared_error(y_val_sub, preds)
        val_mse[name] = mse
        print(f"Model {name} MSE: {mse:.6f}")

    best_name = min(val_mse, key=val_mse.get)
    print(f"Selected best model: {best_name}")

    # Retrain best model on full training data
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, os.path.join(OUT_MODELS_DIR, "best_ml_model.joblib"))
    return best_model

best_ml_model = train_and_select_model(X_train, y_train, features_all, TRAIN_END_DATE)


# **8. Risk & Covariance Utilities:** To calculate the "Mean-Variance" optimization, we need a Covariance Matrix. We use Ledoit-Wolf shrinkage to make the matrix mathematically robust (Positive Semi-Definite) so our optimizer doesn't crash.

# In[ ]:


def make_psd(matrix, eps=1e-6):
    """
    Forces covariance matrix to be Symmetric Positive Semi-Definite (PSD)
    by clipping negative eigenvalues. Essential for CVXPY stability.
    """
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals_clipped = np.clip(vals, eps, None)
    return vecs @ np.diag(vals_clipped) @ vecs.T

def compute_mu_cov_for_universe(prices, active_tickers, reb_date, lookback_days=252):
    """Computes robust Mu (Expected Return) and Sigma (Covariance) using Ledoit-Wolf."""
    px = prices[active_tickers].loc[:reb_date].tail(lookback_days)
    log_ret = np.log(px / px.shift(1)).dropna(how="any")

    if log_ret.shape[0] < 10 or log_ret.shape[1] < 5:
        return None, None, []

    mu = log_ret.mean().values
    lw = LedoitWolf().fit(log_ret.values) # Shrinkage estimation
    cov_psd = make_psd(lw.covariance_)

    return mu, cov_psd, log_ret.columns.tolist()


# **9. Optimization Engine:** This is the core mathematical engine. We use cvxpy to solve for the weights that maximize return minus risk, subject to constraints (no shorting, max 5% per stock).

# In[ ]:


def mean_variance_opt_full(mu, cov, max_weight=0.05, risk_aversion=1.0):
    """
    Solves: maximize (mu.T @ w) - gamma * (w.T @ cov @ w)
    Subject to: sum(w)=1, 0 <= w <= max_weight
    """
    n = len(mu)
    w = cp.Variable(n)

    # We wrap in psd_wrap to assure the solver the matrix is safe
    cov_psd = cp.psd_wrap(cov)

    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cov_psd))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except:
        return np.ones(n) / n # Fallback to Equal Weight on failure

    if w.value is None: return np.ones(n) / n

    # Clean up small numerical noise
    w_val = np.clip(np.array(w.value).ravel(), 0, None)
    return w_val / w_val.sum()


# **10. Monte Carlo Simulation:** For every single rebalance, we run a simulation to see "what could happen" over the next quarter. This gives us our risk metrics (VaR/CVaR).

# In[ ]:


def monte_carlo_sim(mu, cov, weights, horizon=21, n_sims=10000):
    """
    Simulates thousands of potential future price paths to estimate risk.
    """
    try:
        # Generate random market scenarios
        sims = np.random.multivariate_normal(mu, cov, size=(n_sims, horizon))
    except np.linalg.LinAlgError:
        # Fallback for numerical instability
        sims = np.random.multivariate_normal(mu, np.diag(np.diag(cov)), size=(n_sims, horizon))

    port_daily = sims @ weights
    # Calculate cumulative return for every path
    final_returns = np.exp(np.cumsum(port_daily, axis=1))[:, -1] - 1
    return final_returns


# **11. Visualization Utilities:** We separated the plotting logic to keep the backtest loops clean.

# In[ ]:


def plot_mc_sample(mc_returns, out_dir, tag):
    plt.figure(figsize=(8, 5))
    plt.hist(mc_returns, bins=100, density=True, alpha=0.7)
    plt.title(f"Monte Carlo Risk Distribution ({tag})")
    plt.xlabel("Simulated Return")
    plt.savefig(os.path.join(out_dir, f"mc_dist_{tag}.pdf"))
    plt.show()

def plot_final_weights(weights_hist, name, out_dir):
    if not weights_hist: return
    # Extract last rebalance weights
    last_w = weights_hist[-1][1].sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 5))
    last_w.plot(kind="bar")
    plt.title(f"{name}: Top 20 Holdings (Final Rebalance)")
    plt.savefig(os.path.join(out_dir, f"{name}_weights.pdf"))
    plt.show()


# **12. Backtesting Functions:** Here we define the logic for our three strategies: Equal Weight, MVO-Historical, and MVO-ML. The ML strategy blends the model's alpha score with the historical mean to avoid making extreme bets.

# In[ ]:


def run_backtest(strategy_type, prices, sp500_df, start_date, end_date, ml_model=None, features=None):
    """Unified backtesting engine for all 3 strategies."""
    prices_bt = prices.loc[start_date:end_date]
    dates = prices_bt.index
    capital = 1.0
    vals, weights_hist = [], []

    i = 0
    print(f"--- Running Backtest: {strategy_type} ---")

    while i < len(dates):
        reb_date = dates[i]

        # 1. Define Universe (PIT)
        sp_members = get_sp500_members_on(reb_date, sp500_df)
        universe = [t for t in sp_members if t in prices_bt.columns and prices_bt[t].notna().sum() > LOOKBACK_DAYS]

        # 2. Estimate Risk Model
        mu_hist, cov, tickers = compute_mu_cov_for_universe(prices_bt, universe, reb_date)

        if cov is None:
            i += REBALANCE_DAYS
            continue

        # 3. Optimize Weights based on Strategy
        if strategy_type == "Equal":
            w = np.ones(len(tickers)) / len(tickers)

        elif strategy_type == "MV-Hist":
            w = mean_variance_opt_full(mu_hist, cov)

        elif strategy_type == "MV-ML":
            # Filter features for current date
            current_feats = features.loc[features.index <= reb_date].groupby('ticker').last()
            valid_tickers = [t for t in tickers if t in current_feats.index]

            # Predict Alpha
            preds = ml_model.predict(current_feats.loc[valid_tickers].drop(columns=['target'], errors='ignore'))

            # Blend ML Alpha with Historical Mu (Bayesian-style shrinkage)
            # This prevents the model from making wild bets on low-confidence predictions
            alpha_vec = pd.Series(preds, index=valid_tickers).reindex(tickers).fillna(0).values
            mu_blended = 0.4 * alpha_vec + 0.6 * mu_hist

            w = mean_variance_opt_full(mu_blended, cov)

        # 4. Simulate Forward Performance
        hold_days = min(REBALANCE_DAYS, len(dates) - i)
        px_slice = prices_bt.iloc[i : i + hold_days][tickers]
        ret_slice = np.log(px_slice / px_slice.shift(1)).dropna()

        port_ret = ret_slice @ w
        for r in port_ret:
            capital *= np.exp(r)
            vals.append(capital)

        weights_hist.append((reb_date, pd.Series(w, index=tickers)))

        # Plot MC Risk for the first rebalance only
        if i == 0:
            mc_ret = monte_carlo_sim(mu_hist, cov, w)
            plot_mc_sample(mc_ret, OUT_RESULTS_DIR, strategy_type)

        i += REBALANCE_DAYS

    return pd.DataFrame({'portfolio': vals}, index=dates[:len(vals)]), weights_hist


# **13. Execution:** We run all three backtests sequentially. This takes the heavy lifting defined above and actually executes it on the data.

# In[ ]:


# Run the Backtests
port_eq, w_eq = run_backtest("Equal", prices, sp500_df, BACKTEST_START, BACKTEST_END)
port_hist, w_hist = run_backtest("MV-Hist", prices, sp500_df, BACKTEST_START, BACKTEST_END)
port_ml, w_ml = run_backtest("MV-ML", prices, sp500_df, BACKTEST_START, BACKTEST_END,
                             ml_model=best_ml_model, features=features_all)

# Save Results
port_eq.to_csv(os.path.join(OUT_RESULTS_DIR, "port_eq.csv"))
port_hist.to_csv(os.path.join(OUT_RESULTS_DIR, "port_hist.csv"))
port_ml.to_csv(os.path.join(OUT_RESULTS_DIR, "port_ml.csv"))
print("Backtests Complete.")


# **14. Final Comparison & Stats:** We compile the equity curves into one chart and calculate the final Sharpe Ratios to see which strategy won.

# In[ ]:


def compute_stats(df, name):
    ret = df['portfolio'].pct_change().dropna()
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    return {"Strategy": name, "Sharpe": sharpe, "Vol": ann_vol, "Return": ann_ret}

stats = pd.DataFrame([
    compute_stats(port_eq, "Equal Weight"),
    compute_stats(port_hist, "MV-Hist"),
    compute_stats(port_ml, "MV-ML")
]).set_index("Strategy")

print("--- Final Performance Stats ---")
display(stats)

# Plot Equity Curves
plt.figure(figsize=(12, 6))
plt.plot(port_eq, label="Equal Weight")
plt.plot(port_hist, label="MV-Hist")
plt.plot(port_ml, label="MV-ML")
plt.title("Cumulative Portfolio Performance (2021-2024)")
plt.legend()
plt.savefig(os.path.join(OUT_RESULTS_DIR, "final_comparison.pdf"))
plt.show()

# Show Top Holdings
plot_final_weights(w_ml, "MV-ML", OUT_RESULTS_DIR)

