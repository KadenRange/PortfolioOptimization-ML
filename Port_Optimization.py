import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import joblib
import os
import pickle
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.covariance import LedoitWolf
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Suppress LightGBM feature-name warning spam
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# -----------------------
# CONFIG / PATHS
# -----------------------
PRICE_START    = "2012-01-01"
TRAIN_END_DATE = "2020-12-31"   # ML trained on dates <= this
BACKTEST_START = "2021-01-01"
BACKTEST_END   = "2024-12-31"

LOOKBACK_DAYS  = 252            # trailing window for μ, Σ
REBALANCE_DAYS = 63             # ~quarterly (trading days)
MAX_WEIGHT     = 0.05           # per-asset cap
TOP_K          = 75             # ML-selected subset size

RANDOM_SEED    = 42

SP500_HISTORY_CSV = "resources/s&p500_history.csv"   # PIT membership file

OUT_DATA_DIR     = "resources/data"
OUT_MODELS_DIR   = "resources/models"
OUT_RESULTS_DIR  = "resources/results"

os.makedirs(OUT_DATA_DIR, exist_ok=True)
os.makedirs(OUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# Extra ETFs to enrich diversification
ETF_TICKERS = [
    # Sector ETFs
    'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
    'XLRE', 'XLU', 'XLY', 'XLC', 'XLV',

    # Factor / Style ETFs
    'SPLV', 'MTUM', 'QUAL', 'VLUE', 'IWM', 'VTV',

    # Broad Market ETFs
    'SPY', 'QQQ', 'DIA', 'VT', 'VTI',

    # International / EM
    'EFA', 'VWO', 'EWJ', 'EWU', 'EWC',

    # Bonds
    'TLT', 'IEF', 'SHY', 'HYG', 'LQD',

    # Commodities / Alternatives
    'GLD', 'SLV', 'DBC', 'USO', 'UUP',

    # Volatility
    'VIXY'
]

# ---------------------------------------------------
# 0. HYBRID SUCCESSOR MAPPING (STYLE 3)
# ---------------------------------------------------
SUCCESSOR_MAP = {
    # Well-known acquisitions / mergers
    "WFM":  ["AMZN"],     # Whole Foods -> Amazon
    "TWC":  ["CMCSA"],    # Time Warner Cable -> Comcast
    "COV":  ["MDT"],      # Covidien -> Medtronic
    "LVLT": ["LUMN"],     # Level 3 -> Lumen
    "CELG": ["BMY"],      # Celgene -> Bristol-Myers
    "STJ":  ["ABT"],      # St. Jude Medical -> Abbott Labs
    "BXLT": ["TAK"],      # Baxalta -> Takeda (ADR)
    "KRFT": ["KHC"],      # Kraft -> Kraft Heinz
    "HNZ":  ["KHC"],      # Heinz -> Kraft Heinz
    "PCP":  ["BRK-B"],    # Precision Castparts -> Berkshire
    "DTV":  ["T"],        # DirecTV -> AT&T
    "SYMC": ["GEN"],      # Symantec -> Gen Digital
    "RAI":  ["BTI"],      # Reynolds -> BAT (ADR)
    "BRCM": ["AVGO"],     # Broadcom -> Broadcom Ltd.
    "CVC":  ["CHTR"],     # Cablevision -> Charter
    "GAS":  ["SO"],       # AGL Resources -> Southern
    "LLTC": ["ADI"],      # Linear Tech -> Analog Devices
    "ARG":  ["APD"],      # Airgas -> Air Products
    "SIAL": ["DHR"],      # Sigma-Aldrich -> Danaher
    "NYX":  ["ICE"],      # NYSE Euronext -> ICE
    "FDO":  ["DLTR"],     # Family Dollar -> Dollar Tree
    "JOY":  ["CAT"],      # approx: Joy Global -> Komatsu, mapped to CAT
    "CVH":  ["AET"],      # Coventry Health -> Aetna
    "PGN":  ["DUK"],      # Progress Energy -> Duke
    "APC":  ["OXY"],      # Anadarko -> Occidental
    "PXD":  ["XOM"],      # Pioneer -> Exxon
    "TWTR": ["X"],        # Twitter -> X
    "FB":   ["META"],     # FB -> META
    "MXIM": ["ADI"],      # Maxim -> Analog Devices
    "ALTR": ["INTC"],     # Altera -> Intel
    "BCR":  ["BDX"],      # C.R. Bard -> Becton Dickinson
    "CFN":  ["BDX"],      # CareFusion -> BDX
    "GGP":  ["BAM"],      # approx: GGP -> Brookfield
    "WLTW": ["WTW"],      # Willis Towers Watson
    "HCBK": ["MTB"],      # Hudson City -> M&T
    "PBCT": ["MTB"],      # People’s United -> M&T
    "WCG":  ["CNC"],      # WellCare -> Centene
    "MON":  ["BAYRY"],    # Monsanto -> Bayer ADR
    "DISCK":["WBD"],      # Discovery -> Warner Bros. Discovery
    "DISCA":["WBD"],
    "VIAB": ["PARA"],     # Viacom -> Paramount
    "CBS":  ["PARA"],
    "PARA": ["PARA"],     # keep as itself (Yahoo may still fail)
    "HRS":  ["LHX"],      # Harris -> L3Harris
    "RTN":  ["RTX"],      # Raytheon -> RTX
    "UTX":  ["RTX"],      # United Tech -> RTX
    "ABMD": ["MDT"],      # Abiomed -> Medtronic
    "ALXN": ["AZN"],      # Alexion -> AstraZeneca ADR
    "DWDP": ["DOW"],      # DowDuPont -> Dow (approx)
    "XLNX": ["AMD"],      # Xilinx -> AMD
    "AGN":  ["ABBV"],     # Allergan -> AbbVie (approx)
    "TIF":  ["MC.PA"],    # Tiffany -> LVMH (Paris)
    "FRX":  ["AGN"],      # Forest -> Allergan
    "LO":   ["RAI"],
    "GMCR": ["KDP"],      # Keurig -> KDP

    # Explicit drops (old / OTC / bankrupt)
    "APOL": [],           # went private → drop
    "SPLS": [],           # private
    "JCP":  [],           # bankrupt
    "DF":   [],           # bankrupt
    "CHK":  [],           # old CHK
    "MNK":  [],           # distressed
    "DNR":  [],           # old Denbury
}

# -----------------------
# 1. SP500 PIT MEMBERSHIP
# -----------------------
def load_sp500_history(path_csv):
    """
    Expects CSV with columns: ['date', 'tickers']
    where 'tickers' is a comma-separated list of symbols.
    """
    sp = pd.read_csv(path_csv)
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp.sort_values("date").reset_index(drop=True)
    return sp

def normalize_sp500_with_successors(sp_df, mapping):
    """
    Replace delisted tickers with successor tickers (first candidate in list).
    Deduplicate tickers per date.
    """
    sp = sp_df.copy()

    def map_row(ticker_str):
        tickers = [t.strip() for t in ticker_str.split(",") if t.strip()]
        mapped = []
        for t in tickers:
            if t in mapping and mapping[t]:
                mapped.append(mapping[t][0])  # take first candidate
            elif t in mapping and not mapping[t]:
                continue  # explicit drop
            else:
                mapped.append(t)
        mapped = sorted(set(mapped))
        return ",".join(mapped)

    sp["tickers"] = sp["tickers"].apply(map_row)
    return sp

def get_sp500_members_on(date, sp_df):
    """
    Point-in-time S&P members on a given date.
    """
    rows = sp_df[sp_df["date"] <= date]
    if rows.empty:
        return []
    last_row = rows.iloc[-1]
    tickers = [t.strip() for t in last_row["tickers"].split(",") if t.strip()]
    return tickers

sp500_df_raw = load_sp500_history(SP500_HISTORY_CSV)
sp500_df = normalize_sp500_with_successors(sp500_df_raw, SUCCESSOR_MAP)

# Save normalized PIT file for reproducibility
sp500_df.to_csv(os.path.join(OUT_DATA_DIR, "sp500_history_normalized.csv"), index=False)

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

# -----------------------
# 2. PRICE DATA
# -----------------------
def download_prices_yf(tickers, start_date, end_date):
    print(f"Downloading prices for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    if isinstance(data, pd.Series):  # single ticker case
        data = data.to_frame()
    data = data.sort_index()
    print("Raw price shape:", data.shape)
    return data

universe_all = build_universe_from_sp500(
    sp500_df,
    PRICE_START,
    BACKTEST_END,
    extra_tickers=ETF_TICKERS
)

prices_raw = download_prices_yf(universe_all, PRICE_START, BACKTEST_END)

# Optional tiny formatting fix for tickers like BRK.B -> BRK-B if Yahoo
# only supports the dash version and the dot version is all-NaN.
def fix_ticker_formatting(df):
    fixed = {}
    for t in df.columns:
        if df[t].isna().all() and "." in t:
            alt = t.replace(".", "-")
            try:
                test = yf.download(alt, start=PRICE_START, end=BACKTEST_END)["Close"]
                if not test.isna().all():
                    fixed[t] = alt
            except Exception:
                pass
    return fixed

fmt_map = fix_ticker_formatting(prices_raw)
for old, new in fmt_map.items():
    print(f"[FORMAT FIX] {old} → {new}")
    prices_raw[new] = prices_raw[old]
    del prices_raw[old]

prices = prices_raw.dropna(axis=1, how="all")
print("Final price DataFrame shape:", prices.shape)
prices.to_pickle(os.path.join(OUT_DATA_DIR, "prices.pkl"))

# -----------------------
# 3. FEATURES (CROSS-SECTIONAL TARGET)
# -----------------------
def engineer_features(prices, horizon=21):
    """
    Build features & a cross-sectional target:
    - Features: log_ret_1d, log_ret_5d, ma_10d_dev, vol_10d, mom_20d
    - Raw target: future horizon log-return
    - Final target: cross-sectional z-score per date
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

        # Forward horizon log-return
        df_t["target_raw"] = s_log.shift(-horizon).rolling(horizon).sum()

        records.append(df_t)

    feats = pd.concat(records)
    feats.index.name = "date"
    feats = feats.dropna(subset=["target_raw", "log_ret_1d",
                                 "log_ret_5d", "ma_10d_dev",
                                 "vol_10d", "mom_20d"])

    # Cross-sectional z-score per date
    def cs_z(x):
        mu = x.mean()
        sd = x.std(ddof=0)
        return (x - mu) / (sd + 1e-8)

    feats["target"] = feats.groupby("date")["target_raw"].transform(cs_z)
    feats = feats.drop(columns=["target_raw"])
    feats = feats.dropna(subset=["target"])

    print("Feature matrix shape:", feats.shape)
    return feats

features_all = engineer_features(prices, horizon=21)
features_all.to_pickle(os.path.join(OUT_DATA_DIR, "features_all.pkl"))

# Train / backtest split by date
features_train = features_all[features_all.index <= TRAIN_END_DATE]
features_test  = features_all[features_all.index > TRAIN_END_DATE]

X_train = features_train.drop(columns=["target"])
y_train = features_train["target"]

# -----------------------
# 4. ML MODELS (RIDGE, XGB, LGBM) + SELECTION
# -----------------------
def build_ml_models():
    """
    Build three ML models (Ridge, XGBoost, LightGBM) with a shared preprocessor.
    """
    categorical = ["ticker"]
    numeric = ["log_ret_1d", "log_ret_5d", "ma_10d_dev", "vol_10d", "mom_20d"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    # 1) Ridge baseline
    ridge = Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])

    # 2) XGBoost
    xgb = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED
        ))
    ])

    # 3) LightGBM
    lgbm = Pipeline([
        ("prep", preprocessor),
        ("model", LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=RANDOM_SEED
        ))
    ])

    models = {
        "ridge": ridge,
        "xgb": xgb,
        "lgbm": lgbm
    }

    return models

def train_and_select_model(X_train, y_train, features_all, train_end_date, val_frac=0.2):
    """
    Train all models and choose the best based on validation MSE.
    Uses a chronological validation split within the training period.
    """
    feats_train = features_all[features_all.index <= train_end_date]
    n = len(feats_train)
    cutoff = int((1 - val_frac) * n)
    feats_train_sub = feats_train.iloc[:cutoff]
    feats_val_sub   = feats_train.iloc[cutoff:]

    X_train_sub = feats_train_sub.drop(columns=["target"])
    y_train_sub = feats_train_sub["target"]
    X_val_sub   = feats_val_sub.drop(columns=["target"])
    y_val_sub   = feats_val_sub["target"]

    print(f"Train subset: {len(X_train_sub):,} rows, Val subset: {len(X_val_sub):,} rows")

    models = build_ml_models()
    val_mse = {}

    for name, model in models.items():
        print(f"\nFitting {name} model...")
        model.fit(X_train_sub, y_train_sub)
        preds = model.predict(X_val_sub)
        mse = mean_squared_error(y_val_sub, preds)
        val_mse[name] = mse
        print(f"{name} validation MSE: {mse:.6f}")

    best_name = min(val_mse, key=val_mse.get)
    best_model = models[best_name]
    print("\nSelected best model:", best_name, "with MSE =", val_mse[best_name])

    print(f"Refitting best model '{best_name}' on full training data...")
    best_model.fit(X_train, y_train)

    # Save models and metrics
    joblib.dump(best_model, os.path.join(OUT_MODELS_DIR, "best_ml_model.joblib"))
    joblib.dump(models,      os.path.join(OUT_MODELS_DIR, "all_ml_models.joblib"))
    pd.Series(val_mse).to_csv(os.path.join(OUT_RESULTS_DIR, "ml_model_validation_mse.csv"))

    return best_model, val_mse

print("Fitting ML models and selecting best...")
best_ml_model, val_mse = train_and_select_model(
    X_train, y_train, features_all, TRAIN_END_DATE
)
print("Done. Best model saved.")

# Reload from disk to confirm reproducibility
best_ml_model = joblib.load(os.path.join(OUT_MODELS_DIR, "best_ml_model.joblib"))
print("Loaded best_ml_model from disk.")

# -----------------------
# 5. RISK MODEL UTILS
# -----------------------
def make_psd(matrix, eps=1e-6):
    """
    Force covariance matrix to be symmetric PSD by eigenvalue clipping.
    """
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals_clipped = np.clip(vals, eps, None)
    psd = vecs @ np.diag(vals_clipped) @ vecs.T
    return psd

def compute_mu_cov_for_universe(prices, active_tickers, reb_date, lookback_days=252):
    """
    Use Ledoit-Wolf shrinkage + PSD repair for robust μ, Σ.
    """
    px = prices[active_tickers].loc[:reb_date]
    px_window = px.tail(lookback_days)

    log_ret = np.log(px_window / px_window.shift(1)).dropna()
    log_ret = log_ret.dropna(axis=1, how="any")

    if log_ret.shape[0] < 10 or log_ret.shape[1] < 5:
        return None, None, []

    mu = log_ret.mean().values

    lw = LedoitWolf().fit(log_ret.values)
    cov = lw.covariance_
    cov_psd = make_psd(cov)

    tickers_final = log_ret.columns.tolist()
    return mu, cov_psd, tickers_final

def mean_variance_opt_full(mu, cov, max_weight=0.05, risk_aversion=1.0):
    """
    Solve: max_w  μᵀw - λ wᵀΣw
    s.t.   sum w = 1,  0 <= w <= max_weight
    """
    mu = np.array(mu).ravel()
    cov = np.array(cov)

    n = len(mu)
    w = cp.Variable(n)

    cov_psd = make_psd(cov)
    cov_psd = cp.psd_wrap(cov_psd)   # critical to avoid ARPACK failures

    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cov_psd))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception as e:
        print("Optimizer error:", e)
        return np.ones(n) / n

    if w.value is None:
        return np.ones(n) / n

    w_val = np.array(w.value).ravel()
    w_val = np.clip(w_val, 0, None)
    if w_val.sum() == 0:
        w_val = np.ones(n) / n
    else:
        w_val /= w_val.sum()
    return w_val

def monte_carlo_sim(mu, cov, weights, horizon=21, n_sims=10000):
    """
    Simulate forward portfolio returns with a multivariate normal.
    Returns array of final cumulative returns (length n_sims).
    """
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    weights = np.asarray(weights)

    try:
        sims = np.random.multivariate_normal(mu, cov, size=(n_sims, horizon))
    except np.linalg.LinAlgError:
        diag_cov = np.diag(np.diag(cov))
        sims = np.random.multivariate_normal(mu, diag_cov, size=(n_sims, horizon))

    port_daily = sims @ weights
    port_prices = np.exp(np.cumsum(port_daily, axis=1))
    final_returns = port_prices[:, -1] - 1
    return final_returns

# -----------------------
# 5b. RETURN & MC VISUALIZATIONS
# -----------------------
def plot_return_distribution(prices, out_dir):
    log_ret = np.log(prices / prices.shift(1)).stack().dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(log_ret, bins=100, density=True)
    plt.title("Distribution of Daily Log Returns (All Assets)")
    plt.xlabel("Log Return")
    plt.tight_layout()
    fname = os.path.join(out_dir, "dist_daily_log_returns.pdf")
    plt.savefig(fname)
    plt.show()
    print("Saved daily log-return distribution to", fname)

def plot_mc_sample(mc_returns, out_dir, tag):
    plt.figure(figsize=(8, 5))
    plt.hist(mc_returns, bins=100, density=True)
    plt.title(f"Sample Monte Carlo Final Return Distribution ({tag})")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"sample_mc_distribution_{tag}.pdf")
    plt.savefig(fname)
    plt.show()
    print("Saved sample MC distribution to", fname)

def plot_mc_paths(mu, cov, w, out_dir, tag, n_paths=25, horizon=63):
    try:
        sims = np.random.multivariate_normal(mu, cov, size=(n_paths, horizon))
    except np.linalg.LinAlgError:
        diag_cov = np.diag(np.diag(cov))
        sims = np.random.multivariate_normal(mu, diag_cov, size=(n_paths, horizon))

    port = np.exp(np.cumsum(sims @ w, axis=1))
    plt.figure(figsize=(10, 6))
    plt.plot(port.T, alpha=0.5)
    plt.title(f"Monte Carlo Price Paths (Sample, {tag})")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"mc_paths_{tag}.pdf")
    plt.savefig(fname)
    plt.show()
    print("Saved sample MC paths to", fname)

# global return distribution (once)
plot_return_distribution(prices, OUT_RESULTS_DIR)

# -----------------------
# 6. WEIGHT SNAPSHOT UTILS (FOR VISUALIZATION)
# -----------------------
def compress_weights_to_rebalances(weights_hist, tol=1e-10):
    """
    Given weights_hist: list of (date, w_series) for each trading day,
    return a DataFrame of weights at rebalance points by detecting
    changes in the weight vector. Uses a global union of tickers to
    avoid shape mismatches.
    """
    if not weights_hist:
        return pd.DataFrame()

    all_tickers = sorted({t for _, w in weights_hist for t in w.index})
    snaps = []
    prev_vec = None

    for d, w_ser in weights_hist:
        w_full = w_ser.reindex(all_tickers).fillna(0)
        vec = w_full.values

        if prev_vec is None or not np.allclose(prev_vec, vec, atol=tol):
            snaps.append((d, w_full))
            prev_vec = vec.copy()

    if not snaps:
        return pd.DataFrame()

    W = pd.DataFrame({d: w for d, w in snaps}).T
    W.index.name = "date"
    W = W.sort_index()
    return W

def plot_top_holdings(weights_hist, name, out_dir, top_n=15):
    """
    Plot top-N tickers (by average weight) across rebalance snapshots
    as a stacked area chart over time.
    """
    W = compress_weights_to_rebalances(weights_hist)
    if W.empty:
        print(f"[WARN] No weight snapshots for {name}")
        return

    avg_w = W.mean().sort_values(ascending=False)
    top_tickers = avg_w.head(top_n).index
    W_top = W[top_tickers]

    row_sums = W_top.sum(axis=1).replace(0, np.nan)
    W_norm = W_top.div(row_sums, axis=0)

    plt.figure(figsize=(12, 6))
    W_norm.plot.area(ax=plt.gca(), linewidth=0)
    plt.title(f"{name}: Top {top_n} Holdings Over Rebalance Dates (weight share)")
    plt.ylabel("Weight Share")
    plt.xlabel("Date")
    plt.legend(loc="upper left", fontsize=8, ncol=3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{name}_top{top_n}_holdings_area.pdf")
    plt.savefig(fname)
    plt.show()
    print(f"Saved holdings area chart to {fname}")

def plot_final_weights(weights_hist, name, out_dir, top_n=20):
    """
    Bar chart of final rebalance weights (top-N tickers).
    """
    W = compress_weights_to_rebalances(weights_hist)
    if W.empty:
        print(f"[WARN] No weight snapshots for {name}")
        return

    w_last = W.iloc[-1].sort_values(ascending=False)
    w_top = w_last.head(top_n)

    plt.figure(figsize=(10, 5))
    w_top.plot(kind="bar")
    plt.title(f"{name}: Final Rebalance Weights (Top {top_n})")
    plt.ylabel("Weight")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{name}_final_weights_top{top_n}.pdf")
    plt.savefig(fname)
    plt.show()
    print(f"Saved final weights bar chart to {fname}")

# -----------------------
# 7. BACKTESTERS
# -----------------------
def backtest_equal_weight(prices,
                          sp500_df,
                          start_date, end_date,
                          lookback_days=LOOKBACK_DAYS):

    prices_bt = prices.loc[start_date:end_date]
    dates = prices_bt.index

    capital = 1.0
    vals = []
    weights_hist = []
    mc_history = []

    i = 0
    first_mc_plotted = False
    print("Running equal weight backtest")
    while i < len(dates):
        reb_date = dates[i]

        sp_members = get_sp500_members_on(reb_date, sp500_df)
        universe = [
            t for t in sp_members
            if t in prices_bt.columns and prices_bt[t].notna().sum() > lookback_days
        ]

        mu_hist, cov, tickers_final = compute_mu_cov_for_universe(
            prices_bt, universe, reb_date, lookback_days=lookback_days
        )

        if cov is None or len(tickers_final) < 5:
            print(f"[WARN] Skipping {reb_date.date()} — insufficient μ/Σ")
            i += REBALANCE_DAYS
            continue

        n = len(tickers_final)
        w = np.ones(n) / n
        w_ser = pd.Series(w, index=tickers_final)

        mc_returns = monte_carlo_sim(
            mu_hist, cov, w,
            horizon=REBALANCE_DAYS,
            n_sims=10000
        )

        mc_history.append({
            "rebalance_date": reb_date,
            "mean": mc_returns.mean(),
            "std":  mc_returns.std(),
            "sharpe": mc_returns.mean()/mc_returns.std() if mc_returns.std()>0 else np.nan,
            "VaR_5":  np.percentile(mc_returns, 5),
            "CVaR_5": mc_returns[mc_returns <= np.percentile(mc_returns, 5)].mean()
        })

        # Plot MC distribution + paths only once (first successful rebalance)
        if not first_mc_plotted:
            plot_mc_sample(mc_returns, OUT_RESULTS_DIR, "eq")
            plot_mc_paths(mu_hist, cov, w, OUT_RESULTS_DIR, "eq",
                          n_paths=25, horizon=REBALANCE_DAYS)
            first_mc_plotted = True

        hold_dates = dates[i : min(i + REBALANCE_DAYS, len(dates))]
        px = prices_bt.loc[hold_dates, tickers_final]
        lr = np.log(px / px.shift(1)).dropna()

        port_lr = lr.values @ w_ser.values
        for d, r in zip(lr.index, port_lr):
            capital *= np.exp(r)
            vals.append((d, capital))
            weights_hist.append((d, w_ser))

        i += REBALANCE_DAYS

    port_val = pd.DataFrame(vals, columns=["date", "portfolio"]).set_index("date")
    mc_history = pd.DataFrame(mc_history).set_index("rebalance_date")
    return port_val, weights_hist, mc_history

def backtest_mv_hist(prices,
                     sp500_df,
                     start_date, end_date,
                     lookback_days=LOOKBACK_DAYS,
                     max_weight=MAX_WEIGHT,
                     risk_aversion=1.0):

    prices_bt = prices.loc[start_date:end_date]
    dates = prices_bt.index

    capital = 1.0
    vals = []
    weights_hist = []
    mc_history = []

    i = 0
    first_mc_plotted = False
    print("Running MV-Hist backtest")
    while i < len(dates):
        reb_date = dates[i]

        sp_members = get_sp500_members_on(reb_date, sp500_df)
        universe = [
            t for t in sp_members
            if t in prices_bt.columns and prices_bt[t].notna().sum() > lookback_days
        ]

        mu_hist, cov, tickers_final = compute_mu_cov_for_universe(
            prices_bt, universe, reb_date, lookback_days
        )

        if cov is None or len(tickers_final) < 5:
            print(f"[WARN] Skipping {reb_date.date()} — insufficient μ/Σ")
            i += REBALANCE_DAYS
            continue

        w = mean_variance_opt_full(mu_hist, cov,
                                   max_weight=max_weight,
                                   risk_aversion=risk_aversion)
        w_ser = pd.Series(w, index=tickers_final)

        mc_returns = monte_carlo_sim(mu_hist, cov, w,
                                     horizon=REBALANCE_DAYS,
                                     n_sims=10000)

        mc_history.append({
            "rebalance_date": reb_date,
            "mean": mc_returns.mean(),
            "std":  mc_returns.std(),
            "sharpe": mc_returns.mean()/mc_returns.std() if mc_returns.std()>0 else np.nan,
            "VaR_5":  np.percentile(mc_returns, 5),
            "CVaR_5": mc_returns[mc_returns <= np.percentile(mc_returns, 5)].mean()
        })

        if not first_mc_plotted:
            plot_mc_sample(mc_returns, OUT_RESULTS_DIR, "mv_hist")
            plot_mc_paths(mu_hist, cov, w, OUT_RESULTS_DIR, "mv_hist",
                          n_paths=25, horizon=REBALANCE_DAYS)
            first_mc_plotted = True

        hold_dates = dates[i : min(i + REBALANCE_DAYS, len(dates))]
        px = prices_bt.loc[hold_dates, tickers_final]
        lr = np.log(px / px.shift(1)).dropna()

        port_lr = lr.values @ w_ser.values
        for d, r in zip(lr.index, port_lr):
            capital *= np.exp(r)
            vals.append((d, capital))
            weights_hist.append((d, w_ser))

        i += REBALANCE_DAYS

    port_val = pd.DataFrame(vals, columns=["date", "portfolio"]).set_index("date")
    mc_history = pd.DataFrame(mc_history).set_index("rebalance_date")
    return port_val, weights_hist, mc_history

def backtest_mv_ml(prices,
                   sp500_df,
                   features_all,
                   ml_model,
                   start_date, end_date,
                   lookback_days=LOOKBACK_DAYS,
                   max_weight=MAX_WEIGHT,
                   alpha_ml=0.4,
                   risk_aversion=1.0):

    prices_bt = prices.loc[start_date:end_date]
    dates = prices_bt.index

    feats = features_all.copy()
    feats.index = pd.to_datetime(feats.index)
    feats = feats.sort_index()

    capital = 1.0
    vals = []
    weights_hist = []
    mc_summary = []

    i = 0
    first_mc_plotted = False
    print("Running MV-ML backtest")
    while i < len(dates):
        reb_date = dates[i]

        sp_members = get_sp500_members_on(reb_date, sp500_df)
        universe = sorted(set(sp_members) | set(ETF_TICKERS))
        universe = [
            t for t in universe
            if t in prices_bt.columns and prices_bt[t].notna().sum() > lookback_days
        ]

        mu_hist, cov, tickers_final = compute_mu_cov_for_universe(
            prices_bt, universe, reb_date, lookback_days=lookback_days
        )
        if cov is None or len(tickers_final) < 5:
            print(f"[WARN] Skipping {reb_date.date()} — insufficient μ/Σ")
            i += REBALANCE_DAYS
            continue

        rows = []
        used_tickers = []
        for t in tickers_final:
            df_t = feats[(feats["ticker"] == t) & (feats.index <= reb_date)]
            if len(df_t) == 0:
                continue
            df_t = df_t.sort_index()
            rows.append(df_t.iloc[-1])
            used_tickers.append(t)

        if len(used_tickers) < 5:
            print(f"[WARN] Skipping {reb_date.date()} — too few tickers with features")
            i += REBALANCE_DAYS
            continue

        X_reb = pd.DataFrame(rows)

        idx_map = {t: j for j, t in enumerate(tickers_final)}
        idx = [idx_map[t] for t in used_tickers]

        mu_hist_used = mu_hist[idx]
        cov_used     = cov[np.ix_(idx, idx)]

        if np.isnan(mu_hist_used).any() or np.isnan(cov_used).any():
            print(f"[WARN] Skipping {reb_date.date()} — NaNs in μ or Σ")
            i += REBALANCE_DAYS
            continue

        mu_ml_raw = ml_model.predict(X_reb)
        mu_ml_raw = np.array(mu_ml_raw)

        ml_mean = mu_ml_raw.mean()
        ml_std  = mu_ml_raw.std() if mu_ml_raw.std() > 0 else 1.0
        z_ml = (mu_ml_raw - ml_mean) / ml_std
        ranks = pd.Series(z_ml).rank(method="average") / len(z_ml)
        ml_score = ranks.values  # 0..1 increasing with predicted performance

        mu_hist_used = np.array(mu_hist_used)
        h_mean = mu_hist_used.mean()
        h_std  = mu_hist_used.std() if mu_hist_used.std() > 0 else 1.0
        z_hist = (mu_hist_used - h_mean) / h_std
        hist_score = (pd.Series(z_hist).rank(method="average") / len(z_hist)).values

        score_blend = alpha_ml * ml_score + (1 - alpha_ml) * hist_score

        # Restrict to top-K names by blended score (if desired)
        if TOP_K is not None and len(used_tickers) > TOP_K:
            order = np.argsort(-score_blend)  # descending
            top_idx = order[:TOP_K]
            used_tickers = [used_tickers[j] for j in top_idx]
            mu_hist_used = mu_hist_used[top_idx]
            cov_used     = cov_used[np.ix_(top_idx, top_idx)]
            score_blend  = score_blend[top_idx]

        # Convert blended scores to pseudo-returns on the scale of mu_hist_used
        pseudo_mu = score_blend.copy()
        pseudo_mu = pseudo_mu - pseudo_mu.mean()
        pseudo_mu = pseudo_mu / (pseudo_mu.std() + 1e-8)
        pseudo_mu = pseudo_mu * (mu_hist_used.std() + 1e-8) + mu_hist_used.mean()

        w = mean_variance_opt_full(pseudo_mu, cov_used,
                                   max_weight=max_weight,
                                   risk_aversion=risk_aversion)
        w_ser = pd.Series(w, index=used_tickers)

        mc_returns = monte_carlo_sim(
            pseudo_mu,
            cov_used,
            w,
            horizon=REBALANCE_DAYS,
            n_sims=10000
        )
        mc_summary.append({
            "rebalance_date": reb_date,
            "mean": mc_returns.mean(),
            "std": mc_returns.std(),
            "sharpe": mc_returns.mean() / mc_returns.std() if mc_returns.std() > 0 else np.nan,
            "VaR_5": np.percentile(mc_returns, 5),
            "CVaR_5": mc_returns[mc_returns <= np.percentile(mc_returns, 5)].mean()
        })

        if not first_mc_plotted:
            plot_mc_sample(mc_returns, OUT_RESULTS_DIR, "mv_ml")
            plot_mc_paths(pseudo_mu, cov_used, w, OUT_RESULTS_DIR, "mv_ml",
                          n_paths=25, horizon=REBALANCE_DAYS)
            first_mc_plotted = True

        hold_dates = dates[i : min(i + REBALANCE_DAYS, len(dates))]
        px = prices_bt.loc[hold_dates, used_tickers]
        lr = np.log(px / px.shift(1)).dropna()

        port_lr = lr.values @ w_ser.values
        for d, r in zip(lr.index, port_lr):
            capital *= np.exp(r)
            vals.append((d, capital))
            weights_hist.append((d, w_ser))

        i += REBALANCE_DAYS

    port_val = pd.DataFrame(vals, columns=["date", "portfolio"]).set_index("date")
    mc_summary = pd.DataFrame(mc_summary).set_index("rebalance_date")

    return port_val, weights_hist, mc_summary

# -----------------------
# 8. RUN BACKTESTS
# -----------------------
port_eq,   w_eq,   mc_eq   = backtest_equal_weight(
    prices, sp500_df,
    start_date=BACKTEST_START,
    end_date=BACKTEST_END
)

port_hist, w_hist, mc_hist = backtest_mv_hist(
    prices, sp500_df,
    start_date=BACKTEST_START,
    end_date=BACKTEST_END
)

port_ml,   w_ml,   mc_ml   = backtest_mv_ml(
    prices, sp500_df,
    features_all,
    best_ml_model,
    start_date=BACKTEST_START,
    end_date=BACKTEST_END
)

# -----------------------
# 9. STATS + SAVING OUTPUTS
# -----------------------
port_eq.to_csv(os.path.join(OUT_RESULTS_DIR, "port_eq.csv"))
port_hist.to_csv(os.path.join(OUT_RESULTS_DIR, "port_mv_hist.csv"))
port_ml.to_csv(os.path.join(OUT_RESULTS_DIR, "port_mv_ml.csv"))

mc_eq.to_csv(os.path.join(OUT_RESULTS_DIR, "mc_eq_summary.csv"))
mc_hist.to_csv(os.path.join(OUT_RESULTS_DIR, "mc_mv_hist_summary.csv"))
mc_ml.to_csv(os.path.join(OUT_RESULTS_DIR, "mc_mv_ml_summary.csv"))

with open(os.path.join(OUT_RESULTS_DIR, "weights_eq.pkl"), "wb") as f:
    pickle.dump(w_eq, f)
with open(os.path.join(OUT_RESULTS_DIR, "weights_mv_hist.pkl"), "wb") as f:
    pickle.dump(w_hist, f)
with open(os.path.join(OUT_RESULTS_DIR, "weights_mv_ml.pkl"), "wb") as f:
    pickle.dump(w_ml, f)

def compute_stats(port_val, freq=252):
    v = port_val["portfolio"]
    rets = np.log(v / v.shift(1)).dropna()
    total_ret = v.iloc[-1] - 1
    ann_ret = rets.mean() * freq
    ann_vol = rets.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cummax = v.cummax()
    dd = (v - cummax) / cummax
    mdd = dd.min()
    return {
        "Total Return": float(total_ret),
        "Ann. Return": float(ann_ret),
        "Ann. Vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(mdd),
    }

stats_eq   = compute_stats(port_eq)
stats_hist = compute_stats(port_hist)
stats_ml   = compute_stats(port_ml)

print("Equal Weight:", stats_eq)
print("MV-Hist    :", stats_hist)
print("MV-ML      :", stats_ml)

df_plot = pd.concat([
    port_eq["portfolio"].rename("Equal Weight"),
    port_hist["portfolio"].rename("MV-Hist"),
    port_ml["portfolio"].rename("MV-ML"),
], axis=1)

plt.figure(figsize=(12, 6))
df_plot.plot(ax=plt.gca(), title="Portfolio Value Over Time")
plt.ylabel("Portfolio Value")
plt.tight_layout()
eq_curve_path = os.path.join(OUT_RESULTS_DIR, "portfolio_equity_curves.pdf")
plt.savefig(eq_curve_path)
plt.show()
print("Saved equity curves to", eq_curve_path)

stats_df = pd.DataFrame({
    "Equal_Weight": stats_eq,
    "MV_Hist": stats_hist,
    "MV_ML": stats_ml
})
stats_df.to_csv(os.path.join(OUT_RESULTS_DIR, "strategy_stats.csv"))
print("\nSaved stats to:", os.path.join(OUT_RESULTS_DIR, "strategy_stats.csv"))

# -----------------------
# 10. HOLDINGS VISUALIZATIONS
# -----------------------
plot_top_holdings(w_eq,   "Equal_Weight", OUT_RESULTS_DIR, top_n=15)
plot_final_weights(w_eq,  "Equal_Weight", OUT_RESULTS_DIR, top_n=20)

plot_top_holdings(w_hist, "MV_Hist", OUT_RESULTS_DIR, top_n=15)
plot_final_weights(w_hist,"MV_Hist", OUT_RESULTS_DIR, top_n=20)

plot_top_holdings(w_ml,   "MV_ML", OUT_RESULTS_DIR, top_n=15)
plot_final_weights(w_ml,  "MV_ML", OUT_RESULTS_DIR, top_n=20)

print("\nAll outputs saved under:", OUT_RESULTS_DIR)