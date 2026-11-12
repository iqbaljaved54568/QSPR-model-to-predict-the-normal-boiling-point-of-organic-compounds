"""
Optuna tuning and five models on the 35-descriptor set (Table 5, Table S6)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Boiling Point Prediction — FINAL MODEL PIPELINE (re-tuned per descriptor set)
# Linear · Random Forest · XGBoost · LightGBM · ANN (tanh)
# Optuna auto-tuning ON (5-fold CV) · ANN tuned separately for raw & log target
# Target-distribution diagnostics · Homoscedasticity check · Training-stability
# diagnostics · Log Transform · AD · SHAP · Feature Importance
# =============================================================================

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import gaussian_kde, probplot, pearsonr
from scipy.ndimage import uniform_filter1d
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# ── Descriptor-set registry ──────────────────────────────────────────────────
FILES = {
    "94desc": {
        "train": "./BP_FINAL_MODEL/descriptors/train_test_split/train_94desc.csv",
        "test":  "./BP_FINAL_MODEL/descriptors/train_test_split/test_94desc.csv",
    },
    "35desc": {
        "train": "./BP_FINAL_MODEL/descriptors/train_test_split/train_35desc.csv",
        "test":  "./BP_FINAL_MODEL/descriptors/train_test_split/test_35desc.csv",
    },
}


ACTIVE_SET = "35desc"   # <-- change to "94desc" to run the full descriptor set

TRAIN_FILE = FILES[ACTIVE_SET]["train"]
TEST_FILE  = FILES[ACTIVE_SET]["test"]


TARGET_COL_INDEX       = 5
DESCRIPTOR_START_INDEX = 6

OLD_TRAIN_FILE = None

OLD_TARGET_COL_INDEX = 5

N_SPLITS     = 10
RANDOM_STATE = 42
SIGMA_LIMIT  = 3.0
TOP_N_FEAT   = 20

# ── Auto-tuning settings ─────────────────────────────────────────────────────

AUTO_TUNE        = True
OPTUNA_TRIALS    = 25


ANN_OPTUNA_TRIALS = 12

OPTUNA_CV_SPLITS = 5

# ── ANN settings ─────────────────────────────────────────────────────────────
ANN_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ANN_EPOCHS   = 300        # matches Pass-2 'epochs'
ANN_PATIENCE = 30         # matches Pass-2 'patience'

# ── Output directory (per descriptor set) ────────────────────────────────────
OUT_DIR_ROOT = "./BP_FINAL_MODEL/figures/"
OUT_DIR = f"{OUT_DIR_ROOT}{ACTIVE_SET}/"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"✅  Descriptor set        : {ACTIVE_SET}")
print(f"✅  Output directory ready : {OUT_DIR}")
print(f"✅  ANN device             : {ANN_DEVICE}")

# =============================================================================
# 2. FIXED HYPERPARAMETER BLOCKS
# =============================================================================

LINEAR_PARAMS = {
    "alpha": 1.0, "fit_intercept": True, "max_iter": 10000,
}

RF_PARAMS = {
    "n_estimators": 531, "max_depth": 23, "min_samples_split": 2,
    "min_samples_leaf": 2, "max_features": 0.6356199683368842,
    "bootstrap": True, "random_state": 42, "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 1700, "max_depth": 5, "learning_rate": 0.03931653933364233,
    "subsample": 0.9895401816056413, "colsample_bytree": 0.525615149200575,
    "reg_alpha": 0.12041669654452902, "reg_lambda": 0.020658414439233216,
    "min_child_weight": 21, "gamma": 0.10250989672005564,
    "random_state": 42, "n_jobs": -1, "verbosity": 0, "tree_method": "hist",
}

LGBM_PARAMS = {
    "n_estimators": 3003, "learning_rate": 0.04896585817649962,
    "num_leaves": 112, "max_depth": 6, "min_child_samples": 13,
    "subsample": 0.7123282563580081, "colsample_bytree": 0.7388988317925173,
    "reg_alpha": 0.00020542253454559892, "reg_lambda": 1.3151301555825066,
    "random_state": 42, "n_jobs": -1, "verbosity": -1,
}

ANN_PARAMS = {
    "hidden_layers": [64, 448, 192, 480],
    "activation":    "tanh",
    "dropout":       0.1494612957966698,
    "learning_rate": 0.006035415854625059,
    "batch_size":    64,
    "weight_decay":  1.1046379977835825e-05,
    "batch_norm":    False,
    "grad_clip":     1.0,
    "epochs":        ANN_EPOCHS,
    "patience":      ANN_PATIENCE,
}

# =============================================================================
# 3. GLOBAL STYLE
# =============================================================================

MODEL_COLORS = {
    "Linear":       "#4A90D9",
    "RandomForest": "#27AE60",
    "XGBoost":      "#E67E22",
    "LightGBM":     "#C0392B",
    "ANN":          "#8E44AD",
}
MODEL_MARKERS = {
    "Linear": "o", "RandomForest": "s",
    "XGBoost": "^", "LightGBM": "D", "ANN": "P",
}
PALETTE = {
    "light_bg": "#F7F9FB", "grid":      "#DDE3EA",
    "text":     "#1A2A3A", "text_light":"#5A6A7A",
    "perfect":  "#2C3E50", "ad_fill":   "#EBF5FB",
    "train_in": "#2C6E8A", "test_in":   "#27AE60",
    "train_out":"#F39C12", "test_out":  "#C0392B",
}
TYPE_COLORS = {
    "A — High Leverage": "#E67E22",
    "B — High Residual": "#C0392B",
    "C — Both":          "#8E44AD",
    "D — Within AD":     "#27AE60",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 8,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B0BEC5", "axes.linewidth": 0.8,
    "axes.facecolor": PALETTE["light_bg"], "figure.facecolor": "white",
    "grid.color": PALETTE["grid"], "grid.linewidth": 0.55, "grid.linestyle": "--",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.framealpha": 0.92, "legend.edgecolor": "#C5CDD5",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✅  Saved → {path}")

# =============================================================================
# 4. LOAD DATA
# =============================================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_train = train_df.iloc[:, TARGET_COL_INDEX].values
X_test  = test_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_test  = test_df.iloc[:, TARGET_COL_INDEX].values
feature_names = list(train_df.columns[DESCRIPTOR_START_INDEX:])
n_train, p = X_train.shape

print(f"\n{'='*64}")
print(f"  Boiling Point Prediction — FINAL Pipeline | Set: {ACTIVE_SET}")
print(f"{'='*64}")
print(f"  Training : {n_train:,}  |  Test : {X_test.shape[0]:,}  |  Descriptors : {p}")
print(f"  BP range : {y_train.min():.1f} – {y_train.max():.1f} K")
print(f"  Auto-tune: {'YES (' + str(OPTUNA_TRIALS) + ' trials/model, ANN tuned separately raw/log)' if AUTO_TUNE else 'NO — using locked fallback parameters'}")

# =============================================================================
# 4b. TARGET DISTRIBUTION DIAGNOSTICS
# =============================================================================

from scipy.stats import skew, kurtosis

def distribution_report(y, label):
    y = np.asarray(y, dtype=float)
    y_log = np.log(y)
    rows = [
        {"Transform": "Raw (K)",  "Skewness": skew(y),     "Kurtosis": kurtosis(y),
         "Min": y.min(), "Max": y.max(), "Range": y.max()-y.min(), "Mean": y.mean(), "Std": y.std()},
        {"Transform": "Log(K)",   "Skewness": skew(y_log), "Kurtosis": kurtosis(y_log),
         "Min": y_log.min(), "Max": y_log.max(), "Range": y_log.max()-y_log.min(),
         "Mean": y_log.mean(), "Std": y_log.std()},
    ]
    df = pd.DataFrame(rows).set_index("Transform")
    print(f"\n  {'─'*60}")
    print(f"  Target distribution — {label}")
    print(f"  {'─'*60}")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))
    return df

print(f"\n{'='*64}")
print(f"  TARGET DISTRIBUTION DIAGNOSTICS")
print(f"{'='*64}")
new_dist = distribution_report(y_train, f"NEW dataset ({ACTIVE_SET}, n={n_train})")

old_dist = None
if OLD_TRAIN_FILE is not None:
    old_df   = pd.read_csv(OLD_TRAIN_FILE)
    y_old    = old_df.iloc[:, OLD_TARGET_COL_INDEX].values
    old_dist = distribution_report(y_old, f"OLD dataset (n={len(y_old)})")

    delta_skew_raw = new_dist.loc["Raw (K)", "Skewness"] - old_dist.loc["Raw (K)", "Skewness"]
    delta_skew_log = new_dist.loc["Log(K)", "Skewness"] - old_dist.loc["Log(K)", "Skewness"]
    print(f"\n  Δ Skewness (new − old), raw  : {delta_skew_raw:+.4f}")
    print(f"  Δ Skewness (new − old), log  : {delta_skew_log:+.4f}")
    if abs(new_dist.loc['Raw (K)','Skewness']) < abs(old_dist.loc['Raw (K)','Skewness']) * 0.7:
        print("  ⚠  Raw-scale skewness dropped substantially in the new dataset.")
        print("     This alone can explain why log-transform stopped helping: the")
        print("     variance-stabilization problem the log was solving may no longer exist.")
else:
    print("\n  (OLD_TRAIN_FILE not set — skipping old-vs-new comparison. Set it in the")
    print("   CONFIGURATION section to compare against the previous descriptor set.)")

# Visual: histogram + Q-Q plot, raw vs log, side by side
def plot_target_distribution(y, save_path=None, old_y=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Target Distribution Diagnostics — {ACTIVE_SET}",
                 fontsize=13, fontweight="bold", y=1.01)
    y_log = np.log(y)

    for ax, data, title, color in [
        (axes[0,0], y,     f"Raw BP (K)  |  skew={skew(y):.3f}  kurt={kurtosis(y):.3f}",  "#4A90D9"),
        (axes[0,1], y_log, f"Log BP  |  skew={skew(y_log):.3f}  kurt={kurtosis(y_log):.3f}", "#C0392B"),
    ]:
        ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor="white", density=True)
        if old_y is not None:
            old_data = np.log(old_y) if title.startswith("Log") else old_y
            ax.hist(old_data, bins=40, color="gray", alpha=0.35, density=True, label="Old dataset")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10, fontstyle="italic")
        ax.set_ylabel("Density", fontsize=9.5)
        ax.grid(True); ax.set_axisbelow(True)

    for ax, data, title in [
        (axes[1,0], y,     "Q-Q Plot — Raw BP"),
        (axes[1,1], y_log, "Q-Q Plot — Log BP"),
    ]:
        probplot(data, dist="norm", plot=ax)
        ax.set_title(title, fontsize=10, fontstyle="italic")
        ax.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_target_distribution(y_train, save_path=f"{OUT_DIR}fig_target_distribution_diagnostics.png",
                          old_y=(y_old if OLD_TRAIN_FILE is not None else None))

# ── Publication-ready version: clean before/after KDE panel ─────────────────
def plot_log_transform_effect(y, save_path=None):
    y = np.asarray(y, dtype=float)
    y_log = np.log(y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Effect of Log-Transformation on Boiling Point Distribution ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.03)

    panels = [
        (axes[0], y,     "Before — Raw Boiling Point (K)", "BP (K)",   "#4A90D9"),
        (axes[1], y_log, "After — log(Boiling Point)",      "log(BP)",  "#C0392B"),
    ]
    for ax, data, title, xlabel, color in panels:
        n_bins = min(40, max(15, int(np.sqrt(len(data)))))
        ax.hist(data, bins=n_bins, color=color, alpha=0.55, edgecolor="white",
                 linewidth=0.6, density=True, label="Histogram")

        kde_x = np.linspace(data.min(), data.max(), 400)
        kde_y = gaussian_kde(data)(kde_x)
        ax.plot(kde_x, kde_y, color=color, linewidth=2.2, label="KDE")
        ax.fill_between(kde_x, kde_y, alpha=0.10, color=color)

        mean_v, median_v = data.mean(), np.median(data)
        ax.axvline(mean_v,   color="#1A2A3A", linewidth=1.4, linestyle="--", label=f"Mean = {mean_v:.2f}")
        ax.axvline(median_v, color="#1A2A3A", linewidth=1.4, linestyle=":",  label=f"Median = {median_v:.2f}")

        sk, ku = skew(data), kurtosis(data)
        ax.text(0.03, 0.97,
                f"Skewness = {sk:+.3f}\nKurtosis  = {ku:+.3f}\nn = {len(data):,}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5", alpha=0.95))

        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("Density", fontsize=10.5)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_log_transform_effect(y_train, save_path=f"{OUT_DIR}fig_log_transform_effect.png")

# =============================================================================
# 5. SCALING
# =============================================================================

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# 6. ANN ARCHITECTURE
# =============================================================================

ACT_MAP = {
    "tanh": nn.Tanh, "relu": nn.ReLU,
    "elu":  nn.ELU,  "selu": nn.SELU,
}

def build_ann(input_dim, hidden_layers, activation="tanh", dropout=0.2, batch_norm=False):
    act_fn = ACT_MAP.get(activation, nn.Tanh)
    layers = []
    in_dim = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def train_ann(X_tr, y_tr, X_va, y_va, params, device="cpu", verbose=False):
    """Train ANN with early stopping. Returns trained model + training history."""
    model = build_ann(
        X_tr.shape[1],
        params["hidden_layers"],
        params["activation"],
        params["dropout"],
        batch_norm=params.get("batch_norm", False),
    ).to(device)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1).to(device)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=params["batch_size"], shuffle=True
    )
    optim  = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=10, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    grad_clip = params.get("grad_clip", None)

    best_val, best_state, patience_cnt = np.inf, None, 0
    train_hist, val_hist = [], []

    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va_t), y_va_t).item()
            tr_loss  = criterion(model(X_tr_t), y_tr_t).item()

        train_hist.append(tr_loss)
        val_hist.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= params["patience"]:
                if verbose:
                    print(f"    Early stop @ epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist


def ann_predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_t).squeeze().cpu().numpy()


def assess_training_stability(val_hist, tail_frac=0.2):
    """Flags whether the validation loss curve looks like a genuine converged
    fit vs. an unstable optimization (oscillation / divergence). This is what
    distinguishes 'the transform made the problem harder to fit' (optimization
    issue — fixable with LR/clipping/architecture) from 'the transform genuinely
    doesn't help' (a real fit ceiling)."""
    vh = np.asarray(val_hist, dtype=float)
    n_tail = max(3, int(len(vh) * tail_frac))
    tail = vh[-n_tail:]
    diffs = np.diff(tail)
    oscillation_frac = float(np.mean(diffs > 0)) if len(diffs) > 0 else 0.0
    tail_rel_std = float(np.std(tail) / (np.mean(tail) + 1e-12))
    diverging = bool(vh[-1] > vh.min() * 1.5)
    unstable = bool(oscillation_frac > 0.4 or diverging or tail_rel_std > 0.15)
    return {
        "oscillation_frac": oscillation_frac,
        "tail_rel_std": tail_rel_std,
        "diverging": diverging,
        "unstable": unstable,
        "best_val": float(vh.min()),
        "final_val": float(vh[-1]),
    }


def ann_cv(X_tr, y_tr, params, n_splits=5, rs=42, log_transform=False, device="cpu"):
    """K-fold CV for ANN — returns r2 array, mae/rmse lists, and out-of-fold
    predictions (raw K scale). The OOF predictions are what make the
    Applicability Domain residual scale (Section 17) correct: they're genuine
    generalization residuals rather than in-sample/training residuals."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    y_fit = np.log(y_tr) if log_transform else y_tr
    r2_list, mae_list, rmse_list = [], [], []
    preds_oof = np.zeros_like(y_tr, dtype=float)

    for tr_idx, va_idx in cv.split(X_tr):
        m, _, _ = train_ann(
            X_tr[tr_idx], y_fit[tr_idx],
            X_tr[va_idx], y_fit[va_idx],
            params, device=device
        )
        pred = ann_predict(m, X_tr[va_idx], device)
        if log_transform:
            pred = np.exp(pred)
        preds_oof[va_idx] = pred
        r2_list.append(r2_score(y_tr[va_idx], pred))
        mae_list.append(mean_absolute_error(y_tr[va_idx], pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_tr[va_idx], pred)))

    return np.array(r2_list), mae_list, rmse_list, preds_oof

# =============================================================================
# 7. OPTUNA AUTO-TUNING
# =============================================================================

TUNE_STORAGE_DIR = OUT_DIR
os.makedirs(TUNE_STORAGE_DIR, exist_ok=True)

def _study_storage(study_name):
    return f"sqlite:///{TUNE_STORAGE_DIR}{study_name}.db"


def tune_lgbm(X_tr, y_tr, log_transform, n_trials, cv_splits, rs):
    from lightgbm import early_stopping, log_evaluation
    y_fit = np.log(y_tr) if log_transform else y_tr
    cv    = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": rs, "n_jobs": -1, "verbosity": -1,
        }
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m = LGBMRegressor(**params)
            m.fit(X_tr[tr_idx], y_fit[tr_idx],
                  eval_set=[(X_tr[va_idx], y_fit[va_idx])],
                  callbacks=[early_stopping(stopping_rounds=50, verbose=False),
                             log_evaluation(period=0)])
            pred = m.predict(X_tr[va_idx])
            fold_scores.append(r2_score(y_fit[va_idx], pred))
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage=_study_storage(f"optuna_lgbm_{ACTIVE_SET}"),
        study_name=f"lgbm_{ACTIVE_SET}", load_if_exists=True,
    )
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [LightGBM] {len(study.trials)} trial(s) already completed in saved study, "
          f"running {n_remaining} more ...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    print(f"    [LightGBM] tuning wall time this call: {time.time()-t0:.1f}s")
    best = study.best_params
    best.update({"random_state": rs, "n_jobs": -1, "verbosity": -1})
    return best


def tune_xgb(X_tr, y_tr, log_transform, n_trials, cv_splits, rs):
    y_fit = np.log(y_tr) if log_transform else y_tr
    cv    = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": rs, "n_jobs": -1,
            "verbosity": 0, "tree_method": "hist",
            "early_stopping_rounds": 50,
        }
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m = XGBRegressor(**params)
            m.fit(X_tr[tr_idx], y_fit[tr_idx],
                  eval_set=[(X_tr[va_idx], y_fit[va_idx])], verbose=False)
            pred = m.predict(X_tr[va_idx])
            fold_scores.append(r2_score(y_fit[va_idx], pred))
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage=_study_storage(f"optuna_xgb_{ACTIVE_SET}"),
        study_name=f"xgb_{ACTIVE_SET}", load_if_exists=True,
    )
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [XGBoost] {len(study.trials)} trial(s) already completed in saved study, "
          f"running {n_remaining} more ...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    print(f"    [XGBoost] tuning wall time this call: {time.time()-t0:.1f}s")
    best = study.best_params
    best.pop("early_stopping_rounds", None)
    best.update({"random_state": rs, "n_jobs": -1, "verbosity": 0, "tree_method": "hist"})
    return best


def tune_rf(X_tr, y_tr, log_transform, n_trials, cv_splits, rs):
    y_fit = np.log(y_tr) if log_transform else y_tr
    cv    = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1200),
            "max_depth":         trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt","log2",0.5]),
            "random_state": rs, "n_jobs": -1, "bootstrap": True,
        }

        scores = cross_val_score(
            RandomForestRegressor(**params), X_tr, y_fit,
            cv=cv, scoring="r2", n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        storage=_study_storage(f"optuna_rf_{ACTIVE_SET}"),
        study_name=f"rf_{ACTIVE_SET}", load_if_exists=True,
    )
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [RandomForest] {len(study.trials)} trial(s) already completed in saved study, "
          f"running {n_remaining} more ...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    print(f"    [RandomForest] tuning wall time this call: {time.time()-t0:.1f}s")
    best = study.best_params
    best.update({"random_state": rs, "n_jobs": -1, "bootstrap": True})
    return best


def tune_ann(X_tr, y_tr, log_transform, n_trials, cv_splits, rs, device):

    tag = "log" if log_transform else "raw"

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 5)
        hidden   = [
            trial.suggest_int(f"h{i}", 32, 512, step=32)
            for i in range(n_layers)
        ]
        params = {
            "hidden_layers": hidden,
            "activation":    "tanh",   # fixed per your request
            "dropout":       trial.suggest_float("dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "weight_decay":  trial.suggest_float("wd", 1e-5, 1e-2, log=True),
            "batch_norm":    trial.suggest_categorical("batch_norm", [True, False]),
            "grad_clip":     trial.suggest_categorical("grad_clip", [None, 1.0, 5.0]),
            "epochs":        100,
            "patience":      15,
        }
        cv  = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
        r2s = []
        y_fit = np.log(y_tr) if log_transform else y_tr
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m, _, _ = train_ann(
                X_tr[tr_idx], y_fit[tr_idx],
                X_tr[va_idx], y_fit[va_idx],
                params, device=device
            )
            pred = ann_predict(m, X_tr[va_idx], device)
            if log_transform:
                pred = np.exp(pred)
            r2s.append(r2_score(y_tr[va_idx], pred))
            trial.report(np.mean(r2s), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(r2s)

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        storage=_study_storage(f"optuna_ann_{tag}_{ACTIVE_SET}"),
        study_name=f"ann_{tag}_{ACTIVE_SET}", load_if_exists=True,
    )
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [ANN-{tag}] {len(study.trials)} trial(s) already completed in saved study, "
          f"running {n_remaining} more ...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    print(f"    [ANN-{tag}] tuning wall time this call: {time.time()-t0:.1f}s")
    bp = study.best_params
    n_layers = bp["n_layers"]
    best_params = {
        "hidden_layers": [bp[f"h{i}"] for i in range(n_layers)],
        "activation":    "tanh",
        "dropout":       bp["dropout"],
        "learning_rate": bp["lr"],
        "batch_size":    bp["batch_size"],
        "weight_decay":  bp["wd"],
        "batch_norm":    bp["batch_norm"],
        "grad_clip":     bp["grad_clip"],
        "epochs":        ANN_EPOCHS,
        "patience":      ANN_PATIENCE,
    }
    return best_params

# =============================================================================
# 8. EVALUATION ENGINE
# =============================================================================

def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    bias = np.mean(y_pred - y_true)
    return dict(r2=r2, mae=mae, rmse=rmse, mape=mape, bias=bias)


def run_sklearn_model(name, model, params_dict, X_tr, y_tr, X_te, y_te,
                      log_transform=False, n_splits=10, rs=42):
    y_tr_fit = np.log(y_tr) if log_transform else y_tr
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=rs)

    cv_r2 = cross_val_score(model, X_tr, y_tr_fit, cv=cv, scoring="r2", n_jobs=1)


    cv_mae_list, cv_rmse_list = [], []
    cv_oof_pred = np.zeros_like(y_tr, dtype=float)
    for tr_idx, va_idx in cv.split(X_tr):
        _m = type(model)(**model.get_params())
        _m.fit(X_tr[tr_idx], y_tr_fit[tr_idx])
        pred = _m.predict(X_tr[va_idx])
        if log_transform:
            pred = np.exp(pred)
        cv_oof_pred[va_idx] = pred
        cv_mae_list.append(mean_absolute_error(y_tr[va_idx], pred))
        cv_rmse_list.append(np.sqrt(mean_squared_error(y_tr[va_idx], pred)))
    cv_rmse_orig = np.sqrt(mean_squared_error(y_tr, cv_oof_pred))

    model.fit(X_tr, y_tr_fit)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    if log_transform:
        y_pred_tr = np.exp(y_pred_tr)
        y_pred_te = np.exp(y_pred_te)

    tr_met = compute_metrics(y_tr, y_pred_tr)
    te_met = compute_metrics(y_te, y_pred_te)
    label  = f"{name} (log)" if log_transform else name

    print(f"  {'─'*52}")
    print(f"  {label}")
    print(f"  CV  R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  Test R²: {te_met['r2']:.4f}  MAE: {te_met['mae']:.3f} K  "
          f"RMSE: {te_met['rmse']:.3f} K  MAPE: {te_met['mape']:.2f}%")

    return {
        "name": label, "base_name": name, "log_transform": log_transform,
        "model": model, "params": params_dict,
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_r2_vals": cv_r2, "cv_mae": np.mean(cv_mae_list),
        "cv_rmse": np.mean(cv_rmse_list),
        "cv_rmse_orig": cv_rmse_orig,   # ★ correct AD residual scale
        "cv_oof_pred": cv_oof_pred,     # ★ correct AD train-side residuals
        "train_r2": tr_met["r2"], "test_r2": te_met["r2"],
        "test_mae": te_met["mae"], "test_rmse": te_met["rmse"],
        "test_mape": te_met["mape"], "test_bias": te_met["bias"],
        "s_train": np.sqrt(mean_squared_error(y_tr, y_pred_tr)),
        "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te,
        "residuals": y_te - y_pred_te, "errors": y_pred_te - y_te,
    }


def run_ann_model(X_tr, y_tr, X_te, y_te, params_dict,
                  log_transform=False, n_splits=10, rs=42, device="cpu"):
    print(f"  {'─'*52}")
    label = "ANN (log)" if log_transform else "ANN"
    print(f"  {label}  [tanh, layers={params_dict['hidden_layers']}]")

    cv_r2, cv_mae_list, cv_rmse_list, cv_oof_pred = ann_cv(
        X_tr, y_tr, params_dict,
        n_splits=n_splits, rs=rs, log_transform=log_transform, device=device
    )
    cv_rmse_orig = np.sqrt(mean_squared_error(y_tr, cv_oof_pred))

    # Final model on full training set; use 10% of train as validation for early stopping
    val_size  = max(1, int(0.1 * len(X_tr)))
    idx_perm  = np.random.RandomState(rs).permutation(len(X_tr))
    va_idx    = idx_perm[:val_size]
    tr_idx    = idx_perm[val_size:]

    y_fit = np.log(y_tr) if log_transform else y_tr
    final_model, train_hist, val_hist = train_ann(
        X_tr[tr_idx], y_fit[tr_idx],
        X_tr[va_idx], y_fit[va_idx],
        params_dict, device=device, verbose=True
    )

    y_pred_tr = ann_predict(final_model, X_tr, device)
    y_pred_te = ann_predict(final_model, X_te, device)
    if log_transform:
        y_pred_tr = np.exp(y_pred_tr)
        y_pred_te = np.exp(y_pred_te)

    tr_met = compute_metrics(y_tr, y_pred_tr)
    te_met = compute_metrics(y_te, y_pred_te)
    stability = assess_training_stability(val_hist)

    print(f"  CV  R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  Test R²: {te_met['r2']:.4f}  MAE: {te_met['mae']:.3f} K  "
          f"RMSE: {te_met['rmse']:.3f} K  MAPE: {te_met['mape']:.2f}%")
    print(f"  Training stability: oscillation={stability['oscillation_frac']*100:.0f}% of tail epochs, "
          f"tail rel.std={stability['tail_rel_std']:.3f}, diverging={stability['diverging']}")
    if stability["unstable"]:
        print(f"  ⚠  {label} training curve looks UNSTABLE (oscillating/diverging), not just a lower-R²")
        print(f"     fit. This points to an OPTIMIZATION problem (try lower LR, gradient clipping,")
        print(f"     batch norm, or a smaller/simpler architecture) rather than a genuine ceiling on")
        print(f"     what this target transform can achieve.")

    return {
        "name": label, "base_name": "ANN", "log_transform": log_transform,
        "model": final_model, "params": params_dict,
        "train_hist": train_hist, "val_hist": val_hist,
        "stability": stability,
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_r2_vals": cv_r2, "cv_mae": np.mean(cv_mae_list),
        "cv_rmse": np.mean(cv_rmse_list),
        "cv_rmse_orig": cv_rmse_orig,   # ★ correct AD residual scale
        "cv_oof_pred": cv_oof_pred,     # ★ correct AD train-side residuals
        "train_r2": tr_met["r2"], "test_r2": te_met["r2"],
        "test_mae": te_met["mae"], "test_rmse": te_met["rmse"],
        "test_mape": te_met["mape"], "test_bias": te_met["bias"],
        "s_train": np.sqrt(mean_squared_error(y_tr, y_pred_tr)),  # kept for reference only — do not use for AD
        "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te,
        "residuals": y_te - y_pred_te, "errors": y_pred_te - y_te,
    }

# =============================================================================
# 8b. HOMOSCEDASTICITY DIAGNOSTIC
# =============================================================================

from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr

def homoscedasticity_check(X_tr_sc, y_tr, rs=42, n_splits=10, save_path=None):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=rs)

    # Raw-scale fit
    oof_raw = cross_val_predict(Ridge(alpha=1.0, max_iter=10000), X_tr_sc, y_tr, cv=cv)
    resid_raw = y_tr - oof_raw

    # Log-scale fit, back-transformed residuals (raw K units) for apples-to-apples comparison
    oof_log = cross_val_predict(Ridge(alpha=1.0, max_iter=10000), X_tr_sc, np.log(y_tr), cv=cv)
    pred_log_bt = np.exp(oof_log)
    resid_log = y_tr - pred_log_bt


    rho_raw, _ = spearmanr(np.abs(resid_raw), oof_raw)
    rho_log, _ = spearmanr(np.abs(resid_log), pred_log_bt)

    print(f"\n{'='*64}")
    print(f"  HOMOSCEDASTICITY CHECK — Linear (Ridge) OOF residuals, {ACTIVE_SET}")
    print(f"{'='*64}")
    print(f"  Raw target : |resid| vs fitted Spearman ρ = {rho_raw:+.4f}   "
          f"OOF R² = {r2_score(y_tr, oof_raw):.4f}")
    print(f"  Log target : |resid| vs fitted Spearman ρ = {rho_log:+.4f}   "
          f"OOF R² = {r2_score(y_tr, pred_log_bt):.4f}  (back-transformed to K)")
    if abs(rho_raw) < 0.15:
        print("  ⚠  Raw-target residuals already look close to homoscedastic (|ρ| < 0.15).")
        print("     The log-transform's variance-stabilization rationale may no longer")
        print("     apply with this descriptor set — consistent with log hurting R².")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Residuals vs Fitted — Linear Model, Raw vs Log Target ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.02)
    for ax, fitted, resid, rho, title in [
        (axes[0], oof_raw,    resid_raw, rho_raw, "Raw target"),
        (axes[1], pred_log_bt, resid_log, rho_log, "Log target (back-transformed to K)"),
    ]:
        ax.scatter(fitted, resid, s=10, alpha=0.5, color="#4A90D9", edgecolors="none")
        ax.axhline(0, color="#333", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Fitted BP (K)", fontsize=9.5)
        ax.set_ylabel("Residual (K)", fontsize=9.5)
        ax.set_title(f"{title}  (|resid| vs fitted ρ={rho:+.3f})",
                     fontsize=10, fontstyle="italic")
        ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

    return {"rho_raw": rho_raw, "rho_log": rho_log,
            "oof_r2_raw": r2_score(y_tr, oof_raw), "oof_r2_log": r2_score(y_tr, pred_log_bt)}

homoscedasticity_results = homoscedasticity_check(
    X_train_sc, y_train, rs=RANDOM_STATE, n_splits=N_SPLITS,
    save_path=f"{OUT_DIR}fig_homoscedasticity_check.png"
)

# =============================================================================
# 9. TRAIN ALL MODELS WITH THE LOCKED PASS-2 PARAMETERS
# =============================================================================

if AUTO_TUNE:
    print(f"\n{'='*64}")
    print(f"  AUTO-TUNING with Optuna ({OPTUNA_TRIALS} trials/model, {ACTIVE_SET})")
    print(f"{'='*64}")

    print("\n  Tuning LightGBM ...")
    LGBM_PARAMS  = tune_lgbm(X_train_sc, y_train, True, OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE)
    LGBM_PARAMS.update({"verbosity": -1})

    print("  Tuning XGBoost ...")
    XGB_PARAMS   = tune_xgb(X_train_sc, y_train, True, OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE)

    print("  Tuning Random Forest ...")
    RF_PARAMS    = tune_rf(X_train_sc, y_train, True, OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE)


    print("  Tuning ANN (raw target) ...")
    ANN_PARAMS_RAW = tune_ann(X_train_sc, y_train, False, ANN_OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE, ANN_DEVICE)
    print("  Tuning ANN (log target) ...")
    ANN_PARAMS_LOG = tune_ann(X_train_sc, y_train, True,  ANN_OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE, ANN_DEVICE)

    print("\n  ✅  Tuning complete. Best parameters found:")
    for name, p_dict in [("LightGBM", LGBM_PARAMS), ("XGBoost", XGB_PARAMS),
                          ("Random Forest", RF_PARAMS),
                          ("ANN (raw target)", ANN_PARAMS_RAW),
                          ("ANN (log target)", ANN_PARAMS_LOG)]:
        print(f"\n  {name}: {p_dict}")
else:
    print(f"\n{'='*64}")
    print("  USING LOCKED FALLBACK PARAMETERS (no re-tuning)")
    print(f"{'='*64}")
    ANN_PARAMS_RAW = ANN_PARAMS
    ANN_PARAMS_LOG = ANN_PARAMS
    for name, p_dict in [("LightGBM", LGBM_PARAMS), ("XGBoost", XGB_PARAMS),
                          ("Random Forest", RF_PARAMS), ("ANN", ANN_PARAMS)]:
        print(f"\n  {name}: {p_dict}")

# Train all models (both raw and log)
results_raw, results_log = {}, {}

for transform, results_dict, lt in [("original", results_raw, False),
                                      ("log",      results_log, True)]:
    print(f"\n{'─'*64}")
    print(f"  Training — {transform} scale")
    print(f"{'─'*64}")

    ann_params_for_transform = ANN_PARAMS_LOG if lt else ANN_PARAMS_RAW

    results_dict["Linear"] = run_sklearn_model(
        "Linear", Ridge(**LINEAR_PARAMS), LINEAR_PARAMS,
        X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE
    )
    results_dict["RandomForest"] = run_sklearn_model(
        "RandomForest", RandomForestRegressor(**RF_PARAMS), RF_PARAMS,
        X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE
    )
    results_dict["XGBoost"] = run_sklearn_model(
        "XGBoost", XGBRegressor(**XGB_PARAMS), XGB_PARAMS,
        X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE
    )
    results_dict["LightGBM"] = run_sklearn_model(
        "LightGBM", LGBMRegressor(**LGBM_PARAMS), LGBM_PARAMS,
        X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE
    )
    results_dict["ANN"] = run_ann_model(
        X_train_sc, y_train, X_test_sc, y_test,
        ann_params_for_transform, lt, N_SPLITS, RANDOM_STATE, ANN_DEVICE
    )

# Pick best overall model (LightGBM, best transform)
best_res   = results_log["LightGBM"] if \
             results_log["LightGBM"]["test_r2"] > results_raw["LightGBM"]["test_r2"] \
             else results_raw["LightGBM"]
best_model = best_res["model"]
tag_str    = " (Log)" if best_res["log_transform"] else ""
print(f"\n  ✅  Best model: LightGBM{tag_str}  [{ACTIVE_SET}]  Test R² = {best_res['test_r2']:.4f}")

# =============================================================================
# 10. HYPERPARAMETER TABLES
# =============================================================================

def print_hyperparameter_tables():
    all_params = {
        "Linear (Ridge)": LINEAR_PARAMS,
        "Random Forest":  RF_PARAMS,
        "XGBoost":        XGB_PARAMS,
        "LightGBM":       LGBM_PARAMS,
        "ANN (raw target)": {k: str(v) for k, v in ANN_PARAMS_RAW.items()},
        "ANN (log target)": {k: str(v) for k, v in ANN_PARAMS_LOG.items()},
    }

    print(f"\n{'═'*64}")
    print(f"  HYPERPARAMETER TABLES — {ACTIVE_SET}")
    print(f"{'═'*64}")
    rows = []
    for model_name, params in all_params.items():
        for k, v in params.items():
            rows.append({"Model": model_name, "Hyperparameter": k, "Value": str(v)})
        rows.append({"Model": "", "Hyperparameter": "", "Value": ""})  # spacer

    hp_df = pd.DataFrame(rows)
    print(hp_df[hp_df["Model"] != ""].to_string(index=False))
    print("\n  LaTeX:\n")
    print(hp_df[hp_df["Model"] != ""].to_latex(
        index=False,
        caption=f"Locked Pass-2 Optuna hyperparameter settings for all models ({ACTIVE_SET}).",
        label=f"tab:hyperparameters_{ACTIVE_SET}"
    ))

    # Also plot as a visual table figure
    fig, ax = plt.subplots(figsize=(13, 0.38 * len(rows) + 2))
    ax.axis("off")
    data_rows = [[r["Model"], r["Hyperparameter"], r["Value"]]
                 for r in rows if r["Model"] != ""]
    tbl = ax.table(
        cellText=data_rows,
        colLabels=["Model", "Hyperparameter", "Value"],
        cellLoc="left", loc="center",
        colWidths=[0.25, 0.40, 0.35]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)

    # Colour header and model-name cells
    header_color  = "#2C6E8A"
    model_colors_tbl = {
        "Linear (Ridge)": "#EBF5FB", "Random Forest": "#EAFAF1",
        "XGBoost": "#FEF9E7", "LightGBM": "#FDEDEC",
        "ANN (raw target)": "#F5EEF8", "ANN (log target)": "#F0E6F6",
    }
    current_color = "white"
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if col == 0 and cell.get_text().get_text() in model_colors_tbl:
                current_color = model_colors_tbl[cell.get_text().get_text()]
                cell.set_text_props(fontweight="bold")
            cell.set_facecolor(current_color)
        cell.set_edgecolor("#D5D8DC")

    ax.set_title(f"Hyperparameter Settings — All Models ({ACTIVE_SET})",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, f"{OUT_DIR}fig_hyperparameter_table.png")
    plt.show(); plt.close(fig)

print_hyperparameter_tables()

# =============================================================================
# 11. PERFORMANCE SUMMARY TABLE
# =============================================================================

def build_summary(results_raw, results_log):
    rows = []
    for name in results_raw:
        for tag, res in [("Original", results_raw[name]), ("Log", results_log[name])]:
            rows.append({
                "Model": name, "Transform": tag,
                "CV R² (±std)":  f"{res['cv_r2_mean']:.4f} ±{res['cv_r2_std']:.4f}",
                "CV MAE (K)":    f"{res['cv_mae']:.3f}",
                "CV RMSE (K)":   f"{res['cv_rmse']:.3f}",
                "Test R²":       f"{res['test_r2']:.4f}",
                "Test MAE (K)":  f"{res['test_mae']:.3f}",
                "Test RMSE (K)": f"{res['test_rmse']:.3f}",
                "Test MAPE (%)": f"{res['test_mape']:.2f}",
                "Test Bias (K)": f"{res['test_bias']:.3f}",
            })
    df_out = pd.DataFrame(rows).set_index(["Model", "Transform"])
    print(f"\n{'═'*90}")
    print(f"  PERFORMANCE SUMMARY — All Models, Original vs Log | {ACTIVE_SET}")
    print(f"{'═'*90}")
    print(df_out.to_string())
    print("\n  LaTeX:\n")
    print(df_out.to_latex(
        caption=f"Predictive performance of all models with and without "
                f"log-transformation ({ACTIVE_SET}, {p} descriptors). All metrics on original Kelvin scale.",
        label=f"tab:model_comparison_{ACTIVE_SET}", escape=False
    ))
    return df_out

summary_df = build_summary(results_raw, results_log)
print("\n  NOTE (apples-to-apples check): all 'Test R²/MAE/RMSE/MAPE' values above — for")
print("  BOTH the 'Original' and 'Log' rows — are computed on the RAW Kelvin scale.")
print("  Log-target models' predictions are exponentiated back (np.exp) BEFORE any metric")
print("  is computed (see run_sklearn_model / run_ann_model), so this table already compares")
print("  the two transforms fairly rather than comparing R² computed on different scales.")
summary_df.to_csv(f"{OUT_DIR}performance_summary_{ACTIVE_SET}.csv")
print(f"  ✅  Saved → {OUT_DIR}performance_summary_{ACTIVE_SET}.csv")

# =============================================================================
# 12. ANN TRAINING CURVE
# =============================================================================

def plot_ann_training(res_raw, res_log, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"ANN Learning Curves — Original vs Log Transform ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, res, title in [
        (axes[0], res_raw, "Original Scale"),
        (axes[1], res_log, "Log Transform"),
    ]:
        th = res["train_hist"]
        vh = res["val_hist"]
        epochs = np.arange(1, len(th)+1)
        ax.plot(epochs, th, color=MODEL_COLORS["ANN"], linewidth=1.8, label="Train MSE")
        ax.plot(epochs, vh, color="#E67E22", linewidth=1.8,
                linestyle="--", label="Val MSE")
        ax.axvline(np.argmin(vh)+1, color="#333", linewidth=1.0,
                   linestyle=":", label=f"Best epoch={np.argmin(vh)+1}")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("MSE Loss", fontsize=10)
        ax.set_title(f"ANN — {title}  (R²={res['test_r2']:.4f})",
                     fontsize=10, fontstyle="italic")
        st = res.get("stability", None)
        if st is not None:
            flag = "⚠ UNSTABLE" if st["unstable"] else "✓ stable"
            ax.text(0.02, 0.05,
                    f"{flag}\noscillation={st['oscillation_frac']*100:.0f}%  "
                    f"tail σ/μ={st['tail_rel_std']:.3f}",
                    transform=ax.transAxes, fontsize=7.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C5CDD5", alpha=0.9))
        ax.legend(fontsize=8.5)
        ax.set_yscale("log")
        ax.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_ann_training(results_raw["ANN"], results_log["ANN"],
                  save_path=f"{OUT_DIR}fig_ann_training_curve.png")

# =============================================================================
# 13. FIGURE — Log vs Original comparison (all 5 models)
# =============================================================================

def plot_log_comparison(results_raw, results_log, save_path=None):
    names   = list(results_raw.keys())
    metrics = ["test_r2", "test_mae", "test_rmse", "test_mape"]
    titles  = ["Test R²", "Test MAE (K)", "Test RMSE (K)", "Test MAPE (%)"]
    ylabels = ["R²", "MAE (K)", "RMSE (K)", "MAPE (%)"]
    higher_better = [True, False, False, False]

    x, w = np.arange(len(names)), 0.35
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    fig.suptitle(f"Effect of Log-Transformation on Model Performance — All Models ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, metric, title, ylabel, hb in zip(axes, metrics, titles, ylabels, higher_better):
        vals_raw = [results_raw[n][metric] for n in names]
        vals_log = [results_log[n][metric] for n in names]
        b1 = ax.bar(x - w/2, vals_raw, w, color=[MODEL_COLORS[n] for n in names],
                    alpha=0.55, edgecolor="white", linewidth=0.6)
        b2 = ax.bar(x + w/2, vals_log, w, color=[MODEL_COLORS[n] for n in names],
                    alpha=0.95, edgecolor="white", linewidth=0.6, hatch="///")

        for i, (vr, vl) in enumerate(zip(vals_raw, vals_log)):
            improved    = (vl > vr) if hb else (vl < vr)
            arrow_color = "#27AE60" if improved else "#C0392B"
            ax.annotate("", xy=(x[i]+w/2, vl), xytext=(x[i]-w/2, vr),
                        arrowprops=dict(arrowstyle="->", color=arrow_color,
                                        lw=1.2, connectionstyle="arc3,rad=0.25"))

        for bar, val in zip(list(b1)+list(b2), vals_raw+vals_log):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontstyle="italic")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        for tick, c in zip(ax.get_xticklabels(), [MODEL_COLORS[n] for n in names]):
            tick.set_color(c)

    legend_elem = [Patch(facecolor="#888", alpha=0.55, label="Original"),
                   Patch(facecolor="#888", alpha=0.95, hatch="///", label="Log")]
    fig.legend(handles=legend_elem, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_log_comparison(results_raw, results_log,
                    save_path=f"{OUT_DIR}fig_log_comparison.png")

# =============================================================================
# 14. FIGURE — 3×2 Pred vs Exp (all 5 models, best transform)
# =============================================================================

def plot_all_pred_vs_exp(results_raw, results_log, y_test, save_path=None):
    names = list(results_raw.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes.flat[-1].set_visible(False)   # 5 models → hide 6th cell
    fig.suptitle(f"Predicted vs Experimental Boiling Point — All Models ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, name in zip(axes.flat[:5], names):
        res   = results_log[name] if results_log[name]["test_r2"] > results_raw[name]["test_r2"] \
                else results_raw[name]
        color = MODEL_COLORS[name]
        y_pred= res["y_pred_test"]
        rmse, r2, mae = res["test_rmse"], res["test_r2"], res["test_mae"]
        tag   = " (log)" if res["log_transform"] else ""

        xy  = np.vstack([y_test, y_pred])
        kde = gaussian_kde(xy)(xy)
        idx = kde.argsort()
        sc  = ax.scatter(y_test[idx], y_pred[idx], c=kde[idx], cmap="viridis",
                         s=10, alpha=0.7, edgecolors="none", rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.78, label="Density")

        lims = [min(y_test.min(), y_pred.min())-5, max(y_test.max(), y_pred.max())+5]
        ax.plot(lims, lims, "--", color=PALETTE["perfect"], linewidth=1.5)
        ax.fill_between(lims, [l-rmse for l in lims], [l+rmse for l in lims],
                        alpha=0.08, color=color)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")

        ax_ins = ax.inset_axes([0.63, 0.04, 0.35, 0.25])
        errors = res["errors"]
        ax_ins.hist(errors, bins=25, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.4, density=True)
        ax_ins.axvline(0, color="#333", linewidth=0.9, linestyle="--")
        ax_ins.set_xlabel("Error (K)", fontsize=6)
        ax_ins.tick_params(labelsize=5.5)
        ax_ins.set_facecolor("#EAEFF4")
        for sp in ["top","right"]: ax_ins.spines[sp].set_visible(False)

        ax.text(0.04, 0.97, f"R² = {r2:.4f}\nMAE = {mae:.2f} K\nRMSE = {rmse:.2f} K",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C5CDD5", alpha=0.95))
        ax.set_xlabel("Experimental BP (K)", fontsize=9.5)
        ax.set_ylabel("Predicted BP (K)", fontsize=9.5)
        ax.set_title(f"{name}{tag}", fontsize=11, fontweight="bold", color=color)
        ax.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_all_pred_vs_exp(results_raw, results_log, y_test,
                     save_path=f"{OUT_DIR}fig_all_pred_vs_exp.png")

# =============================================================================
# 15. FIGURE — CV folds + Residual KDE
# =============================================================================

def plot_cv_and_kde(results_raw, results_log, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"Cross-Validation Stability & Residual Error Distributions ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.02)

    fold_nums = np.arange(1, N_SPLITS+1)
    for name in results_raw:
        color, marker = MODEL_COLORS[name], MODEL_MARKERS[name]
        for tag, res_dict, ls in [("Orig", results_raw, "-"), ("Log", results_log, "--")]:
            res = res_dict[name]
            ax1.plot(fold_nums, res["cv_r2_vals"], color=color, linewidth=1.6,
                     linestyle=ls, marker=marker, markersize=5,
                     label=f"{name} ({tag})")

    ax1.set_xlabel("Fold", fontsize=10); ax1.set_ylabel("CV R²", fontsize=10)
    ax1.set_title("(A)  10-Fold CV R²", loc="left", fontsize=10, fontstyle="italic")
    ax1.set_xticks(fold_nums); ax1.legend(fontsize=7, ncol=2)
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)

    for name in results_raw:
        color = MODEL_COLORS[name]
        res   = results_log[name] if results_log[name]["test_r2"] > results_raw[name]["test_r2"] \
                else results_raw[name]
        tag   = " (log)" if res["log_transform"] else ""
        err   = res["errors"]
        kde_x = np.linspace(err.min()-5, err.max()+5, 400)
        kde_y = gaussian_kde(err, bw_method="scott")(kde_x)
        ax2.plot(kde_x, kde_y, color=color, linewidth=2.2,
                 label=f"{name}{tag}  RMSE={res['test_rmse']:.2f} K")
        ax2.fill_between(kde_x, kde_y, alpha=0.08, color=color)

    ax2.axvline(0, color="#333", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Prediction Error (Pred − Exp, K)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.set_title("(B)  Residual Error KDE", loc="left", fontsize=10, fontstyle="italic")
    ax2.legend(fontsize=8); ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_cv_and_kde(results_raw, results_log,
                save_path=f"{OUT_DIR}fig_cv_residuals.png")

# =============================================================================
# 16. FEATURE IMPORTANCE + SHAP (LightGBM only — tree-based)
# =============================================================================

# Feature importance
def plot_feature_importance(model, feature_names, top_n=20,
                            title_suffix="", save_path=None):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    feat_df = pd.DataFrame({
        "Feature":    [feature_names[i] for i in idx],
        "Importance": imp[idx],
    }).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.42*top_n + 2.2))
    norm  = feat_df["Importance"] / feat_df["Importance"].max()
    bar_c = plt.cm.RdYlBu_r(0.25 + 0.65 * norm.values)
    bars  = ax.barh(feat_df["Feature"], feat_df["Importance"],
                    color=bar_c, edgecolor="white", linewidth=0.5, height=0.72)
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(bar.get_width() + feat_df["Importance"].max()*0.012,
                bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=10)
    ax.set_title(f"Top {top_n} Descriptor Importances — LightGBM{title_suffix} ({ACTIVE_SET})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, feat_df["Importance"].max()*1.22)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)
    return feat_df

feat_df = plot_feature_importance(
    best_model, feature_names, top_n=TOP_N_FEAT, title_suffix=tag_str,
    save_path=f"{OUT_DIR}fig_feature_importance_lgbm.png"
)

# SHAP
print("\n  Computing SHAP values ...")
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train_sc)
shap_df     = pd.DataFrame(shap_values, columns=feature_names)
mean_shap   = shap_df.abs().mean().sort_values(ascending=False)
top_feats   = mean_shap.head(TOP_N_FEAT).index.tolist()
print(f"  ✅  SHAP done. Top feature: {top_feats[0]}")

# SHAP combined panel
def plot_shap_combined(shap_values, X_train_sc, mean_shap,
                       feature_names, top_n=20, save_path=None):
    top_idx  = [feature_names.index(f) for f in mean_shap.head(top_n).index]
    shap_top = shap_values[:, top_idx]
    X_top    = X_train_sc[:, top_idx]
    feat_top = [feature_names[i] for i in top_idx]

    fig = plt.figure(figsize=(16, 0.45*top_n + 3))
    fig.suptitle(f"SHAP Analysis — LightGBM{tag_str} | Top {top_n} Descriptors ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.02)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45,
                             left=0.05, right=0.97, top=0.92, bottom=0.10)

    ax_bar = fig.add_subplot(gs[0, 0])
    data   = mean_shap.head(top_n).iloc[::-1]
    norm   = (data-data.min())/(data.max()-data.min())
    bc     = plt.cm.Oranges(0.35 + 0.60*norm.values)
    bars   = ax_bar.barh(data.index, data.values, color=bc,
                         edgecolor="white", linewidth=0.5, height=0.72)
    for bar, val in zip(bars, data.values):
        ax_bar.text(bar.get_width()+data.max()*0.015,
                    bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="left", fontsize=7.8)
    ax_bar.set_xlabel("Mean |SHAP| (K)", fontsize=10)
    ax_bar.set_title("(A)  Global Feature Importance", loc="left",
                     fontsize=10, fontstyle="italic")
    ax_bar.set_xlim(0, data.max()*1.22)
    ax_bar.tick_params(axis="y", length=0, labelsize=8.5)
    ax_bar.xaxis.grid(True); ax_bar.set_axisbelow(True)

    ax_bee = fig.add_subplot(gs[0, 1])
    plt.sca(ax_bee)
    shap.summary_plot(shap_top, X_top, feature_names=feat_top,
                      plot_type="dot", max_display=top_n,
                      show=False, color_bar=True, plot_size=None)
    ax_bee = plt.gca()   # re-grab current axes in case SHAP created a new one internally
    ax_bee.set_title("(B)  Direction & Magnitude (Beeswarm)",
                     loc="left", fontsize=10, fontstyle="italic")
    ax_bee.set_xlabel("SHAP Value — Impact on BP (K)", fontsize=10)

    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_shap_combined(shap_values, X_train_sc, mean_shap,
                   feature_names, top_n=TOP_N_FEAT,
                   save_path=f"{OUT_DIR}fig_shap_combined.png")

# SHAP dependence plots
def plot_shap_dependence(shap_values, X_train_sc, feature_names,
                         top_feats, n_plots=4, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"SHAP Dependence Plots — Top {n_plots} Features | LightGBM{tag_str} ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, feat in zip(axes.flat, top_feats[:n_plots]):
        fi   = feature_names.index(feat)
        sv   = shap_values[:, fi]
        fv   = X_train_sc[:, fi]
        inter= top_feats[1] if feat == top_feats[0] else top_feats[0]
        ci   = feature_names.index(inter)
        cv   = X_train_sc[:, ci]
        sc   = ax.scatter(fv, sv, c=cv, cmap="coolwarm", s=8,
                          alpha=0.55, edgecolors="none", rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.85, label=f"{inter} (scaled)")
        sort_i = np.argsort(fv)
        smooth = uniform_filter1d(sv[sort_i], size=max(1, len(sv)//40))
        ax.plot(fv[sort_i], smooth, color="#1A2A3A", linewidth=1.8, alpha=0.8, label="Trend")
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{feat} (scaled)", fontsize=9.5)
        ax.set_ylabel("SHAP Value (K)", fontsize=9.5)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5); ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_shap_dependence(shap_values, X_train_sc, feature_names, top_feats,
                     n_plots=4, save_path=f"{OUT_DIR}fig_shap_dependence.png")

# =============================================================================
# 16b. CHECKPOINT
# =============================================================================

import joblib

checkpoint_path = f"{OUT_DIR}checkpoint_{ACTIVE_SET}.pkl"
checkpoint = {
    "ACTIVE_SET": ACTIVE_SET,
    "OUT_DIR": OUT_DIR,
    "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
    "y_train": y_train, "y_test": y_test,
    "feature_names": feature_names, "n_train": n_train, "p": p,
    "tag_str": tag_str,

    "best_res": {k: v for k, v in best_res.items() if k != "model"},
    "feat_df": feat_df,
    "mean_shap": mean_shap,
    "SIGMA_LIMIT": SIGMA_LIMIT, "TOP_N_FEAT": TOP_N_FEAT,
    "test_df": test_df,
}
joblib.dump(checkpoint, checkpoint_path)
print(f"\n  ✅  AD checkpoint saved → {checkpoint_path}")
print(f"     Re-run AD_analysis_standalone.py against this file to redo AD")
print(f"     analysis (Williams plot, 4-panel, exports) in seconds, without")
print(f"     retraining anything.")

# =============================================================================
# 17. APPLICABILITY DOMAIN
# =============================================================================

def compute_leverage(X_ref, X_query):
    U, S, Vt = np.linalg.svd(X_ref, full_matrices=False)

    tol   = S.max() * max(X_ref.shape) * np.finfo(float).eps * 100
    S_inv = np.where(S > tol, 1.0/S**2, 0.0)
    return (X_query @ Vt.T)**2 @ S_inv

h_train = compute_leverage(X_train_sc, X_train_sc)
h_test  = compute_leverage(X_train_sc, X_test_sc)
h_star  = 3.0 * (p + 1) / n_train

# ★ s = CV RMSE (out-of-fold) of the best model — the correct AD scale
s = best_res["cv_rmse_orig"]

# ★ Train-side residuals use CV out-of-fold predictions (unbiased); test-side

resid_train = y_train - best_res["cv_oof_pred"]
resid_test  = y_test  - best_res["y_pred_test"]
sr_train = resid_train / s
sr_test  = resid_test  / s

in_ad_train = (h_train <= h_star) & (np.abs(sr_train) <= SIGMA_LIMIT)
in_ad_test  = (h_test  <= h_star) & (np.abs(sr_test)  <= SIGMA_LIMIT)

print(f"\n  AD: h* = {h_star:.5f}  ({ACTIVE_SET}, {p} descriptors)")
print(f"  s (CV RMSE, OOF)  : {s:.3f} K   [NOT in-sample train RMSE — see comment above]")
print(f"  3σ absolute limit : ±{3*s:.1f} K")
print(f"  Leverage max test : {h_test.max():.5f}  "
      f"{'✅ valid (≤1.0)' if h_test.max() <= 1.0 else '❌ check scaler/tolerance'}")
print(f"  Train within AD: {in_ad_train.mean()*100:.1f}%")
print(f"  Test  within AD: {in_ad_test.mean()*100:.1f}%")

# AD performance breakdown — within vs. outside vs. overall
print(f"\n  {'':22}  {'R²':>7}  {'MAE (K)':>9}  {'RMSE (K)':>10}  {'N':>6}")
print(f"  {'─'*60}")
for mask, ad_label in [
    (in_ad_test,                       "Test — Within AD"),
    (~in_ad_test,                      "Test — Outside AD"),
    (np.ones(len(y_test), dtype=bool), "Test — Overall"),
]:
    if mask.sum() < 2:
        continue
    _r2   = r2_score(y_test[mask], best_res["y_pred_test"][mask])
    _mae  = mean_absolute_error(y_test[mask], best_res["y_pred_test"][mask])
    _rmse = np.sqrt(mean_squared_error(y_test[mask], best_res["y_pred_test"][mask]))
    print(f"  {ad_label:22}  {_r2:7.4f}  {_mae:9.3f}  {_rmse:10.3f}  {mask.sum():6,}")

# =============================================================================
# 17b. OUTLIER TYPE CLASSIFICATION
# =============================================================================

mask_high_h  = h_test >  h_star
mask_high_sr = np.abs(sr_test) > SIGMA_LIMIT

type_A = mask_high_h  & ~mask_high_sr
type_B = ~mask_high_h &  mask_high_sr
type_C = mask_high_h  &  mask_high_sr
type_D = in_ad_test

outlier_type = np.where(type_D, "D — Within AD",
               np.where(type_A, "A — High Leverage",
               np.where(type_B, "B — High Residual",
                                "C — Both")))

print(f"\n  Outlier type breakdown:")
print(f"  Type A (h > h* only)  : {type_A.sum():,}  — structurally novel, ok prediction")
print(f"  Type B (|σ| > 3 only) : {type_B.sum():,}  — known structure, poor prediction")
print(f"  Type C (both)         : {type_C.sum():,}  — true outliers")

abs_error = np.abs(resid_test)

# Williams plot
def plot_williams(h_train, sr_train, h_test, sr_test,
                  in_ad_train, in_ad_test, h_star, s, sigma_lim=3.0, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 8))
    sr_max = min(max(np.abs(sr_train).max(), np.abs(sr_test).max()) * 1.12, 12)
    x_max  = max(h_train.max(), h_test.max()) * 1.18

    ax.fill_betweenx([-sigma_lim, sigma_lim], 0, h_star,
                     color=PALETTE["ad_fill"], alpha=0.55, zorder=0, label="AD Region")
    ax.axvline(h_star, color=PALETTE["perfect"], linewidth=1.8,
               linestyle="--", zorder=4, label=f"h* = {h_star:.5f}")
    ax.axhline( sigma_lim, color=PALETTE["perfect"], linewidth=1.4, linestyle=":", zorder=4,
                label=f"±{sigma_lim:.0f}σ  (±{sigma_lim*s:.1f} K)")
    ax.axhline(-sigma_lim, color=PALETTE["perfect"], linewidth=1.4, linestyle=":", zorder=4)
    ax.axhline(0, color="#AAB7B8", linewidth=0.8)

    sk = dict(s=18, alpha=0.55, edgecolors="none", rasterized=True, zorder=5)
    ax.scatter(h_train[in_ad_train],  sr_train[in_ad_train],
               color=PALETTE["train_in"], marker="o",
               label=f"Train – within AD ({in_ad_train.sum():,})", **sk)
    ax.scatter(h_train[~in_ad_train], sr_train[~in_ad_train],
               color=PALETTE["train_out"], marker="^", s=22, alpha=0.65,
               edgecolors="none", zorder=5, label=f"Train – outside ({(~in_ad_train).sum():,})")
    ax.scatter(h_test[in_ad_test],    sr_test[in_ad_test],
               color=PALETTE["test_in"], marker="o",
               label=f"Test – within AD ({in_ad_test.sum():,})", **sk)
    ax.scatter(h_test[~in_ad_test],   sr_test[~in_ad_test],
               color=PALETTE["test_out"], marker="s", s=22, alpha=0.75,
               edgecolors="none", zorder=5, label=f"Test – outside ({(~in_ad_test).sum():,})")

    q_kw = dict(fontsize=8, color="#7F8C8D", fontstyle="italic", ha="center", va="center")
    y_q = min(sr_max * 0.82, 10)
    ax.text(h_star*0.45,  y_q, "✓  Reliable\npredictions",   **q_kw)
    ax.text(h_star*1.55,  y_q, "⚠  Structurally\ninfluential",**q_kw)
    ax.text(h_star*0.45, -y_q, "⚠  Response\noutliers",       **q_kw)
    ax.text(h_star*1.55, -y_q, "✗  Outside AD",               **q_kw)
    ax.text(0.02, 0.97,
            f"h* = {h_star:.5f}\ns = CV RMSE = {s:.2f} K\n3σ = {3*s:.1f} K\n"
            f"Train AD: {in_ad_train.mean()*100:.1f}%\n"
            f"Test AD: {in_ad_test.mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#C5CDD5", alpha=0.96))

    ax.set_xlabel("Leverage $h_i$", fontsize=11)
    ax.set_ylabel("Standardised Residual $\\hat{\\sigma}_i$", fontsize=11)
    ax.set_title(f"Williams Plot — AD | LightGBM{tag_str} ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-0.002, x_max); ax.set_ylim(-sr_max, sr_max)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.xaxis.grid(True); ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_williams(h_train, sr_train, h_test, sr_test,
              in_ad_train, in_ad_test, h_star, s, SIGMA_LIMIT,
              save_path=f"{OUT_DIR}fig_williams_plot.png")

# AD + SHAP + FeatImp 4-panel
def plot_ad_feature_panel(h_test, sr_test, in_ad_test, h_star,
                           mean_shap, feat_df, y_test, y_pred_test,
                           top_n=20, save_path=None):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"LightGBM{tag_str} — AD · SHAP · Feature Importance ({ACTIVE_SET})",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                           left=0.07, right=0.97, top=0.96, bottom=0.07)
    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0]); ax4 = fig.add_subplot(gs[1,1])

    sr_max = min(np.abs(sr_test).max() * 1.12, 10)
    ax1.fill_betweenx([-SIGMA_LIMIT, SIGMA_LIMIT], 0, h_star,
                      color=PALETTE["ad_fill"], alpha=0.55)
    ax1.axvline(h_star, color=PALETTE["perfect"], linewidth=1.6, linestyle="--",
                label=f"h*={h_star:.5f}")
    ax1.axhline( SIGMA_LIMIT, color=PALETTE["perfect"], linewidth=1.2, linestyle=":")
    ax1.axhline(-SIGMA_LIMIT, color=PALETTE["perfect"], linewidth=1.2, linestyle=":")
    ax1.axhline(0, color="#AAB7B8", linewidth=0.8)
    ax1.scatter(h_test[in_ad_test],  sr_test[in_ad_test],  color=PALETTE["test_in"],
                s=14, alpha=0.55, edgecolors="none", label=f"Within ({in_ad_test.sum():,})")
    ax1.scatter(h_test[~in_ad_test], sr_test[~in_ad_test], color=PALETTE["test_out"],
                s=18, marker="^", alpha=0.7, edgecolors="none",
                label=f"Outside ({(~in_ad_test).sum():,})")
    ax1.set_xlabel("Leverage $h_i$", fontsize=9.5)
    ax1.set_ylabel("Std Residual $\\hat{\\sigma}_i$", fontsize=9.5)
    ax1.set_title("(A)  Williams Plot (Test)", loc="left", fontsize=10, fontstyle="italic")
    ax1.legend(fontsize=8); ax1.grid(True); ax1.set_axisbelow(True)
    ax1.set_ylim(-sr_max, sr_max)

    for mask, label, color, mk in [
        (in_ad_test,  "Within AD",  PALETTE["test_in"],  "o"),
        (~in_ad_test, "Outside AD", PALETTE["test_out"], "^"),
    ]:
        ax2.scatter(y_test[mask], y_pred_test[mask], color=color, marker=mk,
                    s=14, alpha=0.6, edgecolors="none", label=label, rasterized=True)
    lims = [min(y_test.min(), y_pred_test.min())-5, max(y_test.max(), y_pred_test.max())+5]
    ax2.plot(lims, lims, "--", color=PALETTE["perfect"], linewidth=1.4)
    ax2.set_xlim(lims); ax2.set_ylim(lims); ax2.set_aspect("equal")
    if in_ad_test.sum()>1 and (~in_ad_test).sum()>1:
        r2_in  = r2_score(y_test[in_ad_test], y_pred_test[in_ad_test])
        mae_in = mean_absolute_error(y_test[in_ad_test], y_pred_test[in_ad_test])
        r2_out  = r2_score(y_test[~in_ad_test], y_pred_test[~in_ad_test])
        mae_out = mean_absolute_error(y_test[~in_ad_test], y_pred_test[~in_ad_test])
        ax2.text(0.04, 0.97,
                 f"Within AD : R²={r2_in:.4f}, MAE={mae_in:.2f} K\n"
                 f"Outside AD: R²={r2_out:.4f}, MAE={mae_out:.2f} K",
                 transform=ax2.transAxes, fontsize=8.5, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5"))
    ax2.set_xlabel("Experimental BP (K)", fontsize=9.5)
    ax2.set_ylabel("Predicted BP (K)", fontsize=9.5)
    ax2.set_title("(B)  Pred vs Exp — AD Membership", loc="left",
                  fontsize=10, fontstyle="italic")
    ax2.legend(fontsize=8.5); ax2.grid(True); ax2.set_axisbelow(True)

    shap_data = mean_shap.head(top_n).iloc[::-1]
    norm_s = (shap_data-shap_data.min())/(shap_data.max()-shap_data.min())
    bc_s   = plt.cm.Oranges(0.35+0.60*norm_s.values)
    bars   = ax3.barh(shap_data.index, shap_data.values,
                      color=bc_s, edgecolor="white", linewidth=0.4, height=0.72)
    for bar, val in zip(bars, shap_data.values):
        ax3.text(bar.get_width()+shap_data.max()*0.015,
                 bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", ha="left", fontsize=7)
    ax3.set_xlabel("Mean |SHAP| (K)", fontsize=9.5)
    ax3.set_title(f"(C)  SHAP Importance — Top {top_n}", loc="left",
                  fontsize=10, fontstyle="italic")
    ax3.set_xlim(0, shap_data.max()*1.22)
    ax3.tick_params(axis="y", length=0, labelsize=7.5)
    ax3.xaxis.grid(True); ax3.set_axisbelow(True)

    fi_data = feat_df.set_index("Feature")["Importance"].iloc[::-1]
    norm_f  = (fi_data-fi_data.min())/(fi_data.max()-fi_data.min())
    bc_f    = plt.cm.Blues(0.35+0.60*norm_f.values)
    bars_f  = ax4.barh(fi_data.index, fi_data.values,
                       color=bc_f, edgecolor="white", linewidth=0.4, height=0.72)
    for bar, val in zip(bars_f, fi_data.values):
        ax4.text(bar.get_width()+fi_data.max()*0.015,
                 bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", ha="left", fontsize=7)
    ax4.set_xlabel("Feature Importance (Gain)", fontsize=9.5)
    ax4.set_title(f"(D)  Feature Importance — Top {top_n}", loc="left",
                  fontsize=10, fontstyle="italic")
    ax4.set_xlim(0, fi_data.max()*1.22)
    ax4.tick_params(axis="y", length=0, labelsize=7.5)
    ax4.xaxis.grid(True); ax4.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_ad_feature_panel(
    h_test, sr_test, in_ad_test, h_star,
    mean_shap, feat_df, y_test, best_res["y_pred_test"],
    top_n=TOP_N_FEAT, save_path=f"{OUT_DIR}fig_ad_feature_shap_panel.png"
)

# =============================================================================
# 17c. OUTLIER OVERVIEW
# =============================================================================

fig_ov = plt.figure(figsize=(16, 12))
fig_ov.suptitle(f"Applicability Domain — Outlier Analysis\nLightGBM{tag_str} | {ACTIVE_SET}",
                fontsize=14, fontweight="bold", y=1.01)
gs_ov = gridspec.GridSpec(2, 2, figure=fig_ov, hspace=0.40, wspace=0.35,
                          left=0.08, right=0.97, top=0.96, bottom=0.08)
axo1 = fig_ov.add_subplot(gs_ov[0,0]); axo2 = fig_ov.add_subplot(gs_ov[0,1])
axo3 = fig_ov.add_subplot(gs_ov[1,0]); axo4 = fig_ov.add_subplot(gs_ov[1,1])

# (A) Williams by type
axo1.fill_betweenx([-SIGMA_LIMIT, SIGMA_LIMIT], 0, h_star,
                   color=PALETTE["ad_fill"], alpha=0.55, zorder=0)
axo1.axvline(h_star, color=PALETTE["perfect"], linewidth=1.6, linestyle="--", zorder=4,
             label=f"h* = {h_star:.5f}")
axo1.axhline( SIGMA_LIMIT, color="#E67E22", linewidth=1.3, linestyle=":", zorder=4,
              label=f"±{SIGMA_LIMIT:.0f}σ")
axo1.axhline(-SIGMA_LIMIT, color="#E67E22", linewidth=1.3, linestyle=":", zorder=4)
axo1.axhline(0, color="#AAB7B8", linewidth=0.8)
for typ, mask_typ, marker, sz in [
    ("D — Within AD",     type_D, "o", 12),
    ("A — High Leverage", type_A, "^", 22),
    ("B — High Residual", type_B, "s", 22),
    ("C — Both",          type_C, "X", 30),
]:
    if mask_typ.sum() == 0:
        continue
    axo1.scatter(h_test[mask_typ], sr_test[mask_typ],
                 color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.35 if typ == "D — Within AD" else 0.80,
                 edgecolors="none", rasterized=True, zorder=5,
                 label=f"{typ}  (n={mask_typ.sum()})")
axo1.set_xlabel("Leverage $h_i$", fontsize=10)
axo1.set_ylabel("Standardised Residual $\\hat{\\sigma}_i$", fontsize=10)
axo1.set_title("(A)  Williams Plot — Outlier Types", loc="left", fontsize=10, fontstyle="italic")
axo1.set_xlim(-0.001, h_test.max()*1.18)
sr_lim = min(np.abs(sr_test).max()*1.15, 10)
axo1.set_ylim(-sr_lim, sr_lim)
axo1.legend(fontsize=7.5, loc="upper right"); axo1.grid(True); axo1.set_axisbelow(True)

# (B) Error KDE by type
for typ, mask_typ in [("D — Within AD", type_D), ("A — High Leverage", type_A),
                      ("B — High Residual", type_B), ("C — Both", type_C)]:
    if mask_typ.sum() < 3:
        continue
    err   = abs_error[mask_typ]
    kde_x = np.linspace(0, err.max()*1.1, 300)
    try:
        kde_y = gaussian_kde(err, bw_method="scott")(kde_x)
        axo2.plot(kde_x, kde_y, color=TYPE_COLORS[typ], linewidth=2.0,
                  label=f"{typ}  MAE={err.mean():.1f} K  (n={mask_typ.sum()})")
        axo2.fill_between(kde_x, kde_y, alpha=0.10, color=TYPE_COLORS[typ])
    except Exception:
        axo2.axvline(err.mean(), color=TYPE_COLORS[typ], linewidth=2.0, linestyle="--",
                     label=f"{typ}  n={mask_typ.sum()}")
axo2.axvline(3*s, color="#333", linewidth=1.4, linestyle=":", label=f"3σ = {3*s:.1f} K")
axo2.set_xlabel("Absolute Prediction Error (K)", fontsize=10)
axo2.set_ylabel("Probability Density", fontsize=10)
axo2.set_title("(B)  Error Distribution by Outlier Type", loc="left", fontsize=10, fontstyle="italic")
axo2.legend(fontsize=7.5); axo2.grid(True); axo2.set_axisbelow(True)

# (C) Outlier frequency by BP bin
bins  = [0, 300, 350, 400, 450, 500, 600, 700, 1000]
lbls  = ["<300","300–350","350–400","400–450","450–500","500–600","600–700",">700"]
out_bins = pd.cut(y_test[~in_ad_test], bins=bins, labels=lbls)
all_bins = pd.cut(y_test, bins=bins, labels=lbls)
out_cnt  = out_bins.value_counts().reindex(lbls, fill_value=0)
all_cnt  = all_bins.value_counts().reindex(lbls, fill_value=0)

x_b, w_b = np.arange(len(lbls)), 0.35
axo3.bar(x_b - w_b/2, all_cnt.values, w_b, color="#5D8AA8", alpha=0.55,
         edgecolor="white", label="All test")
axo3.bar(x_b + w_b/2, out_cnt.values, w_b, color="#C0392B", alpha=0.85,
         edgecolor="white", label="Outliers")
for i, (all_v, out_v) in enumerate(zip(all_cnt.values, out_cnt.values)):
    pct = out_v / all_v * 100 if all_v > 0 else 0
    if out_v > 0:
        axo3.text(i + w_b/2, out_v + 0.5, f"{pct:.0f}%",
                  ha="center", fontsize=7.5, color="#C0392B", fontweight="bold")
axo3.set_xticks(x_b)
axo3.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8.5)
axo3.set_xlabel("Boiling Point Range (K)", fontsize=10)
axo3.set_ylabel("Count", fontsize=10)
axo3.set_title("(C)  Outlier Frequency by BP Range\n(% = fraction of range that are outliers)",
               loc="left", fontsize=10, fontstyle="italic")
axo3.legend(fontsize=8.5); axo3.yaxis.grid(True); axo3.set_axisbelow(True)

# (D) Pred vs Exp by type
for typ, mask_typ, marker, sz in [
    ("D — Within AD",     type_D, "o", 10),
    ("A — High Leverage", type_A, "^", 35),
    ("B — High Residual", type_B, "s", 35),
    ("C — Both",          type_C, "X", 45),
]:
    if mask_typ.sum() == 0:
        continue
    axo4.scatter(y_test[mask_typ], best_res["y_pred_test"][mask_typ],
                 color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.30 if typ == "D — Within AD" else 0.85,
                 edgecolors="none", rasterized=True, zorder=5,
                 label=f"{typ}  (n={mask_typ.sum()})")
lims_ov = [y_test.min()-5, y_test.max()+5]
axo4.plot(lims_ov, lims_ov, "--", color=PALETTE["perfect"], linewidth=1.5, label="y = x")
axo4.fill_between(lims_ov, [l-3*s for l in lims_ov], [l+3*s for l in lims_ov],
                  alpha=0.05, color="#C0392B", label=f"±3σ = ±{3*s:.0f} K")
axo4.set_xlim(lims_ov); axo4.set_ylim(lims_ov); axo4.set_aspect("equal")
axo4.set_xlabel("Experimental BP (K)", fontsize=10)
axo4.set_ylabel("Predicted BP (K)", fontsize=10)
axo4.set_title("(D)  Predicted vs Experimental — Outlier Types", loc="left",
               fontsize=10, fontstyle="italic")
axo4.legend(fontsize=7.5, loc="upper left"); axo4.grid(True); axo4.set_axisbelow(True)

plt.tight_layout()
save_fig(fig_ov, f"{OUT_DIR}fig_outlier_overview_{ACTIVE_SET}.png")
plt.show(); plt.close(fig_ov)

# =============================================================================
# 17d. PCA CHEMICAL SPACE
# =============================================================================

print("\n  Computing PCA ...")
pca       = PCA(n_components=2, random_state=42)
X_all_pca = np.vstack([X_train_sc, X_test_sc])
pca_all   = pca.fit_transform(X_all_pca)
pca_train = pca_all[:n_train]
pca_test  = pca_all[n_train:]

fig_pca, axes_pca = plt.subplots(1, 2, figsize=(14, 6))
fig_pca.suptitle(f"Chemical Space Coverage — PCA of Descriptor Space ({ACTIVE_SET})",
                 fontsize=13, fontweight="bold", y=1.02)

axp = axes_pca[0]
axp.scatter(pca_train[:,0], pca_train[:,1], s=7, alpha=0.20,
            color="#2C6E8A", rasterized=True, label="Train")
axp.scatter(pca_test[in_ad_test, 0], pca_test[in_ad_test, 1],
            s=12, alpha=0.50, color="#27AE60", rasterized=True, label="Test — Within AD")
axp.scatter(pca_test[~in_ad_test, 0], pca_test[~in_ad_test, 1],
            s=35, alpha=0.85, color="#C0392B", marker="^",
            edgecolors="white", linewidths=0.4, zorder=5, label="Test — Outside AD")
axp.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
axp.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
axp.set_title("(A)  AD Membership in Chemical Space", loc="left", fontsize=10, fontstyle="italic")
axp.legend(fontsize=8.5); axp.grid(True); axp.set_axisbelow(True)

axp2 = axes_pca[1]
axp2.scatter(pca_train[:,0], pca_train[:,1], s=5, alpha=0.12,
             color="#2C6E8A", rasterized=True, label="Train (background)")
for typ, mask_typ, marker, sz in [
    ("D — Within AD",     type_D, "o", 10),
    ("A — High Leverage", type_A, "^", 45),
    ("B — High Residual", type_B, "s", 45),
    ("C — Both",          type_C, "X", 60),
]:
    if mask_typ.sum() == 0:
        continue
    axp2.scatter(pca_test[mask_typ,0], pca_test[mask_typ,1],
                 color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.25 if typ == "D — Within AD" else 0.85,
                 edgecolors="none" if typ == "D — Within AD" else "white",
                 linewidths=0.3, rasterized=True, zorder=5,
                 label=f"{typ}  (n={mask_typ.sum()})")
axp2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
axp2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
axp2.set_title("(B)  Outlier Types in Chemical Space", loc="left", fontsize=10, fontstyle="italic")
axp2.legend(fontsize=7.5); axp2.grid(True); axp2.set_axisbelow(True)

plt.tight_layout()
save_fig(fig_pca, f"{OUT_DIR}fig_outlier_pca_{ACTIVE_SET}.png")
plt.show(); plt.close(fig_pca)

# =============================================================================
# 17e. NEAREST-NEIGHBOUR DISTANCE vs ERROR
# =============================================================================

print("\n  Computing nearest-neighbour distances ...")
nn_model = NearestNeighbors(n_neighbors=1, n_jobs=-1)
nn_model.fit(X_train_sc)
dists, _ = nn_model.kneighbors(X_test_sc)
dists = dists.flatten()

r_val, pval = pearsonr(dists, abs_error)

fig_nn, axnn = plt.subplots(figsize=(9, 6))
for typ, mask_typ, marker, sz in [
    ("D — Within AD",     type_D, "o", 10),
    ("A — High Leverage", type_A, "^", 35),
    ("B — High Residual", type_B, "s", 35),
    ("C — Both",          type_C, "X", 45),
]:
    if mask_typ.sum() == 0:
        continue
    axnn.scatter(dists[mask_typ], abs_error[mask_typ],
                 color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.25 if typ == "D — Within AD" else 0.85,
                 edgecolors="none", rasterized=True,
                 label=f"{typ}  (n={mask_typ.sum()})", zorder=5)
axnn.axhline(3*s, color="#333", linewidth=1.3, linestyle=":", label=f"3σ = {3*s:.1f} K")
axnn.set_xlabel("Distance to Nearest Training Compound (scaled descriptor space)", fontsize=10)
axnn.set_ylabel("Absolute Prediction Error (K)", fontsize=10)
axnn.set_title(f"Nearest-Neighbour Distance vs Prediction Error ({ACTIVE_SET})",
               fontsize=12, fontweight="bold")
axnn.text(0.04, 0.96, f"Pearson r = {r_val:.3f}  (p = {pval:.2e})",
          transform=axnn.transAxes, fontsize=9, va="top",
          bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C5CDD5"))
axnn.legend(fontsize=8.5); axnn.grid(True); axnn.set_axisbelow(True)
plt.tight_layout()
save_fig(fig_nn, f"{OUT_DIR}fig_outlier_nn_distance_{ACTIVE_SET}.png")
plt.show(); plt.close(fig_nn)

# =============================================================================
# 18. EXPORT
# =============================================================================

test_out = test_df.copy()
test_out["y_actual"]     = y_test
test_out["y_predicted"]  = best_res["y_pred_test"]
test_out["Residual_K"]   = resid_test              # y_actual − y_predicted (K)
test_out["Std_Residual"] = sr_test                 # (y − ŷ) / s_cv
test_out["Z_Score"]      = sr_test                 # identical to Std_Residual, kept for compatibility
test_out["Abs_Error_K"]  = abs_error                # |y − ŷ|
test_out["Leverage_h"]   = h_test
test_out["Within_AD"]    = in_ad_test
test_out["Outlier_Type"] = outlier_type
test_out.to_csv(f"{OUT_DIR}test_with_AD_{ACTIVE_SET}.csv", index=False)
print(f"\n  ✅  Full output saved → {OUT_DIR}test_with_AD_{ACTIVE_SET}.csv")

outliers_df = test_out[~in_ad_test].copy().sort_values("Abs_Error_K", ascending=False)
outliers_df.to_csv(f"{OUT_DIR}outliers_outside_AD_{ACTIVE_SET}.csv", index=False)
print(f"  ✅  Outliers saved      → {OUT_DIR}outliers_outside_AD_{ACTIVE_SET}.csv  "
      f"({len(outliers_df)} compounds)")

# =============================================================================
# 18b. TOP 20 WORST PREDICTIONS
# =============================================================================

top20 = outliers_df.head(20).copy()
if len(top20) > 0:
    fig_top, axt = plt.subplots(figsize=(12, 7))
    colors_bar = [TYPE_COLORS.get(t, "#888") for t in top20["Outlier_Type"]]
    axt.barh(range(len(top20)), top20["Abs_Error_K"],
             color=colors_bar, edgecolor="white", linewidth=0.5, height=0.72)
    for i, (_, row) in enumerate(top20.iterrows()):
        axt.text(row["Abs_Error_K"] + top20["Abs_Error_K"].max()*0.01, i,
                 f"BP={row['y_actual']:.0f}K  err={row['Abs_Error_K']:.1f}K  {row['Outlier_Type'][0]}",
                 va="center", ha="left", fontsize=7.5)
    axt.axvline(3*s, color="#333", linewidth=1.4, linestyle=":",
                label=f"3σ threshold = {3*s:.1f} K")
    axt.set_yticks(range(len(top20)))
    if "smiles" in top20.columns:
        ylabels = [str(sm)[:28]+"…" if len(str(sm))>28 else str(sm) for sm in top20["smiles"]]
    else:
        ylabels = [f"Compound {i+1}" for i in range(len(top20))]
    axt.set_yticklabels(ylabels, fontsize=7.5)
    axt.set_xlabel("Absolute Prediction Error (K)", fontsize=10)
    axt.set_title(f"Top 20 Worst Predicted Compounds — Outside AD ({ACTIVE_SET})",
                  fontsize=12, fontweight="bold")
    axt.set_xlim(0, top20["Abs_Error_K"].max()*1.28)
    legend_patches = [Patch(color=TYPE_COLORS[t], label=t)
                      for t in ["A — High Leverage","B — High Residual","C — Both"]
                      if t in top20["Outlier_Type"].values]
    legend_patches.append(plt.Line2D([0],[0], color="#333", linestyle=":",
                                       linewidth=1.4, label=f"3σ = {3*s:.1f} K"))
    axt.legend(handles=legend_patches, fontsize=8.5, loc="lower right")
    axt.xaxis.grid(True); axt.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig_top, f"{OUT_DIR}fig_outlier_top20_{ACTIVE_SET}.png")
    plt.show(); plt.close(fig_top)
else:
    print("  (No compounds outside AD — skipping top-20 figure.)")

shap_table = pd.DataFrame({
    "Rank": range(1, TOP_N_FEAT+1),
    "Descriptor": mean_shap.head(TOP_N_FEAT).index,
    "Mean |SHAP| (K)": mean_shap.head(TOP_N_FEAT).values,
    "Gain Importance": feat_df.set_index("Feature")
                              .reindex(mean_shap.head(TOP_N_FEAT).index)["Importance"].values
}).set_index("Rank")
print(f"\n  LaTeX — SHAP + Importance Table ({ACTIVE_SET}):\n")
print(shap_table.to_latex(
    caption=f"Top {TOP_N_FEAT} molecular descriptors by SHAP and LightGBM gain importance ({ACTIVE_SET}).",
    label=f"tab:shap_importance_{ACTIVE_SET}", float_format="%.5f"
))

if len(outliers_df) >= 1:
    n_top = min(10, len(outliers_df))
    top10 = outliers_df[["y_actual","y_predicted","Abs_Error_K",
                          "Leverage_h","Std_Residual","Outlier_Type"]].head(n_top).copy()
    top10.columns = ["BP$_{exp}$ (K)","BP$_{pred}$ (K)","|Error| (K)",
                     "Leverage $h_i$","$\\hat{\\sigma}_i$","Type"]
    top10.index = range(1, n_top+1)
    print(f"\n  LaTeX — Top {n_top} Outliers ({ACTIVE_SET}):\n")
    print(top10.to_latex(
        caption=f"Top {n_top} compounds outside the applicability domain ranked by "
                f"absolute prediction error ({ACTIVE_SET}). Type A: high leverage; "
                f"Type B: high residual; Type C: both.",
        label=f"tab:ad_outliers_{ACTIVE_SET}", escape=False, float_format="%.3f"
    ))

print(f"\n{'='*64}")
print(f"  PIPELINE COMPLETE — {ACTIVE_SET}")
print(f"{'='*64}")
print(f"""
  Figures saved to: {OUT_DIR}
  ├── fig_target_distribution_diagnostics.png ← skew/kurtosis/Q-Q, raw vs log
  ├── fig_log_transform_effect.png   ← clean before/after KDE panel (paper-ready)
  ├── fig_homoscedasticity_check.png ← Linear residuals vs fitted, raw vs log
  ├── fig_hyperparameter_table.png   ← visual HP table (ANN shown raw+log separately)
  ├── fig_ann_training_curve.png     ← ANN loss curves + stability flags (raw + log)
  ├── fig_log_comparison.png         ← log vs original (all 5 models)
  ├── fig_all_pred_vs_exp.png        ← 3×2 pred vs exp
  ├── fig_cv_residuals.png           ← CV folds + residual KDE
  ├── fig_feature_importance_lgbm.png← top {TOP_N_FEAT} LightGBM gain
  ├── fig_shap_combined.png          ← SHAP bar + beeswarm
  ├── fig_shap_dependence.png        ← SHAP dependence (top 4)
  ├── fig_williams_plot.png          ← Williams plot (train+test)
  ├── fig_ad_feature_shap_panel.png  ← 4-panel AD·SHAP·FeatImp
  ├── fig_outlier_overview_{ACTIVE_SET}.png     ← 4-panel: type-coded Williams/KDE/BP-bins/pred-vs-exp
  ├── fig_outlier_pca_{ACTIVE_SET}.png          ← PCA chemical space, AD & outlier-type membership
  ├── fig_outlier_top20_{ACTIVE_SET}.png        ← 20 worst-predicted compounds
  ├── fig_outlier_nn_distance_{ACTIVE_SET}.png  ← nearest-neighbour distance vs error
  ├── test_with_AD_{ACTIVE_SET}.csv
  ├── outliers_outside_AD_{ACTIVE_SET}.csv       ({len(outliers_df)} compounds)
  └── performance_summary_{ACTIVE_SET}.csv

  Descriptor set : {ACTIVE_SET}  ({p} descriptors, n_train={n_train:,}, n_test={X_test.shape[0]:,})
  Best model     : LightGBM{tag_str}  R²={best_res['test_r2']:.4f}  MAE={best_res['test_mae']:.3f} K
  ANN (raw)      : R²={results_raw['ANN']['test_r2']:.4f}   unstable={results_raw['ANN']['stability']['unstable']}
  ANN (log)      : R²={results_log['ANN']['test_r2']:.4f}   unstable={results_log['ANN']['stability']['unstable']}
  Homoscedasticity ρ (raw / log): {homoscedasticity_results['rho_raw']:+.3f} / {homoscedasticity_results['rho_log']:+.3f}
  AD             : {in_ad_test.mean()*100:.1f}% of test set within domain (s = CV RMSE = {s:.2f} K, NOT in-sample)
  Outlier types (test, outside AD) — A: {type_A.sum():,}  B: {type_B.sum():,}  C: {type_C.sum():,}
  NN-distance vs error correlation: r = {r_val:.3f}  (p = {pval:.2e})
  Auto-tune      : {'YES — Optuna 5-fold CV, ANN tuned separately for raw & log target' if AUTO_TUNE else 'NO — locked fallback parameters'}
""")
