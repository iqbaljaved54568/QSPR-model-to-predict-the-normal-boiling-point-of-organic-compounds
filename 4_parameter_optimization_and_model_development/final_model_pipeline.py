# =============================================================================
# Boiling Point Prediction — FULL PIPELINE ON OPTIMAL DESCRIPTOR SET
# Linear · Random Forest · XGBoost · LightGBM · ANN (raw + log)
# Optuna auto-tuning (5-fold CV) · Genuine best-model selection (not assumed
# to be LightGBM) · Model-adaptive SHAP · AD · Y-Randomization
#
# Run this AFTER backward_elimination_35desc.py and
# build_optimal_dataset_from_checkpoint.py have produced
# train_optimal.csv / test_optimal.csv.
# =============================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, probplot, pearsonr, skew, kurtosis, spearmanr
from rdkit import Chem
from scipy.ndimage import uniform_filter1d
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap
import joblib

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TRAIN_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/train_optimal.csv"
TEST_FILE  = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/test_optimal.csv"

TARGET_COL_INDEX       = 5
DESCRIPTOR_START_INDEX = 6

N_SPLITS     = 10
RANDOM_STATE = 42
SIGMA_LIMIT  = 3.0

OPTUNA_TRIALS     = 25
ANN_OPTUNA_TRIALS = 12
OPTUNA_CV_SPLITS  = 5

ANN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANN_EPOCHS, ANN_PATIENCE = 300, 30

OUT_DIR = "/content/drive/MyDrive/BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
os.makedirs(OUT_DIR, exist_ok=True)
TUNE_STORAGE_DIR = OUT_DIR
os.makedirs(TUNE_STORAGE_DIR, exist_ok=True)

print(f"Train file: {TRAIN_FILE}")
print(f"Test file : {TEST_FILE}")
print(f"ANN device: {ANN_DEVICE}")

# =============================================================================
# 2. STYLE
# =============================================================================

MODEL_COLORS  = {"Linear": "#4A90D9", "RandomForest": "#27AE60", "XGBoost": "#E67E22",
                 "LightGBM": "#C0392B", "ANN": "#8E44AD"}
MODEL_MARKERS = {"Linear": "o", "RandomForest": "s", "XGBoost": "^", "LightGBM": "D", "ANN": "P"}
PALETTE = {"light_bg": "#F7F9FB", "grid": "#DDE3EA", "text": "#1A2A3A",
           "perfect": "#2C3E50", "ad_fill": "#EBF5FB",
           "train_in": "#2C6E8A", "test_in": "#27AE60",
           "train_out": "#F39C12", "test_out": "#C0392B"}
TYPE_COLORS = {"A — High Leverage": "#E67E22", "B — High Residual": "#C0392B",
               "C — Both": "#8E44AD", "D — Within AD": "#27AE60"}
TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 8,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B0BEC5", "axes.linewidth": 0.8, "axes.facecolor": PALETTE["light_bg"],
    "figure.facecolor": "white", "grid.color": PALETTE["grid"], "grid.linewidth": 0.55,
    "grid.linestyle": "--", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.framealpha": 0.92, "legend.edgecolor": "#C5CDD5",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

def save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {path}")

# =============================================================================
# 3. LOAD DATA
# =============================================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_train = train_df.iloc[:, TARGET_COL_INDEX].values
X_test  = test_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_test  = test_df.iloc[:, TARGET_COL_INDEX].values
feature_names = list(train_df.columns[DESCRIPTOR_START_INDEX:])
n_train, p = X_train.shape
TOP_N_FEAT = min(20, p)

print(f"\n{'='*64}\n  Optimal-Descriptor Pipeline\n{'='*64}")
print(f"  Training : {n_train:,}  |  Test : {X_test.shape[0]:,}  |  Descriptors : {p}")
print(f"  BP range : {y_train.min():.1f} - {y_train.max():.1f} K")

# =============================================================================
# 4. TARGET DISTRIBUTION DIAGNOSTICS
# =============================================================================

def distribution_report(y, label):
    y = np.asarray(y, dtype=float); y_log = np.log(y)
    rows = [
        {"Transform": "Raw (K)", "Skewness": skew(y), "Kurtosis": kurtosis(y),
         "Min": y.min(), "Max": y.max(), "Mean": y.mean(), "Std": y.std()},
        {"Transform": "Log(K)", "Skewness": skew(y_log), "Kurtosis": kurtosis(y_log),
         "Min": y_log.min(), "Max": y_log.max(), "Mean": y_log.mean(), "Std": y_log.std()},
    ]
    df = pd.DataFrame(rows).set_index("Transform")
    print(f"\n  Target distribution — {label}\n{df.to_string(float_format=lambda v: f'{v:.4f}')}")
    return df

new_dist = distribution_report(y_train, f"optimal set (n={n_train})")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Target Distribution Diagnostics — Optimal Descriptor Set", fontsize=13, fontweight="bold", y=1.01)
y_log = np.log(y_train)
for ax, data, title, color in [
    (axes[0,0], y_train, f"Raw BP (K) | skew={skew(y_train):.3f} kurt={kurtosis(y_train):.3f}", "#4A90D9"),
    (axes[0,1], y_log,   f"Log BP | skew={skew(y_log):.3f} kurt={kurtosis(y_log):.3f}", "#C0392B"),
]:
    ax.hist(data, bins=30, color=color, alpha=0.75, edgecolor="white", density=True)
    ax.set_title(title, fontsize=10, fontstyle="italic")
    ax.grid(True); ax.set_axisbelow(True)
for ax, data, title in [(axes[1,0], y_train, "Q-Q Plot — Raw BP"), (axes[1,1], y_log, "Q-Q Plot — Log BP")]:
    probplot(data, dist="norm", plot=ax)
    ax.set_title(title, fontsize=10, fontstyle="italic")
    ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_target_distribution_diagnostics.png")
plt.show(); plt.close(fig)

# =============================================================================
# 5. SCALING
# =============================================================================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# 6. ANN ARCHITECTURE
# =============================================================================

ACT_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "selu": nn.SELU}

def build_ann(input_dim, hidden_layers, activation="tanh", dropout=0.2, batch_norm=False):
    act_fn = ACT_MAP.get(activation, nn.Tanh)
    layers, in_dim = [], input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        if batch_norm: layers.append(nn.BatchNorm1d(h))
        layers.append(act_fn())
        if dropout > 0: layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)

def train_ann(X_tr, y_tr, X_va, y_va, params, device="cpu", verbose=False):
    model = build_ann(X_tr.shape[1], params["hidden_layers"], params["activation"],
                       params["dropout"], batch_norm=params.get("batch_norm", False)).to(device)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=params["batch_size"], shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.5, min_lr=1e-6)
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
            tr_loss = criterion(model(X_tr_t), y_tr_t).item()
        train_hist.append(tr_loss); val_hist.append(val_loss)
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, patience_cnt = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= params["patience"]:
                if verbose: print(f"    Early stop @ epoch {epoch+1}")
                break
    model.load_state_dict(best_state)
    return model, train_hist, val_hist

def ann_predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()

def assess_training_stability(val_hist, tail_frac=0.2):
    vh = np.asarray(val_hist, dtype=float)
    n_tail = max(3, int(len(vh) * tail_frac))
    tail = vh[-n_tail:]
    diffs = np.diff(tail)
    oscillation_frac = float(np.mean(diffs > 0)) if len(diffs) > 0 else 0.0
    tail_rel_std = float(np.std(tail) / (np.mean(tail) + 1e-12))
    diverging = bool(vh[-1] > vh.min() * 1.5)
    unstable = bool(oscillation_frac > 0.4 or diverging or tail_rel_std > 0.15)
    return {"oscillation_frac": oscillation_frac, "tail_rel_std": tail_rel_std,
            "diverging": diverging, "unstable": unstable,
            "best_val": float(vh.min()), "final_val": float(vh[-1])}

def ann_cv(X_tr, y_tr, params, n_splits=5, rs=42, log_transform=False, device="cpu"):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    y_fit = np.log(y_tr) if log_transform else y_tr
    r2_list, mae_list, rmse_list = [], [], []
    preds_oof = np.zeros_like(y_tr, dtype=float)
    for tr_idx, va_idx in cv.split(X_tr):
        m, _, _ = train_ann(X_tr[tr_idx], y_fit[tr_idx], X_tr[va_idx], y_fit[va_idx], params, device=device)
        pred = ann_predict(m, X_tr[va_idx], device)
        if log_transform: pred = np.exp(pred)
        preds_oof[va_idx] = pred
        r2_list.append(r2_score(y_tr[va_idx], pred))
        mae_list.append(mean_absolute_error(y_tr[va_idx], pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_tr[va_idx], pred)))
    return np.array(r2_list), mae_list, rmse_list, preds_oof

# =============================================================================
# 7. OPTUNA AUTO-TUNING
# =============================================================================

def _study_storage(study_name):
    return f"sqlite:///{TUNE_STORAGE_DIR}{study_name}.db"

def tune_lgbm(X_tr, y_tr, log_transform, n_trials, cv_splits, rs):
    from lightgbm import early_stopping, log_evaluation
    y_fit = np.log(y_tr) if log_transform else y_tr
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": rs, "n_jobs": -1, "verbosity": -1,
        }
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m = LGBMRegressor(**params)
            m.fit(X_tr[tr_idx], y_fit[tr_idx], eval_set=[(X_tr[va_idx], y_fit[va_idx])],
                  callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)])
            pred = m.predict(X_tr[va_idx])
            fold_scores.append(r2_score(y_fit[va_idx], pred))
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune(): raise optuna.TrialPruned()
        return np.mean(fold_scores)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
                                 storage=_study_storage("optuna_lgbm_optimal"), study_name="lgbm_optimal", load_if_exists=True)
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [LightGBM] {len(study.trials)} trial(s) done, running {n_remaining} more ...")
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    best = study.best_params
    best.update({"random_state": rs, "n_jobs": -1, "verbosity": -1})
    return best

def tune_xgb(X_tr, y_tr, log_transform, n_trials, cv_splits, rs):
    y_fit = np.log(y_tr) if log_transform else y_tr
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": rs, "n_jobs": -1, "verbosity": 0, "tree_method": "hist",
            "early_stopping_rounds": 50,
        }
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m = XGBRegressor(**params)
            m.fit(X_tr[tr_idx], y_fit[tr_idx], eval_set=[(X_tr[va_idx], y_fit[va_idx])], verbose=False)
            pred = m.predict(X_tr[va_idx])
            fold_scores.append(r2_score(y_fit[va_idx], pred))
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune(): raise optuna.TrialPruned()
        return np.mean(fold_scores)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
                                 storage=_study_storage("optuna_xgb_optimal"), study_name="xgb_optimal", load_if_exists=True)
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [XGBoost] {len(study.trials)} trial(s) done, running {n_remaining} more ...")
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    best = study.best_params
    best.pop("early_stopping_rounds", None)
    best.update({"random_state": rs, "n_jobs": -1, "verbosity": 0, "tree_method": "hist"})
    return best

def tune_rf(X_tr, y_tr, log_transform, n_trials, cv_splits, rs):
    y_fit = np.log(y_tr) if log_transform else y_tr
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "random_state": rs, "n_jobs": -1, "bootstrap": True,
        }
        scores = cross_val_score(RandomForestRegressor(**params), X_tr, y_fit, cv=cv, scoring="r2", n_jobs=1)
        return scores.mean()
    study = optuna.create_study(direction="maximize", storage=_study_storage("optuna_rf_optimal"),
                                 study_name="rf_optimal", load_if_exists=True)
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [RandomForest] {len(study.trials)} trial(s) done, running {n_remaining} more ...")
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    best = study.best_params
    best.update({"random_state": rs, "n_jobs": -1, "bootstrap": True})
    return best

def tune_ann(X_tr, y_tr, log_transform, n_trials, cv_splits, rs, device):
    tag = "log" if log_transform else "raw"
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden = [trial.suggest_int(f"h{i}", 32, 512, step=32) for i in range(n_layers)]
        params = {
            "hidden_layers": hidden, "activation": "tanh",
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "weight_decay": trial.suggest_float("wd", 1e-5, 1e-2, log=True),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "grad_clip": trial.suggest_categorical("grad_clip", [None, 1.0, 5.0]),
            "epochs": 100, "patience": 15,
        }
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
        r2s = []
        y_fit = np.log(y_tr) if log_transform else y_tr
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_tr)):
            m, _, _ = train_ann(X_tr[tr_idx], y_fit[tr_idx], X_tr[va_idx], y_fit[va_idx], params, device=device)
            pred = ann_predict(m, X_tr[va_idx], device)
            if log_transform: pred = np.exp(pred)
            r2s.append(r2_score(y_tr[va_idx], pred))
            trial.report(np.mean(r2s), fold_i)
            if trial.should_prune(): raise optuna.TrialPruned()
        return np.mean(r2s)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
                                 storage=_study_storage(f"optuna_ann_{tag}_optimal"), study_name=f"ann_{tag}_optimal",
                                 load_if_exists=True)
    n_remaining = max(0, n_trials - len(study.trials))
    print(f"    [ANN-{tag}] {len(study.trials)} trial(s) done, running {n_remaining} more ...")
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    bp = study.best_params
    n_layers = bp["n_layers"]
    return {"hidden_layers": [bp[f"h{i}"] for i in range(n_layers)], "activation": "tanh",
            "dropout": bp["dropout"], "learning_rate": bp["lr"], "batch_size": bp["batch_size"],
            "weight_decay": bp["wd"], "batch_norm": bp["batch_norm"], "grad_clip": bp["grad_clip"],
            "epochs": ANN_EPOCHS, "patience": ANN_PATIENCE}

# =============================================================================
# 8. EVALUATION ENGINE
# =============================================================================

def compute_metrics(y_true, y_pred):
    return dict(r2=r2_score(y_true, y_pred), mae=mean_absolute_error(y_true, y_pred),
                rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
                mape=np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                bias=np.mean(y_pred - y_true))

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
        if log_transform: pred = np.exp(pred)
        cv_oof_pred[va_idx] = pred
        cv_mae_list.append(mean_absolute_error(y_tr[va_idx], pred))
        cv_rmse_list.append(np.sqrt(mean_squared_error(y_tr[va_idx], pred)))
    cv_rmse_orig = np.sqrt(mean_squared_error(y_tr, cv_oof_pred))

    model.fit(X_tr, y_tr_fit)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    if log_transform:
        y_pred_tr, y_pred_te = np.exp(y_pred_tr), np.exp(y_pred_te)

    tr_met, te_met = compute_metrics(y_tr, y_pred_tr), compute_metrics(y_te, y_pred_te)
    label = f"{name} (log)" if log_transform else name
    print(f"  {'-'*52}\n  {label}\n  CV  R2 : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}\n"
          f"  Test R2: {te_met['r2']:.4f}  MAE: {te_met['mae']:.3f} K  RMSE: {te_met['rmse']:.3f} K")

    return {"name": label, "base_name": name, "log_transform": log_transform,
            "model": model, "params": params_dict,
            "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(), "cv_r2_vals": cv_r2,
            "cv_mae": np.mean(cv_mae_list), "cv_rmse": np.mean(cv_rmse_list),
            "cv_rmse_orig": cv_rmse_orig, "cv_oof_pred": cv_oof_pred,
            "train_r2": tr_met["r2"], "test_r2": te_met["r2"], "test_mae": te_met["mae"],
            "test_rmse": te_met["rmse"], "test_mape": te_met["mape"], "test_bias": te_met["bias"],
            "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te,
            "residuals": y_te - y_pred_te, "errors": y_pred_te - y_te}

def run_ann_model(X_tr, y_tr, X_te, y_te, params_dict, log_transform=False, n_splits=10, rs=42, device="cpu"):
    label = "ANN (log)" if log_transform else "ANN"
    print(f"  {'-'*52}\n  {label}  [layers={params_dict['hidden_layers']}]")
    cv_r2, cv_mae_list, cv_rmse_list, cv_oof_pred = ann_cv(X_tr, y_tr, params_dict, n_splits, rs, log_transform, device)
    cv_rmse_orig = np.sqrt(mean_squared_error(y_tr, cv_oof_pred))

    val_size = max(1, int(0.1 * len(X_tr)))
    idx_perm = np.random.RandomState(rs).permutation(len(X_tr))
    va_idx, tr_idx = idx_perm[:val_size], idx_perm[val_size:]
    y_fit = np.log(y_tr) if log_transform else y_tr
    final_model, train_hist, val_hist = train_ann(X_tr[tr_idx], y_fit[tr_idx], X_tr[va_idx], y_fit[va_idx],
                                                    params_dict, device=device, verbose=True)
    y_pred_tr = ann_predict(final_model, X_tr, device)
    y_pred_te = ann_predict(final_model, X_te, device)
    if log_transform:
        y_pred_tr, y_pred_te = np.exp(y_pred_tr), np.exp(y_pred_te)
    tr_met, te_met = compute_metrics(y_tr, y_pred_tr), compute_metrics(y_te, y_pred_te)
    stability = assess_training_stability(val_hist)
    print(f"  CV  R2 : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}\n"
          f"  Test R2: {te_met['r2']:.4f}  MAE: {te_met['mae']:.3f} K  RMSE: {te_met['rmse']:.3f} K")
    if stability["unstable"]:
        print(f"  WARNING: {label} training curve looks unstable (oscillating/diverging).")

    return {"name": label, "base_name": "ANN", "log_transform": log_transform,
            "model": final_model, "params": params_dict, "train_hist": train_hist, "val_hist": val_hist,
            "stability": stability, "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(), "cv_r2_vals": cv_r2,
            "cv_mae": np.mean(cv_mae_list), "cv_rmse": np.mean(cv_rmse_list), "cv_rmse_orig": cv_rmse_orig,
            "cv_oof_pred": cv_oof_pred, "train_r2": tr_met["r2"], "test_r2": te_met["r2"],
            "test_mae": te_met["mae"], "test_rmse": te_met["rmse"], "test_mape": te_met["mape"],
            "test_bias": te_met["bias"], "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te,
            "residuals": y_te - y_pred_te, "errors": y_pred_te - y_te}

# =============================================================================
# 8b. HOMOSCEDASTICITY CHECK
# =============================================================================

def homoscedasticity_check(X_tr_sc, y_tr, rs=42, n_splits=10, save_path=None):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    oof_raw = cross_val_predict(Ridge(alpha=1.0, max_iter=10000), X_tr_sc, y_tr, cv=cv)
    resid_raw = y_tr - oof_raw
    oof_log = cross_val_predict(Ridge(alpha=1.0, max_iter=10000), X_tr_sc, np.log(y_tr), cv=cv)
    pred_log_bt = np.exp(oof_log)
    resid_log = y_tr - pred_log_bt
    rho_raw, _ = spearmanr(np.abs(resid_raw), oof_raw)
    rho_log, _ = spearmanr(np.abs(resid_log), pred_log_bt)
    print(f"\n  Homoscedasticity: raw rho={rho_raw:+.4f} (R2={r2_score(y_tr, oof_raw):.4f})  "
          f"log rho={rho_log:+.4f} (R2={r2_score(y_tr, pred_log_bt):.4f})")
    if abs(rho_raw) < 0.15:
        print("  NOTE: raw-target residuals already look close to homoscedastic (|rho|<0.15) --")
        print("        the log-transform's variance-stabilization rationale may not apply here.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Residuals vs Fitted — Linear, Raw vs Log (Optimal Set)", fontsize=13, fontweight="bold", y=1.02)
    for ax, fitted, resid, rho, title in [(axes[0], oof_raw, resid_raw, rho_raw, "Raw target"),
                                            (axes[1], pred_log_bt, resid_log, rho_log, "Log target (back-transformed)")]:
        ax.scatter(fitted, resid, s=10, alpha=0.5, color="#4A90D9", edgecolors="none")
        ax.axhline(0, color="#333", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Fitted BP (K)"); ax.set_ylabel("Residual (K)")
        ax.set_title(f"{title} (rho={rho:+.3f})", fontsize=10, fontstyle="italic")
        ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path: save_fig(fig, save_path)
    plt.show(); plt.close(fig)
    return {"rho_raw": rho_raw, "rho_log": rho_log}

homoscedasticity_results = homoscedasticity_check(X_train_sc, y_train, RANDOM_STATE, N_SPLITS,
                                                    f"{OUT_DIR}fig_homoscedasticity_check.png")

# =============================================================================
# 9. TRAIN ALL MODELS WITH LOCKED HYPERPARAMETERS, THEN PICK THE GENUINE BEST
#    OVERALL.
# =============================================================================

AUTO_TUNE = False

LOCKED_PARAMS = {
    "LGBM": {
        "n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
        "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
        "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
        "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1,
    },
    "XGB": {
        "n_estimators": 853, "max_depth": 5, "learning_rate": 0.013927409266246972,
        "subsample": 0.6376191271414946, "colsample_bytree": 0.9630582089322561,
        "reg_alpha": 0.042954597225917636, "reg_lambda": 0.6007292047894542,
        "min_child_weight": 8, "gamma": 0.002534291177932309,
        "random_state": 42, "n_jobs": -1, "verbosity": 0, "tree_method": "hist",
    },
    "RF": {
        "n_estimators": 1200, "max_depth": 16, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": 0.5,
        "random_state": 42, "n_jobs": -1, "bootstrap": True,
    },
    "ANN_RAW": {
        "hidden_layers": [224, 512, 512, 512], "activation": "tanh",
        "dropout": 0.05780630823856324, "learning_rate": 0.006334516693674627,
        "batch_size": 128, "weight_decay": 9.976043673518242e-05,
        "batch_norm": True, "grad_clip": 5.0, "epochs": ANN_EPOCHS, "patience": ANN_PATIENCE,
    },
    "ANN_LOG": {
        "hidden_layers": [352, 384], "activation": "tanh",
        "dropout": 0.18407386081838017, "learning_rate": 0.0007013466196636101,
        "batch_size": 32, "weight_decay": 0.00025362535610134546,
        "batch_norm": False, "grad_clip": 5.0, "epochs": ANN_EPOCHS, "patience": ANN_PATIENCE,
    },
}
LINEAR_PARAMS = {"alpha": 1.0, "fit_intercept": True, "max_iter": 10000}

if AUTO_TUNE:
    print(f"\n{'='*64}\n  AUTO-TUNING ({OPTUNA_TRIALS} trials/model, optimal set)\n{'='*64}")
    print("\n  Tuning LightGBM ...")
    LGBM_PARAMS = tune_lgbm(X_train_sc, y_train, True, OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE)
    LGBM_PARAMS.update({"verbosity": -1})
    print("  Tuning XGBoost ...")
    XGB_PARAMS = tune_xgb(X_train_sc, y_train, True, OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE)
    print("  Tuning Random Forest ...")
    RF_PARAMS = tune_rf(X_train_sc, y_train, True, OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE)
    print("  Tuning ANN (raw target) ...")
    ANN_PARAMS_RAW = tune_ann(X_train_sc, y_train, False, ANN_OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE, ANN_DEVICE)
    print("  Tuning ANN (log target) ...")
    ANN_PARAMS_LOG = tune_ann(X_train_sc, y_train, True, ANN_OPTUNA_TRIALS, OPTUNA_CV_SPLITS, RANDOM_STATE, ANN_DEVICE)
    print("\n  Tuning complete.")
else:
    print(f"\n{'='*64}\n  USING LOCKED HYPERPARAMETERS (tuning already completed)\n{'='*64}")
    LGBM_PARAMS    = LOCKED_PARAMS["LGBM"]
    XGB_PARAMS     = LOCKED_PARAMS["XGB"]
    RF_PARAMS      = LOCKED_PARAMS["RF"]
    ANN_PARAMS_RAW = LOCKED_PARAMS["ANN_RAW"]
    ANN_PARAMS_LOG = LOCKED_PARAMS["ANN_LOG"]

for nm, pd_ in [("LightGBM", LGBM_PARAMS), ("XGBoost", XGB_PARAMS), ("RandomForest", RF_PARAMS),
                ("ANN (raw)", ANN_PARAMS_RAW), ("ANN (log)", ANN_PARAMS_LOG)]:
    print(f"\n  {nm}: {pd_}")

results_raw, results_log = {}, {}
for transform, results_dict, lt in [("original", results_raw, False), ("log", results_log, True)]:
    print(f"\n{'-'*64}\n  Training — {transform} scale\n{'-'*64}")
    ann_params = ANN_PARAMS_LOG if lt else ANN_PARAMS_RAW
    results_dict["Linear"] = run_sklearn_model("Linear", Ridge(**LINEAR_PARAMS), LINEAR_PARAMS,
                                                X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE)
    results_dict["RandomForest"] = run_sklearn_model("RandomForest", RandomForestRegressor(**RF_PARAMS), RF_PARAMS,
                                                       X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE)
    results_dict["XGBoost"] = run_sklearn_model("XGBoost", XGBRegressor(**XGB_PARAMS), XGB_PARAMS,
                                                  X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE)
    results_dict["LightGBM"] = run_sklearn_model("LightGBM", LGBMRegressor(**LGBM_PARAMS), LGBM_PARAMS,
                                                   X_train_sc, y_train, X_test_sc, y_test, lt, N_SPLITS, RANDOM_STATE)
    results_dict["ANN"] = run_ann_model(X_train_sc, y_train, X_test_sc, y_test, ann_params, lt, N_SPLITS, RANDOM_STATE, ANN_DEVICE)

_all_candidates = []
for tag, rd in [("Original", results_raw), ("Log", results_log)]:
    for name, res in rd.items():
        _all_candidates.append((res["test_r2"], name, tag, res))
_all_candidates.sort(key=lambda t: t[0], reverse=True)
_best_r2, best_model_name, best_transform_tag, best_res = _all_candidates[0]
best_model = best_res["model"]
tag_str = " (Log)" if best_res["log_transform"] else ""

if best_model_name in TREE_MODELS:
    IMPORTANCE_LABEL = "Feature Importance (Gain)"
elif best_model_name == "Linear":
    IMPORTANCE_LABEL = "Feature Importance (|Coefficient|)"
else:
    IMPORTANCE_LABEL = "Feature Importance (n/a — ANN)"

print(f"\n  Best model overall: {best_model_name}{tag_str}  Test R2 = {best_res['test_r2']:.4f}")
print(f"  (chosen by comparing all 5 models x 2 transforms -- not assumed)")
print(f"\n  Full ranking (Test R2, all models x both transforms):")
for r2v, name, tag, _ in _all_candidates:
    marker = " <- BEST" if (name, tag) == (best_model_name, best_transform_tag) else ""
    print(f"    {name:14} {tag:9} Test R2 = {r2v:.4f}{marker}")

# =============================================================================
# 10. HYPERPARAMETER + PERFORMANCE SUMMARY TABLES
# =============================================================================

all_params = {"Linear (Ridge)": LINEAR_PARAMS, "Random Forest": RF_PARAMS, "XGBoost": XGB_PARAMS,
              "LightGBM": LGBM_PARAMS, "ANN (raw target)": ANN_PARAMS_RAW, "ANN (log target)": ANN_PARAMS_LOG}
hp_rows = [{"Model": m, "Hyperparameter": k, "Value": str(v)} for m, params in all_params.items() for k, v in params.items()]
hp_df = pd.DataFrame(hp_rows)
print(f"\n{'='*64}\n  HYPERPARAMETER TABLE — Optimal Descriptor Set\n{'='*64}")
print(hp_df.to_string(index=False))
hp_df.to_csv(f"{OUT_DIR}hyperparameter_table_optimal.csv", index=False)

perf_rows = []
for name in results_raw:
    for tag, res in [("Original", results_raw[name]), ("Log", results_log[name])]:
        perf_rows.append({"Model": name, "Transform": tag,
                           "CV_R2_mean": res["cv_r2_mean"], "CV_R2_std": res["cv_r2_std"],
                           "CV_MAE": res["cv_mae"], "CV_RMSE": res["cv_rmse"],
                           "Test_R2": res["test_r2"], "Test_MAE": res["test_mae"],
                           "Test_RMSE": res["test_rmse"], "Test_MAPE": res["test_mape"],
                           "Test_Bias": res["test_bias"]})
summary_df = pd.DataFrame(perf_rows)
print(f"\n{'='*90}\n  PERFORMANCE SUMMARY — Optimal Descriptor Set (p={p})\n{'='*90}")
print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
summary_df.to_csv(f"{OUT_DIR}performance_summary_optimal.csv", index=False)

# =============================================================================
# 11. FIGURES
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("ANN Learning Curves — Original vs Log Transform (Optimal Set)", fontsize=13, fontweight="bold", y=1.02)
for ax, res, title in [(axes[0], results_raw["ANN"], "Original Scale"), (axes[1], results_log["ANN"], "Log Transform")]:
    th, vh = res["train_hist"], res["val_hist"]
    epochs = np.arange(1, len(th) + 1)
    ax.plot(epochs, th, color=MODEL_COLORS["ANN"], linewidth=1.8, label="Train MSE")
    ax.plot(epochs, vh, color="#E67E22", linewidth=1.8, linestyle="--", label="Val MSE")
    ax.axvline(np.argmin(vh) + 1, color="#333", linewidth=1.0, linestyle=":", label=f"Best epoch={np.argmin(vh)+1}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.set_yscale("log")
    ax.set_title(f"ANN — {title} (R2={res['test_r2']:.4f})", fontsize=10, fontstyle="italic")
    st = res["stability"]
    flag = "UNSTABLE" if st["unstable"] else "stable"
    ax.text(0.02, 0.05, f"{flag}\noscillation={st['oscillation_frac']*100:.0f}%",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C5CDD5"))
    ax.legend(fontsize=8.5); ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout(); save_fig(fig, f"{OUT_DIR}fig_ann_training_curve.png"); plt.show(); plt.close(fig)

metrics, titles = ["test_r2", "test_mae", "test_rmse", "test_mape"], ["Test R2", "Test MAE (K)", "Test RMSE (K)", "Test MAPE (%)"]
names = list(results_raw.keys())
x, w = np.arange(len(names)), 0.35
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
fig.suptitle("Effect of Log-Transformation on Model Performance — Optimal Set", fontsize=13, fontweight="bold", y=1.02)
for ax, metric, title in zip(axes, metrics, titles):
    vals_raw = [results_raw[n][metric] for n in names]
    vals_log = [results_log[n][metric] for n in names]
    ax.bar(x - w/2, vals_raw, w, color=[MODEL_COLORS[n] for n in names], alpha=0.55, label="Original")
    ax.bar(x + w/2, vals_log, w, color=[MODEL_COLORS[n] for n in names], alpha=0.95, hatch="///", label="Log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(title, fontsize=10, fontstyle="italic"); ax.yaxis.grid(True); ax.set_axisbelow(True)
axes[0].legend(fontsize=8)
plt.tight_layout(); save_fig(fig, f"{OUT_DIR}fig_log_comparison.png"); plt.show(); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes.flat[-1].set_visible(False)
fig.suptitle("Predicted vs Experimental BP — Optimal Descriptor Set", fontsize=13, fontweight="bold", y=1.01)
for ax, name in zip(axes.flat[:5], names):
    res = results_log[name] if results_log[name]["test_r2"] > results_raw[name]["test_r2"] else results_raw[name]
    color = MODEL_COLORS[name]; y_pred = res["y_pred_test"]
    rmse, r2v, mae = res["test_rmse"], res["test_r2"], res["test_mae"]
    tag = " (log)" if res["log_transform"] else ""
    ax.scatter(y_test, y_pred, s=12, alpha=0.6, color=color, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min())-5, max(y_test.max(), y_pred.max())+5]
    ax.plot(lims, lims, "--", color=PALETTE["perfect"], linewidth=1.5)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.text(0.04, 0.97, f"R2={r2v:.4f}\nMAE={mae:.2f} K\nRMSE={rmse:.2f} K", transform=ax.transAxes,
            fontsize=8.5, va="top", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C5CDD5"))
    ax.set_title(f"{name}{tag}", fontsize=11, fontweight="bold", color=color)
    ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout(); save_fig(fig, f"{OUT_DIR}fig_all_pred_vs_exp.png"); plt.show(); plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
fold_nums = np.arange(1, N_SPLITS + 1)
for name in names:
    color, marker = MODEL_COLORS[name], MODEL_MARKERS[name]
    for tag, res_dict, ls in [("Orig", results_raw, "-"), ("Log", results_log, "--")]:
        res = res_dict[name]
        ax1.plot(fold_nums, res["cv_r2_vals"], color=color, linewidth=1.6, linestyle=ls,
                  marker=marker, markersize=5, label=f"{name} ({tag})")
ax1.set_xlabel("Fold"); ax1.set_ylabel("CV R2"); ax1.set_title("(A) 10-Fold CV R2", loc="left", fontstyle="italic")
ax1.legend(fontsize=7, ncol=2); ax1.yaxis.grid(True); ax1.set_axisbelow(True)
for name in names:
    color = MODEL_COLORS[name]
    res = results_log[name] if results_log[name]["test_r2"] > results_raw[name]["test_r2"] else results_raw[name]
    tag = " (log)" if res["log_transform"] else ""
    err = res["errors"]
    kde_x = np.linspace(err.min()-5, err.max()+5, 400)
    ax2.plot(kde_x, gaussian_kde(err, bw_method="scott")(kde_x), color=color, linewidth=2.2,
              label=f"{name}{tag} RMSE={res['test_rmse']:.2f} K")
ax2.axvline(0, color="#333", linewidth=1.5, linestyle="--")
ax2.set_xlabel("Prediction Error (K)"); ax2.set_ylabel("Density")
ax2.set_title("(B) Residual Error KDE", loc="left", fontstyle="italic")
ax2.legend(fontsize=8); ax2.yaxis.grid(True); ax2.set_axisbelow(True)
plt.tight_layout(); save_fig(fig, f"{OUT_DIR}fig_cv_residuals.png"); plt.show(); plt.close(fig)

# =============================================================================
# 12. FEATURE IMPORTANCE + SHAP
# =============================================================================

def plot_tree_importance(model, feature_names, top_n, save_path=None):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    feat_df = pd.DataFrame({"Feature": [feature_names[i] for i in idx], "Importance": imp[idx]}).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.42*top_n + 2.2))
    norm_v = feat_df["Importance"] / feat_df["Importance"].max()
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=plt.cm.RdYlBu_r(0.25 + 0.65*norm_v.values),
                    edgecolor="white", linewidth=0.5, height=0.72)
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(bar.get_width()+feat_df["Importance"].max()*0.012, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel(IMPORTANCE_LABEL)
    ax.set_title(f"Top {top_n} Descriptor Importances — {best_model_name}{tag_str} (Optimal Set)", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout(); save_fig(fig, save_path); plt.show(); plt.close(fig)
    return feat_df

def plot_linear_importance(model, feature_names, top_n, save_path=None):
    imp = np.abs(model.coef_)
    idx = np.argsort(imp)[::-1][:top_n]
    feat_df = pd.DataFrame({"Feature": [feature_names[i] for i in idx], "Importance": imp[idx]}).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.42*top_n + 2.2))
    norm_v = feat_df["Importance"] / feat_df["Importance"].max()
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=plt.cm.RdYlBu_r(0.25 + 0.65*norm_v.values),
                    edgecolor="white", linewidth=0.5, height=0.72)
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(bar.get_width()+feat_df["Importance"].max()*0.012, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel(IMPORTANCE_LABEL)
    ax.set_title(f"Top {top_n} Descriptor Importances — {best_model_name}{tag_str} (Optimal Set)", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout(); save_fig(fig, save_path); plt.show(); plt.close(fig)
    return feat_df

if best_model_name in TREE_MODELS:
    feat_df = plot_tree_importance(best_model, feature_names, TOP_N_FEAT,
                                    f"{OUT_DIR}fig_feature_importance_{best_model_name.lower()}.png")
    print("\n  Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_sc)
elif best_model_name == "Linear":
    feat_df = plot_linear_importance(best_model, feature_names, TOP_N_FEAT, f"{OUT_DIR}fig_feature_importance_linear.png")
    print("\n  Computing SHAP values (LinearExplainer) ...")
    explainer = shap.LinearExplainer(best_model, X_train_sc)
    shap_values = explainer.shap_values(X_train_sc)
else:
    print("\n  Best model is ANN -- skipping SHAP + gain-importance figures.")
    feat_df = pd.DataFrame({"Feature": [], "Importance": []})
    shap_values = None

if shap_values is not None:
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_feats = mean_shap.head(TOP_N_FEAT).index.tolist()
    print(f"  SHAP done. Top feature: {top_feats[0]}")
else:
    mean_shap = pd.Series(dtype=float)
    top_feats = []

def plot_shap_combined(shap_values, X_train_sc, mean_shap, feature_names, top_n, save_path=None):
    top_idx = [feature_names.index(f) for f in mean_shap.head(top_n).index]
    shap_top, X_top, feat_top = shap_values[:, top_idx], X_train_sc[:, top_idx], [feature_names[i] for i in top_idx]
    fig = plt.figure(figsize=(16, 0.45*top_n + 3))
    fig.suptitle(f"SHAP Analysis — {best_model_name}{tag_str} | Top {top_n} (Optimal Set)", fontsize=13, fontweight="bold", y=1.02)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45, left=0.05, right=0.97, top=0.92, bottom=0.10)
    ax_bar = fig.add_subplot(gs[0, 0])
    data = mean_shap.head(top_n).iloc[::-1]
    norm_v = (data - data.min()) / (data.max() - data.min())
    bars = ax_bar.barh(data.index, data.values, color=plt.cm.Oranges(0.35 + 0.60*norm_v.values), edgecolor="white", height=0.72)
    for bar, val in zip(bars, data.values):
        ax_bar.text(bar.get_width()+data.max()*0.015, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=7.8)
    ax_bar.set_xlabel("Mean |SHAP| (K)"); ax_bar.set_title("(A) Global Feature Importance", loc="left", fontstyle="italic")
    ax_bar.xaxis.grid(True); ax_bar.set_axisbelow(True)
    ax_bee = fig.add_subplot(gs[0, 1]); plt.sca(ax_bee)
    shap.summary_plot(shap_top, X_top, feature_names=feat_top, plot_type="dot", max_display=top_n,
                       show=False, color_bar=True, plot_size=None)
    ax_bee = plt.gca(); ax_bee.set_title("(B) Direction & Magnitude (Beeswarm)", loc="left", fontstyle="italic")
    ax_bee.set_xlabel("SHAP Value — Impact on BP (K)")
    save_fig(fig, save_path); plt.show(); plt.close(fig)

if shap_values is not None:
    plot_shap_combined(shap_values, X_train_sc, mean_shap, feature_names, TOP_N_FEAT, f"{OUT_DIR}fig_shap_combined.png")
else:
    print("  (Skipping fig_shap_combined.png -- no SHAP values for ANN.)")

# =============================================================================
# 13. CHECKPOINT
# =============================================================================

def _strip_for_pickle(res):
    drop = {"model", "train_hist", "val_hist"}
    return {k: v for k, v in res.items() if k not in drop}

checkpoint = {
    "X_train_sc": X_train_sc, "X_test_sc": X_test_sc, "y_train": y_train, "y_test": y_test,
    "feature_names": feature_names, "n_train": n_train, "p": p, "tag_str": tag_str,
    "best_model_name": best_model_name, "best_res": {k: v for k, v in best_res.items() if k != "model"},
    "results_raw": {n: _strip_for_pickle(r) for n, r in results_raw.items()},
    "results_log": {n: _strip_for_pickle(r) for n, r in results_log.items()},
    "feat_df": feat_df, "mean_shap": mean_shap, "SIGMA_LIMIT": SIGMA_LIMIT, "TOP_N_FEAT": TOP_N_FEAT,
    "test_df": test_df, "LGBM_PARAMS": LGBM_PARAMS, "XGB_PARAMS": XGB_PARAMS, "RF_PARAMS": RF_PARAMS,
    "LINEAR_PARAMS": LINEAR_PARAMS, "ANN_PARAMS_RAW": ANN_PARAMS_RAW, "ANN_PARAMS_LOG": ANN_PARAMS_LOG,
}
joblib.dump(checkpoint, f"{OUT_DIR}checkpoint_optimal.pkl")
print(f"\n  Checkpoint saved -> {OUT_DIR}checkpoint_optimal.pkl")

# =============================================================================
# 14. APPLICABILITY DOMAIN
# =============================================================================

def compute_leverage(X_ref, X_query):
    U, S, Vt = np.linalg.svd(X_ref, full_matrices=False)
    tol = S.max() * max(X_ref.shape) * np.finfo(float).eps * 100
    S_inv = np.where(S > tol, 1.0/S**2, 0.0)
    return (X_query @ Vt.T)**2 @ S_inv

h_train = compute_leverage(X_train_sc, X_train_sc)
h_test  = compute_leverage(X_train_sc, X_test_sc)
h_star  = 3.0 * (p + 1) / n_train
s = best_res["cv_rmse_orig"]
resid_train = y_train - best_res["cv_oof_pred"]
resid_test  = y_test  - best_res["y_pred_test"]
sr_train, sr_test = resid_train / s, resid_test / s
in_ad_train = (h_train <= h_star) & (np.abs(sr_train) <= SIGMA_LIMIT)
in_ad_test  = (h_test  <= h_star) & (np.abs(sr_test)  <= SIGMA_LIMIT)

print(f"\n  AD: h*={h_star:.5f}  Best model: {best_model_name}{tag_str}  s(CV RMSE)={s:.3f} K")
print(f"  Train AD={in_ad_train.mean()*100:.1f}%  Test AD={in_ad_test.mean()*100:.1f}%")

print(f"\n  {'':22}  {'R2':>7}  {'MAE (K)':>9}  {'RMSE (K)':>10}  {'N':>6}")
for mask, ad_label in [(in_ad_test, "Test - Within AD"), (~in_ad_test, "Test - Outside AD"),
                         (np.ones(len(y_test), dtype=bool), "Test - Overall")]:
    if mask.sum() < 2: continue
    _r2, _mae = r2_score(y_test[mask], best_res["y_pred_test"][mask]), mean_absolute_error(y_test[mask], best_res["y_pred_test"][mask])
    _rmse = np.sqrt(mean_squared_error(y_test[mask], best_res["y_pred_test"][mask]))
    print(f"  {ad_label:22}  {_r2:7.4f}  {_mae:9.3f}  {_rmse:10.3f}  {mask.sum():6,}")

mask_high_h, mask_high_sr = h_test > h_star, np.abs(sr_test) > SIGMA_LIMIT
type_A = mask_high_h & ~mask_high_sr
type_B = ~mask_high_h & mask_high_sr
type_C = mask_high_h & mask_high_sr
type_D = in_ad_test
outlier_type = np.where(type_D, "D — Within AD", np.where(type_A, "A — High Leverage",
                np.where(type_B, "B — High Residual", "C — Both")))
abs_error = np.abs(resid_test)
print(f"\n  Outlier types: A={type_A.sum()}  B={type_B.sum()}  C={type_C.sum()}")

fig, ax = plt.subplots(figsize=(11, 8))
sr_max = min(max(np.abs(sr_train).max(), np.abs(sr_test).max())*1.12, 12)
x_max = max(h_train.max(), h_test.max())*1.18
ax.fill_betweenx([-SIGMA_LIMIT, SIGMA_LIMIT], 0, h_star, color=PALETTE["ad_fill"], alpha=0.55, zorder=0)
ax.axvline(h_star, color=PALETTE["perfect"], linewidth=1.8, linestyle="--", label=f"h*={h_star:.5f}")
ax.axhline(SIGMA_LIMIT, color=PALETTE["perfect"], linewidth=1.4, linestyle=":", label=f"+/-{SIGMA_LIMIT:.0f}sigma ({SIGMA_LIMIT*s:.1f} K)")
ax.axhline(-SIGMA_LIMIT, color=PALETTE["perfect"], linewidth=1.4, linestyle=":")
ax.scatter(h_train[in_ad_train], sr_train[in_ad_train], color=PALETTE["train_in"], s=18, alpha=0.55, label=f"Train in AD ({in_ad_train.sum()})")
ax.scatter(h_train[~in_ad_train], sr_train[~in_ad_train], color=PALETTE["train_out"], marker="^", s=22, alpha=0.65, label=f"Train out ({(~in_ad_train).sum()})")
ax.scatter(h_test[in_ad_test], sr_test[in_ad_test], color=PALETTE["test_in"], s=18, alpha=0.55, label=f"Test in AD ({in_ad_test.sum()})")
ax.scatter(h_test[~in_ad_test], sr_test[~in_ad_test], color=PALETTE["test_out"], marker="s", s=22, alpha=0.75, label=f"Test out ({(~in_ad_test).sum()})")
ax.set_xlabel("Leverage $h_i$"); ax.set_ylabel("Standardised Residual")
ax.set_title(f"Williams Plot — {best_model_name}{tag_str} (Optimal Set, p={p})", fontsize=13, fontweight="bold")
ax.set_xlim(-0.002, x_max); ax.set_ylim(-sr_max, sr_max); ax.legend(fontsize=8.5); ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout(); save_fig(fig, f"{OUT_DIR}fig_williams_plot.png"); plt.show(); plt.close(fig)

fig = plt.figure(figsize=(16, 14))
fig.suptitle(f"{best_model_name}{tag_str} — AD, SHAP, Feature Importance (Optimal Set)", fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38, left=0.07, right=0.97, top=0.96, bottom=0.07)
ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i, j]) for i in (0,1) for j in (0,1)]
sr_max2 = min(np.abs(sr_test).max()*1.12, 10)
ax1.fill_betweenx([-SIGMA_LIMIT, SIGMA_LIMIT], 0, h_star, color=PALETTE["ad_fill"], alpha=0.55)
ax1.axvline(h_star, color=PALETTE["perfect"], linewidth=1.6, linestyle="--", label=f"h*={h_star:.5f}")
ax1.axhline(SIGMA_LIMIT, color=PALETTE["perfect"], linewidth=1.2, linestyle=":")
ax1.axhline(-SIGMA_LIMIT, color=PALETTE["perfect"], linewidth=1.2, linestyle=":")
ax1.scatter(h_test[in_ad_test], sr_test[in_ad_test], color=PALETTE["test_in"], s=14, alpha=0.55, label=f"Within ({in_ad_test.sum()})")
ax1.scatter(h_test[~in_ad_test], sr_test[~in_ad_test], color=PALETTE["test_out"], s=18, marker="^", alpha=0.7, label=f"Outside ({(~in_ad_test).sum()})")
ax1.set_xlabel("Leverage $h_i$"); ax1.set_ylabel("Std Residual")
ax1.set_title("(A) Williams Plot (Test)", loc="left", fontstyle="italic")
ax1.legend(fontsize=8); ax1.grid(True); ax1.set_axisbelow(True); ax1.set_ylim(-sr_max2, sr_max2)

for mask, label, color, mk in [(in_ad_test, "Within AD", PALETTE["test_in"], "o"), (~in_ad_test, "Outside AD", PALETTE["test_out"], "^")]:
    ax2.scatter(y_test[mask], best_res["y_pred_test"][mask], color=color, marker=mk, s=14, alpha=0.6, label=label)
lims = [min(y_test.min(), best_res["y_pred_test"].min())-5, max(y_test.max(), best_res["y_pred_test"].max())+5]
ax2.plot(lims, lims, "--", color=PALETTE["perfect"], linewidth=1.4)
ax2.set_xlim(lims); ax2.set_ylim(lims); ax2.set_aspect("equal")
ax2.set_xlabel("Experimental BP (K)"); ax2.set_ylabel("Predicted BP (K)")
ax2.set_title("(B) Pred vs Exp — AD Membership", loc="left", fontstyle="italic")
ax2.legend(fontsize=8.5); ax2.grid(True); ax2.set_axisbelow(True)

if len(mean_shap) > 0:
    shap_data = mean_shap.head(TOP_N_FEAT).iloc[::-1]
    norm_s = (shap_data-shap_data.min())/(shap_data.max()-shap_data.min())
    ax3.barh(shap_data.index, shap_data.values, color=plt.cm.Oranges(0.35+0.60*norm_s.values), edgecolor="white", height=0.72)
else:
    ax3.text(0.5, 0.5, "SHAP not computed for this model", ha="center", va="center", transform=ax3.transAxes)
ax3.set_xlabel("Mean |SHAP| (K)"); ax3.set_title(f"(C) SHAP Importance — Top {TOP_N_FEAT}", loc="left", fontstyle="italic")
ax3.xaxis.grid(True); ax3.set_axisbelow(True)

if len(feat_df) > 0:
    fi_data = feat_df.set_index("Feature")["Importance"].iloc[::-1]
    norm_f = (fi_data-fi_data.min())/(fi_data.max()-fi_data.min())
    ax4.barh(fi_data.index, fi_data.values, color=plt.cm.Blues(0.35+0.60*norm_f.values), edgecolor="white", height=0.72)
else:
    ax4.text(0.5, 0.5, "Feature importance not computed for this model", ha="center", va="center", transform=ax4.transAxes)
ax4.set_xlabel(IMPORTANCE_LABEL); ax4.set_title(f"(D) Feature Importance — Top {TOP_N_FEAT}", loc="left", fontstyle="italic")
ax4.xaxis.grid(True); ax4.set_axisbelow(True)
plt.tight_layout(); save_fig(fig, f"{OUT_DIR}fig_ad_feature_shap_panel.png"); plt.show(); plt.close(fig)

# =============================================================================
# 14b. OUTLIER OVERVIEW — 4 PANELS
# =============================================================================

fig_ov = plt.figure(figsize=(16, 12))
fig_ov.suptitle(f"Applicability Domain — Outlier Analysis\n{best_model_name}{tag_str} | Optimal Set",
                fontsize=14, fontweight="bold", y=1.01)
gs_ov = gridspec.GridSpec(2, 2, figure=fig_ov, hspace=0.40, wspace=0.35,
                          left=0.08, right=0.97, top=0.96, bottom=0.08)
axo1 = fig_ov.add_subplot(gs_ov[0,0]); axo2 = fig_ov.add_subplot(gs_ov[0,1])
axo3 = fig_ov.add_subplot(gs_ov[1,0]); axo4 = fig_ov.add_subplot(gs_ov[1,1])

axo1.fill_betweenx([-SIGMA_LIMIT, SIGMA_LIMIT], 0, h_star, color=PALETTE["ad_fill"], alpha=0.55, zorder=0)
axo1.axvline(h_star, color=PALETTE["perfect"], linewidth=1.6, linestyle="--", zorder=4, label=f"h* = {h_star:.5f}")
axo1.axhline(SIGMA_LIMIT, color="#E67E22", linewidth=1.3, linestyle=":", zorder=4, label=f"±{SIGMA_LIMIT:.0f}σ")
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
    axo1.scatter(h_test[mask_typ], sr_test[mask_typ], color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.35 if typ == "D — Within AD" else 0.80, edgecolors="none", rasterized=True,
                 zorder=5, label=f"{typ}  (n={mask_typ.sum()})")
axo1.set_xlabel("Leverage $h_i$", fontsize=10)
axo1.set_ylabel("Standardised Residual", fontsize=10)
axo1.set_title("(A)  Williams Plot — Outlier Types", loc="left", fontsize=10, fontstyle="italic")
axo1.set_xlim(-0.001, h_test.max()*1.18)
sr_lim = min(np.abs(sr_test).max()*1.15, 10)
axo1.set_ylim(-sr_lim, sr_lim)
axo1.legend(fontsize=7.5, loc="upper right"); axo1.grid(True); axo1.set_axisbelow(True)

for typ, mask_typ in [("D — Within AD", type_D), ("A — High Leverage", type_A),
                      ("B — High Residual", type_B), ("C — Both", type_C)]:
    if mask_typ.sum() < 3:
        continue
    err = abs_error[mask_typ]
    kde_x = np.linspace(0, err.max()*1.1, 300)
    try:
        kde_y = gaussian_kde(err, bw_method="scott")(kde_x)
        axo2.plot(kde_x, kde_y, color=TYPE_COLORS[typ], linewidth=2.0,
                  label=f"{typ}  MAE={err.mean():.1f} K  (n={mask_typ.sum()})")
        axo2.fill_between(kde_x, kde_y, alpha=0.10, color=TYPE_COLORS[typ])
    except Exception:
        axo2.axvline(err.mean(), color=TYPE_COLORS[typ], linewidth=2.0, linestyle="--", label=f"{typ}  n={mask_typ.sum()}")
axo2.axvline(3*s, color="#333", linewidth=1.4, linestyle=":", label=f"3σ = {3*s:.1f} K")
axo2.set_xlabel("Absolute Prediction Error (K)", fontsize=10)
axo2.set_ylabel("Probability Density", fontsize=10)
axo2.set_title("(B)  Error Distribution by Outlier Type", loc="left", fontsize=10, fontstyle="italic")
axo2.legend(fontsize=7.5); axo2.grid(True); axo2.set_axisbelow(True)

bins = [0, 300, 350, 400, 450, 500, 600, 700, 1000]
lbls = ["<300","300-350","350-400","400-450","450-500","500-600","600-700",">700"]
out_bins = pd.cut(y_test[~in_ad_test], bins=bins, labels=lbls)
all_bins = pd.cut(y_test, bins=bins, labels=lbls)
out_cnt = out_bins.value_counts().reindex(lbls, fill_value=0)
all_cnt = all_bins.value_counts().reindex(lbls, fill_value=0)
x_b, w_b = np.arange(len(lbls)), 0.35
axo3.bar(x_b - w_b/2, all_cnt.values, w_b, color="#5D8AA8", alpha=0.55, edgecolor="white", label="All test")
axo3.bar(x_b + w_b/2, out_cnt.values, w_b, color="#C0392B", alpha=0.85, edgecolor="white", label="Outliers")
for i, (all_v, out_v) in enumerate(zip(all_cnt.values, out_cnt.values)):
    pct = out_v / all_v * 100 if all_v > 0 else 0
    if out_v > 0:
        axo3.text(i + w_b/2, out_v + 0.5, f"{pct:.0f}%", ha="center", fontsize=7.5, color="#C0392B", fontweight="bold")
axo3.set_xticks(x_b); axo3.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8.5)
axo3.set_xlabel("Boiling Point Range (K)", fontsize=10)
axo3.set_ylabel("Count", fontsize=10)
axo3.set_title("(C)  Outlier Frequency by BP Range\n(% = fraction of range that are outliers)",
               loc="left", fontsize=10, fontstyle="italic")
axo3.legend(fontsize=8.5); axo3.yaxis.grid(True); axo3.set_axisbelow(True)

for typ, mask_typ, marker, sz in [
    ("D — Within AD",     type_D, "o", 10),
    ("A — High Leverage", type_A, "^", 35),
    ("B — High Residual", type_B, "s", 35),
    ("C — Both",          type_C, "X", 45),
]:
    if mask_typ.sum() == 0:
        continue
    axo4.scatter(y_test[mask_typ], best_res["y_pred_test"][mask_typ], color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.30 if typ == "D — Within AD" else 0.85, edgecolors="none", rasterized=True,
                 zorder=5, label=f"{typ}  (n={mask_typ.sum()})")
lims_ov = [y_test.min()-5, y_test.max()+5]
axo4.plot(lims_ov, lims_ov, "--", color=PALETTE["perfect"], linewidth=1.5, label="y = x")
axo4.fill_between(lims_ov, [l-3*s for l in lims_ov], [l+3*s for l in lims_ov], alpha=0.05, color="#C0392B", label=f"±3σ = ±{3*s:.0f} K")
axo4.set_xlim(lims_ov); axo4.set_ylim(lims_ov); axo4.set_aspect("equal")
axo4.set_xlabel("Experimental BP (K)", fontsize=10)
axo4.set_ylabel("Predicted BP (K)", fontsize=10)
axo4.set_title("(D)  Predicted vs Experimental — Outlier Types", loc="left", fontsize=10, fontstyle="italic")
axo4.legend(fontsize=7.5, loc="upper left"); axo4.grid(True); axo4.set_axisbelow(True)

plt.tight_layout()
save_fig(fig_ov, f"{OUT_DIR}fig_outlier_overview_optimal.png")
plt.show(); plt.close(fig_ov)

print("\n  Computing PCA and nearest-neighbour distances ...")
pca = PCA(n_components=2, random_state=42)
pca_all = pca.fit_transform(np.vstack([X_train_sc, X_test_sc]))
pca_train, pca_test = pca_all[:n_train], pca_all[n_train:]

fig_pca, axes_pca = plt.subplots(1, 2, figsize=(14, 6))
fig_pca.suptitle("Chemical Space Coverage — PCA of Descriptor Space (Optimal Set)", fontsize=13, fontweight="bold", y=1.02)

axp = axes_pca[0]
axp.scatter(pca_train[:, 0], pca_train[:, 1], s=7, alpha=0.20, color="#2C6E8A", rasterized=True, label="Train")
axp.scatter(pca_test[in_ad_test, 0], pca_test[in_ad_test, 1], s=12, alpha=0.50, color="#27AE60", rasterized=True, label="Test — Within AD")
axp.scatter(pca_test[~in_ad_test, 0], pca_test[~in_ad_test, 1], s=35, alpha=0.85, color="#C0392B", marker="^",
            edgecolors="white", linewidths=0.4, zorder=5, label="Test — Outside AD")
axp.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
axp.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
axp.set_title("(A)  AD Membership in Chemical Space", loc="left", fontsize=10, fontstyle="italic")
axp.legend(fontsize=8.5); axp.grid(True); axp.set_axisbelow(True)

axp2 = axes_pca[1]
axp2.scatter(pca_train[:, 0], pca_train[:, 1], s=5, alpha=0.12, color="#2C6E8A", rasterized=True, label="Train (background)")
for typ, mask_typ, marker, sz in [
    ("D — Within AD",     type_D, "o", 10),
    ("A — High Leverage", type_A, "^", 45),
    ("B — High Residual", type_B, "s", 45),
    ("C — Both",          type_C, "X", 60),
]:
    if mask_typ.sum() == 0:
        continue
    axp2.scatter(pca_test[mask_typ, 0], pca_test[mask_typ, 1], color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.25 if typ == "D — Within AD" else 0.85,
                 edgecolors="none" if typ == "D — Within AD" else "white",
                 linewidths=0.3, rasterized=True, zorder=5, label=f"{typ}  (n={mask_typ.sum()})")
axp2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
axp2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
axp2.set_title("(B)  Outlier Types in Chemical Space", loc="left", fontsize=10, fontstyle="italic")
axp2.legend(fontsize=7.5); axp2.grid(True); axp2.set_axisbelow(True)

plt.tight_layout()
save_fig(fig_pca, f"{OUT_DIR}fig_pca_chemical_space.png")
plt.show(); plt.close(fig_pca)

nn_model = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_train_sc)
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
    axnn.scatter(dists[mask_typ], abs_error[mask_typ], color=TYPE_COLORS[typ], marker=marker, s=sz,
                 alpha=0.25 if typ == "D — Within AD" else 0.85, edgecolors="none", rasterized=True,
                 label=f"{typ}  (n={mask_typ.sum()})", zorder=5)
axnn.axhline(3*s, color="#333", linewidth=1.3, linestyle=":", label=f"3σ = {3*s:.1f} K")
axnn.set_xlabel("Distance to Nearest Training Compound (scaled descriptor space)", fontsize=10)
axnn.set_ylabel("Absolute Prediction Error (K)", fontsize=10)
axnn.set_title(f"Nearest-Neighbour Distance vs Prediction Error (Optimal Set)", fontsize=12, fontweight="bold")
axnn.text(0.04, 0.96, f"Pearson r = {r_val:.3f}  (p = {pval:.2e})", transform=axnn.transAxes, fontsize=9, va="top",
          bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C5CDD5"))
axnn.legend(fontsize=8.5); axnn.grid(True); axnn.set_axisbelow(True)
plt.tight_layout()
save_fig(fig_nn, f"{OUT_DIR}fig_nn_distance_vs_error.png")
plt.show(); plt.close(fig_nn)

# =============================================================================
# 14c. ORGANIC COMPOUND CLASSIFICATION
# =============================================================================

print(f"\n{'='*64}\n  ORGANIC COMPOUND CLASSIFICATION\n{'='*64}")

def _find_smiles_col(df):
    for c in ["smiles", "SMILES", "Smiles", "canonical_smiles", "smi"]:
        if c in df.columns:
            return c
    return None

smiles_col = _find_smiles_col(test_df)
if smiles_col is None:
    print("  WARNING: no SMILES column found in test_df -- skipping compound classification.")
    test_out_class = pd.Series(["Unknown"] * len(test_df), index=test_df.index)
else:
    print(f"  Using SMILES column: '{smiles_col}'")

    CLASS_PATTERNS = [
        ("Carboxylic acid",         "[CX3](=O)[OX2H1]"),
        ("Ester",                   "[#6][CX3](=O)[OX2H0][#6]"),
        ("Amide",                   "[CX3](=O)[NX3]"),
        ("Aldehyde",                "[CX3H1](=O)[#6]"),
        ("Ketone",                  "[#6][CX3](=O)[#6]"),
        ("Nitrile",                 "[NX1]#[CX2]"),
        ("Nitro compound",          "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
        ("Amine",                   "[NX3;H2,H1,H0;!$(NC=O);!$(N=*)]"),
        ("Alcohol",                 "[OX2H][CX4]"),
        ("Ether",                   "[OD2]([#6])[#6]"),
        ("Halogenated hydrocarbon", "[F,Cl,Br,I]"),
        ("Alkyne",                  "[CX2]#[CX2]"),
        ("Aromatic hydrocarbon",    "c1ccccc1"),
        ("Alkene",                  "[CX3]=[CX3]"),
    ]
    _compiled = [(label, Chem.MolFromSmarts(smarts)) for label, smarts in CLASS_PATTERNS]

    def classify_compound(smi):
        try:
            mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        if mol is None:
            return "Unknown"
        for label, patt in _compiled:
            if patt is not None and mol.HasSubstructMatch(patt):
                return label
        return "Alkane / saturated hydrocarbon"

    test_out_class = test_df[smiles_col].apply(classify_compound)
    print(f"  Classified {len(test_out_class):,} test compounds into "
          f"{test_out_class.nunique()} classes.")
    print(test_out_class.value_counts().to_string())

# =============================================================================
# 15. EXPORT AD RESULTS
# =============================================================================

test_out = test_df.copy()
test_out["y_actual"] = y_test; test_out["y_predicted"] = best_res["y_pred_test"]
test_out["Residual_K"] = resid_test; test_out["Std_Residual"] = sr_test
test_out["Abs_Error_K"] = abs_error; test_out["Leverage_h"] = h_test
test_out["Within_AD"] = in_ad_test; test_out["Outlier_Type"] = outlier_type
test_out["Compound_Class"] = test_out_class.values
test_out.to_csv(f"{OUT_DIR}test_with_AD_optimal.csv", index=False)
print(f"\n  Saved -> {OUT_DIR}test_with_AD_optimal.csv")

outliers_df = test_out[~in_ad_test].copy().sort_values("Abs_Error_K", ascending=False)
outliers_df.to_csv(f"{OUT_DIR}outliers_outside_AD_optimal.csv", index=False)
print(f"  Saved -> {OUT_DIR}outliers_outside_AD_optimal.csv ({len(outliers_df)} compounds)")

# =============================================================================
# 15c. OUTLIER RATE BY COMPOUND CLASS
# =============================================================================

if smiles_col is not None:
    class_stats = (
        test_out.groupby("Compound_Class")
        .agg(N=("Compound_Class", "size"),
             N_outliers=("Within_AD", lambda s: int((~s).sum())),
             Mean_Abs_Error_K=("Abs_Error_K", "mean"),
             Mean_Std_Residual=("Std_Residual", "mean"))
        .reset_index()
    )
    class_stats["Outlier_rate_pct"] = class_stats["N_outliers"] / class_stats["N"] * 100
    class_stats = class_stats.sort_values("Outlier_rate_pct", ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}\n  OUTLIER RATE BY COMPOUND CLASS (Test set, n={len(test_out)})\n{'='*70}")
    print(class_stats.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    class_stats.to_csv(f"{OUT_DIR}class_outlier_analysis_optimal.csv", index=False)
    print(f"\n  Saved -> {OUT_DIR}class_outlier_analysis_optimal.csv")

    plot_df = class_stats[class_stats["N"] >= 3].copy()
    if len(plot_df) > 0:
        fig_cls, ax_cls = plt.subplots(figsize=(10, 0.5*len(plot_df) + 2))
        rate_range = max(plot_df["Outlier_rate_pct"].max() - plot_df["Outlier_rate_pct"].min(), 1e-9)
        norm_rate = (plot_df["Outlier_rate_pct"] - plot_df["Outlier_rate_pct"].min()) / rate_range
        bars = ax_cls.barh(plot_df["Compound_Class"], plot_df["Outlier_rate_pct"],
                            color=plt.cm.RdYlGn_r(0.25 + 0.70*norm_rate.values),
                            edgecolor="white", linewidth=0.5, height=0.7)
        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            ax_cls.text(bar.get_width() + plot_df["Outlier_rate_pct"].max()*0.015,
                        bar.get_y() + bar.get_height()/2,
                        f"{row['Outlier_rate_pct']:.1f}%  (n={int(row['N'])}, MAE={row['Mean_Abs_Error_K']:.1f} K)",
                        va="center", fontsize=8)
        ax_cls.set_xlabel("Outlier Rate (%)", fontsize=10)
        ax_cls.set_title(f"Outlier Rate by Organic Compound Class — {best_model_name}{tag_str} (Optimal Set)",
                          fontsize=12, fontweight="bold")
        ax_cls.xaxis.grid(True); ax_cls.set_axisbelow(True)
        plt.tight_layout()
        save_fig(fig_cls, f"{OUT_DIR}fig_outlier_rate_by_class_optimal.png")
        plt.show(); plt.close(fig_cls)
else:
    class_stats = pd.DataFrame()

# =============================================================================
# 15b. TOP 20 WORST PREDICTIONS — full table + annotated figure
# =============================================================================

top20 = outliers_df.head(20).copy()
if len(top20) > 0:
    top20_table = pd.DataFrame({
        "SMILES": top20[smiles_col].values if smiles_col is not None else ["N/A"] * len(top20),
        "Compound_Class": top20["Compound_Class"].values,
        "Exp_BP_K": top20["y_actual"].values,
        "Pred_BP_K": top20["y_predicted"].values,
        "Abs_Error_K": top20["Abs_Error_K"].values,
        "Z_score": top20["Std_Residual"].values,
        "Outlier_Type": top20["Outlier_Type"].values,
    })
    print(f"\n{'='*95}\n  TOP 20 WORST PREDICTIONS — Outside AD ({best_model_name}{tag_str}, Optimal Set)\n{'='*95}")
    print(top20_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    top20_table.to_csv(f"{OUT_DIR}top20_worst_predictions_optimal.csv", index=False)
    print(f"\n  Saved -> {OUT_DIR}top20_worst_predictions_optimal.csv")

    fig_top, axt = plt.subplots(figsize=(13, 8))
    colors_bar = [TYPE_COLORS.get(t, "#888") for t in top20["Outlier_Type"]]
    axt.barh(range(len(top20)), top20["Abs_Error_K"], color=colors_bar, edgecolor="white", linewidth=0.5, height=0.72)
    for i, (_, row) in enumerate(top20_table.iterrows()):
        axt.text(row["Abs_Error_K"] + top20_table["Abs_Error_K"].max()*0.012, i,
                 f"Exp={row['Exp_BP_K']:.0f}K  Pred={row['Pred_BP_K']:.0f}K  Z={row['Z_score']:+.2f}  [{row['Compound_Class']}]",
                 va="center", ha="left", fontsize=7.3)
    axt.axvline(3*s, color="#333", linewidth=1.4, linestyle=":", label=f"3σ threshold = {3*s:.1f} K")
    axt.set_yticks(range(len(top20)))
    ylabels = [(str(sm)[:26]+"…" if len(str(sm)) > 26 else str(sm)) for sm in top20_table["SMILES"]]
    axt.set_yticklabels(ylabels, fontsize=7.5)
    axt.invert_yaxis()
    axt.set_xlabel("Absolute Prediction Error (K)", fontsize=10)
    axt.set_title(f"Top 20 Worst Predicted Compounds — Outside AD ({best_model_name}{tag_str}, Optimal Set)",
                  fontsize=12, fontweight="bold")
    axt.set_xlim(0, top20["Abs_Error_K"].max()*1.55)
    legend_patches = [Patch(color=TYPE_COLORS[t], label=t) for t in
                      ["A — High Leverage", "B — High Residual", "C — Both"] if t in top20["Outlier_Type"].values]
    legend_patches.append(plt.Line2D([0], [0], color="#333", linestyle=":", linewidth=1.4, label=f"3σ = {3*s:.1f} K"))
    axt.legend(handles=legend_patches, fontsize=8.5, loc="lower right")
    axt.xaxis.grid(True); axt.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig_top, f"{OUT_DIR}fig_outlier_top20_optimal.png")
    plt.show(); plt.close(fig_top)
else:
    print("  (No compounds outside AD -- skipping top-20 table/figure.)")
    top20_table = pd.DataFrame()

if len(mean_shap) > 0:
    shap_table = pd.DataFrame({
        "Rank": range(1, TOP_N_FEAT + 1), "Descriptor": mean_shap.head(TOP_N_FEAT).index,
        "Mean_abs_SHAP": mean_shap.head(TOP_N_FEAT).values,
        "Importance": feat_df.set_index("Feature").reindex(mean_shap.head(TOP_N_FEAT).index)["Importance"].values,
    })
    print(f"\n  Top {TOP_N_FEAT} descriptors (SHAP + {IMPORTANCE_LABEL}):")
    print(shap_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    shap_table.to_csv(f"{OUT_DIR}shap_importance_table_optimal.csv", index=False)

# =============================================================================
# Y-RANDOMIZATION (Y-SCRAMBLING) VALIDATION
#
# Deliberately not run inline here. Y-randomization is a slow, many-iteration
# procedure (100+ scrambled refits) that has nothing to do with tuning or
# selecting the model above -- keeping it in this script made re-runs slower
# and the two analyses easy to conflate. Run it separately with:
#
#   python 5_applicability_domain_and_y_randomization/run_y_randomization.py \
#       --checkpoint checkpoint_optimal.pkl --out_dir ./yrand_outputs/
#
# It loads everything it needs (locked hyperparameters, scaled train/test
# data, the winning model's name) straight from the checkpoint saved in
# Section 13 above -- no retraining required, and no need to re-run this
# script to validate the model afterward.
# =============================================================================

print(f"\n{'='*64}\n  PIPELINE COMPLETE — OPTIMAL DESCRIPTOR SET\n{'='*64}")
print(f"""
  Descriptors    : {p}
  Best model     : {best_model_name}{tag_str}  Test R2={best_res['test_r2']:.4f}  MAE={best_res['test_mae']:.3f} K
  AD             : {in_ad_test.mean()*100:.1f}% of test set within domain
  Figures + CSVs saved to: {OUT_DIR}

  Y-randomization was not run as part of this script -- see
  5_applicability_domain_and_y_randomization/run_y_randomization.py
""")
