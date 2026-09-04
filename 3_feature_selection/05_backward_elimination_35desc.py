# BACKWARD FEATURE ELIMINATION (Stage_3 Desc_reduction)
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TRAIN_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/train_35desc.csv"
TEST_FILE  = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/test_35desc.csv"

TARGET_COL_INDEX       = 5
DESCRIPTOR_START_INDEX = 6

RANDOM_STATE  = 42
CV_SPLITS_RFE = 5


OUT_DIR = "/content/drive/MyDrive/BP_FINAL_MODEL/figures/backward_elimination_35desc/"
os.makedirs(OUT_DIR, exist_ok=True)

ANN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which models to include in the sweep. Turn ANN off first if you just want a
# quick look — it's by far the slowest part.
# ANN_log dropped: log transform was found to significantly hurt R^2 for this
# descriptor set, so only the raw-target ANN is swept. ANN_PARAMS_LOG is kept
# below (unused) in case you want to re-enable it later.
INCLUDE_MODELS = ["Linear", "RandomForest", "XGBoost", "LightGBM", "ANN_raw"]

# =============================================================================
# 2. LOCKED HYPERPARAMETERS (as supplied — treated as fixed for this sweep)
# =============================================================================

LGBM_PARAMS = {
    "n_estimators": 2835, "learning_rate": 0.026811730129178632, "num_leaves": 145,
    "max_depth": 4, "min_child_samples": 33, "subsample": 0.9509625022217238,
    "colsample_bytree": 0.7544040195312076, "reg_alpha": 0.002770059887158264,
    "reg_lambda": 4.0236971995361515, "random_state": 42, "n_jobs": -1, "verbosity": -1,
}

XGB_PARAMS = {
    "n_estimators": 659, "max_depth": 4, "learning_rate": 0.01895242515278871,
    "subsample": 0.8310238495458468, "colsample_bytree": 0.7698722276112926,
    "reg_alpha": 0.0035336141962440154, "reg_lambda": 0.003230321800809352,
    "min_child_weight": 9, "gamma": 0.008093148207921756,
    "random_state": 42, "n_jobs": -1, "verbosity": 0, "tree_method": "hist",
}

RF_PARAMS = {
    "n_estimators": 871, "max_depth": 28, "min_samples_split": 2,
    "min_samples_leaf": 1, "max_features": 0.5,
    "random_state": 42, "n_jobs": -1, "bootstrap": True,
}

LINEAR_PARAMS = {"alpha": 1.0, "fit_intercept": True, "max_iter": 10000}

ANN_PARAMS_RAW = {
    "hidden_layers": [224, 224, 64, 512], "activation": "tanh",
    "dropout": 0.31349354716246197, "learning_rate": 0.005051372065900885,
    "batch_size": 64, "weight_decay": 0.0009527245200758531,
    "batch_norm": True, "grad_clip": 1.0, "epochs": 300, "patience": 30,
}

ANN_PARAMS_LOG = {
    "hidden_layers": [480, 480, 320, 512], "activation": "tanh",
    "dropout": 0.18048147781772694, "learning_rate": 0.0016358694582320331,
    "batch_size": 64, "weight_decay": 4.866604105656277e-05,
    "batch_norm": False, "grad_clip": None, "epochs": 300, "patience": 30,
}
# During the elimination SWEEP only (not the final check), epochs/patience are
# capped lower for speed. The dicts above are used as-is for the ONE-TIME
# full-feature model used to build the permutation-importance ranking.
ANN_SWEEP_EPOCHS   = 150
ANN_SWEEP_PATIENCE  = 20

MODEL_COLORS = {
    "Linear": "#4A90D9", "RandomForest": "#27AE60", "XGBoost": "#E67E22",
    "LightGBM": "#C0392B", "ANN_raw": "#8E44AD", "ANN_log": "#5B2C6F",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "#F7F9FB", "figure.facecolor": "white",
    "grid.color": "#DDE3EA", "grid.linewidth": 0.6, "grid.linestyle": "--",
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

X_train_raw = train_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_train     = train_df.iloc[:, TARGET_COL_INDEX].values
X_test_raw  = test_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_test      = test_df.iloc[:, TARGET_COL_INDEX].values
feature_names = list(train_df.columns[DESCRIPTOR_START_INDEX:])
p = X_train_raw.shape[1]

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train_raw)
X_test   = scaler.transform(X_test_raw)

print(f"Loaded 35-descriptor set: n_train={len(y_train)}, n_test={len(y_test)}, p={p}")

# =============================================================================
# 4. FEATURE-COUNT GRID (backward schedule)
# =============================================================================

def build_feature_grid(p_start, stops=None):
    """Descending list of feature counts at which we record a data point.
    Rationale for using the 35-descriptor set as the starting point rather
    than 94: prior full-model runs showed negligible performance difference
    between the 94- and 35-descriptor sets, so the far cheaper 35-descriptor
    set is used as the base, and backward elimination explores whether an
    even smaller subset holds up just as well.
    Schedule: 3-at-a-time down to 15 descriptors, then 1-at-a-time down to a
    single descriptor -- coarse enough to keep runtime down, fine enough near
    the low end (where the answer to "how few can I get away with" lives)."""
    if stops is None:
        stops = [(p_start, 15, 3), (15, 1, 1)]
    grid = {p_start}
    n = p_start
    for _, lo, step in stops:
        while n > lo:
            n = max(lo, n - step)
            grid.add(n)
    return sorted(grid, reverse=True)

FEATURE_GRID = build_feature_grid(p)
print(f"Feature-count checkpoints ({len(FEATURE_GRID)} points): {FEATURE_GRID}")

# =============================================================================
# 5. TRANSFORM SELECTION (for Linear/RF/XGB/LightGBM only)
#    Only one hyperparameter set was supplied per model, so we check once,
#    at full feature count, whether raw or log target performs better under
#    those locked hyperparameters, and lock that choice for the whole sweep.
# =============================================================================

def quick_transform_check(build_fn, X, y):
    cv = KFold(n_splits=CV_SPLITS_RFE, shuffle=True, random_state=RANDOM_STATE)
    r2_raw = cross_val_score(build_fn(), X, y, cv=cv, scoring="r2", n_jobs=1).mean()
    r2_log = cross_val_score(build_fn(), X, np.log(y), cv=cv, scoring="r2", n_jobs=1).mean()
    # r2_log above is on the log scale (fair only as a relative signal here,
    # not compared directly to r2_raw's K-scale R^2 in absolute terms, but a
    # log-target CV R^2 that is much higher/lower than the raw-target one is
    # still informative for THIS purpose: is the model even trainable well
    # after transform, given these fixed hyperparameters).
    # For a scale-fair comparison, back-transform log predictions and rescore:
    y_pred_log_oof = np.zeros_like(y, dtype=float)
    for tr_idx, va_idx in cv.split(X):
        m = build_fn()
        m.fit(X[tr_idx], np.log(y[tr_idx]))
        y_pred_log_oof[va_idx] = np.exp(m.predict(X[va_idx]))
    r2_log_k = r2_score(y, y_pred_log_oof)
    print(f"    raw CV R^2={r2_raw:.4f}   log CV R^2 (back-transformed, K-scale)={r2_log_k:.4f}")
    return "log" if r2_log_k > r2_raw else "original"

# =============================================================================
# 6. RECURSIVE BACKWARD ELIMINATION — Linear / RF / XGBoost / LightGBM
# =============================================================================

def build_ridge():   return Ridge(**LINEAR_PARAMS)
def build_rf():       return RandomForestRegressor(**RF_PARAMS)
def build_xgb():      return XGBRegressor(**XGB_PARAMS)
def build_lgbm():     return LGBMRegressor(**LGBM_PARAMS)

TREE_MODEL_BUILDERS = {
    "Linear": build_ridge, "RandomForest": build_rf,
    "XGBoost": build_xgb, "LightGBM": build_lgbm,
}

def recursive_backward_elimination(model_name, build_fn, X, y, X_te, y_te,
                                    feature_grid, log_transform, cv_splits, rs=42):
    remaining = list(range(X.shape[1]))
    grid_set  = set(feature_grid)
    records   = []
    y_fit_full = np.log(y) if log_transform else y

    while True:
        n_current = len(remaining)
        X_sub = X[:, remaining]

        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rs)
        cv_r2 = cross_val_score(build_fn(), X_sub, y_fit_full, cv=cv, scoring="r2", n_jobs=1)

        model = build_fn()
        model.fit(X_sub, y_fit_full)
        importances = model.feature_importances_ if hasattr(model, "feature_importances_") \
            else np.abs(model.coef_)

        y_pred_te = model.predict(X_te[:, remaining])
        if log_transform:
            y_pred_te = np.exp(y_pred_te)

        if n_current in grid_set:
            records.append({
                "model": model_name, "log_transform": log_transform,
                "n_features": n_current,
                "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
                "test_r2": r2_score(y_te, y_pred_te),
                "test_mae": mean_absolute_error(y_te, y_pred_te),
                "test_rmse": np.sqrt(mean_squared_error(y_te, y_pred_te)),
                "features": [feature_names[i] for i in remaining],
            })
            print(f"    [{model_name}] n={n_current:3d}  CV R2={cv_r2.mean():.4f}  Test R2={records[-1]['test_r2']:.4f}")

        if n_current <= 1:
            break
        next_targets = [g for g in feature_grid if g < n_current]
        next_n  = max(next_targets) if next_targets else 1
        n_drop  = n_current - next_n
        drop_positions = np.argsort(importances)[:n_drop]
        drop_global = {remaining[i] for i in drop_positions}
        remaining = [i for i in remaining if i not in drop_global]

    return pd.DataFrame(records)

# =============================================================================
# 7. ANN — build/train helpers (same architecture logic as main pipeline)
# =============================================================================

ACT_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "selu": nn.SELU}

def build_ann(input_dim, hidden_layers, activation="tanh", dropout=0.2, batch_norm=False):
    act_fn = ACT_MAP.get(activation, nn.Tanh)
    layers, in_dim = [], input_dim
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

def train_ann(X_tr, y_tr, X_va, y_va, params, device="cpu"):
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
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, patience_cnt = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= params["patience"]:
                break
    model.load_state_dict(best_state)
    return model

def ann_predict(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()

def permutation_importance_ann(model, X_va, y_va_raw, device, log_transform, n_repeats=3, rs=42):
    base_pred = ann_predict(model, X_va, device)
    if log_transform:
        base_pred = np.exp(base_pred)
    base_mse = mean_squared_error(y_va_raw, base_pred)
    rng = np.random.RandomState(rs)
    importances = np.zeros(X_va.shape[1])
    for j in range(X_va.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_va.copy()
            rng.shuffle(X_perm[:, j])
            pred = ann_predict(model, X_perm, device)
            if log_transform:
                pred = np.exp(pred)
            scores.append(mean_squared_error(y_va_raw, pred))
        importances[j] = np.mean(scores) - base_mse
    return importances

def static_backward_elimination_ann(model_name, params, X, y_raw, X_te, y_te, feature_grid,
                                     log_transform, device, rs=42, val_frac=0.15):
    idx_perm = np.random.RandomState(rs).permutation(len(X))
    n_val = max(1, int(val_frac * len(X)))
    va_idx, tr_idx = idx_perm[:n_val], idx_perm[n_val:]
    y_fit_full = np.log(y_raw) if log_transform else y_raw

    print(f"    [{model_name}] building full-feature model for permutation ranking ...")
    m0 = train_ann(X[tr_idx], y_fit_full[tr_idx], X[va_idx], y_fit_full[va_idx], params, device=device)
    importances = permutation_importance_ann(m0, X[va_idx], y_raw[va_idx], device, log_transform)
    ranking = list(np.argsort(importances))  # ascending: least important first

    sweep_params = dict(params)
    sweep_params["epochs"], sweep_params["patience"] = ANN_SWEEP_EPOCHS, ANN_SWEEP_PATIENCE

    remaining = list(range(X.shape[1]))
    grid_set  = set(feature_grid)
    records   = []

    while True:
        n_current = len(remaining)
        if n_current in grid_set:
            X_tr_sub, X_va_sub, X_te_sub = X[tr_idx][:, remaining], X[va_idx][:, remaining], X_te[:, remaining]
            m = train_ann(X_tr_sub, y_fit_full[tr_idx], X_va_sub, y_fit_full[va_idx], sweep_params, device=device)

            y_pred_va = ann_predict(m, X_va_sub, device)
            y_pred_te = ann_predict(m, X_te_sub, device)
            if log_transform:
                y_pred_va, y_pred_te = np.exp(y_pred_va), np.exp(y_pred_te)

            records.append({
                "model": model_name, "log_transform": log_transform,
                "n_features": n_current,
                "cv_r2_mean": r2_score(y_raw[va_idx], y_pred_va),  # holdout, not true CV -- see header note
                "cv_r2_std": np.nan,
                "test_r2": r2_score(y_te, y_pred_te),
                "test_mae": mean_absolute_error(y_te, y_pred_te),
                "test_rmse": np.sqrt(mean_squared_error(y_te, y_pred_te)),
                "features": [feature_names[i] for i in remaining],
            })
            print(f"    [{model_name}] n={n_current:3d}  Holdout R2={records[-1]['cv_r2_mean']:.4f}  Test R2={records[-1]['test_r2']:.4f}")

        if n_current <= 1:
            break
        next_targets = [g for g in feature_grid if g < n_current]
        next_n = max(next_targets) if next_targets else 1
        n_drop = n_current - next_n
        drop_set, dropped = set(), 0
        for idx in ranking:
            if idx in remaining and dropped < n_drop:
                drop_set.add(idx); dropped += 1
            if dropped >= n_drop:
                break
        remaining = [i for i in remaining if i not in drop_set]

    return pd.DataFrame(records)

# =============================================================================
# 8. RUN THE SWEEP
# =============================================================================

all_results = []

for name in INCLUDE_MODELS:
    if name in TREE_MODEL_BUILDERS:
        build_fn = TREE_MODEL_BUILDERS[name]
        print(f"\nChecking raw vs. log target for {name} (full 35-feature set) ...")
        best_transform = quick_transform_check(build_fn, X_train, y_train)
        log_flag = (best_transform == "log")
        print(f"  -> locking '{best_transform}' target for {name}'s full sweep")

        print(f"Running recursive backward elimination for {name} ...")
        t0 = time.time()
        df_res = recursive_backward_elimination(
            name, build_fn, X_train, y_train, X_test, y_test,
            FEATURE_GRID, log_flag, CV_SPLITS_RFE, RANDOM_STATE
        )
        print(f"  {name} sweep done in {time.time()-t0:.1f}s")
        all_results.append(df_res)

    elif name == "ANN_raw":
        print(f"\nRunning static backward elimination for ANN (raw target) ...")
        t0 = time.time()
        df_res = static_backward_elimination_ann(
            "ANN_raw", ANN_PARAMS_RAW, X_train, y_train, X_test, y_test,
            FEATURE_GRID, log_transform=False, device=ANN_DEVICE, rs=RANDOM_STATE
        )
        print(f"  ANN_raw sweep done in {time.time()-t0:.1f}s")
        all_results.append(df_res)

results_df = pd.concat(all_results, ignore_index=True)
results_df.drop(columns=["features"]).to_csv(f"{OUT_DIR}backward_elimination_results_35desc.csv", index=False)
print(f"\nSaved -> {OUT_DIR}backward_elimination_results_35desc.csv")

# Full record of which descriptors survive at each step, per model. Feature
# lists are joined with '|' in the summary CSV (easy to eyeball / filter),
# plus a long-format version (one row per model/n_features/feature) for
# programmatic use (e.g. checking if a specific descriptor was ever dropped).
results_df_with_feats = results_df.copy()
results_df_with_feats["features_kept"] = results_df_with_feats["features"].apply(lambda f: "|".join(f))
results_df_with_feats.drop(columns=["features"]).to_csv(
    f"{OUT_DIR}backward_elimination_features_by_step_35desc.csv", index=False
)
print(f"Saved -> {OUT_DIR}backward_elimination_features_by_step_35desc.csv")

long_rows = []
for _, row in results_df.iterrows():
    for feat in row["features"]:
        long_rows.append({"model": row["model"], "n_features": row["n_features"], "feature": feat})
pd.DataFrame(long_rows).to_csv(f"{OUT_DIR}backward_elimination_features_long_35desc.csv", index=False)
print(f"Saved -> {OUT_DIR}backward_elimination_features_long_35desc.csv")

# =============================================================================
# 9. FIGURE — R^2 vs number of descriptors (the headline plot)
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 7))
for name in results_df["model"].unique():
    sub = results_df[results_df["model"] == name].sort_values("n_features")
    ax.plot(sub["n_features"], sub["test_r2"], marker="o", markersize=4,
            color=MODEL_COLORS.get(name, "#333"), linewidth=1.8, label=f"{name} (Test)")
    ax.fill_between(sub["n_features"],
                     sub["cv_r2_mean"] - sub["cv_r2_std"].fillna(0),
                     sub["cv_r2_mean"] + sub["cv_r2_std"].fillna(0),
                     color=MODEL_COLORS.get(name, "#333"), alpha=0.08)
    ax.plot(sub["n_features"], sub["cv_r2_mean"], linestyle="--", linewidth=1.0,
            color=MODEL_COLORS.get(name, "#333"), alpha=0.6, label=f"{name} (CV/Holdout)")

ax.set_xlabel("Number of Descriptors Retained", fontsize=11)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("Backward Feature Elimination — R² vs. Number of Descriptors (from 35)",
             fontsize=13, fontweight="bold")
ax.invert_xaxis()  # 35 -> 1, reading left to right as elimination progresses
ax.legend(fontsize=7.5, ncol=2, loc="lower left")
ax.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_r2_vs_nfeatures.png")
plt.show(); plt.close(fig)

# =============================================================================
# 10. FIGURE — RMSE and MAE vs number of descriptors
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for name in results_df["model"].unique():
    sub = results_df[results_df["model"] == name].sort_values("n_features")
    c = MODEL_COLORS.get(name, "#333")
    axes[0].plot(sub["n_features"], sub["test_rmse"], marker="o", markersize=4, color=c, label=name)
    axes[1].plot(sub["n_features"], sub["test_mae"],  marker="o", markersize=4, color=c, label=name)

for ax, title, ylabel in [(axes[0], "Test RMSE vs. Descriptors", "RMSE (K)"),
                          (axes[1], "Test MAE vs. Descriptors", "MAE (K)")]:
    ax.set_xlabel("Number of Descriptors Retained", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=11, fontstyle="italic")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True); ax.set_axisbelow(True)

fig.suptitle("Backward Feature Elimination — Error Metrics vs. Number of Descriptors",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_rmse_mae_vs_nfeatures.png")
plt.show(); plt.close(fig)

# =============================================================================
# 11. OPTIMAL N PER MODEL — smallest n reaching ~99% of that model's own peak R^2
# =============================================================================

TOLERANCE = 0.99  # smallest n_features whose test R^2 is >= 99% of the model's best R^2

summary_rows = []
for name in results_df["model"].unique():
    sub = results_df[results_df["model"] == name].sort_values("n_features", ascending=False)
    peak_r2 = sub["test_r2"].max()
    threshold = peak_r2 * TOLERANCE if peak_r2 > 0 else peak_r2 / TOLERANCE
    candidates = sub[sub["test_r2"] >= threshold]
    best_row = candidates.sort_values("n_features").iloc[0]  # smallest n meeting threshold
    summary_rows.append({
        "model": name, "peak_test_r2": peak_r2, "peak_at_n": sub.loc[sub["test_r2"].idxmax(), "n_features"],
        "optimal_n_99pct": best_row["n_features"], "test_r2_at_optimal_n": best_row["test_r2"],
        "test_rmse_at_optimal_n": best_row["test_rmse"], "test_mae_at_optimal_n": best_row["test_mae"],
        "features_at_optimal_n": "|".join(best_row["features"]),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("optimal_n_99pct")
summary_df.to_csv(f"{OUT_DIR}optimal_n_summary_35desc.csv", index=False)
print(f"\nSaved -> {OUT_DIR}optimal_n_summary_35desc.csv")
print(summary_df.drop(columns=["features_at_optimal_n"]).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = [MODEL_COLORS.get(m, "#333") for m in summary_df["model"]]
bars = ax.bar(summary_df["model"], summary_df["optimal_n_99pct"], color=colors, edgecolor="white")
for bar, n_opt, r2 in zip(bars, summary_df["optimal_n_99pct"], summary_df["test_r2_at_optimal_n"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"n={n_opt}\nR²={r2:.3f}", ha="center", fontsize=8.5)
ax.set_ylabel("Smallest n Reaching 99% of Peak Test R²", fontsize=10.5)
ax.set_title("Optimal Descriptor Count per Model (99%-of-peak-R² rule)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y"); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_optimal_n_summary.png")
plt.show(); plt.close(fig)

print(f"\n{'='*64}")
print("BACKWARD ELIMINATION SWEEP COMPLETE")
print(f"{'='*64}")
print(summary_df.drop(columns=["features_at_optimal_n"]).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = [MODEL_COLORS.get(m, "#333") for m in summary_df["model"]]
bars = ax.bar(summary_df["model"], summary_df["optimal_n_99pct"], color=colors, edgecolor="white")
max_height = summary_df["optimal_n_99pct"].max()
label_offset = max_height * 0.03
for bar, n_opt, r2 in zip(bars, summary_df["optimal_n_99pct"], summary_df["test_r2_at_optimal_n_99pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset,
            f"n={n_opt}\nR²={r2:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, max_height * 1.25)
ax.set_ylabel("Smallest n Reaching 99% of Peak Test R²", fontsize=10.5)
ax.set_title("Optimal Descriptor Count per Model (99%-of-peak-R² rule)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y"); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_optimal_n_summary.png")
plt.show(); plt.close(fig)

import shap

TOP_N_OVERALL = 20

lgbm_row_summary = summary_df[summary_df["model"] == "LightGBM"].iloc[0]
lgbm_log_transform = bool(results_df.loc[results_df["model"] == "LightGBM", "log_transform"].iloc[0])
n_opt_lgbm = int(lgbm_row_summary["optimal_n_99pct"])
lgbm_top10_features = lgbm_row_summary["features_at_optimal_n_99pct"].split("|")

print(f"\n{'='*64}")
print(f"(A) LightGBM TOP-{n_opt_lgbm} DESCRIPTORS (its own 99%-of-peak elimination optimum)")
print(f"{'='*64}")
print(f"  Target transform locked for LightGBM during the sweep: "
      f"{'log' if lgbm_log_transform else 'original'}")

lgbm_top10_idx = [feature_names.index(f) for f in lgbm_top10_features]
X_train_lgbm10 = X_train[:, lgbm_top10_idx]
X_test_lgbm10  = X_test[:, lgbm_top10_idx]
y_fit_lgbm10   = np.log(y_train) if lgbm_log_transform else y_train

lgbm_final10 = LGBMRegressor(**LGBM_PARAMS)
lgbm_final10.fit(X_train_lgbm10, y_fit_lgbm10)
imp10 = lgbm_final10.feature_importances_
order10 = np.argsort(imp10)[::-1]

lgbm_top10_table = pd.DataFrame({
    "Rank": range(1, len(order10) + 1),
    "Descriptor": [lgbm_top10_features[i] for i in order10],
    "Gain_Importance": imp10[order10],
})
lgbm_top10_table.to_csv(f"{OUT_DIR}lightgbm_top10_descriptors.csv", index=False)
print(lgbm_top10_table.to_string(index=False))
print(f"  Saved -> {OUT_DIR}lightgbm_top10_descriptors.csv")

fig, ax = plt.subplots(figsize=(9, 5))
plot_data = lgbm_top10_table.iloc[::-1]
norm = plot_data["Gain_Importance"] / plot_data["Gain_Importance"].max()
bar_c = plt.cm.RdYlBu_r(0.25 + 0.65 * norm.values)
bars = ax.barh(plot_data["Descriptor"], plot_data["Gain_Importance"],
               color=bar_c, edgecolor="white", linewidth=0.5, height=0.7)
for bar, val in zip(bars, plot_data["Gain_Importance"]):
    ax.text(bar.get_width() + plot_data["Gain_Importance"].max()*0.02,
            bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", fontsize=8.5)
ax.set_xlabel("Gain Importance", fontsize=10.5)
ax.set_title(f"LightGBM — Top {n_opt_lgbm} Descriptors (elimination optimum)",
             fontsize=12, fontweight="bold")
ax.xaxis.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_lightgbm_top10_importance.png")
plt.show(); plt.close(fig)

# ---- (B) Top-20 overall: LightGBM refit on the full 35-descriptor set ----
print(f"\n{'='*64}")
print(f"(B) TOP-{TOP_N_OVERALL} DESCRIPTORS OVERALL (LightGBM, full 35-descriptor set)")
print(f"{'='*64}")

y_fit_full_lgbm = np.log(y_train) if lgbm_log_transform else y_train
lgbm_full = LGBMRegressor(**LGBM_PARAMS)
lgbm_full.fit(X_train, y_fit_full_lgbm)
imp_full = lgbm_full.feature_importances_

print("  Computing SHAP values on full 35-descriptor LightGBM model ...")
explainer = shap.TreeExplainer(lgbm_full)
shap_values_full = explainer.shap_values(X_train)
shap_df_full = pd.DataFrame(shap_values_full, columns=feature_names)
mean_shap_full = shap_df_full.abs().mean().sort_values(ascending=False)
top20_shap_feats = mean_shap_full.head(TOP_N_OVERALL).index.tolist()

top20_table = pd.DataFrame({
    "Rank": range(1, TOP_N_OVERALL + 1),
    "Descriptor": top20_shap_feats,
    "Mean_abs_SHAP": mean_shap_full.head(TOP_N_OVERALL).values,
    "Gain_Importance": [imp_full[feature_names.index(f)] for f in top20_shap_feats],
})
top20_table.to_csv(f"{OUT_DIR}lightgbm_top20_overall_shap_importance.csv", index=False)
print(top20_table.to_string(index=False))
print(f"  Saved -> {OUT_DIR}lightgbm_top20_overall_shap_importance.csv")

# Combined bar (SHAP) + beeswarm figure, matching the style of the main pipeline
fig = plt.figure(figsize=(16, 0.45*TOP_N_OVERALL + 3))
fig.suptitle(f"SHAP Analysis — LightGBM (Full 35-Descriptor Set) | Top {TOP_N_OVERALL}",
             fontsize=13, fontweight="bold", y=1.02)
gs = plt.GridSpec(1, 2, wspace=0.45, left=0.06, right=0.97, top=0.90, bottom=0.10)

ax_bar = fig.add_subplot(gs[0, 0])
data = mean_shap_full.head(TOP_N_OVERALL).iloc[::-1]
norm = (data - data.min()) / (data.max() - data.min())
bc = plt.cm.Oranges(0.35 + 0.60 * norm.values)
bars = ax_bar.barh(data.index, data.values, color=bc, edgecolor="white", linewidth=0.5, height=0.72)
for bar, val in zip(bars, data.values):
    ax_bar.text(bar.get_width() + data.max()*0.02, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7.8)
ax_bar.set_xlabel("Mean |SHAP| (K)", fontsize=10)
ax_bar.set_title("(A)  Global Feature Importance", loc="left", fontsize=10, fontstyle="italic")
ax_bar.tick_params(axis="y", length=0, labelsize=8.5)
ax_bar.xaxis.grid(True); ax_bar.set_axisbelow(True)

ax_bee = fig.add_subplot(gs[0, 1])
plt.sca(ax_bee)
top_idx = [feature_names.index(f) for f in top20_shap_feats]
shap.summary_plot(shap_values_full[:, top_idx], X_train[:, top_idx],
                   feature_names=top20_shap_feats, plot_type="dot",
                   max_display=TOP_N_OVERALL, show=False, color_bar=True, plot_size=None)
ax_bee = plt.gca()
ax_bee.set_title("(B)  Direction & Magnitude (Beeswarm)", loc="left", fontsize=10, fontstyle="italic")
ax_bee.set_xlabel("SHAP Value — Impact on BP (K)", fontsize=10)

save_fig(fig, f"{OUT_DIR}fig_lightgbm_top20_shap_combined.png")
plt.show(); plt.close(fig)

print(f"\n{'='*64}")
print("FINAL DESCRIPTOR ANALYSIS COMPLETE")
print(f"{'='*64}")
print(f"  LightGBM top-{n_opt_lgbm} (own elimination optimum) -> lightgbm_top10_descriptors.csv")
print(f"  Top-{TOP_N_OVERALL} overall (SHAP + gain, full 35-descriptor set) -> lightgbm_top20_overall_shap_importance.csv")

# =============================================================================
# 13. CHECKPOINT — everything the NEXT pipeline cell needs
#     Saves selected descriptor sets, transform decisions, and locked
#     hyperparameters so the full tuning -> AD -> Y-randomization pipeline
#     can be re-run on the chosen optimal descriptor set WITHOUT re-running
#     this entire elimination sweep. Load this in the next cell with:
#         import joblib
#         ckpt = joblib.load(".../checkpoint_backward_elimination_35desc.pkl")
# =============================================================================

import joblib

per_model_transform = dict(zip(results_df["model"], results_df["log_transform"]))
per_model_selected_features_99pct = {
    row["model"]: row["features_at_optimal_n_99pct"].split("|") for _, row in summary_df.iterrows()
}
per_model_selected_features_stat = {
    row["model"]: (row["features_at_optimal_n_stat"].split("|") if row["features_at_optimal_n_stat"] else None)
    for _, row in summary_df.iterrows()
}

checkpoint = {
    "source": "backward_elimination_35desc.py",
    "base_descriptor_set": "35desc",
    "feature_names_35": feature_names,
    "results_df": results_df.drop(columns=["cv_r2_folds"], errors="ignore"),
    "summary_df": summary_df,
    "anchor_df": anchor_df,
    "per_model_transform": per_model_transform,                       # e.g. {"LightGBM": False, ...}
    "per_model_selected_features_99pct": per_model_selected_features_99pct,
    "per_model_selected_features_stat": per_model_selected_features_stat,
    "lightgbm_top10_features": lgbm_top10_features,
    "lightgbm_top10_optimal_n": n_opt_lgbm,
    "lightgbm_top20_overall_features": top20_shap_feats,
    # locked hyperparameters used throughout this sweep (35-descriptor tuned)
    "LINEAR_PARAMS": LINEAR_PARAMS, "RF_PARAMS": RF_PARAMS,
    "XGB_PARAMS": XGB_PARAMS, "LGBM_PARAMS": LGBM_PARAMS,
    "ANN_PARAMS_RAW": ANN_PARAMS_RAW, "ANN_PARAMS_LOG": ANN_PARAMS_LOG,
    "TRAIN_FILE": TRAIN_FILE, "TEST_FILE": TEST_FILE,
    "TARGET_COL_INDEX": TARGET_COL_INDEX, "DESCRIPTOR_START_INDEX": DESCRIPTOR_START_INDEX,
}

checkpoint_path = f"{OUT_DIR}checkpoint_backward_elimination_35desc.pkl"
joblib.dump(checkpoint, checkpoint_path)
print(f"\n  Checkpoint saved -> {checkpoint_path}")

# =============================================================================
