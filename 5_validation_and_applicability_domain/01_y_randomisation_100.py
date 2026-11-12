"""
Y-randomisation, first 100 permutations

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Y-Randomisation (Y-Scrambling) Test
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TRAIN_FILE = "./BP_FINAL_MODEL/descriptors/train_test_split/train_optimal.csv"
TEST_FILE  = "./BP_FINAL_MODEL/descriptors/train_test_split/test_optimal.csv"

TARGET_COL_INDEX       = 5
DESCRIPTOR_START_INDEX = 6
ACTIVE_SET = "10desc"

# ── Y-randomisation settings ──────────────────────────────────────────────────
N_RANDOM      = 100    # number of randomisation iterations (≥100 for publication)
BATCH_SIZE    = 20     # checkpoint saved every this many iterations
N_JOBS        = -1     # parallel workers across iterations (use all cores)
VAL_FRAC      = 0.15   # held-out fraction of TRAIN used for the internal "CV-like" split
EARLY_STOP_ROUNDS = 30
RANDOM_STATE  = 42

# ── Real 10-fold CV (for the genuine, unscrambled performance baseline) ──────
REAL_CV_SPLITS = 10

OUT_DIR = "./BP_FINAL_MODEL/figures/10desc/"
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT_PATH = f"{OUT_DIR}yrand_checkpoint_{ACTIVE_SET}.npz"

# ── Final tuned LightGBM hyperparameters (10-descriptor optimal set) ────────
LGBM_PARAMS = {
    "n_estimators":      1784,
    "learning_rate":     0.06686808497475842,
    "num_leaves":        18,
    "max_depth":         6,
    "min_child_samples": 32,
    "subsample":         0.9921028774475024,
    "colsample_bytree":  0.7660499104221706,
    "reg_alpha":         0.06008952920081579,
    "reg_lambda":        0.001212337250189692,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,     # used for the ONE-OFF real-performance fit only
    "verbosity":         -1,
}

# =============================================================================
# 2. STYLE
# =============================================================================

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":   11, "axes.titleweight": "bold", "axes.titlepad": 8,
    "axes.labelsize":   10,
    "axes.spines.top":  False, "axes.spines.right": False,
    "axes.edgecolor":   "#B0BEC5", "axes.linewidth":  0.8,
    "axes.facecolor":   "#F7F9FB", "figure.facecolor": "white",
    "grid.color":       "#DDE3EA", "grid.linewidth":   0.55,
    "grid.linestyle":   "--",
    "xtick.labelsize":  8.5, "ytick.labelsize": 8.5,
    "legend.fontsize":  8.5, "legend.framealpha": 0.92,
    "legend.edgecolor": "#C5CDD5",
    "savefig.dpi":      300, "savefig.bbox": "tight",
})
COLOR_REAL = "#2C6E8A"
COLOR_RND  = "#7F8C8D"

def save_fig(fig, fname):
    path = f"{OUT_DIR}{fname}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✅  Saved → {path}")

# =============================================================================
# 3. LOAD DATA
# =============================================================================

print(f"{'='*64}")
print(f"  Y-Randomisation Test — LightGBM | {ACTIVE_SET}")
print(f"  {N_RANDOM} iterations, batched every {BATCH_SIZE}, {N_JOBS} parallel workers")
print(f"{'='*64}\n")

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_train = train_df.iloc[:, TARGET_COL_INDEX].values
X_test  = test_df.iloc[:, DESCRIPTOR_START_INDEX:].values
y_test  = test_df.iloc[:, TARGET_COL_INDEX].values

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

n_train, p = X_train.shape
print(f"  {ACTIVE_SET:<10}  Train: {n_train:,} × {p}  |  Test: {X_test.shape[0]:,} × {p}"
      f"  |  BP: {y_train.min():.0f}–{y_train.max():.0f} K")

if p != 10:
    print(f"  ⚠️   Warning: expected 10 descriptors for the optimal set, "
          f"but found {p} columns from DESCRIPTOR_START_INDEX={DESCRIPTOR_START_INDEX}. "
          f"Double-check TARGET_COL_INDEX / DESCRIPTOR_START_INDEX against train_optimal.csv.")

# =============================================================================
# 4. REAL (UNSCRAMBLED) MODEL PERFORMANCE — computed once, not the bottleneck
# =============================================================================

print(f"\n  Computing real (unscrambled) model performance ...")
real_model = LGBMRegressor(**LGBM_PARAMS)
real_cv = KFold(n_splits=REAL_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# n_jobs=1 here deliberately — LGBM_PARAMS already sets n_jobs=-1 on the model
# itself, so wrapping it in another n_jobs=-1 CV call oversubscribes the CPU.
y_oof = cross_val_predict(real_model, X_train_sc, y_train, cv=real_cv, n_jobs=1)
real_cv_r2 = r2_score(y_train, y_oof)

real_model.fit(X_train_sc, y_train)
y_pred_test_real = real_model.predict(X_test_sc)
real_test_r2   = r2_score(y_test, y_pred_test_real)
real_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_real))
real_test_mae  = mean_absolute_error(y_test, y_pred_test_real)

print(f"  Real CV R²   = {real_cv_r2:.4f}")
print(f"  Real Test R² = {real_test_r2:.4f}  RMSE = {real_test_rmse:.2f} K  MAE = {real_test_mae:.2f} K")

# =============================================================================
# 5. FAST Y-RANDOMISATION ENGINE
#    One fit per iteration (single train/val split, not full k-fold), early
#    stopping to avoid building unnecessary trees on pure noise, and the 100
#    iterations run in parallel across all cores via joblib.
# =============================================================================

def _one_iteration(seed, X_tr, y_tr, X_te, y_te, params, val_frac, es_rounds):
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_tr)

    # Single internal validation split — cheap "CV-like" proxy, not full k-fold
    n = len(y_tr)
    idx = rng.permutation(n)
    n_val = max(10, int(val_frac * n))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    m = LGBMRegressor(**{**params, "random_state": seed, "n_jobs": 1})
    m.fit(
        X_tr[fit_idx], y_shuffled[fit_idx],
        eval_set=[(X_tr[val_idx], y_shuffled[val_idx])],
        callbacks=[early_stopping(stopping_rounds=es_rounds, verbose=False),
                   log_evaluation(period=0)],
    )

    # CV-like R² on the held-out (shuffled-label) validation split
    pred_val   = m.predict(X_tr[val_idx])
    cv_like_r2 = r2_score(y_shuffled[val_idx], pred_val)

    # External test R² — evaluated against the REAL (unshuffled) y_test
    pred_test = m.predict(X_te)
    test_r2   = r2_score(y_te, pred_test)

    return cv_like_r2, test_r2


def run_y_randomisation(X_tr, y_tr, X_te, y_te, n_random, batch_size, n_jobs,
                        params, val_frac, es_rounds, rs, checkpoint_path):
    # Resume from checkpoint if one exists
    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        cv_rand   = list(ckpt["cv_rand"])
        test_rand = list(ckpt["test_rand"])
        n_done = len(cv_rand)
        print(f"  Resuming from checkpoint: {n_done}/{n_random} iterations already done")
    else:
        cv_rand, test_rand = [], []
        n_done = 0

    if n_done >= n_random:
        print(f"  ✅  Already have {n_done} iterations — nothing more to run.")
        return np.array(cv_rand[:n_random]), np.array(test_rand[:n_random])

    remaining = n_random - n_done
    print(f"\n  Running Y-randomisation: {remaining} more iterations "
          f"(batches of {batch_size}, {n_jobs} parallel workers) ...")

    seeds_remaining = [rs + n_done + i for i in range(remaining)]
    t_start = time.time()

    for batch_start in range(0, remaining, batch_size):
        batch_seeds = seeds_remaining[batch_start: batch_start + batch_size]
        t0 = time.time()
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_one_iteration)(seed, X_tr, y_tr, X_te, y_te, params, val_frac, es_rounds)
            for seed in batch_seeds
        )
        for cv_r2_i, test_r2_i in batch_results:
            cv_rand.append(cv_r2_i)
            test_rand.append(test_r2_i)

        # Save checkpoint after every batch — a disconnect now costs at most
        # one batch's worth of iterations, not the whole run.
        np.savez(checkpoint_path,
                 cv_rand=np.array(cv_rand), test_rand=np.array(test_rand))

        n_completed = len(cv_rand)
        elapsed_batch = time.time() - t0
        elapsed_total = time.time() - t_start
        print(f"    {n_completed}/{n_random}  "
              f"mean CV-like R² = {np.mean(cv_rand):.4f}  "
              f"mean test R² = {np.mean(test_rand):.4f}  "
              f"[batch {elapsed_batch:.1f}s, total {elapsed_total:.1f}s]")

    return np.array(cv_rand[:n_random]), np.array(test_rand[:n_random])


cv_rand, test_rand = run_y_randomisation(
    X_train_sc, y_train, X_test_sc, y_test,
    n_random=N_RANDOM, batch_size=BATCH_SIZE, n_jobs=N_JOBS,
    params=LGBM_PARAMS, val_frac=VAL_FRAC, es_rounds=EARLY_STOP_ROUNDS,
    rs=RANDOM_STATE, checkpoint_path=CHECKPOINT_PATH,
)

print(f"\n  Done. Scrambled CV-like R² : {cv_rand.mean():.4f} ± {cv_rand.std():.4f}"
      f"  (max: {cv_rand.max():.4f})")
print(f"        Scrambled test R²    : {test_rand.mean():.4f} ± {test_rand.std():.4f}"
      f"  (max: {test_rand.max():.4f})")

# =============================================================================
# 6. STATISTICAL SUMMARY
# =============================================================================

def y_rand_stats(real_r2, rand_r2, label="", split="CV"):
    n         = len(rand_r2)
    mean_rand = rand_r2.mean()
    std_rand  = rand_r2.std()
    max_rand  = rand_r2.max()
    delta     = real_r2 - mean_rand
    z_score   = (real_r2 - mean_rand) / (std_rand / np.sqrt(n))
    p_value   = 1 - norm.cdf(z_score)

    divider = "─" * 62
    print(f"\n{divider}")
    print(f"  Y-Randomisation Stats — {label} | {split}")
    print(divider)
    print(f"  Iterations             : {n}")
    print(f"  Real R²                : {real_r2:.4f}")
    print(f"  Scrambled R² mean      : {mean_rand:.4f}")
    print(f"  Scrambled R² std       : {std_rand:.4f}")
    print(f"  Scrambled R² max       : {max_rand:.4f}")
    print(f"  ΔR² (real − scrambled) : {delta:.4f}")
    print(f"  Z-score                : {z_score:.2f}")
    print(f"  p-value (one-tailed)   : {p_value:.2e}")
    validity = "✅  VALID — genuine structure–property relationship" \
               if mean_rand < 0.3 and delta > 0.5 \
               else "⚠️   CHECK — scrambled performance too high"
    print(f"  Verdict                : {validity}")
    print(divider)

    return {"n": n, "real_r2": real_r2, "mean_rand": mean_rand,
            "std_rand": std_rand, "max_rand": max_rand,
            "delta": delta, "z_score": z_score, "p_value": p_value}

stats_cv   = y_rand_stats(real_cv_r2,   cv_rand,   ACTIVE_SET, "CV-like (single split)")
stats_test = y_rand_stats(real_test_r2, test_rand, ACTIVE_SET, "Test")

# =============================================================================
# 7. FIGURE 1 — Y-Randomisation Distribution (2-panel: CV-like | Test)
# =============================================================================

def plot_y_rand_panel(real_cv, real_test, cv_rand, test_rand, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Y-Randomisation Test — LightGBM | {ACTIVE_SET}\n"
                 "Scrambled R² distribution vs real model performance",
                 fontsize=13, fontweight="bold", y=1.03)

    configs = [
        (axes[0], real_cv,   cv_rand,   f"{ACTIVE_SET} — CV-like R² (single split)"),
        (axes[1], real_test, test_rand, f"{ACTIVE_SET} — External Test R²"),
    ]

    for ax, real_r2, rand_r2, title in configs:
        ax.hist(rand_r2, bins=25, color=COLOR_RND, alpha=0.65,
                edgecolor="white", linewidth=0.5, density=True,
                label=f"Scrambled  (n={len(rand_r2)})")

        x_fit = np.linspace(rand_r2.min() - 0.05, max(rand_r2.max(), real_r2) + 0.05, 300)
        pdf   = norm.pdf(x_fit, rand_r2.mean(), rand_r2.std())
        ax.plot(x_fit, pdf, color=COLOR_RND, linewidth=1.5, linestyle="--", alpha=0.8)

        ax.axvline(real_r2, color=COLOR_REAL, linewidth=2.2,
                   linestyle="-", label=f"Real R² = {real_r2:.4f}", zorder=5)
        ax.axvline(rand_r2.mean(), color="#555", linewidth=1.2, linestyle=":",
                   label=f"Mean scrambled = {rand_r2.mean():.4f}")

        delta = real_r2 - rand_r2.mean()
        stats_txt = (f"Scrambled: {rand_r2.mean():.4f} ± {rand_r2.std():.4f}\n"
                     f"ΔR² = {delta:.4f}\n"
                     f"Max scrambled = {rand_r2.max():.4f}")
        ax.text(0.04, 0.96, stats_txt, transform=ax.transAxes, fontsize=8.5,
                va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                     ec="#C5CDD5", alpha=0.95))

        ax.set_xlabel("R²", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.set_title(title, loc="left", fontsize=10, fontstyle="italic")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_y_rand_panel(real_cv_r2, real_test_r2, cv_rand, test_rand,
                  save_path=f"fig_y_randomisation_panel_{ACTIVE_SET}.png")

# =============================================================================
# 8. FIGURE 2 — Convergence plot (running mean of scrambled R²)
# =============================================================================

def plot_convergence(cv_rand, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    iters = np.arange(1, len(cv_rand) + 1)
    run_mean = np.cumsum(cv_rand) / iters
    run_std  = pd.Series(cv_rand).expanding().std().values

    ax.plot(iters, run_mean, color=COLOR_REAL, linewidth=2, label=f"{ACTIVE_SET} running mean")
    ax.fill_between(iters, run_mean - run_std, run_mean + run_std,
                    alpha=0.12, color=COLOR_REAL)
    ax.axhline(0, color="#AAB7B8", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Number of Randomisation Iterations", fontsize=10)
    ax.set_ylabel("Running Mean Scrambled R²", fontsize=10)
    ax.set_title(f"Y-Randomisation Convergence — {ACTIVE_SET}\n"
                 "Running mean stabilises as iterations accumulate",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.xaxis.grid(True); ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.show(); plt.close(fig)

plot_convergence(cv_rand, save_path=f"fig_y_rand_convergence_{ACTIVE_SET}.png")

# =============================================================================
# 9. SUMMARY TABLE + LaTeX
# =============================================================================

def build_y_rand_table(stats_cv, stats_test, label):
    rows = []
    for split, s in [("CV-like", stats_cv), ("Test", stats_test)]:
        rows.append({
            "Descriptors": label, "Split": split,
            "Real R²": f"{s['real_r2']:.4f}",
            "Scrambled R² (mean)": f"{s['mean_rand']:.4f}",
            "Scrambled R² (std)":  f"{s['std_rand']:.4f}",
            "Scrambled R² (max)":  f"{s['max_rand']:.4f}",
            "ΔR²": f"{s['delta']:.4f}",
            "Z-score": f"{s['z_score']:.2f}",
            "p-value": f"{s['p_value']:.2e}",
        })
    df_table = pd.DataFrame(rows).set_index(["Descriptors", "Split"])
    print(f"\n{'═'*90}")
    print(f"  Y-RANDOMISATION SUMMARY TABLE — {label}")
    print(f"{'═'*90}")
    print(df_table.to_string())
    print("\n  LaTeX:\n")
    print(df_table.to_latex(
        caption=f"Y-randomisation test results for the final LightGBM model "
                f"({label} descriptors). Real R² represents genuine model "
                f"performance; scrambled R² (mean, std, max) is derived from "
                f"{N_RANDOM} random permutations of the target variable. "
                f"ΔR² = Real R² − Mean scrambled R².",
        label=f"tab:y_randomisation_{label}", escape=False
    ))
    return df_table

summary_table = build_y_rand_table(stats_cv, stats_test, ACTIVE_SET)
summary_table.to_csv(f"{OUT_DIR}y_randomisation_summary_{ACTIVE_SET}.csv")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================

print(f"""
{'='*64}
  Y-RANDOMISATION COMPLETE — {ACTIVE_SET}
{'='*64}

  Figures saved:
    fig_y_randomisation_panel_{ACTIVE_SET}.png   CV-like | Test distribution plots
    fig_y_rand_convergence_{ACTIVE_SET}.png      Running-mean convergence

  Key results:
  ┌──────────────────────────────────────────────────────┐
  │  {ACTIVE_SET} LightGBM                                    │
  │    Real CV R²          : {real_cv_r2:.4f}
  │    Scrambled CV-like R² : {cv_rand.mean():.4f} ± {cv_rand.std():.4f}
  │    Real Test R²        : {real_test_r2:.4f}
  │    Scrambled Test R²   : {test_rand.mean():.4f} ± {test_rand.std():.4f}
  └──────────────────────────────────────────────────────┘

  Publication criterion:
    Scrambled R² << Real R²  →  {'✅ PASSED' if cv_rand.mean() < 0.3 else '⚠️  CHECK'}

  Suggested paper sentence:
  ─────────────────────────────────────────────────────────
  "To confirm that the LightGBM model captures genuine
  structure–property relationships rather than chance
  correlations, a Y-randomisation test was performed using
  {N_RANDOM} permutations of the target variable. The mean
  scrambled R² was {cv_rand.mean():.4f} ± {cv_rand.std():.4f} (max {cv_rand.max():.4f}),
  substantially below the real R² of {real_cv_r2:.4f}
  (ΔR² = {real_cv_r2 - cv_rand.mean():.4f}), confirming that model
  performance is not attributable to chance
  (p < {stats_cv['p_value']:.0e})."
  ─────────────────────────────────────────────────────────
{'='*64}
""")
