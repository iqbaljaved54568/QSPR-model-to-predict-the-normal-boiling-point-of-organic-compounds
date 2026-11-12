"""
Applicability-domain analysis from a saved checkpoint

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# AD_analysis_standalone.py — Re-run AD analysis from checkpoint (35desc)
# No retraining. Uses checkpoint_35desc.pkl saved by the main pipeline.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib

# -----------------------------------------------------------------------------
# STEP 1 — Load checkpoint (no retraining)
# -----------------------------------------------------------------------------
checkpoint_path = "./BP_FINAL_MODEL/figures/35desc/checkpoint_35desc.pkl"
ckpt = joblib.load(checkpoint_path)

ACTIVE_SET     = ckpt["ACTIVE_SET"]
OUT_DIR        = ckpt["OUT_DIR"]
X_train_sc     = ckpt["X_train_sc"]
X_test_sc      = ckpt["X_test_sc"]
y_train        = ckpt["y_train"]
y_test         = ckpt["y_test"]
feature_names  = ckpt["feature_names"]
n_train        = ckpt["n_train"]
p              = ckpt["p"]
tag_str        = ckpt["tag_str"]
best_res       = ckpt["best_res"]           # dict, no 'model' key (fine — not needed below)
feat_df        = ckpt["feat_df"]
mean_shap      = ckpt["mean_shap"]
SIGMA_LIMIT    = ckpt["SIGMA_LIMIT"]
TOP_N_FEAT     = ckpt["TOP_N_FEAT"]
test_df        = ckpt["test_df"]

print(f"✅  Loaded checkpoint for {ACTIVE_SET}  (p={p}, n_train={n_train})")

# -----------------------------------------------------------------------------
# STEP 2 — Style / palette (same as main pipeline, needed for plots)
# -----------------------------------------------------------------------------
PALETTE = {
    "light_bg": "#F7F9FB", "grid": "#DDE3EA", "text": "#1A2A3A",
    "text_light": "#5A6A7A", "perfect": "#2C3E50", "ad_fill": "#EBF5FB",
    "train_in": "#2C6E8A", "test_in": "#27AE60",
    "train_out": "#F39C12", "test_out": "#C0392B",
}
TYPE_COLORS = {
    "A — High Leverage": "#E67E22",
    "B — High Residual": "#C0392B",
    "C — Both": "#8E44AD",
    "D — Within AD": "#27AE60",
}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
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

# -----------------------------------------------------------------------------
# STEP 3 — Leverage function (unchanged — the math was always correct)
# -----------------------------------------------------------------------------
def compute_leverage(X_ref, X_query):
    U, S, Vt = np.linalg.svd(X_ref, full_matrices=False)
    tol   = S.max() * max(X_ref.shape) * np.finfo(float).eps * 100
    S_inv = np.where(S > tol, 1.0/S**2, 0.0)
    return (X_query @ Vt.T)**2 @ S_inv

h_train = compute_leverage(X_train_sc, X_train_sc)
h_test  = compute_leverage(X_train_sc, X_test_sc)
h_star  = 3.0 * (p + 1) / n_train

s = best_res["cv_rmse_orig"]
resid_train = y_train - best_res["cv_oof_pred"]
resid_test  = y_test  - best_res["y_pred_test"]
sr_train = resid_train / s
sr_test  = resid_test  / s

in_ad_train = (h_train <= h_star) & (np.abs(sr_train) <= SIGMA_LIMIT)
in_ad_test  = (h_test  <= h_star) & (np.abs(sr_test)  <= SIGMA_LIMIT)

# -----------------------------------------------------------------------------
# STEP 4 — CORRECTED diagnostic printout (this is the actual fix)
# -----------------------------------------------------------------------------
n_extreme_test  = int((h_test  > 1.0).sum())
n_extreme_train = int((h_train > 1.0 + 1e-6).sum())

print(f"\n  AD: h* = {h_star:.5f}  ({ACTIVE_SET}, {p} descriptors)")
print(f"  s (CV RMSE, OOF)  : {s:.3f} K   [NOT in-sample train RMSE]")
print(f"  3σ absolute limit : ±{3*s:.1f} K")
print(f"  Leverage max test : {h_test.max():.5f}  "
      f"({n_extreme_test} test compound(s) exceed 1.0 — expected for strong "
      f"extrapolation points, already captured by h > h* classification, "
      f"NOT a scaling/tolerance error)")

if n_extreme_train > 0:
    print(f"  ⚠  Train leverage max = {h_train.max():.5f} exceeds 1.0 for "
          f"{n_extreme_train} training point(s) — this WOULD indicate a real "
          f"numerical/tolerance issue, since training points must satisfy h_ii ≤ 1.")
else:
    print(f"  ✅  Train leverage max = {h_train.max():.5f} — within valid [0,1] bound, as expected.")

print(f"  Train within AD: {in_ad_train.mean()*100:.1f}%")
print(f"  Test  within AD: {in_ad_test.mean()*100:.1f}%")

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

# -----------------------------------------------------------------------------
# STEP 5 — Outlier type classification (unchanged)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# STEP 6 — Williams plot (regenerated, with corrected annotation AND the
# label-overlap fix: stats box moved down / made narrower, quadrant labels
# repositioned so nothing sits under the box anymore)
# -----------------------------------------------------------------------------
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

    # ── FIX: give the plot more headroom above +sigma_lim so the top labels
    # and the stats box both have clear room, and push the stats box to the
    # right-hand side (below the legend) instead of overlapping the top-left
    # quadrant label. ──────────────────────────────────────────────────────
    sr_max = sr_max * 1.15  # extra headroom
    q_kw = dict(fontsize=8, color="#7F8C8D", fontstyle="italic", ha="center", va="center")

    # Quadrant labels — kept close to the axhline bands, well clear of the
    # top-left corner where the stats box now lives is avoided by moving
    # the stats box to bottom-left instead (see below).
    y_q_top = min(sr_max * 0.60, 7)
    y_q_bot = min(sr_max * 0.60, 7)
    ax.text(h_star*0.45,  y_q_top, "✓  Reliable\npredictions",    **q_kw)
    ax.text(h_star*1.55,  y_q_top, "⚠  Structurally\ninfluential", **q_kw)
    ax.text(h_star*0.45, -y_q_bot, "⚠  Response\noutliers",        **q_kw)
    ax.text(h_star*1.55, -y_q_bot, "✗  Outside AD",                **q_kw)

    # Stats box moved to bottom-left, well below the "Response outliers"
    # quadrant label, with va="bottom" so it grows upward from the anchor
    # instead of downward off the axis.
    ax.text(0.02, 0.03,
            f"h* = {h_star:.5f}\ns = CV RMSE = {s:.2f} K\n3σ = {3*s:.1f} K\n"
            f"Train AD: {in_ad_train.mean()*100:.1f}%\n"
            f"Test AD: {in_ad_test.mean()*100:.1f}%\n"
            f"(h>1 = valid extrapolation flag,\nnot a scaling error)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C5CDD5", alpha=0.96),
            zorder=6)

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
              save_path=f"{OUT_DIR}fig_williams_plot_corrected.png")

# -----------------------------------------------------------------------------
# STEP 7 — Re-export CSVs with corrected AD flags (in case anything downstream
# reads these files — the underlying flags were already correct, just re-saving
# for consistency / to timestamp the corrected run)
# -----------------------------------------------------------------------------
test_out = test_df.copy()
test_out["y_actual"]     = y_test
test_out["y_predicted"]  = best_res["y_pred_test"]
test_out["Residual_K"]   = resid_test
test_out["Std_Residual"] = sr_test
test_out["Abs_Error_K"]  = abs_error
test_out["Leverage_h"]   = h_test
test_out["Within_AD"]    = in_ad_test
test_out["Outlier_Type"] = outlier_type
test_out.to_csv(f"{OUT_DIR}test_with_AD_{ACTIVE_SET}_corrected.csv", index=False)
print(f"\n  ✅  Saved → {OUT_DIR}test_with_AD_{ACTIVE_SET}_corrected.csv")

outliers_df = test_out[~in_ad_test].copy().sort_values("Abs_Error_K", ascending=False)
outliers_df.to_csv(f"{OUT_DIR}outliers_outside_AD_{ACTIVE_SET}_corrected.csv", index=False)
print(f"  ✅  Saved → {OUT_DIR}outliers_outside_AD_{ACTIVE_SET}_corrected.csv  "
      f"({len(outliers_df)} compounds)")

print(f"\n{'='*64}")
print(f"  AD RE-ANALYSIS COMPLETE — {ACTIVE_SET}")
print(f"{'='*64}")
print(f"  No retraining performed. Leverage math unchanged — only the")
print(f"  diagnostic message was corrected. h_test.max()={h_test.max():.5f} is a")
print(f"  legitimate extrapolation flag, already reflected in your Type A/C outliers.")
