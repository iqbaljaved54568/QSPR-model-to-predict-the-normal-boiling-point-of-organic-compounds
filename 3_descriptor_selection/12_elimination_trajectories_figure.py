"""
Backward-elimination trajectories of all five models (Figure 6)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Figure 6: backward-elimination trajectories of all five models
# Uses ONLY the CSV your sweep already saved -- nothing is retrained.
# Also re-selects each model's optimal descriptor count from CROSS-VALIDATION
# (not the test set), so the test set stays independent.
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = ("./BP_FINAL_MODEL/figures/backward_elimination_35desc/"
               "backward_elimination_results_35desc.csv")
OUT_DIR = "./BP_FINAL_MODEL/figures/backward_elimination_35desc/"
TOL = 0.99            # same 99%-of-peak rule as before, but applied to CV R2
N_FOLDS = 5           # CV_SPLITS_RFE used in the sweep

df = pd.read_csv(RESULTS_CSV).sort_values(["model", "n_features"])
order  = ["LightGBM", "XGBoost", "RandomForest", "ANN_raw", "Linear"]
labels = {"LightGBM": "LightGBM", "XGBoost": "XGBoost", "RandomForest": "Random Forest",
          "ANN_raw": "ANN (hold-out)*", "Linear": "Ridge"}
colors = {"LightGBM": "#1b9e77", "XGBoost": "#d95f02", "RandomForest": "#7570b3",
          "ANN_raw": "#e7298a", "Linear": "#666666"}

# ---- 1. Optimal n per model, chosen on CV (training data only) --------------
rows = []
for m in order:
    sub = df[df["model"] == m]
    peak = sub["cv_r2_mean"].max()
    ok = sub[sub["cv_r2_mean"] >= TOL * peak]
    best = ok.loc[ok["n_features"].idxmin()]
    # 1-SE rule as a second, common criterion (tree models / Ridge only: ANN has no SE)
    pk = sub.loc[sub["cv_r2_mean"].idxmax()]
    se = pk["cv_r2_std"] / np.sqrt(N_FOLDS) if pd.notna(pk["cv_r2_std"]) else np.nan
    n_1se = (sub[sub["cv_r2_mean"] >= pk["cv_r2_mean"] - se]["n_features"].min()
             if pd.notna(se) else np.nan)
    rows.append({"Model": labels[m],
                 "Peak CV R2": round(peak, 4),
                 "n at peak": int(pk["n_features"]),
                 "Optimal n (99% of peak CV R2)": int(best["n_features"]),
                 "CV R2 at optimal n": round(best["cv_r2_mean"], 4),
                 "Test R2 at optimal n": round(best["test_r2"], 4),
                 "Optimal n (1-SE rule)": n_1se})
opt = pd.DataFrame(rows)
print(opt.to_string(index=False))
opt.to_csv(f"{OUT_DIR}optimal_n_by_CV_all_models.csv", index=False)

# ---- 2. Overlay figure: (a) CV R2, (b) test R2 ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
for m in order:
    sub = df[df["model"] == m]
    c = colors[m]
    ls = "--" if m == "ANN_raw" else "-"
    axes[0].plot(sub["n_features"], sub["cv_r2_mean"], ls, marker="o", ms=3.5, c=c, label=labels[m])
    if sub["cv_r2_std"].notna().any():
        axes[0].fill_between(sub["n_features"], sub["cv_r2_mean"] - sub["cv_r2_std"],
                             sub["cv_r2_mean"] + sub["cv_r2_std"], color=c, alpha=0.12, lw=0)
    axes[1].plot(sub["n_features"], sub["test_r2"], ls, marker="o", ms=3.5, c=c, label=labels[m])
    n_opt = opt.loc[opt["Model"] == labels[m], "Optimal n (99% of peak CV R2)"].iloc[0]
    y_opt = sub.loc[sub["n_features"] == n_opt, "cv_r2_mean"].iloc[0]
    axes[0].scatter([n_opt], [y_opt], s=90, facecolor="none", edgecolor=c, lw=2, zorder=5)

axes[0].set_title("(a) Cross-validated R$^2$ (5-fold, mean ± SD)", loc="left")
axes[1].set_title("(b) Test-set R$^2$", loc="left")
for ax in axes:
    ax.set_xlabel("Number of descriptors retained")
    ax.invert_xaxis()                     # read left-to-right as elimination proceeds
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)                 # adjust if low-n points fall below
axes[0].set_ylabel("R$^2$")
axes[1].legend(loc="lower left", fontsize=9)
fig.text(0.01, -0.02, "Open circles: smallest descriptor count reaching 99% of each model's peak CV R². "
         "*ANN: single 15% hold-out of the training set, not k-fold CV; descriptors ranked once "
         "by permutation importance.", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}fig5_elimination_all_models.png", dpi=300, bbox_inches="tight")
plt.show()
