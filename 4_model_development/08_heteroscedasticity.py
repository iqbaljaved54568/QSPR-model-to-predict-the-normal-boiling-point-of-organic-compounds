"""
Breusch-Pagan tests and residual-vs-fitted plots (Table S8, Figure S13)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Heteroscedasticity diagnostics -- STANDALONE version (no retraining needed)
# Loads the checkpoint that the main pipeline cell saved to Google Drive.
# Change ACTIVE_SET to switch.
# =============================================================================
ACTIVE_SET = "94desc"          # <-- change to "35desc" for the 35-descriptor set

pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

OUT_DIR = f"./BP_FINAL_MODEL/figures/{ACTIVE_SET}/"
CKPT = f"{OUT_DIR}checkpoint_{ACTIVE_SET}.pkl"
if not os.path.exists(CKPT):
    raise FileNotFoundError(f"No checkpoint at {CKPT}. Check the folder name in the data folder.")
ckpt = joblib.load(CKPT)
print("Checkpoint contains:", sorted(ckpt.keys()))
y_train = np.asarray(ckpt["y_train"], float)

# Newer checkpoints hold every model; older ones hold only the best model (LightGBM, raw).
if "results_raw" in ckpt and "results_log" in ckpt:
    sources = [("Raw", ckpt["results_raw"]), ("Log", ckpt["results_log"])]
    MODELS = [m for m in ["Linear", "LightGBM"] if m in ckpt["results_raw"]]
    print("Models available:", list(ckpt["results_raw"].keys()))
else:
    print("NOTE: this checkpoint only stores the best model -- running the test for it alone.")
    b = ckpt["best_res"]
    name = ckpt.get("best_model_name", "LightGBM")
    sources = [("Log" if b.get("log_transform") else "Raw", {name: b})]
    MODELS = [name]

N_BINS = 5
rows, plot_data = [], []
for scale, rd in sources:
    for m in MODELS:
        if m not in rd:
            continue
        fitted = np.asarray(rd[m]["cv_oof_pred"], float)    # out-of-fold predictions, K
        resid = y_train - fitted
        lm, lm_p, f, f_p = het_breuschpagan(resid, sm.add_constant(fitted), robust=True)
        rho, rho_p = spearmanr(np.abs(resid), fitted)
        bins = pd.qcut(fitted, N_BINS, labels=False)
        sd = pd.Series(resid).groupby(bins).std()
        mid = pd.Series(fitted).groupby(bins).median()
        rows.append({"Model": m, "Target": scale, "n": len(resid),
                     "BP LM stat": round(lm, 2), "BP p-value": f"{lm_p:.2g}",
                     "Spearman rho": round(rho, 3), "rho p-value": f"{rho_p:.2g}",
                     "Resid SD, lowest quintile (K)": round(sd.iloc[0], 2),
                     "Resid SD, highest quintile (K)": round(sd.iloc[-1], 2),
                     "SD ratio (high/low)": round(sd.iloc[-1] / sd.iloc[0], 2)})
        plot_data.append((m, scale, fitted, resid, mid, sd))

table = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(table.to_string(index=False))
table.to_csv(f"{OUT_DIR}heteroscedasticity_tests_{ACTIVE_SET}.csv", index=False)

# ---- Residual-vs-fitted figure ---------------------------------------------
n = len(plot_data)
fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.4), sharey=True, squeeze=False)
for (m, scale, fitted, resid, mid, sd), ax in zip(plot_data, axes[0]):
    ax.scatter(fitted, resid, s=6, alpha=0.3, color="#4A90D9", edgecolors="none")
    ax.axhline(0, color="#333", lw=1, ls="--")
    ax.plot(mid, sd, "o-", color="#d95f02", lw=2, label="±1 SD per quintile")
    ax.plot(mid, -sd, "o-", color="#d95f02", lw=2)
    r = table[(table.Model == m) & (table.Target == scale)].iloc[0]
    ax.set_title(f"{m}, {scale.lower()} target\nBP p = {r['BP p-value']}, "
                 f"SD ratio = {r['SD ratio (high/low)']}", fontsize=9.5)
    ax.set_xlabel("Fitted boiling point, out-of-fold (K)")
    ax.grid(alpha=0.3)
axes[0, 0].set_ylabel("Residual (K)")
axes[0, 0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}fig_residual_vs_fitted_BP_{ACTIVE_SET}.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved table and figure to {OUT_DIR}")
