"""
Residual statistics and predicted-vs-experimental figure (Table S12, Figure 4)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Figure 4 + residual-distribution statistics (no retraining)
# Loads the saved checkpoint of the 94-descriptor run.
#
# Changes vs. the original figure:
#   - larger residual insets with a symmetric (zero-centred) x-axis and a bold line at 0
#   - each inset annotated with mean residual, skewness and excess kurtosis
#   - shaded band = +/- 2 RMSE (as the caption states; the original code shaded +/- 1 RMSE)
# Also prints/saves a table: bias, median, SD, skewness (+ SE), excess kurtosis,
# D'Agostino-Pearson and Shapiro-Wilk normality tests, and % of points inside +/- 2 RMSE.
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, normaltest, shapiro, gaussian_kde

ACTIVE_SET = "94desc"
DIR = f"./BP_FINAL_MODEL/figures/{ACTIVE_SET}/"
ck = joblib.load(DIR + f"checkpoint_{ACTIVE_SET}.pkl")
y = np.asarray(ck["y_test"], float); R = ck["results_raw"]
order = [m for m in ["Linear", "RandomForest", "XGBoost", "LightGBM", "ANN"] if m in R]
labels = {"Linear": "Linear (Ridge)", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
          "LightGBM": "LightGBM", "ANN": "ANN"}
colors = {"Linear": "#4C72B0", "RandomForest": "#55A868", "XGBoost": "#DD8452", "LightGBM": "#C44E52", "ANN": "#8172B3"}

rows = []
for m in order:
    p = np.asarray(R[m]["y_pred_test"], float); e = p - y; n = len(e)
    rmse = np.sqrt(np.mean(e ** 2))
    rows.append({"Model": labels[m], "n": n, "Test R2": R[m]["test_r2"], "RMSE (K)": rmse,
                 "Mean residual (K)": e.mean(), "Median residual (K)": np.median(e), "SD (K)": e.std(ddof=1),
                 "Skewness": skew(e), "SE skew": np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))),
                 "Excess kurtosis": kurtosis(e), "D'Agostino p": normaltest(e).pvalue,
                 "Shapiro-Wilk p": shapiro(e).pvalue, "% within ±2 RMSE": 100 * np.mean(np.abs(e) <= 2 * rmse)})
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.round(4).to_string(index=False))
T.round(4).to_csv(DIR + f"residual_statistics_{ACTIVE_SET}.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(16, 11)); axes.flat[-1].set_visible(False)
for ax, m, r in zip(axes.flat, order, rows):
    p = np.asarray(R[m]["y_pred_test"], float); e = p - y; rmse = r["RMSE (K)"]
    xy = np.vstack([y, p]); k = gaussian_kde(xy)(xy); i = k.argsort()
    sc = ax.scatter(y[i], p[i], c=k[i], cmap="viridis", s=10, alpha=0.7, edgecolors="none", rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.78, label="Point density")
    lims = [min(y.min(), p.min()) - 5, max(y.max(), p.max()) + 5]
    ax.plot(lims, lims, "--", color="#333", lw=1.4)
    ax.fill_between(lims, [l - 2 * rmse for l in lims], [l + 2 * rmse for l in lims], alpha=0.10, color=colors[m])
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.text(0.04, 0.97, f"$R^2$ = {r['Test R2']:.4f}\nMAE = {np.mean(np.abs(e)):.2f} K\nRMSE = {rmse:.2f} K",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#C5CDD5", alpha=0.95))
    ins = ax.inset_axes([0.56, 0.15, 0.42, 0.28])
    lim = np.ceil(np.percentile(np.abs(e), 99.5) / 10) * 10
    ins.hist(np.clip(e, -lim, lim), bins=np.linspace(-lim, lim, 31), color=colors[m], alpha=0.8,
             edgecolor="white", lw=0.4, density=True)
    ins.axvline(0, color="black", lw=1.6)
    ins.axvline(e.mean(), color="#D55E00", lw=1.2, ls="--")
    ins.set_xlim(-lim, lim); ins.set_yticks([])
    ins.set_xlabel("Residual, pred. − exp. (K)", fontsize=7); ins.tick_params(labelsize=7)
    ins.text(0.02, 0.97, f"mean {e.mean():+.1f} K\nskew {r['Skewness']:+.2f}\nex. kurt. {r['Excess kurtosis']:.1f}",
             transform=ins.transAxes, fontsize=6.5, va="top")
    ins.set_facecolor("#F2F4F7"); [ins.spines[s].set_visible(False) for s in ("top", "right", "left")]
    ax.set_xlabel("Experimental $T_b$ (K)"); ax.set_ylabel("Predicted $T_b$ (K)")
    ax.set_title(labels[m], fontsize=11, fontweight="bold", color=colors[m]); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DIR + f"fig3_pred_vs_exp_revised_{ACTIVE_SET}.png", dpi=300, bbox_inches="tight")
plt.savefig(DIR + f"fig3_pred_vs_exp_revised_{ACTIVE_SET}.pdf", bbox_inches="tight"); plt.show()
print("Saved revised figure and residual_statistics CSV to", DIR)
