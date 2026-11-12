"""
Original vs log-transformed target, 10 descriptors (Figure 7)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# Figure 7 -- 10-descriptor set, original vs log-transformed Tb (test set).
# Values from 4_model_development/05_final_pipeline_10desc.py output (optimal descriptor pipeline).
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

models = ["Ridge", "Random\nForest", "XGBoost", "LightGBM", "ANN"]
orig = {"R2":  [0.6167, 0.9496, 0.9565, 0.9596, 0.9348],
        "MAE": [45.1535, 13.9691, 12.8044, 12.3215, 16.8949],
        "RMSE":[56.6495, 20.5506, 19.0777, 18.3871, 23.3584],
        "MAPE":[9.2811, 2.8033, 2.5487, 2.4570, 3.4486]}
logt = {"R2":  [0.5743, 0.9495, 0.9560, 0.9585, 0.8495],
        "MAE": [46.3994, 13.9203, 12.9200, 12.3465, 26.9488],
        "RMSE":[59.6978, 20.5547, 19.2014, 18.6357, 35.4940],
        "MAPE":[9.3849, 2.7773, 2.5585, 2.4488, 5.4037]}
CO, CL = "#3a7d6b", "#8172b3"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(13, 8.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85], hspace=0.42, wspace=0.32)
x = np.arange(len(models)); w = 0.38

def bars(ax, key, title, ylab, fmt, ylim=None):
    ax.bar(x - w/2, orig[key], w, color=CO, zorder=3)
    ax.bar(x + w/2, logt[key], w, color=CL, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(["Ridge","RF","XGB","LGBM","ANN"], fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylab); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim: ax.set_ylim(*ylim)

ax1 = fig.add_subplot(gs[0, 0]); bars(ax1, "R2", "(a) Test $R^2$", "$R^2$", "{:.3f}", (0, 1))
ax2 = fig.add_subplot(gs[0, 1]); bars(ax2, "MAE", "(b) Test MAE", "Error (K)", "{:.1f}", (0, 62))
ax3 = fig.add_subplot(gs[0, 2], sharey=ax2); bars(ax3, "RMSE", "(c) Test RMSE", "", "{:.1f}")
plt.setp(ax3.get_yticklabels(), visible=True)
ax4 = fig.add_subplot(gs[0, 3]); bars(ax4, "MAPE", "(d) Test MAPE", "MAPE (%)", "{:.1f}", (0, 10))

# (e) normalised panel: % degradation under log transform (positive = log worse)
keys = ["R2", "MAE", "RMSE", "MAPE"]
deg = np.array([[ (orig[k][i]-logt[k][i])/orig[k][i]*100 if k == "R2"
                  else (logt[k][i]-orig[k][i])/orig[k][i]*100 for k in keys]
                for i in range(len(models))])
ax5 = fig.add_subplot(gs[1, :])
mcols = ["#4c4c4c", "#7a7a7a", "#a6a6a6", "#d0d0d0"]
w2 = 0.2
for j, k in enumerate(keys):
    xs = x + (j - 1.5) * w2
    ax5.bar(xs, deg[:, j], w2, color=mcols[j], edgecolor="white", zorder=3,
            label={"R2": "$R^2$ (decrease)", "MAE": "MAE", "RMSE": "RMSE", "MAPE": "MAPE"}[k])
    for xi, v in zip(xs, deg[:, j]):
        ax5.text(xi, v + (1.2 if v >= 0 else -1.2), f"{v:+.1f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=7.5)
ax5.axhline(0, color="#333", lw=1)
ax5.set_xticks(x); ax5.set_xticklabels([m.replace("\n", " ") for m in models])
ax5.set_ylabel("Change under log transform (%)")
ax5.set_ylim(-8, 68)
ax5.set_title("(e) Relative change in each metric with log-transformed $T_b$ "
              "(positive = worse than original scale)", loc="left", fontsize=10.5, fontweight="bold")
ax5.grid(axis="y", alpha=0.3, zorder=0); ax5.spines[["top", "right"]].set_visible(False)
ax5.legend(loc="upper left", frameon=False, ncol=4, fontsize=9)

fig.legend(handles=[Patch(color=CO, label="Original $T_b$ (K)"),
                    Patch(color=CL, label="Log-transformed $T_b$ (back-transformed to K)")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
plt.savefig("figure6_revised.png", dpi=300, bbox_inches="tight")
print(np.round(deg, 2))
