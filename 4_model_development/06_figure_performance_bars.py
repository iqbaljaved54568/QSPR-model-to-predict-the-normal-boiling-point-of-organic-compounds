"""
Performance bar chart

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# Figure 5 -- values taken from Table 5 (raw target).
# Option A: bar chart, y-axis from 0.85, label clipping fixed.
# Option B: dot plot of test R2 with CV R2 mean +/- SD (no baseline problem).
import numpy as np
import matplotlib.pyplot as plt

models = ["Linear\n(Ridge)", "Random\nForest", "XGBoost", "LightGBM", "ANN"]
test94 = [0.9267, 0.9506, 0.9589, 0.9663, 0.9581]
test35 = [0.8801, 0.9524, 0.9596, 0.9644, 0.9431]
cv94   = [0.9270, 0.9465, 0.9568, 0.9642, 0.9608]
cv94sd = [0.0086, 0.0069, 0.0066, 0.0039, 0.0056]
cv35   = [0.8738, 0.9479, 0.9581, 0.9628, 0.9399]
cv35sd = [0.0118, 0.0070, 0.0067, 0.0058, 0.0071]
C94, C35 = "#2c5d8f", "#e08a2e"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

# ---------------- Option A ----------------
x = np.arange(len(models)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 4.6))
b1 = ax.bar(x - w/2, test94, w, color=C94, label="94 descriptors", zorder=3)
b2 = ax.bar(x + w/2, test35, w, color=C35, label="35 descriptors", zorder=3)
ax.axvspan(2.5, 3.5, color="#fff3c4", zorder=0)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0012, f"{b.get_height():.4f}",
                ha="center", va="bottom", fontsize=7.5, rotation=90)
ax.set_ylim(0.85, 0.985)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Test $R^2$")
ax.grid(axis="y", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.0, -0.17, "Note: y-axis starts at 0.85; bar heights are not proportional to $R^2$.",
        transform=ax.transAxes, fontsize=8, style="italic")
ax.legend(loc="upper left", frameon=False, ncol=2)
plt.tight_layout()
plt.savefig("figure4_optionA_bars.png", dpi=300, bbox_inches="tight")

# ---------------- Option B ----------------
fig, ax = plt.subplots(figsize=(8, 4.6))
off = 0.13
ax.errorbar(x - off, cv94, yerr=cv94sd, fmt="none", ecolor=C94, elinewidth=1.4, capsize=3, alpha=0.6, zorder=2)
ax.errorbar(x + off, cv35, yerr=cv35sd, fmt="none", ecolor=C35, elinewidth=1.4, capsize=3, alpha=0.6, zorder=2)
ax.scatter(x - off, cv94, marker="_", s=120, color=C94, zorder=3)
ax.scatter(x + off, cv35, marker="_", s=120, color=C35, zorder=3)
ax.scatter(x - off, test94, s=55, color=C94, edgecolor="white", zorder=4, label="94 descriptors")
ax.scatter(x + off, test35, s=55, color=C35, edgecolor="white", marker="s", zorder=4, label="35 descriptors")
for xi, t in zip(x - off, test94): ax.text(xi - 0.07, t, f"{t:.3f}", ha="right", va="center", fontsize=7.5)
for xi, t in zip(x + off, test35): ax.text(xi + 0.07, t, f"{t:.3f}", ha="left", va="center", fontsize=7.5)
ax.set_xlim(-0.6, 4.6); ax.set_ylim(0.86, 0.98)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("$R^2$")
ax.grid(axis="y", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=C94, label="94 descriptors"),
     Line2D([], [], marker="s", ls="", color=C35, label="35 descriptors"),
     Line2D([], [], marker="o", ls="", color="#444", label="Test $R^2$"),
     Line2D([], [], color="#444", alpha=0.6, label="10-fold CV $R^2$ (mean ± SD)")]
ax.legend(handles=h, loc="lower right", frameon=False, fontsize=8.5, ncol=2)
plt.tight_layout()
plt.savefig("figure4_optionB_dots.png", dpi=300, bbox_inches="tight")
print("done")
