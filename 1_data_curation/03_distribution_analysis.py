"""
Boiling point and molecular weight distributions; descriptive statistics (Table 1)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =========================
# Step 3: Distribution Analysis (BP + Molecular Weight)
# Journal-Standard Figures
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, shapiro

# -------------------------
# Global plotting style
# -------------------------
sns.set_theme(
    style="white",
    context="paper",
    font="Times New Roman",
    rc={
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    },
)

# Color palette (color-blind safe)
MAIN_COLOR = "#4C72B0"   # muted blue
ACCENT_COLOR = "#DD8452" # muted orange

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(
    "./BP_FINAL_MODEL/Final_clean-organic.csv"
)

# -------------------------
# Column detection
# -------------------------
def detect_column(candidates, columns):
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None

bp_candidates = ["boiling point (K)", "bp_k", " boiling point k", "boiling point_k", "bo_k"]
mw_candidates = ["mol.wt", "mol weight", "molecular weight", "mw"]

bp_col = detect_column(bp_candidates, df.columns)
mw_col = detect_column(mw_candidates, df.columns)

if bp_col is None or mw_col is None:
    raise ValueError("❌ Could not automatically detect BP or MW column")

print(f"✅ Boiling Point column: {bp_col}")
print(f"✅ Molecular Weight column: {mw_col}")

# -------------------------
# Distribution analysis
# -------------------------
def analyze_distribution(series, label, unit, bins=40):
    series = series.dropna()

    # ---- Descriptive statistics ----
    stats = {
        "Count": series.count(),
        "Mean": series.mean(),
        "Median": series.median(),
        "Std Dev": series.std(),
        "Min": series.min(),
        "Max": series.max(),
        "Skewness": skew(series),
        "Kurtosis": kurtosis(series),
    }

    print(f"\n===== {label} Statistics =====")
    for k, v in stats.items():
        print(f"{k:<10}: {v:.3f}")

    # ---- Normality test ----
    sample = series.sample(min(5000, len(series)), random_state=42)
    stat, p = shapiro(sample)

    print("\nShapiro–Wilk Test")
    print(f"Statistic = {stat:.4f}, p-value = {p:.3e}")

    # =========================
    # Figure 1: Histogram + KDE
    # =========================
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    sns.histplot(
        series,
        bins=bins,
        stat="density",
        color=MAIN_COLOR,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    sns.kdeplot(
        series,
        color=ACCENT_COLOR,
        linewidth=2.2,
        ax=ax,
    )

    ax.axvline(series.mean(), color="black", linestyle="--", linewidth=1.2)
    ax.axvline(series.median(), color="black", linestyle=":", linewidth=1.2)

    ax.set_xlabel(f"{label} ({unit})", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(f"Distribution of {label}", fontsize=13, weight="bold")

    # Annotation box
    textstr = (
        f"N = {len(series)}\n"
        f"Mean = {series.mean():.2f}\n"
        f"Median = {series.median():.2f}\n"
        f"Skew = {skew(series):.2f}"
    )

    ax.text(
        0.98, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=0.6),
    )

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

    # =========================
    # Figure 2: Box + Violin
    # =========================
    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    sns.violinplot(
        x=series,
        color=MAIN_COLOR,
        inner=None,
        linewidth=0,
        alpha=0.4,
        ax=ax,
    )

    sns.boxplot(
        x=series,
        width=0.25,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        medianprops={"color": ACCENT_COLOR, "linewidth": 2},
        whiskerprops={"linewidth": 1},
        ax=ax,
    )

    ax.set_xlabel(f"{label} ({unit})", fontsize=11)
    ax.set_title(f"{label}: Spread and Outliers", fontsize=12, weight="bold")

    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    plt.show()

# -------------------------
# Run analysis
# -------------------------
analyze_distribution(df[bp_col], "Boiling Point", "K")
analyze_distribution(df[mw_col], "Molecular Weight", "g mol$^{-1}$")

