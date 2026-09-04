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
    "/content/drive/MyDrive/BP_FINAL_MODEL/Final_clean-organic.csv"
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

#============================================
#Simplified Functional Group Classification
#============================================
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/Final_clean-organic.csv"
SMILES_COL = "smiles"
BP_COL = "boiling point (K)"

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=[SMILES_COL]).copy()
df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()
print(f"Dataset loaded: {len(df):,} compounds")

# ---------------------------------------------------------------------
# SMARTS pattern registry: (display_name, SMARTS, chemical_family)
# Ordered from most specific to most general within each family.
# ---------------------------------------------------------------------
PATTERN_REGISTRY = [
    ("Aromatic",           "c1ccccc1",                     "Hydrocarbon"),
    ("Alkyne",             "[CX2]#[CX2]",                  "Hydrocarbon"),
    ("Alkene",             "[CX3]=[CX3]",                  "Hydrocarbon"),
    ("Alkane",             "[CX4]",                        "Hydrocarbon"),

    ("Phenol",             "c[OX2H]",                      "Oxygen"),
    ("Carboxylic Acid",    "[CX3](=O)[OX2H1]",             "Oxygen"),
    ("Ester",              "[CX3](=O)[OX2][#6]",           "Oxygen"),
    ("Aldehyde",           "[CX3H1](=O)[#6]",              "Oxygen"),
    ("Ketone",             "[#6][CX3](=O)[#6]",            "Oxygen"),
    ("Ether",              "[OD2]([#6])[#6]",              "Oxygen"),
    ("Alcohol",            "[CX4][OX2H]",                  "Oxygen"),

    ("Amide",              "[NX3][CX3](=O)[#6]",           "Nitrogen"),
    ("Nitrile",            "[NX1]#[CX2]",                  "Nitrogen"),
    ("Nitro",              "[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "Nitrogen"),
    ("Tertiary Amine",     "[NX3H0]([#6])([#6])[#6]",      "Nitrogen"),
    ("Secondary Amine",    "[NX3H1]([#6])[#6]",            "Nitrogen"),
    ("Primary Amine",      "[NX3H2][#6]",                  "Nitrogen"),

    ("Thiol",              "[#16X2H]",                     "Sulfur"),
    ("Thioether",          "[#16X2]([#6])[#6]",            "Sulfur"),

    ("Fluoride",           "[FX1][#6]",                    "Halogen"),
    ("Chloride",           "[ClX1][#6]",                   "Halogen"),
    ("Bromide",            "[BrX1][#6]",                   "Halogen"),
    ("Iodide",             "[IX1][#6]",                    "Halogen"),

    ("Pyridine",           "n1ccccc1",                     "Heteroaromatic"),
    ("Furan",              "o1cccc1",                      "Heteroaromatic"),
    ("Thiophene",          "s1cccc1",                      "Heteroaromatic"),
]

compiled = [(name, Chem.MolFromSmarts(sm), fam) for name, sm, fam in PATTERN_REGISTRY]
compiled = [(n, p, f) for n, p, f in compiled if p is not None]

def classify(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], "Invalid SMILES"
    matches = [name for name, patt, fam in compiled if mol.HasSubstructMatch(patt)]
    if not matches:
        symbols = {a.GetSymbol() for a in mol.GetAtoms()}
        primary = "Alkane" if symbols.issubset({"C", "H"}) else "Unclassified"
        matches = [primary]
    return matches, matches[0]

results = df[SMILES_COL].apply(classify)
df["Classes"] = results.apply(lambda x: x[0])
df["Primary_Class"] = results.apply(lambda x: x[1])
df["N_Classes"] = df["Classes"].apply(len)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
class_counts = df["Classes"].explode().value_counts()
class_pct = (class_counts / len(df) * 100).round(1)
summary = pd.DataFrame({"Count": class_counts, "% of dataset": class_pct})

print("\nFunctional group frequency (compounds may belong to >1 class):")
print(summary.to_string())

print(f"\nNumber of distinct functional-group classes represented: {len(class_counts)}")
print(f"Average number of functional groups per compound: {df['N_Classes'].mean():.2f}")
print(f"Compounds with more than one functional group: "
      f"{(df['N_Classes']>1).sum():,} ({(df['N_Classes']>1).mean()*100:.1f}%)")

summary.to_csv("functional_group_summary.csv")
df[["Name","CAS No.","smiles","Primary_Class","N_Classes",BP_COL]].to_csv(
    "compounds_with_class_labels.csv", index=False
)

# ---------------------------------------------------------------------
# One simple, publication-ready bar chart
# ---------------------------------------------------------------------
top = summary.sort_values("Count", ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top.index, top["Count"], color="#2C6E8A", edgecolor="white")
for i, (cnt, pct) in enumerate(zip(top["Count"], top["% of dataset"])):
    ax.text(cnt + 20, i, f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=8)
ax.set_xlabel("Number of compounds")
ax.set_title(f"Functional-group diversity across the final dataset (N={len(df):,})",
             fontsize=11, loc="left", fontweight="bold")
ax.set_xlim(0, top["Count"].max()*1.25)
plt.tight_layout()
plt.savefig("functional_group_diversity.png", dpi=300, bbox_inches="tight")
print("\nFigure saved: functional_group_diversity.png")

# !pip install mordred

