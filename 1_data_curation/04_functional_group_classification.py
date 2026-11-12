"""
Functional-group classification (SMARTS)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

#============================================
#Simplified Functional Group Classification
#============================================
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = "./BP_FINAL_MODEL/Final_clean-organic.csv"
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
