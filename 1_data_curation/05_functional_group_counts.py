"""
Functional-group definitions and counts (Table S3)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Functional-group definitions and counts for the final dataset
# No model is involved.
#
# Starts from the 26 SMARTS patterns of the original classification cell (1_data_curation/04_functional_group_classification.py) and
# makes three corrections:
#   1. "Alkane" was the pattern [CX4] (any sp3 carbon), which matched 92% of compounds
#      and inflated the number of groups per compound. Alkane is now a CLASS rule:
#      only C and H, no multiple bonds, no aromatic atoms (cyclic or acyclic).
#   2. Two classes named in the manuscript but never searched for are added:
#      imidazole and organophosphorus compounds (any P; P is a retained element).
#   3. "Fluoride" is reported but cannot occur (F was excluded during curation).
# Outputs: multi-label counts (a compound can have several groups), a single primary
# class per compound (priority order = order of the list), counts per family, and the
# compounds still unclassified. Saves functional_group_counts.csv and a LaTeX table.
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

INPUT_FILE = "./BP_FINAL_MODEL/Final_clean-organic.csv"
OUT = "./BP_FINAL_MODEL/"
df = pd.read_csv(INPUT_FILE).dropna(subset=["smiles"]).copy()
print(f"Dataset: {len(df):,} compounds")

# (name, SMARTS, family) -- priority order for the primary class (most specific first)
REG = [
    ("Organophosphorus",  "[#15]",                                "Phosphorus"),       # added
    ("Imidazole",         "c1ncc[nX3]1",                          "Heteroaromatic"),   # added
    ("Pyridine",          "n1ccccc1",                             "Heteroaromatic"),
    ("Furan",             "o1cccc1",                              "Heteroaromatic"),
    ("Thiophene",         "s1cccc1",                              "Heteroaromatic"),
    ("Carboxylic acid",   "[CX3](=O)[OX2H1]",                     "Oxygen"),
    ("Amide",             "[NX3][CX3](=O)[#6]",                   "Nitrogen"),
    ("Nitro",             "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",   "Nitrogen"),
    ("Nitrile",           "[NX1]#[CX2]",                          "Nitrogen"),
    ("Ester",             "[CX3](=O)[OX2][#6]",                   "Oxygen"),
    ("Aldehyde",          "[CX3H1](=O)[#6]",                      "Oxygen"),
    ("Ketone",            "[#6][CX3](=O)[#6]",                    "Oxygen"),
    ("Phenol",            "c[OX2H]",                              "Oxygen"),
    ("Alcohol",           "[CX4][OX2H]",                          "Oxygen"),
    ("Primary amine",     "[NX3H2][#6]",                          "Nitrogen"),
    ("Secondary amine",   "[NX3H1]([#6])[#6]",                    "Nitrogen"),
    ("Tertiary amine",    "[NX3H0]([#6])([#6])[#6]",              "Nitrogen"),
    ("Thiol",             "[#16X2H]",                             "Sulfur"),
    ("Thioether",         "[#16X2]([#6])[#6]",                    "Sulfur"),
    ("Ether",             "[OD2]([#6])[#6]",                      "Oxygen"),
    ("Iodide",            "[IX1][#6]",                            "Halogen"),
    ("Bromide",           "[BrX1][#6]",                           "Halogen"),
    ("Chloride",          "[ClX1][#6]",                           "Halogen"),
    ("Fluoride",          "[FX1][#6]",                            "Halogen"),
    ("Benzene ring",      "c1ccccc1",                             "Hydrocarbon"),
    ("Alkyne",            "[CX2]#[CX2]",                          "Hydrocarbon"),
    ("Alkene",            "[CX3]=[CX3]",                          "Hydrocarbon"),
]
PATT = [(n, Chem.MolFromSmarts(s), f, s) for n, s, f in REG]
assert all(p is not None for _, p, _, _ in PATT)

def is_alkane(m):
    return ({a.GetSymbol() for a in m.GetAtoms()} <= {"C", "H"}
            and not any(a.GetIsAromatic() for a in m.GetAtoms())
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in m.GetBonds()))

def classify(smi):
    m = Chem.MolFromSmiles(smi)
    hits = [n for n, p, _, _ in PATT if m.HasSubstructMatch(p)]
    if is_alkane(m):
        hits.append("Alkane (saturated hydrocarbon)")
    return hits

df["Groups"] = df["smiles"].apply(classify)
df["N_groups"] = df["Groups"].apply(len)
df["Primary"] = df["Groups"].apply(lambda g: g[0] if g else "Unclassified")
fam_of = {n: f for n, _, f, _ in PATT} | {"Alkane (saturated hydrocarbon)": "Hydrocarbon",
                                          "Unclassified": "Unclassified"}
smarts_of = {n: s for n, _, _, s in PATT} | {"Alkane (saturated hydrocarbon)": "C/H only, all single bonds, non-aromatic"}

multi = df["Groups"].explode().value_counts()
prim = df["Primary"].value_counts()
names = [n for n, *_ in REG] + ["Alkane (saturated hydrocarbon)"]
T = pd.DataFrame({"Family": [fam_of[n] for n in names], "Group": names,
                  "Definition (SMARTS)": [smarts_of[n] for n in names],
                  "Compounds containing (n)": [int(multi.get(n, 0)) for n in names],
                  "Primary class (n)": [int(prim.get(n, 0)) for n in names]})
T["Containing (%)"] = (100 * T["Compounds containing (n)"] / len(df)).round(1)
T = T.sort_values(["Family", "Compounds containing (n)"], ascending=[True, False])
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
print(T.to_string(index=False))

present = T[T["Compounds containing (n)"] > 0]
print(f"\nGroups searched: {len(T)}; groups present in the dataset: {len(present)}")
print(f"Unclassified compounds: {int((df.Primary == 'Unclassified').sum())}")
print(f"Primary-class counts sum to {int(T['Primary class (n)'].sum() + (df.Primary=='Unclassified').sum())} (should be {len(df)})")
print(f"Mean number of groups per compound: {df.N_groups.mean():.2f}; "
      f"compounds with > 1 group: {(df.N_groups > 1).sum():,} ({100*(df.N_groups > 1).mean():.1f}%)")
fam = df.assign(Family=df["Primary"].map(fam_of)).groupby("Family").size().sort_values(ascending=False)
print("\nCompounds per family (by primary class):\n", fam.to_string())
print("\nUnclassified compounds (check these):")
print(df[df.Primary == "Unclassified"][["Name", "smiles"]].to_string(index=False))

T.to_csv(OUT + "functional_group_counts.csv", index=False)
with open(OUT + "functional_group_counts.tex", "w") as f:
    for _, r in T.iterrows():
        sm = r["Definition (SMARTS)"].replace("#", r"\#").replace("$", r"\$").replace("_", r"\_")
        f.write(f"{r['Family']} & {r['Group']} & \\texttt{{{sm}}} & {r['Compounds containing (n)']:,} & "
                f"{r['Containing (%)']:.1f} & {r['Primary class (n)']:,} \\\\\n")
print("\nSaved functional_group_counts.csv and functional_group_counts.tex to", OUT)
