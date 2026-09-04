# ==================================
# Raw Data Selection
# ==================================
import pandas as pd
import numpy as np

# ================= CONFIG =================
INPUT_FILE = "/content/drive/MyDrive/BP_FINAL_MODEL/DIPPER.csv"
OUTPUT_FILE = "/content/drive/MyDrive/BP_ND_Analysis/BP_Dipper_1500.csv"

TARGET_SIZE = 1500
BP_COL = "boiling point(K)"      # adjust if needed
MW_COL = "molecular weight"

BP_BINS = 10         # quantiles
MW_BINS = 10
RANDOM_STATE = 42
# =========================================

# ---------- Load data ----------
df = pd.read_csv(INPUT_FILE)

df = df[[BP_COL, MW_COL] + [c for c in df.columns if c not in [BP_COL, MW_COL]]]

df = df.dropna(subset=[BP_COL, MW_COL]).copy()

N = len(df)
print(f"Total compounds available: {N}")

if N <= TARGET_SIZE:
    print("⚠ Dataset already smaller than target size")
    df.to_csv(OUTPUT_FILE, index=False)
    exit()

# ---------- Create quantile bins ----------
df["bp_bin"] = pd.qcut(df[BP_COL], BP_BINS, duplicates="drop")
df["mw_bin"] = pd.qcut(df[MW_COL], MW_BINS, duplicates="drop")

# ---------- Joint stratification ----------
df["stratum"] = df["bp_bin"].astype(str) + "_" + df["mw_bin"].astype(str)

# ---------- Allocate samples proportionally ----------
stratum_counts = df["stratum"].value_counts()
stratum_frac = stratum_counts / len(df)

stratum_target = (stratum_frac * TARGET_SIZE).round().astype(int)

# Fix rounding mismatch
diff = TARGET_SIZE - stratum_target.sum()
if diff != 0:
    stratum_target.iloc[:abs(diff)] += np.sign(diff)

# ---------- Sample ----------
samples = []
for stratum, n in stratum_target.items():
    group = df[df["stratum"] == stratum]
    if n > 0:
        samples.append(group.sample(min(n, len(group)), random_state=RANDOM_STATE))

subset = pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE)

# ---------- Cleanup ----------
subset = subset.drop(columns=["bp_bin", "mw_bin", "stratum"])

subset.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Selected {len(subset)} representative compounds")
print(f"Saved to: {OUTPUT_FILE}")

# =========================
# Step 2: Data Curation pipeline
# =========================
import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

RAW_PATH = "/content/drive/MyDrive/BP_FINAL_MODEL/BP_subset_4500.csv"
MANUAL_EXCLUSION_PATH = "/content/drive/MyDrive/BP_FINAL_MODEL/manual_exclusion_list.csv"


def canonicalize(smiles):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def embeds_successfully(smiles, attempts=50, seed=42):
    """Attempt 3D conformer generation (proxy for SDF/3D-descriptor feasibility)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        cid = AllChem.EmbedMolecule(mol, randomSeed=seed, maxAttempts=attempts, useRandomCoords=True)
        return cid != -1
    except Exception:
        return False


def main():
    raw = pd.read_csv(RAW_PATH)
    raw["canonical_smiles"] = raw["SMILES"].apply(canonicalize)
    print(f"Step 0 - Raw master file loaded: {len(raw)} compounds "
          f"(internally comprising the NIST- and DIPPR-sourced compounds, "
          f"combined with ~274 overlapping/duplicate entries between the two)")

    manual = pd.read_csv(MANUAL_EXCLUSION_PATH)
    manual["canonical_smiles"] = manual["Name"].map(
        raw.set_index("Name")["canonical_smiles"].to_dict()
    )
    excl_set = set(manual["canonical_smiles"].dropna())
    step1 = raw[~raw["canonical_smiles"].isin(excl_set)].copy()
    print(f"Step 1 - Manual exclusion ({len(manual)} documented compounds, "
          f"removed wherever they occur): {len(step1)} "
          f"(removed {len(raw) - len(step1)})")

    # ---- Step 2: remove invalid/unparsable SMILES
    step2 = step1.dropna(subset=["canonical_smiles"]).copy()
    print(f"Step 2 - Valid SMILES only: {len(step2)} "
          f"(removed {len(step1) - len(step2)})")

    # ---- Step 3: remove duplicate structures (canonical SMILES, keep first)
    step3 = step2.drop_duplicates(subset=["canonical_smiles"], keep="first").copy()
    print(f"Step 3 - Duplicate structures removed: {len(step3)} "
          f"(removed {len(step2) - len(step3)})")

    # ---- Step 4: elemental scope filter (C, H, N, O, P, S, Br, Cl, I only)
    ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "Br", "Cl", "I"}

    def elements_of(formula):
        return set(re.findall(r"[A-Z][a-z]?", str(formula)))

    step3["elements_ok"] = step3["Formula"].apply(
        lambda f: elements_of(f).issubset(ALLOWED_ELEMENTS)
    )
    step4 = step3[step3["elements_ok"]].copy()
    print(f"Step 4 - Elemental scope filter (CHNOPS + Br/Cl/I only): {len(step4)} "
          f"(removed {len(step3) - len(step4)})")

    # ---- Step 5: 3D embedding (SDF generation) feasibility
    step4["embed_ok"] = step4["canonical_smiles"].apply(embeds_successfully)
    step5 = step4[step4["embed_ok"]].copy()
    print(f"Step 5 - 3D embedding feasibility: {len(step5)} "
          f"(removed {len(step4) - len(step5)})")

    # ---- Step 6: heavy-atom count range filter (2-28)
    def heavy_atom_count(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else None

    step5["heavy_atoms"] = step5["canonical_smiles"].apply(heavy_atom_count)
    step6 = step5[(step5["heavy_atoms"] >= 2) & (step5["heavy_atoms"] <= 28)].copy()
    print(f"Step 6 - Heavy-atom count range (2-28): {len(step6)} "
          f"(removed {len(step5) - len(step6)})")

    # ---- Step 7: boiling-point IQR outlier removal (1.5x IQR)
    bp = step6["boiling point (K)"]
    Q1, Q3 = bp.quantile(0.25), bp.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    step7 = step6[(bp >= lower) & (bp <= upper)].copy()
    print(f"Step 7 - Boiling-point IQR outlier removal "
          f"(bounds: {lower:.1f}-{upper:.1f} K): {len(step7)} "
          f"(removed {len(step6) - len(step7)})")

    print(f"\nFinal curated dataset: {len(step7)} compounds "
          f"(target: 5,071)")

    out_cols = ["Name", "CAS No.", "SMILES", "Formula", "molecular weight", "boiling point (K)"]
    step7[out_cols].to_csv("final_dataset_reconstructed.csv", index=False)
    return step7


if __name__ == "__main__":
    main()

