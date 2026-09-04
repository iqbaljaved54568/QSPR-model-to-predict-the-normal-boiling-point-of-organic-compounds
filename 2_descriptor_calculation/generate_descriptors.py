#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Unified_descriptors_generation_pipeline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.Descriptors import descList
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from mordred import Calculator, descriptors as mord_desc

# ============================================================
# CONFIG - edit these paths for your environment
# ============================================================
INPUT_CSV   = "/content/drive/MyDrive/BP_FINAL_MODEL/Final_clean-organic.csv"
OUTPUT_DIR  = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors"
SDF_FILE    = "final_organic_3D.sdf"
COMBINED_CSV = "descriptors_combined_2D3D.csv"
SUMMARY_CSV  = "descriptor_summary.csv"

RANDOM_SEED = 42
MORDRED_CHUNK_SIZE = 250   # smaller chunks = more frequent progress updates
MORDRED_NPROC = 4          # set to 1 if you hit multiprocessing issues on Colab

DRAGON_SUMMARY_CSV = "descriptor_dragon_style_summary.csv"
DRAGON_LABELED_CSV = "descriptor_dragon_style_labels.csv"

# SMILES column auto-detection order (first match wins)
SMILES_COL_CANDIDATES = ["smiles", "canonical_smiles", "SMILES"]

DRAGON_CLASSES = {
    "Constitutional": ["n", "Num", "MolWt", "HeavyAtom", "nAtom", "nC", "nH", "nO", "nN",
                       "nRing", "nHeavyAtom", "nBonds", "nArom", "nRot"],
    "Topological": ["Chi", "Kappa", "Path", "Wiener", "Harary", "Zagreb", "Hosoya",
                    "Balaban", "Topo", "Adjacency"],
    "Walk/Path Counting": ["Walk", "Randic", "EigenPath"],
    "Connectivity": ["Kier", "HallKier", "Connectivity", "Bertz", "BalabanJ", "Zagreb", "Hosoya"],
    "Information Content": ["IC", "Information", "Entropy", "Shannon"],
    "2D Autocorrelation": ["ATS", "AATS", "MATS", "GATS", "AUTOC"],
    "Burden Eigenvalues / BCUT": ["BCUT", "burden"],
    "Charge / Electronic": ["PEOE", "Gasteiger", "PartialCharge", "MaxPartialCharge", "MinPartialCharge"],
    "Surface/Volume": ["VSA", "ASA", "Surface", "Volume"],
    "3D Geometry / Shape": ["Asphericity", "Eccentricity", "RadiusOfGyration",
                            "InertialShapeFactor", "NPR1", "NPR2", "Spherocity",
                            "WHIM", "WE", "MoRSE", "RDF", "3D", "G3D"],
    "GETAWAY": ["H", "R", "G", "GETAWAY"],
    "RDF Descriptors": ["RDF"],
    "3D Autocorrelation": ["3DA", "3DM"],
    "Fragment / Functional Groups": ["fr_", "C-", "O-", "N-", "FG", "Joback"],
    "Hydrogen Bonding": ["HBD", "HBA", "HBond"],
    "Hybrid Descriptors": ["MolLogP", "MolMR", "TPSA"],
    "Other": [],
}


def classify_dragon_style(descriptor_columns):
    """Classify each descriptor column into a DRAGON-style category.
    Returns (counts_df, column_to_class dict)."""
    counts = {cls: 0 for cls in DRAGON_CLASSES}
    col_to_class = {}
    for col in descriptor_columns:
        assigned = False
        for cls, keywords in DRAGON_CLASSES.items():
            if keywords and any(k in col for k in keywords):
                counts[cls] += 1
                col_to_class[col] = cls
                assigned = True
                break
        if not assigned:
            counts["Other"] += 1
            col_to_class[col] = "Other"
    counts_df = pd.DataFrame({
        "Descriptor_Class": list(counts.keys()),
        "Count": list(counts.values()),
    })
    counts_df["Percentage"] = (counts_df["Count"] / counts_df["Count"].sum() * 100).round(2)
    return counts_df, col_to_class


def detect_smiles_col(df):
    for c in SMILES_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No SMILES column found. Looked for {SMILES_COL_CANDIDATES}, "
        f"available columns: {list(df.columns)}"
    )


def build_3d_mol(smiles, seed=RANDOM_SEED):
    """Parse SMILES, embed a 3D conformer, and UFF-optimize. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol3d = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(
        mol3d, randomSeed=seed,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
    )
    if status != 0:
        return None
    opt_status = AllChem.UFFOptimizeMolecule(mol3d, maxIters=500)
    if opt_status != 0:
        return None
    return mol3d


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    smiles_col = detect_smiles_col(df)
    print(f"Loaded {len(df):,} compounds. Using SMILES column: '{smiles_col}'")

    metadata_cols = df.columns.tolist()  # keep ALL original metadata columns
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    print(f"After dropping null SMILES: {len(df):,} compounds")

    # ------------------------------------------------------
    # 2. Build 2D + 3D mols for every compound
    # ------------------------------------------------------
    print("Generating 3D conformers (embedding + UFF optimization)...")
    mols_2d, mols_3d, kept_idx = [], [], []
    for idx, smi in enumerate(tqdm(df[smiles_col].astype(str).tolist(), desc="3D embed")):
        m2d = Chem.MolFromSmiles(smi)
        m3d = build_3d_mol(smi)
        if m2d is None or m3d is None:
            continue
        # attach metadata as SDF properties on the 3D mol
        for col in metadata_cols:
            val = df.at[idx, col]
            if pd.notna(val):
                m3d.SetProp(col, str(val))
        mols_2d.append(m2d)
        mols_3d.append(m3d)
        kept_idx.append(idx)

    n_input = len(df)
    n_kept = len(kept_idx)
    print(f"3D embedding succeeded: {n_kept} / {n_input} "
          f"(failed: {n_input - n_kept})")

    metadata_df = df.loc[kept_idx].reset_index(drop=True)

    # ------------------------------------------------------
    # 3. Write SDF (archival / reproducibility)
    # ------------------------------------------------------
    sdf_path = os.path.join(OUTPUT_DIR, SDF_FILE)
    writer = Chem.SDWriter(sdf_path)
    for mol in mols_3d:
        writer.write(mol)
    writer.close()
    print(f"SDF saved: {sdf_path}")

    # ------------------------------------------------------
    # 4a. RDKit 2D descriptors
    # ------------------------------------------------------
    print("Computing RDKit 2D descriptors...")
    rdkit_2d_names = [name for name, _ in descList]

    def calc_rdkit_2d(mol):
        out = {}
        for name, func in descList:
            try:
                out[name] = func(mol)
            except Exception:
                out[name] = np.nan
        return out

    rdkit_2d_data = [calc_rdkit_2d(m) for m in tqdm(mols_2d, desc="RDKit 2D")]
    rdkit_2d_df = pd.DataFrame(rdkit_2d_data).add_prefix("rdkit2d_")
    print(f"  -> {rdkit_2d_df.shape[1]} RDKit 2D descriptors")

    # ------------------------------------------------------
    # 4b. RDKit 3D descriptors
    # ------------------------------------------------------
    print("Computing RDKit 3D descriptors...")

    def calc_rdkit_3d(mol):
        try:
            return Descriptors3D.CalcMolDescriptors3D(mol)
        except Exception:
            return {}

    rdkit_3d_data = [calc_rdkit_3d(m) for m in tqdm(mols_3d, desc="RDKit 3D")]
    rdkit_3d_df = pd.DataFrame(rdkit_3d_data).add_prefix("rdkit3d_")
    print(f"  -> {rdkit_3d_df.shape[1]} RDKit 3D descriptors")

    # ------------------------------------------------------
    # 4c. Mordred descriptors - SINGLE PASS, 2D+3D together
    # ------------------------------------------------------
    print("Computing Mordred descriptors (2D+3D combined, single pass)...")
    calc = Calculator(mord_desc, ignore_3D=False)

    mordred_chunks = []
    start = time.time()
    for i in tqdm(range(0, n_kept, MORDRED_CHUNK_SIZE), desc="Mordred chunks"):
        chunk = mols_3d[i:i + MORDRED_CHUNK_SIZE]
        try:
            df_chunk = calc.pandas(chunk, nproc=MORDRED_NPROC)
        except TypeError:
            df_chunk = calc.pandas(chunk)
        df_chunk.index = range(i, i + len(chunk))
        mordred_chunks.append(df_chunk)

    mordred_df = pd.concat(mordred_chunks, axis=0).reindex(range(n_kept))
    mordred_df.columns = [f"mordred_{c}" for c in mordred_df.columns]
    elapsed = (time.time() - start) / 60
    print(f"  -> {mordred_df.shape[1]} Mordred descriptors ({elapsed:.1f} min)")

    # ------------------------------------------------------
    # 5. Combine all descriptor blocks
    # ------------------------------------------------------
    descriptor_df = pd.concat(
        [rdkit_2d_df.reset_index(drop=True),
         rdkit_3d_df.reset_index(drop=True),
         mordred_df.reset_index(drop=True)],
        axis=1
    )
    print(f"\nRaw combined descriptor count: {descriptor_df.shape[1]}")

    # ------------------------------------------------------
    # 6. Clean: drop all-NaN and constant columns
    # ------------------------------------------------------
    before = descriptor_df.shape[1]
    descriptor_df = descriptor_df.dropna(axis=1, how="all")
    after_nan = descriptor_df.shape[1]

    const_cols = [
        c for c in descriptor_df.columns
        if pd.api.types.is_numeric_dtype(descriptor_df[c])
        and descriptor_df[c].nunique(dropna=True) <= 1
    ]
    descriptor_df = descriptor_df.drop(columns=const_cols)
    after_const = descriptor_df.shape[1]

    print(f"Dropped all-NaN columns:   {before - after_nan}")
    print(f"Dropped constant columns:  {after_nan - after_const}")
    print(f"Final descriptor count:    {after_const}")

    # ------------------------------------------------------
    # 7. Final combined file: metadata first, then descriptors
    # ------------------------------------------------------
    combined_df = pd.concat(
        [metadata_df.reset_index(drop=True), descriptor_df.reset_index(drop=True)],
        axis=1
    )
    out_path = os.path.join(OUTPUT_DIR, COMBINED_CSV)
    combined_df.to_csv(out_path, index=False)
    print(f"\nCombined dataset saved: {out_path}")
    print(f"Shape: {combined_df.shape[0]} rows x {combined_df.shape[1]} columns "
          f"({len(metadata_cols)} metadata + {after_const} descriptors)")

    # ------------------------------------------------------
    # 8. Summary (by source: RDKit 2D / RDKit 3D / Mordred)
    # ------------------------------------------------------
    summary = pd.DataFrame([
        {"source": "RDKit 2D", "raw_count": rdkit_2d_df.shape[1]},
        {"source": "RDKit 3D", "raw_count": rdkit_3d_df.shape[1]},
        {"source": "Mordred (2D+3D)", "raw_count": mordred_df.shape[1]},
        {"source": "TOTAL (raw)", "raw_count": before},
        {"source": "TOTAL (after cleanup)", "raw_count": after_const},
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, SUMMARY_CSV), index=False)
    print("\nSummary (by source):\n", summary.to_string(index=False))

    # ------------------------------------------------------
    # 9. DRAGON-style descriptor classification
    #    (Constitutional, Topological, 3D Geometry, etc.)
    # ------------------------------------------------------
    dragon_summary_df, col_to_class = classify_dragon_style(descriptor_df.columns.tolist())
    dragon_summary_df.to_csv(os.path.join(OUTPUT_DIR, DRAGON_SUMMARY_CSV), index=False)

    labels_df = pd.DataFrame({
        "descriptor": list(col_to_class.keys()),
        "dragon_class": list(col_to_class.values()),
    })
    labels_df.to_csv(os.path.join(OUTPUT_DIR, DRAGON_LABELED_CSV), index=False)

    print("\nDRAGON-style descriptor classification:\n",
          dragon_summary_df.to_string(index=False))
    print(f"\nPer-descriptor class labels saved: "
          f"{os.path.join(OUTPUT_DIR, DRAGON_LABELED_CSV)}")
    print(f"Class summary saved: {os.path.join(OUTPUT_DIR, DRAGON_SUMMARY_CSV)}")

    return combined_df


if __name__ == "__main__":
    main()
