"""
Stage at which each of the 2,054 descriptors was removed or kept (Table S5)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Complete descriptor mapping: 2,054 raw descriptors -> final 10
#
# Uses the stage files saved by the selection scripts, plus the RDKit/Mordred name
# lists (to recover the 138 constant descriptors that were never saved).
# Output (in BP_FINAL_MODEL/descriptors/):
#   descriptor_mapping_2054.csv / .xlsx   one row per descriptor: source, the stage
#                                          it reached, why it was removed, key values
#   descriptor_stage_summary.csv           counts per stage and source
#   table_SI_descriptor_stages.tex         LaTeX summary table
# Also checks whether running the Spearman filter on the TRAINING SET ONLY
# would have changed which descriptors passed (test-set leakage check).
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit==2025.9.3", "mordredcommunity", "openpyxl"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.Descriptors import descList
from mordred import Calculator, descriptors as mord_desc

D = "./BP_FINAL_MODEL/descriptors/"
S = D + "train_test_split/"
F = "./BP_FINAL_MODEL/figures/backward_elimination_35desc/"
TB = "boiling point (K)"
hdr = lambda path: [c for c in pd.read_csv(path, nrows=0).columns]

# ---- 1. Raw 2,054 names (from the libraries, same prefixes as the pipeline) --
m = Chem.AddHs(Chem.MolFromSmiles("CCO")); AllChem.EmbedMolecule(m, randomSeed=42)
raw = ([("rdkit2d_" + n, "RDKit 2D") for n, _ in descList] +
       [("rdkit3d_" + n, "RDKit 3D") for n in Descriptors3D.CalcMolDescriptors3D(m)] +
       [("mordred_" + str(d), "Mordred") for d in Calculator(mord_desc, ignore_3D=False).descriptors])
raw = pd.DataFrame(raw, columns=["Descriptor", "Source"]).drop_duplicates("Descriptor")
print("Raw counts by source:", raw["Source"].value_counts().to_dict(), "| total", len(raw), "(expected 217/11/1826 = 2054)")

# ---- 2. Stage membership from the saved files --------------------------------
comb = pd.read_csv(D + "descriptors_combined_2D3D.csv", low_memory=False)
meta = list(comb.columns[:6]); desc_cols = list(comb.columns[6:])
print("Combined file:", len(desc_cols), "descriptors (expected 1916)")
missing = set(desc_cols) - set(raw["Descriptor"])
if missing:
    print(f"NOTE: {len(missing)} names in the combined file not in the library list "
          f"(e.g. {sorted(missing)[:5]}) -- added as-is.")
    raw = pd.concat([raw, pd.DataFrame({"Descriptor": sorted(missing), "Source": "unknown"})])

X = comb[desc_cols]
numeric = set(X.select_dtypes(include=[np.number]).columns)
Xn = X[[c for c in desc_cols if c in numeric]]
zero_frac = (Xn == 0).mean()
after_zero = set(zero_frac[zero_frac <= 0.50].index)
stage = {
    "constant": set(desc_cols),                                  # survived constant removal
    "numeric": numeric,
    "zeros": after_zero,
    "variance": set(hdr(D + "stage_variance_filtered.csv")[6:]),
    "spearman": set(hdr(D + "stage_spearman_filtered.csv")[6:]),
    "vif": set(hdr(D + "stage2_reduced.csv")[6:]),
    "top35": set(hdr(S + "train_35desc.csv" if __import__("os").path.exists(S + "train_35desc.csv")
                     else S + "train_38desc.csv")[6:]),
    "final10": set(c for c in hdr(S + "train_optimal.csv")[6:]),
}
order = [("constant", "Constant across all compounds"),
         ("numeric", "Failed to calculate for one or more compounds (non-numeric)"),
         ("zeros", "More than 50% zero values"),
         ("variance", "Variance <= 0.01"),
         ("spearman", "|Spearman rho| with Tb < 0.10"),
         ("vif", "Removed by iterative VIF filter (VIF >= 10)"),
         ("top35", "Not among top 35 by RF/SHAP importance"),
         ("final10", "Removed by LightGBM backward elimination")]
for k, _ in order:
    print(f"  {k:9s}: {len(stage[k])}")

def fate(d):
    for k, reason in order:
        if d not in stage[k]:
            return k, reason
    return "final", "Retained in final model"
raw[["Last_stage_failed", "Fate"]] = raw["Descriptor"].apply(lambda d: pd.Series(fate(d)))
for k, _ in order:
    raw["In_" + k] = raw["Descriptor"].isin(stage[k])

# ---- 3. Key values per descriptor --------------------------------------------
y = comb[TB].values
var_set = sorted(stage["variance"])
Xv = comb[var_set].fillna(comb[var_set].median())
raw["Zero_fraction"] = raw["Descriptor"].map(zero_frac)
raw["Variance"] = raw["Descriptor"].map(Xn.fillna(Xn.median()).var(ddof=0))
rho_all = {c: spearmanr(Xv[c], y)[0] for c in var_set}
raw["Spearman_rho_all_5071"] = raw["Descriptor"].map(rho_all)

# leakage check: Spearman on training compounds only
train_smiles = set(pd.read_csv(S + "train_optimal.csv", usecols=["smiles"])["smiles"])
tr = comb["smiles"].isin(train_smiles).values
print(f"\nTraining compounds matched in combined file: {tr.sum()} (expected 4056)")
rho_tr = {c: spearmanr(Xv.loc[tr, c], y[tr])[0] for c in var_set}
raw["Spearman_rho_train_only"] = raw["Descriptor"].map(rho_tr)
pass_all = {c for c in var_set if abs(rho_all[c]) >= 0.10}
pass_tr = {c for c in var_set if abs(rho_tr[c]) >= 0.10}
print(f"Reproduces saved Spearman stage: {pass_all == stage['spearman']} ({len(pass_all)} vs {len(stage['spearman'])})")
print(f"Spearman filter, all 5,071: {len(pass_all)} pass | training only: {len(pass_tr)} pass | "
      f"differ: {len(pass_all ^ pass_tr)} descriptors {sorted(pass_all ^ pass_tr)[:10]}")

vif94 = sorted(stage["vif"]); C = comb[vif94].fillna(comb[vif94].median()).corr().values
raw["VIF_in_final_94"] = raw["Descriptor"].map(dict(zip(vif94, np.diag(np.linalg.inv(C + np.eye(len(C))*1e-10)))))
try:
    rf = pd.read_csv(D + "rf_descriptor_ranking.csv")
    rf["RF_rank"] = range(1, len(rf) + 1)
    raw["RF_importance_rank_(of_94)"] = raw["Descriptor"].map(rf.set_index("descriptor")["RF_rank"])
except Exception as e:
    print("RF ranking not added:", e)

# ---- 4. Save ------------------------------------------------------------------
stage_order = {k: i for i, (k, _) in enumerate(order)}; stage_order["final"] = 99
raw = raw.sort_values(by=["Last_stage_failed", "Source", "Descriptor"],
                      key=lambda s: s.map(stage_order) if s.name == "Last_stage_failed" else s,
                      ascending=[False, True, True])
raw.to_csv(D + "descriptor_mapping_2054.csv", index=False)
raw.to_excel(D + "descriptor_mapping_2054.xlsx", index=False)

labels = ["Raw (calculated)", "After removing constant descriptors", "After removing non-numeric",
          "After >50% zero filter", "After variance filter (> 0.01)", "After Spearman filter (|rho| >= 0.10)",
          "After VIF filter (< 10)", "After RF/SHAP importance (top 35)", "After backward elimination (final)"]
rows = [raw["Source"].value_counts()]
for k, _ in order:
    rows.append(raw.loc[raw["In_" + k], "Source"].value_counts())
summ = pd.DataFrame(rows, index=labels).fillna(0).astype(int)
summ["Total"] = summ.sum(axis=1)
print("\n", summ.to_string())
summ.to_csv(D + "descriptor_stage_summary.csv")

cols = [c for c in ["RDKit 2D", "RDKit 3D", "Mordred"] if c in summ.columns] + ["Total"]
tex = ("\\begin{table}[htbp]\n\\centering\n\\caption{Number of descriptors retained at each stage of the "
       "descriptor-selection pipeline, by source (RDKit 2025.9.3; Mordred 1.2.0). The complete per-descriptor "
       "mapping is given in descriptor\\_mapping\\_2054.xlsx.}\n\\label{tab:descriptor_stages}\n\\small\n"
       "\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\nStage & " + " & ".join(cols) + " \\\\\n\\midrule\n" +
       "\n".join(f"{i} & " + " & ".join(str(summ.loc[i, c]) for c in cols) + " \\\\" for i in summ.index) +
       "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n").replace(">=", "$\\geq$").replace("rho", "$\\rho$").replace("> 0", "$>$ 0").replace("< 10", "$<$ 10").replace(">50%", "$>$50\\%")
open(D + "table_SI_descriptor_stages.tex", "w").write(tex)
print("\nFinal 10:", sorted(stage["final10"]))
print("Saved descriptor_mapping_2054.csv/.xlsx, descriptor_stage_summary.csv, table_SI_descriptor_stages.tex to", D)
