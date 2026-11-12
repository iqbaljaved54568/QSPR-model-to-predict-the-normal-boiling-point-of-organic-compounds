"""
Export of the 839 isomer pairs

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Export the complete isomer-pair list (for deposit) -- final 10-descriptor LightGBM
# Reproduces the isomer analysis (7_isomer_analysis/01_isomer_analysis.py)
# exactly (same model, same grouping rules, same pair rule) and checks that it
# gets 160 groups / 520 compounds / 672 of 839 pairs / p = 8.86e-73.
#
# Saves (for Zenodo / SI):
#   isomer_pairs_839.csv      one row per pair: both compounds' name, CAS, SMILES,
#                             experimental and predicted Tb, differences, correct?
#   isomer_compounds_520.csv  one row per compound in an isomer group
#   isomer_groups_160.csv     per-group summary (with SMILES)
# Also prints: accuracy by size of the experimental difference, and a
# group-level permutation test (does not assume pairs are independent).
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)

import itertools, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import binomtest, spearmanr
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT_DIR = "./BP_FINAL_MODEL/figures/10desc/"
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}
MIN_SIZE, MIN_SPREAD = 2, 3.0

train = pd.read_csv(P + "train_optimal.csv"); test = pd.read_csv(P + "test_optimal.csv").reset_index(drop=True)
feats = [c for c in train.columns if c.startswith(("rdkit2d_", "mordred_"))]   # the 10 descriptors, by name
TB = train.columns[5]
print("VERSION 2 |", len(feats), "descriptors | target:", TB)
sc = StandardScaler().fit(train[feats].values)
model = LGBMRegressor(**LGBM_PARAMS).fit(sc.transform(train[feats].values), train[TB].values)
test["Tb_exp_K"] = test[TB].values
test["Tb_pred_K"] = model.predict(sc.transform(test[feats].values))
print(f"Test R2 = {r2_score(test['Tb_exp_K'], test['Tb_pred_K']):.4f} (published 0.9596)")
test["Formula_RDKit"] = test["smiles"].apply(lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)))

groups = {f: g for f, g in test.groupby("Formula_RDKit")
          if g["smiles"].nunique() >= MIN_SIZE and g["Tb_exp_K"].max() - g["Tb_exp_K"].min() >= MIN_SPREAD}

pairs, comp_rows, grp_rows = [], [], []
for f, g in groups.items():
    g = g.sort_values("Tb_exp_K").reset_index(drop=True)
    for _, r in g.iterrows():
        comp_rows.append({"Formula": f, "Name": r["Name"], "CAS": r["CAS No."], "SMILES": r["smiles"],
                          "Tb_exp_K": r["Tb_exp_K"], "Tb_pred_K": r["Tb_pred_K"]})
    for i, j in itertools.combinations(range(len(g)), 2):       # i has the lower experimental Tb
        a, b = g.loc[i], g.loc[j]
        d_exp, d_pred = b["Tb_exp_K"] - a["Tb_exp_K"], b["Tb_pred_K"] - a["Tb_pred_K"]
        pairs.append({"Formula": f,
                      "A_name": a["Name"], "A_CAS": a["CAS No."], "A_SMILES": a["smiles"],
                      "A_Tb_exp_K": a["Tb_exp_K"], "A_Tb_pred_K": a["Tb_pred_K"],
                      "B_name": b["Name"], "B_CAS": b["CAS No."], "B_SMILES": b["smiles"],
                      "B_Tb_exp_K": b["Tb_exp_K"], "B_Tb_pred_K": b["Tb_pred_K"],
                      "dTb_exp_K": d_exp, "dTb_pred_K": d_pred,
                      "Correct_order": bool(d_exp * d_pred > 0),       # ties count as incorrect
                      "Exp_tie": bool(d_exp == 0)})
    e, p_ = g["Tb_exp_K"].values, g["Tb_pred_K"].values
    rho = spearmanr(e, p_)[0] if len(g) >= 3 else (1.0 if (e[1]-e[0])*(p_[1]-p_[0]) > 0 else -1.0)
    grp_rows.append({"Formula": f, "N_isomers": len(g), "SMILES": " | ".join(g["smiles"]),
                     "Exp_spread_K": e.max()-e.min(), "Pred_spread_K": p_.max()-p_.min(), "Spearman_rho": rho})

pairs = pd.DataFrame(pairs); comps = pd.DataFrame(comp_rows); grps = pd.DataFrame(grp_rows)
k, n = int(pairs["Correct_order"].sum()), len(pairs)
bt = binomtest(k, n, 0.5, alternative="greater")
print(f"Groups: {len(grps)} (160) | compounds: {len(comps)} (520) | correct pairs: {k}/{n} (672/839) "
      f"= {100*k/n:.1f}% | binomial one-sided p = {bt.pvalue:.3g} (8.86e-73)")
print(f"Pairs with tied experimental Tb (counted as incorrect): {int(pairs['Exp_tie'].sum())}")

print("\nAccuracy by size of the experimental difference:")
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 1e9)]:
    m = pairs["dTb_exp_K"].abs().between(lo, hi, inclusive="left")
    if m.any():
        print(f"  |dTb_exp| {lo:>3}-{'inf' if hi > 1e8 else hi:<4} K: {pairs.loc[m,'Correct_order'].mean()*100:5.1f}%  (n = {m.sum()})")

# Group-level permutation test: shuffle predictions WITHIN each group (structure-blind
# null that keeps every group's size and value distribution); pairs need not be independent.
rng = np.random.default_rng(42); N_PERM = 10000
gl = [(g.sort_values("Tb_exp_K")["Tb_exp_K"].values, g.sort_values("Tb_exp_K")["Tb_pred_K"].values)
      for g in groups.values()]
def n_correct(gl_):
    return sum(sum((e[j]-e[i])*(p_[j]-p_[i]) > 0 for i, j in itertools.combinations(range(len(e)), 2))
               for e, p_ in gl_)
null = np.array([n_correct([(e, rng.permutation(p_)) for e, p_ in gl]) for _ in range(N_PERM)])
n_ge = int((null >= k).sum())
p_perm = f"p < {1/(N_PERM+1):.1e}" if n_ge == 0 else f"p = {(n_ge+1)/(N_PERM+1):.2g}"
print(f"\nGroup-level permutation test ({N_PERM} permutations): null mean accuracy = "
      f"{100*null.mean()/n:.1f}%, max = {100*null.max()/n:.1f}%; {p_perm}")

pairs.to_csv(OUT_DIR + "isomer_pairs_839.csv", index=False)
comps.to_csv(OUT_DIR + "isomer_compounds_520.csv", index=False)
grps.to_csv(OUT_DIR + "isomer_groups_160.csv", index=False)
print("\nSaved isomer_pairs_839.csv, isomer_compounds_520.csv, isomer_groups_160.csv to", OUT_DIR)
