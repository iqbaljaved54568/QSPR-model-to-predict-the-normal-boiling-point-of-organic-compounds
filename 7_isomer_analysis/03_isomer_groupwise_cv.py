"""
Isomer groups kept together in cross-validation (Table 13)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Isomer discrimination with constitutional-isomer groups kept together
#.
# No re-tuning: the final LightGBM hyperparameters are used throughout.
#
# Part 1  Leakage check on the published analysis: for each of the 160 test-set
#         isomer groups, how many training compounds share its molecular formula?
#         Accuracy on the groups whose formula is absent from training.
# Part 2  Group-wise re-evaluation: 10-fold cross-validation on all 5,071 compounds
#         in which every molecular formula sits entirely in one fold
#         (StratifiedGroupKFold, stratified by Tb). Every compound is predicted by a
#         model that has seen NONE of its isomers. Pairwise ranking accuracy is then
#         computed on ALL isomer groups in the dataset, with the same rules as before
#         (>= 2 distinct SMILES, experimental spread >= 3 K, ties count as wrong).
#         Control: the same pairs predicted by ordinary (random) 10-fold CV, where
#         isomers can be in training. The difference = effect of seeing isomers.
# Outputs -> BP_FINAL_MODEL/figures/10desc/
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import itertools, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from scipy.stats import binomtest, spearmanr
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RDLogger.DisableLog("rdApp.*")

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/10desc/"
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}
MIN_SIZE, MIN_SPREAD, N_BOOT, N_PERM = 2, 3.0, 2000, 2000
rng = np.random.default_rng(42)

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
d = pd.concat([tr.assign(Set="train"), te.assign(Set="test")], ignore_index=True)
d["Formula"] = d["smiles"].apply(lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)))
X, y = d[feats].values, d[TB].values
print(f"{len(d)} compounds ({(d.Set=='train').sum()} train / {(d.Set=='test').sum()} test), "
      f"{d.Formula.nunique()} distinct formulas, {len(feats)} descriptors")

def fit_predict(tr_idx, te_idx):
    sc = StandardScaler().fit(X[tr_idx])
    return LGBMRegressor(**LGBM_PARAMS).fit(sc.transform(X[tr_idx]), y[tr_idx]).predict(sc.transform(X[te_idx]))

def isomer_groups(df):
    return {f: g.sort_values(TB) for f, g in df.groupby("Formula")
            if g["smiles"].nunique() >= MIN_SIZE and g[TB].max() - g[TB].min() >= MIN_SPREAD}

def evaluate(groups, pred_col, label):
    """Pairwise ranking accuracy, binomial + within-group permutation test, group bootstrap CI."""
    G = [(g[TB].values, g[pred_col].values) for g in groups.values()]
    def correct(e, p):
        return sum((e[j] - e[i]) * (p[j] - p[i]) > 0 for i, j in itertools.combinations(range(len(e)), 2))
    c = np.array([correct(e, p) for e, p in G]); n = np.array([len(e) * (len(e) - 1) // 2 for e, _ in G])
    k, N = int(c.sum()), int(n.sum())
    rho = np.array([spearmanr(e, p)[0] if len(e) >= 3 else (1.0 if (e[1]-e[0])*(p[1]-p[0]) > 0 else -1.0)
                    for e, p in G])
    boot = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(G), len(G)); boot.append(c[b].sum() / n[b].sum())
    null = np.array([sum(correct(e, rng.permutation(p)) for e, p in G) for _ in range(N_PERM)])
    n_ge = int((null >= k).sum())
    return {"Analysis": label, "Groups": len(G), "Compounds": int(sum(len(e) for e, _ in G)),
            "Pairs": N, "Correct": k, "Accuracy %": round(100 * k / N, 1),
            "95% CI (group bootstrap)": f"{100*np.percentile(boot, 2.5):.1f}-{100*np.percentile(boot, 97.5):.1f}",
            "Binomial p": f"{binomtest(k, N, 0.5, alternative='greater').pvalue:.2g}",
            "Permutation p": f"< {1/(N_PERM+1):.1e}" if n_ge == 0 else f"{(n_ge+1)/(N_PERM+1):.2g}",
            "Mean Spearman rho": round(np.nanmean(rho), 3),
            "Mean exp. spread K": round(np.mean([e.max()-e.min() for e, _ in G]), 1),
            "Mean pred. spread K": round(np.mean([p.max()-p.min() for _, p in G]), 1)}

# ---- Part 1: published analysis + leakage check ----------------------------------
itr, ite = np.where(d.Set == "train")[0], np.where(d.Set == "test")[0]
d.loc[ite, "Pred_final"] = fit_predict(itr, ite)
test_groups = isomer_groups(d.loc[ite])
train_counts = d.loc[itr, "Formula"].value_counts()
seen = {f: int(train_counts.get(f, 0)) for f in test_groups}
s = pd.Series(seen)
print(f"\nPART 1. Test-set isomer groups: {len(test_groups)} (published: 160)")
print(f"  Groups with >= 1 isomer in the training set: {(s > 0).sum()} ({100*(s > 0).mean():.1f}%)")
print(f"  Training isomers per group: median {s.median():.0f}, max {s.max()}")
unseen = {f: g for f, g in test_groups.items() if seen[f] == 0}
rows = [evaluate(test_groups, "Pred_final", "A. Published: test-set groups, final model")]
if len(unseen) >= 5:
    rows.append(evaluate(unseen, "Pred_final", "B. Subset of A whose formula is absent from training"))
else:
    print(f"  Only {len(unseen)} groups have no training isomer - too few to evaluate separately.")

# ---- Part 2: group-wise cross-validation over the whole dataset ------------------
bins = pd.qcut(y, 10, labels=False, duplicates="drop")
d["Pred_groupCV"] = np.nan; d["Pred_randomCV"] = np.nan
for a, b in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(X, bins, groups=d.Formula):
    assert not set(d.Formula.iloc[a]) & set(d.Formula.iloc[b])         # no formula on both sides
    d.loc[b, "Pred_groupCV"] = fit_predict(a, b)
for a, b in KFold(10, shuffle=True, random_state=42).split(X):
    d.loc[b, "Pred_randomCV"] = fit_predict(a, b)
for col, lab in [("Pred_groupCV", "group-wise (formula) 10-fold CV"), ("Pred_randomCV", "random 10-fold CV")]:
    e = d[col] - d[TB]
    print(f"  Overall, {lab}: R2 = {r2_score(d[TB], d[col]):.4f}, MAE = {e.abs().mean():.2f} K, "
          f"RMSE = {np.sqrt((e**2).mean()):.2f} K")

all_groups = isomer_groups(d)
rows.append(evaluate(all_groups, "Pred_groupCV", "C. All isomer groups, group-wise CV (no isomer seen)"))
rows.append(evaluate(all_groups, "Pred_randomCV", "D. Same pairs as C, random CV (control)"))
tg = {f: g for f, g in isomer_groups(d).items() if f in test_groups}
rows.append(evaluate(tg, "Pred_groupCV", "E. Formulas of the 160 test groups, group-wise CV"))

R = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n", R.to_string(index=False))
R.to_csv(OUT + "isomer_groupwise_summary.csv", index=False)
d[["Name", "CAS No.", "smiles", "Formula", "Set", TB, "Pred_final", "Pred_groupCV", "Pred_randomCV"]].to_csv(
    OUT + "isomer_groupwise_predictions.csv", index=False)
s.rename("n_training_isomers").rename_axis("Formula").to_csv(OUT + "isomer_test_groups_training_overlap.csv")
print("\nSaved isomer_groupwise_summary.csv, isomer_groupwise_predictions.csv, "
      "isomer_test_groups_training_overlap.csv to", OUT)
