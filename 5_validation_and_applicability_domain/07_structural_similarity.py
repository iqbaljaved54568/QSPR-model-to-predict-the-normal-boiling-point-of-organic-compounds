"""
Structural similarity and error vs similarity (Table 10, Figure 9)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Structural similarity between training and prediction sets, and error vs similarity
#. No retraining.
#
# Similarity: Tanimoto on Morgan fingerprints (radius 2, 2048 bits, RDKit).
# For every test compound and every external (OPERA) compound: similarity to its
# NEAREST training compound (max Tanimoto) and mean similarity to its 5 nearest.
# Reports: distribution of nearest-neighbour similarity; MAE/RMSE per similarity bin;
# Spearman correlation between similarity and |error|; % of test compounds with a
# near-analogue (>= 0.7) in training. Figure: (a) histogram of nearest-neighbour
# similarity, test vs external; (b) MAE per similarity bin for both sets.
# Needs external_validation_predictions.csv from external_validation_opera.py.
# Outputs -> BP_FINAL_MODEL/similarity/
# =============================================================================
import subprocess, sys, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog("rdApp.*")

B = "./BP_FINAL_MODEL/"
P = B + "descriptors/train_test_split/"
EXT = B + "external_validation/external_validation_predictions.csv"
OUT = B + "similarity/"; os.makedirs(OUT, exist_ok=True)
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}
BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0001]
LABELS = ["< 0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">= 0.9"]

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
sc = StandardScaler().fit(tr[feats].values)
model = LGBMRegressor(**LGBM_PARAMS).fit(sc.transform(tr[feats].values), tr[TB].values)
test = pd.DataFrame({"smiles": te["smiles"], "Tb": te[TB],
                     "Pred": model.predict(sc.transform(te[feats].values)), "Set": "Internal test"})
ex = pd.read_csv(EXT)
ext = pd.DataFrame({"smiles": ex["smiles"], "Tb": ex["Tb_K"], "Pred": ex["Tb_pred_K"],
                    "Set": "External (OPERA)", "Source": ex["Source"], "Inside_AD": ex["Inside_AD"]})

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = lambda s: gen.GetFingerprint(Chem.MolFromSmiles(s))
fp_tr = [fp(s) for s in tr["smiles"]]
def nn_sim(smiles):
    best, top5 = [], []
    for s in smiles:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp(s), fp_tr))
        best.append(sims.max()); top5.append(np.sort(sims)[-5:].mean())
    return np.array(best), np.array(top5)
for d in (test, ext):
    d["NN_sim"], d["Top5_sim"] = nn_sim(d["smiles"])
    d["AbsErr"] = (d["Pred"] - d["Tb"]).abs(); d["Err"] = d["Pred"] - d["Tb"]
    d["Bin"] = pd.cut(d["NN_sim"], BINS, labels=LABELS, right=False)

pd.set_option("display.width", 250)
print("Nearest-neighbour Tanimoto similarity to the training set (Morgan r=2, 2048 bits):")
for d in (test, ext):
    s = d["NN_sim"]
    print(f"  {d.Set.iloc[0]:17s} n={len(d):5d}  median {s.median():.2f}  IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f}  "
          f">=0.9: {100*(s>=0.9).mean():.1f}%  >=0.7: {100*(s>=0.7).mean():.1f}%  <0.5: {100*(s<0.5).mean():.1f}%")

def by_bin(d):
    g = d.groupby("Bin", observed=False)
    return pd.DataFrame({"n": g.size(), "% of set": (100 * g.size() / len(d)).round(1),
                         "MAE (K)": g["AbsErr"].mean().round(2),
                         "RMSE (K)": g["Err"].apply(lambda e: np.sqrt((e**2).mean()) if len(e) else np.nan).round(2),
                         "Median |e| (K)": g["AbsErr"].median().round(2)})
rows = []
for d in [test, ext, ext[ext.Source == "OPERA PHYSPROP (BP_QR)"].assign(Set="External PHYSPROP only")]:
    t = by_bin(d); t.insert(0, "Set", d.Set.iloc[0]); rows.append(t.reset_index())
    rho, p = spearmanr(d["NN_sim"], d["AbsErr"])
    print(f"\n{d.Set.iloc[0]}: Spearman rho(similarity, |error|) = {rho:+.3f} (p = {p:.2g})")
    print(t.to_string())
T = pd.concat(rows, ignore_index=True); T.to_csv(OUT + "error_by_similarity.csv", index=False)

# within the AD: does low similarity still raise the error?
e_ad = ext[ext.Inside_AD]
print("\nExternal compounds INSIDE the leverage AD, by similarity bin:\n", by_bin(e_ad).to_string())

pd.concat([test, ext]).to_csv(OUT + "similarity_per_compound.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
cols = {"Internal test": "#C44E52", "External (OPERA)": "#4C72B0"}
for d in (test, ext):
    ax[0].hist(d["NN_sim"], bins=np.linspace(0, 1, 41), density=True, alpha=0.55,
               color=cols[d.Set.iloc[0]], label=f"{d.Set.iloc[0]} (median {d.NN_sim.median():.2f})")
ax[0].set_xlabel("Tanimoto similarity to nearest training compound"); ax[0].set_ylabel("Density")
ax[0].set_title("(a) Similarity to the training set", loc="left"); ax[0].legend(fontsize=8.5)
x = np.arange(len(LABELS)); w = 0.38
for k, d in enumerate((test, ext)):
    t = by_bin(d)
    bars = ax[1].bar(x + (k - 0.5) * w, t["MAE (K)"], w, color=cols[d.Set.iloc[0]], alpha=0.85, label=d.Set.iloc[0])
    for b, n in zip(bars, t["n"]):
        if n: ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center", va="bottom", fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels(LABELS); ax[1].set_xlabel("Similarity to nearest training compound")
ax[1].set_ylabel("MAE (K)"); ax[1].set_title("(b) Prediction error by similarity", loc="left"); ax[1].legend(fontsize=8.5)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + "fig_similarity_error.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + "fig_similarity_error.pdf", bbox_inches="tight"); plt.show()
print("\nSaved error_by_similarity.csv, similarity_per_compound.csv and fig_similarity_error.png/pdf to", OUT)
