"""
Chemprop D-MPNN and CheMeleon benchmark (Table S20)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Deep-learning benchmark
# Time: ~30-60 min on a T4 (2 models x 3 seeds). Resumable: finished runs are skipped.
#
# Models (graph neural networks that learn their own representation from the structure,
# i.e. no hand-picked descriptors), trained on EXACTLY the same training set and
# evaluated on the same test set (1,015) and external OPERA set (4,820) as the final model:
#   1. Chemprop D-MPNN, trained from scratch (Heid et al., JCIM 2024; Chemprop v2)
#   2. Chemprop + CheMeleon: D-MPNN foundation model pre-trained on Mordred descriptors
#      of 1M PubChem molecules, fine-tuned here (Burns et al., arXiv:2506.15792, 2025)
# Validation set for early stopping: random 10% of the TRAINING set (seed 42); the test
# set is never used during training. Default Chemprop architecture/hyperparameters,
# max 150 epochs, early stopping patience 20. Three random seeds per model.
# Outputs -> BP_FINAL_MODEL/deep_learning_benchmark/
# =============================================================================
import subprocess, sys, os, glob, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chemprop>=2.2"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

B = "./BP_FINAL_MODEL/"
P = B + "descriptors/train_test_split/"
OUT = B + "deep_learning_benchmark/"; os.makedirs(OUT, exist_ok=True)
SEEDS, EPOCHS, PATIENCE = [0, 1, 2], 150, 20
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - this will be very slow")

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
ext = pd.read_csv(B + "external_validation/external_validation_predictions.csv")

# data files (same split as the paper; 10% of training kept aside for early stopping)
rng = np.random.default_rng(42)
val_mask = np.zeros(len(tr), bool); val_mask[rng.choice(len(tr), int(0.1 * len(tr)), replace=False)] = True
cp = pd.concat([pd.DataFrame({"smiles": tr["smiles"], "Tb": tr[TB], "split": np.where(val_mask, "val", "train")}),
                pd.DataFrame({"smiles": te["smiles"], "Tb": te[TB], "split": "test"})], ignore_index=True)
cp.to_csv(OUT + "chemprop_data.csv", index=False)
te[["smiles"]].to_csv(OUT + "test_smiles.csv", index=False)
ext[["smiles"]].to_csv(OUT + "external_smiles.csv", index=False)

MODELS = {"Chemprop D-MPNN (from scratch)": [],
          "Chemprop + CheMeleon (fine-tuned)": ["--from-foundation", "CHEMELEON"]}

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError("chemprop failed")

def metrics(y, p):
    e = p - y
    return r2_score(y, p), np.abs(e).mean(), np.sqrt((e ** 2).mean())

phys = (ext["Source"] == "OPERA PHYSPROP (BP_QR)").values
rows, preds = [], {}
for name, extra in MODELS.items():
    tag = "chemeleon" if extra else "dmpnn"
    for s in SEEDS:
        d = f"{OUT}{tag}_seed{s}/"
        if not glob.glob(d + "**/best.pt", recursive=True):
            t0 = time.time()
            run(["chemprop", "train", "-i", OUT + "chemprop_data.csv", "-s", "smiles", "--target-columns", "Tb",
                 "--splits-column", "split", "-t", "regression", "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
                 "--pytorch-seed", str(s), "--metrics", "mae", "rmse", "r2", "-o", d] + extra)
            print(f"  trained {name}, seed {s} in {(time.time() - t0) / 60:.1f} min")
        for split, f in [("test", "test_smiles.csv"), ("ext", "external_smiles.csv")]:
            pf = f"{d}pred_{split}.csv"
            if not os.path.exists(pf):
                run(["chemprop", "predict", "-i", OUT + f, "-s", "smiles", "--model-paths", d, "-o", pf])
        pt = pd.read_csv(d + "pred_test.csv").iloc[:, -1].values
        pe = pd.read_csv(d + "pred_ext.csv").iloc[:, -1].values
        preds[(name, s)] = (pt, pe)
        rt, re_, rp = metrics(te[TB].values, pt), metrics(ext["Tb_K"].values, pe), metrics(ext["Tb_K"].values[phys], pe[phys])
        rows.append({"Model": name, "Seed": s, "Test R2": rt[0], "Test MAE": rt[1], "Test RMSE": rt[2],
                     "Ext. PHYSPROP R2": rp[0], "Ext. PHYSPROP MAE": rp[1], "Ext. all R2": re_[0], "Ext. all MAE": re_[1]})
        print(f"  {name:35s} seed {s}: test R2 {rt[0]:.4f}  MAE {rt[1]:.2f} K | PHYSPROP R2 {rp[0]:.3f} MAE {rp[1]:.2f} K")

# reference: final 10-descriptor LightGBM (same training data)
sc = StandardScaler().fit(tr[feats].values)
lgbm = LGBMRegressor(n_estimators=1784, learning_rate=0.06686808497475842, num_leaves=18, max_depth=6,
    min_child_samples=32, subsample=0.9921028774475024, colsample_bytree=0.7660499104221706,
    reg_alpha=0.06008952920081579, reg_lambda=0.001212337250189692, random_state=42, n_jobs=-1,
    verbosity=-1).fit(sc.transform(tr[feats].values), tr[TB].values)
pt = lgbm.predict(sc.transform(te[feats].values)); pe = ext["Tb_pred_K"].values
rt, re_, rp = metrics(te[TB].values, pt), metrics(ext["Tb_K"].values, pe), metrics(ext["Tb_K"].values[phys], pe[phys])
ref = {"Model": "LightGBM, 10 descriptors (this work)", "Test R2": rt[0], "Test MAE": rt[1], "Test RMSE": rt[2],
       "Ext. PHYSPROP R2": rp[0], "Ext. PHYSPROP MAE": rp[1], "Ext. all R2": re_[0], "Ext. all MAE": re_[1]}

R = pd.DataFrame(rows); R.to_csv(OUT + "deep_learning_benchmark_by_seed.csv", index=False)
cols = [c for c in R.columns if c not in ("Model", "Seed")]
S = R.groupby("Model", sort=False)[cols].agg(["mean", "std"])
summary = pd.DataFrame({c: [f"{S.loc[m, (c, 'mean')]:.4f} ± {S.loc[m, (c, 'std')]:.4f}" if "R2" in c
                            else f"{S.loc[m, (c, 'mean')]:.2f} ± {S.loc[m, (c, 'std')]:.2f}" for m in S.index]
                        for c in cols}, index=S.index)
summary.loc[ref["Model"]] = [f"{ref[c]:.4f}" if "R2" in c else f"{ref[c]:.2f}" for c in cols]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\nBENCHMARK (mean ± SD over 3 seeds; same training/test/external data)\n", summary.to_string())
summary.to_csv(OUT + "deep_learning_benchmark_summary.csv")

# paired comparison on the test set: per-compound |error|, CheMeleon (seed-averaged) vs LightGBM
from scipy.stats import wilcoxon
for name in MODELS:
    pavg = np.mean([preds[(name, s)][0] for s in SEEDS], axis=0)
    a, b = np.abs(pavg - te[TB].values), np.abs(pt - te[TB].values)
    print(f"Test set, {name} (3-seed average) vs LightGBM: MAE {a.mean():.2f} vs {b.mean():.2f} K; "
          f"Wilcoxon p = {wilcoxon(a, b).pvalue:.2g}")
print("\nSaved deep_learning_benchmark_summary.csv and _by_seed.csv to", OUT)
