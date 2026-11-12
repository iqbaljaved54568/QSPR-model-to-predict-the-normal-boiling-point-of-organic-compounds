"""
Wall-clock time of a single prediction

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

"""
Wall-clock benchmark of a single-SMILES boiling-point prediction (final 10-descriptor LightGBM).
Run on a LAPTOP (not Colab), from a terminal:

    pip install rdkit mordredcommunity lightgbm scikit-learn pandas psutil
    python benchmark_prediction.py train_optimal.csv test_optimal.csv

(download train_optimal.csv and test_optimal.csv from BP_FINAL_MODEL/descriptors/train_test_split/
 in the data folder into the same folder as this script; close other heavy programs first)

For each of N_MOL test-set SMILES it times, separately and in total:
  1. parsing the SMILES (RDKit) and adding hydrogens
  2. the RDKit Ipc descriptor
  3. the nine Mordred descriptors (only these nine are calculated)
  4. scaling + LightGBM prediction
  5. leverage (applicability-domain check)
It also times one-off start-up (loading data, fitting scaler/model, SVD) and, for comparison,
calculating ALL Mordred 2D descriptors (relevant if the app computes the full set).
Results: printed summary + benchmark_timings.csv
"""
import sys, time, platform, os
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors as mord_desc
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import rdkit, lightgbm, sklearn
RDLogger.DisableLog("rdApp.*")

N_MOL, N_FULL = 200, 20
_args = [a for a in sys.argv[1:] if a.endswith(".csv")]       # ignores notebook arguments such as -f
if len(_args) >= 2:
    TRAIN, TEST = _args[:2]
elif os.path.exists("train_optimal.csv"):
    TRAIN, TEST = "train_optimal.csv", "test_optimal.csv"
else:                                                            # Google Colab with Drive
    try:
        pass  # (Google Colab: mounted Google Drive here)
    except ImportError:
        pass
    _D = "./BP_FINAL_MODEL/descriptors/train_test_split/"
    TRAIN, TEST = _D + "train_optimal.csv", _D + "test_optimal.csv"
print("Data:", TRAIN, "|", TEST)
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}

# ---- machine ---------------------------------------------------------------------
try:
    import psutil; ram = f"{psutil.virtual_memory().total / 2**30:.1f} GB"
except ImportError:
    ram = "n/a (pip install psutil)"
print(f"OS: {platform.platform()} | CPU: {platform.processor() or platform.machine()} | "
      f"logical cores: {os.cpu_count()} | RAM: {ram}")
print(f"Python {platform.python_version()} | RDKit {rdkit.__version__} | LightGBM {lightgbm.__version__} | "
      f"scikit-learn {sklearn.__version__}")

# ---- one-off start-up (what an app does once when it starts) ---------------------
t0 = time.perf_counter()
tr = pd.read_csv(TRAIN); te = pd.read_csv(TEST)
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
sc = StandardScaler().fit(tr[feats].values); Xtr = sc.transform(tr[feats].values)
model = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, tr[TB].values)
_, S, Vt = np.linalg.svd(Xtr, full_matrices=False); S_inv = 1.0 / S ** 2
mord_names = [f.replace("mordred_", "") for f in feats if f.startswith("mordred_")]
calc9 = Calculator([d for d in Calculator(mord_desc, ignore_3D=True).descriptors if str(d) in mord_names])
calc_all = Calculator(mord_desc, ignore_3D=True)
t_start = time.perf_counter() - t0
print(f"\nStart-up (load data, fit scaler+model, SVD, build calculators): {t_start:.1f} s (one-off)")
assert len(calc9.descriptors) == len(mord_names), "could not find all nine Mordred descriptors"

def predict_one(smi):
    t = [time.perf_counter()]
    m = Chem.MolFromSmiles(smi); mh = Chem.AddHs(m);                      t.append(time.perf_counter())
    ipc = Descriptors.Ipc(m);                                             t.append(time.perf_counter())
    r = calc9(mh); md = {str(k): float(v) for k, v in r.asdict().items()};t.append(time.perf_counter())
    x = np.array([[ipc if f == "rdkit2d_Ipc" else md[f.replace("mordred_", "")] for f in feats]])
    xs = sc.transform(x); y = float(model.predict(xs)[0]);                t.append(time.perf_counter())
    h = float(((xs @ Vt.T) ** 2 @ S_inv)[0]);                            t.append(time.perf_counter())
    return y, h, np.diff(t)

smiles = te["smiles"].sample(min(N_MOL, len(te)), random_state=42).tolist()
for s in smiles[:5]:
    predict_one(s)                                                       # warm-up
rows = []
for s in smiles:
    y, h, d = predict_one(s)
    rows.append({"smiles": s, "parse_ms": d[0]*1e3, "ipc_ms": d[1]*1e3, "mordred9_ms": d[2]*1e3,
                 "predict_ms": d[3]*1e3, "leverage_ms": d[4]*1e3, "total_ms": d.sum()*1e3,
                 "heavy_atoms": Chem.MolFromSmiles(s).GetNumHeavyAtoms()})
T = pd.DataFrame(rows)
stats = T.drop(columns=["smiles", "heavy_atoms"]).describe(percentiles=[0.5, 0.95]).T[["mean", "50%", "95%", "max"]]
print(f"\nPer-SMILES wall-clock time over {len(T)} test compounds (ms):\n{stats.round(2).to_string()}")
print(f"\nCorrelation of total time with heavy-atom count: r = {T['total_ms'].corr(T['heavy_atoms']):.2f}")

full = []
for s in smiles[:N_FULL]:
    t = time.perf_counter(); calc_all(Chem.AddHs(Chem.MolFromSmiles(s))); full.append((time.perf_counter() - t) * 1e3)
print(f"For comparison - ALL Mordred 2D descriptors per SMILES: mean {np.mean(full):.0f} ms, "
      f"median {np.median(full):.0f} ms (n = {N_FULL})")
T.to_csv("benchmark_timings.csv", index=False)
print("\nSaved benchmark_timings.csv")
