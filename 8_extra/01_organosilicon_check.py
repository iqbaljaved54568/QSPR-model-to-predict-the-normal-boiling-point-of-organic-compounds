"""
Predictions for organosilicon compounds (Table S11)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# How does the final model handle organosilicon compounds? (illustrative test)
#  - calculates the 10 final descriptors for 8 common organosilicon compounds,
#    exactly as for the main dataset (RDKit Ipc on the H-free molecule; Mordred on
#    the molecule with explicit H), and reports any descriptor that cannot be computed;
#  - computes leverage against the training set (same method as the AD analysis,
#    h* = 0.00814) and the model prediction.
# !! The experimental Tb values below are approximate literature values typed in for
#    this check -- VERIFY each against the NIST WebBook before quoting them in the paper.
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "mordredcommunity"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors as mord_desc
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
H_STAR, S_CV = 0.00814, 19.023
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}

SI = [  # name, SMILES, approximate experimental Tb (K) -- VERIFY
    ("Tetramethylsilane",             "C[Si](C)(C)C",                                   299.8),
    ("Chlorotrimethylsilane",         "C[Si](C)(C)Cl",                                  330.2),
    ("Dichlorodimethylsilane",        "C[Si](C)(Cl)Cl",                                 343.2),
    ("Hexamethyldisiloxane",          "C[Si](C)(C)O[Si](C)(C)C",                        373.7),
    ("Triethylsilane",                "CC[SiH](CC)CC",                                  380.2),
    ("Tetraethyl orthosilicate",      "CCO[Si](OCC)(OCC)OCC",                           441.2),
    ("Phenyltrimethylsilane",         "C[Si](C)(C)c1ccccc1",                            442.6),
    ("Octamethylcyclotetrasiloxane",  "C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1",   448.7),
]
# carbon analogues (Si -> C) for comparison: also in-scope organics
C_ANALOG = {"Tetramethylsilane": ("Neopentane", "CC(C)(C)C", 282.6),
            "Hexamethyldisiloxane": ("Di-tert-butyl ether", "CC(C)(C)OC(C)(C)C", 380.4)}

tr = pd.read_csv(P + "train_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
sc = StandardScaler().fit(tr[feats].values); Xtr = sc.transform(tr[feats].values)
model = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, tr[TB].values)
U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
lev = lambda Xq: (Xq @ Vt.T) ** 2 @ (1.0 / S ** 2)

calc = Calculator(mord_desc, ignore_3D=True)
mord_names = [f.replace("mordred_", "") for f in feats if f.startswith("mordred_")]
def desc(smiles):
    m2d = [Chem.MolFromSmiles(s) for s in smiles]
    md = calc.pandas([Chem.AddHs(m) for m in m2d], quiet=True)[mord_names]
    out = pd.DataFrame({f"mordred_{c}": pd.to_numeric(md[c], errors="coerce").values for c in mord_names})
    out["rdkit2d_Ipc"] = [Descriptors.Ipc(m) for m in m2d]
    return out[feats]

rows = [(n, s, t, "organosilicon") for n, s, t in SI] + [(a, s, t, f"C-analogue of {k}") for k, (a, s, t) in C_ANALOG.items()]
D = desc([r[1] for r in rows])
lo, hi = tr[feats].min(), tr[feats].max()
res = []
for (name, smi, tb, kind), (_, d) in zip(rows, D.iterrows()):
    nan = [f.split("_", 1)[1] for f in feats if pd.isna(d[f])]
    outside = [f.split("_", 1)[1] for f in feats if pd.notna(d[f]) and not (lo[f] <= d[f] <= hi[f])]
    if nan:
        res.append({"Compound": name, "Type": kind, "Tb exp (K)": tb, "Not computable": ", ".join(nan)}); continue
    x = sc.transform(d.values[None, :]); h = float(lev(x)[0]); pred = float(model.predict(x)[0])
    res.append({"Compound": name, "Type": kind, "Tb exp (K)": tb, "Tb pred (K)": round(pred, 1),
                "Error (K)": round(pred - tb, 1), "Leverage h": round(h, 4), "h > h*": h > H_STAR,
                "|std. resid| > 3": abs(pred - tb) / S_CV > 3, "Descriptors outside training range": ", ".join(outside) or "-"})
res = pd.DataFrame(res); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(f"h* = {H_STAR}; residual limit = 3 x {S_CV} = {3*S_CV:.1f} K\n")
print(res.to_string(index=False))
print("\nDescriptor values (organosilicons) vs training range:")
print(pd.concat([D.iloc[:len(SI)].set_axis([r[0] for r in rows[:len(SI)]]).T.round(3),
                 lo.rename("train min").round(3), hi.rename("train max").round(3)], axis=1).to_string())
res.to_csv("./BP_FINAL_MODEL/figures/organosilicon_check.csv", index=False)
