"""
External validation on OPERA boiling-point data (Table 9)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# External validation on a public benchmark: EPA OPERA boiling-point data
# No retraining: compounds are predicted with
# the final 10-descriptor LightGBM (fixed hyperparameters, training set only).
#
# Data: OPERA (Mansouri et al., J. Cheminform. 2018, 10, 10; US EPA), public on GitHub
#   - BP_QR.sdf           curated PHYSPROP boiling points used to build OPERA's BP model
#   - BP_2.9_update.csv   compounds added in OPERA v2.9 (eChemPortal, PubChem, OChem, ...)
# Steps
#   1. Download OPERA_Data.zip, read both sets (deg C -> K), merge duplicates (median).
#   2. Keep compounds inside the model's stated scope (as in the web app): one neutral
#      component, only C,H,N,O,P,S,Cl,Br,I, 2-28 heavy atoms, Tb 238.65-743.0 K.
#   3. REMOVE every compound whose skeleton (InChIKey first block) occurs among our
#      5,071 compounds -> a set the model has never seen, from independent sources.
#   4. Check the descriptor code: recompute the ten descriptors for 200 training
#      compounds and compare with the stored values (should agree to ~1e-6).
#   5. Predict; report R2/MAE/RMSE overall, inside/outside the leverage AD, by source.
# Outputs -> BP_FINAL_MODEL/external_validation/
# =============================================================================
import subprocess, sys, os, io, zipfile, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit==2025.9.3", "mordredcommunity"], check=True)
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors as mord_desc
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
RDLogger.DisableLog("rdApp.*")

P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/external_validation/"; os.makedirs(OUT, exist_ok=True)
URL = "https://github.com/kmansouri/OPERA/raw/master/OPERA_Data.zip"
LGBM_PARAMS = {"n_estimators": 1784, "learning_rate": 0.06686808497475842, "num_leaves": 18,
    "max_depth": 6, "min_child_samples": 32, "subsample": 0.9921028774475024,
    "colsample_bytree": 0.7660499104221706, "reg_alpha": 0.06008952920081579,
    "reg_lambda": 0.001212337250189692, "random_state": 42, "n_jobs": -1, "verbosity": -1}
ALLOWED = {"C", "H", "N", "O", "P", "S", "Cl", "Br", "I"}
TB_MIN, TB_MAX, H_STAR = 238.65, 743.0, 0.00814

# ---- 1. OPERA data ----------------------------------------------------------------
zpath = OUT + "OPERA_Data.zip"
if not os.path.exists(zpath):
    print("Downloading OPERA_Data.zip (~79 MB) ..."); urllib.request.urlretrieve(URL, zpath)
z = zipfile.ZipFile(zpath)
name = lambda end: next(n for n in z.namelist() if n.endswith(end))
upd = pd.read_csv(z.open(name("BP_2.9_update.csv")))
upd = pd.DataFrame({"Name": upd["PREFERRED_NAME"], "CAS": upd["CASRN"], "smiles": upd["Canonical_QSARr"],
                    "Tb_K": upd["value_point_estimate"] + 273.15, "Source": "OPERA v2.9 update",
                    "Original source": upd["pub_source_name"]})
sdf = Chem.ForwardSDMolSupplier(io.BytesIO(z.read(name("BP_QR.sdf"))))
qr = pd.DataFrame([{"Name": m.GetProp("preferred_name") if m.HasProp("preferred_name") else "",
                    "CAS": m.GetProp("CAS"), "smiles": m.GetProp("Canonical_QSARr"),
                    "Tb_K": float(m.GetProp("BP")) + 273.15, "Source": "OPERA PHYSPROP (BP_QR)",
                    "Original source": "PHYSPROP"} for m in sdf if m is not None])
ext = pd.concat([qr, upd], ignore_index=True)
print(f"OPERA records: {len(qr)} (BP_QR) + {len(upd)} (v2.9 update) = {len(ext)}")

# ---- 2. scope filter ---------------------------------------------------------------
def prep(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None or len(Chem.GetMolFrags(m)) != 1: return None, "invalid / multi-component"
    if {a.GetSymbol() for a in m.GetAtoms()} - ALLOWED: return None, "element outside scope"
    if not any(a.GetSymbol() == "C" for a in m.GetAtoms()): return None, "no carbon"
    if sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0 or any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        return None, "charged / radical"
    if not 2 <= m.GetNumHeavyAtoms() <= 28: return None, "heavy atoms outside 2-28"
    return Chem.MolToSmiles(m), "ok"
r = ext["smiles"].apply(prep); ext["can"] = r.str[0]; ext["why"] = r.str[1]
ext.loc[(ext.why == "ok") & ~ext.Tb_K.between(TB_MIN, TB_MAX), "why"] = "Tb outside 238.65-743 K"
print("\nScope filter:\n", ext["why"].value_counts().to_string())
ext = ext[ext.why == "ok"].copy()
ext["IK"] = ext["can"].apply(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
g = ext.groupby("IK")
ext = g.agg(Name=("Name", "first"), CAS=("CAS", "first"), smiles=("can", "first"), Tb_K=("Tb_K", "median"),
            Tb_range_K=("Tb_K", lambda v: v.max() - v.min()), n_records=("Tb_K", "size"),
            Source=("Source", lambda s: " + ".join(sorted(set(s)))),
            Original_source=("Original source", "first")).reset_index()
print(f"Unique in-scope compounds: {len(ext)}")

# ---- 3. remove overlap with our 5,071 ---------------------------------------------
tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
ours = pd.concat([tr, te])["smiles"].apply(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14])
before = len(ext)
ext = ext[~ext["IK"].str[:14].isin(set(ours))].reset_index(drop=True)
print(f"Removed {before - len(ext)} compounds whose skeleton is in our dataset -> external set n = {len(ext)}")

# ---- 4. descriptors (same as training: Ipc on the molecule, Mordred on H-added molecule)
mord_names = [f[len("mordred_"):] for f in feats if f.startswith("mordred_")]
calc9 = Calculator([d for d in Calculator(mord_desc, ignore_3D=True).descriptors if str(d) in mord_names])
assert len(calc9.descriptors) == 9
def describe(smi):
    m = Chem.MolFromSmiles(smi); md = calc9(Chem.AddHs(m)).asdict()
    vals = {"rdkit2d_Ipc": Descriptors.Ipc(m)} | {"mordred_" + str(k): v for k, v in md.items()}
    return [float(vals[f]) if isinstance(vals[f], (int, float, np.floating)) else np.nan for f in feats]
chk = tr.sample(200, random_state=1)
Xchk = np.array([describe(s) for s in chk["smiles"]])
rel = np.nanmax(np.abs(Xchk - chk[feats].values) / (np.abs(chk[feats].values) + 1e-9), axis=0)
print("\nDescriptor check on 200 training compounds, max relative difference per descriptor:")
print(pd.Series(rel, index=feats).map("{:.1e}".format).to_string())

Xe = np.array([describe(s) for s in ext["smiles"]])
bad = np.isnan(Xe).any(axis=1)
if bad.any(): print(f"{bad.sum()} compounds dropped: a descriptor could not be calculated")
ext, Xe = ext[~bad].reset_index(drop=True), Xe[~bad]

# ---- 5. predict ----------------------------------------------------------------------
sc = StandardScaler().fit(tr[feats].values); Xtr = sc.transform(tr[feats].values)
model = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, tr[TB].values)
Z = sc.transform(Xe); ext["Tb_pred_K"] = model.predict(Z)
ext["Residual_K"] = ext["Tb_pred_K"] - ext["Tb_K"]
_, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
ext["Leverage"] = ((Z @ Vt.T) ** 2 / S ** 2).sum(axis=1); ext["Inside_AD"] = ext["Leverage"] <= H_STAR

def metrics(d, label):
    e = d["Residual_K"]
    return {"Subset": label, "n": len(d), "R2": round(r2_score(d["Tb_K"], d["Tb_pred_K"]), 4),
            "MAE (K)": round(e.abs().mean(), 2), "RMSE (K)": round(np.sqrt((e ** 2).mean()), 2),
            "Median |error| (K)": round(e.abs().median(), 2), "Bias (K)": round(e.mean(), 2),
            "% within 57.1 K": round(100 * (e.abs() <= 57.1).mean(), 1)}
rows = [metrics(ext, "All external compounds"),
        metrics(ext[ext.Inside_AD], "Inside AD (h <= h*)"), metrics(ext[~ext.Inside_AD], "Outside AD (h > h*)")]
for s_ in sorted(ext["Source"].unique()):
    rows.append(metrics(ext[ext.Source == s_], f"Source: {s_}"))
rows.append(metrics(ext[ext.Tb_range_K <= 5], "Records agreeing within 5 K (or single record)"))
R = pd.DataFrame(rows); pd.set_option("display.width", 250)
print("\nEXTERNAL VALIDATION (OPERA, compounds not in our dataset)\n", R.to_string(index=False))
print(f"\nCoverage: {100*ext.Inside_AD.mean():.1f}% of external compounds inside the leverage AD")
print("\nLargest 15 errors (check for data problems):")
print(ext.reindex(ext.Residual_K.abs().sort_values(ascending=False).index)
      [["Name", "CAS", "smiles", "Tb_K", "Tb_pred_K", "Residual_K", "Inside_AD", "Original_source"]].head(15).to_string(index=False))
R.to_csv(OUT + "external_validation_summary.csv", index=False)
ext.to_csv(OUT + "external_validation_predictions.csv", index=False)
print("\nSaved external_validation_summary.csv and external_validation_predictions.csv to", OUT)
