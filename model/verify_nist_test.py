# Reproduces the NIST-only test metrics from public files only.
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
d = pd.read_csv("dataset_public_5071.csv")
s = json.load(open("final_model_scaler.json"))
m = lgb.Booster(model_file="final_model_lightgbm.txt")
t = d[(d["Split"] == "test") & d["Source"].str.startswith("NIST")]
X = (t[s["features"]].values - np.array(s["mean"])) / np.array(s["scale"])
y, p = t.iloc[:, t.columns.get_loc("Split") + 1].values, m.predict(X)
print(f"NIST test: n={len(t)}  R2={r2_score(y, p):.4f}  MAE={mean_absolute_error(y, p):.3f} K  "
      f"RMSE={mean_squared_error(y, p) ** 0.5:.3f} K")
