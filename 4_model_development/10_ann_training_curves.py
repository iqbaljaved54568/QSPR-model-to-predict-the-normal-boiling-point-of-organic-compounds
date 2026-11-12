"""
ANN training curves over five seeds (Figure S12)

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

# =============================================================================
# Reproducible ANN training curves (final 10-descriptor ANNs, raw and log target)
# Identical architecture, optimiser,
# scheduler, early stopping and validation split to 4_model_development/05_final_pipeline_10desc.py, but now
# SEEDED (PyTorch + NumPy, deterministic CPU) and repeated over N_SEEDS seeds.
#
# Saves (BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/):
#   ann_training_history_<target>_seed<k>.csv   per epoch: train MSE, val MSE, learning rate
#   ann_seed_summary.csv                        stop epoch, best epoch, final LR, test R2/MAE/RMSE
#   fig_ann_training_curves_seeded.png/.pdf
# =============================================================================
pass  # (Google Colab: mounted Google Drive here)
pass  # (Google Colab: mounted Google Drive here)
import random, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

N_SEEDS = 5
P = "./BP_FINAL_MODEL/descriptors/train_test_split/"
OUT = "./BP_FINAL_MODEL/figures/optimal_descriptor_pipeline/"
PARAMS = {
  "raw": {"hidden_layers": [224, 512, 512, 512], "dropout": 0.05780630823856324,
          "learning_rate": 0.006334516693674627, "batch_size": 128, "weight_decay": 9.976043673518242e-05,
          "batch_norm": True, "grad_clip": 5.0, "epochs": 300, "patience": 30},
  "log": {"hidden_layers": [352, 384], "dropout": 0.18407386081838017,
          "learning_rate": 0.0007013466196636101, "batch_size": 32, "weight_decay": 0.00025362535610134546,
          "batch_norm": False, "grad_clip": 5.0, "epochs": 300, "patience": 30}}
torch.use_deterministic_algorithms(True); device = "cpu"

tr = pd.read_csv(P + "train_optimal.csv"); te = pd.read_csv(P + "test_optimal.csv")
feats = [c for c in tr.columns if c.startswith(("rdkit2d_", "mordred_"))]; TB = tr.columns[5]
sc = StandardScaler().fit(tr[feats].values)
Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
ytr, yte = tr[TB].values, te[TB].values
perm = np.random.RandomState(42).permutation(len(Xtr))          # same 10% validation split as 4_model_development/05_final_pipeline_10desc.py
va, fi = perm[:max(1, int(0.1 * len(Xtr)))], perm[max(1, int(0.1 * len(Xtr))):]

def build(inp, p):
    layers, d = [], inp
    for h in p["hidden_layers"]:
        layers += [nn.Linear(d, h)] + ([nn.BatchNorm1d(h)] if p["batch_norm"] else []) + [nn.Tanh()] + \
                  ([nn.Dropout(p["dropout"])] if p["dropout"] > 0 else [])
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 1))

def train(p, y_fit, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = build(Xtr.shape[1], p).to(device)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    Xf, yf, Xv, yv = T(Xtr[fi]), T(y_fit[fi]).unsqueeze(1), T(Xtr[va]), T(y_fit[va]).unsqueeze(1)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xf, yf), batch_size=p["batch_size"], shuffle=True, generator=g)
    opt = torch.optim.Adam(m.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    crit = nn.MSELoss(); best, state, wait, hist = np.inf, None, 0, []
    for ep in range(p["epochs"]):
        m.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), p["grad_clip"]); opt.step()
        m.eval()
        with torch.no_grad():
            vl, tl = crit(m(Xv), yv).item(), crit(m(Xf), yf).item()
        hist.append({"epoch": ep + 1, "train_MSE": tl, "val_MSE": vl, "learning_rate": opt.param_groups[0]["lr"]})
        sch.step(vl)
        if vl < best:
            best, state, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= p["patience"]:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        pred = m(T(Xte)).squeeze().numpy()
    return pd.DataFrame(hist), pred

rows, curves = [], {}
for tgt, p in PARAMS.items():
    y_fit = np.log(ytr) if tgt == "log" else ytr
    for s in range(N_SEEDS):
        h, pred = train(p, y_fit, s)
        if tgt == "log": pred = np.exp(pred)
        h.to_csv(OUT + f"ann_training_history_{tgt}_seed{s}.csv", index=False)
        curves[(tgt, s)] = h
        best_ep = int(h.loc[h["val_MSE"].idxmin(), "epoch"])
        rows.append({"Target": tgt, "Seed": s, "Stopped at epoch": len(h), "Best epoch": best_ep,
                     "LR reductions": int((h["learning_rate"].diff() < 0).sum()),
                     "Final LR": h["learning_rate"].iloc[-1], "Test R2": r2_score(yte, pred),
                     "Test MAE (K)": mean_absolute_error(yte, pred), "Test RMSE (K)": mean_squared_error(yte, pred) ** 0.5})
        print(rows[-1])
summ = pd.DataFrame(rows); summ.to_csv(OUT + "ann_seed_summary.csv", index=False)
print("\n", summ.round(4).to_string(index=False))
print("\nMean ± SD over seeds:\n", summ.groupby("Target")[["Stopped at epoch", "Best epoch", "Test R2", "Test RMSE (K)"]]
      .agg(["mean", "std"]).round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, tgt in zip(axes, ["raw", "log"]):
    ax2 = ax.twinx()
    for s in range(N_SEEDS):
        h = curves[(tgt, s)]
        ax.plot(h["epoch"], h["train_MSE"], color="#0072B2", alpha=0.35 if s else 0.9, lw=1.2,
                label="Training MSE" if s == 0 else None)
        ax.plot(h["epoch"], h["val_MSE"], color="#D55E00", alpha=0.35 if s else 0.9, lw=1.2,
                label="Validation MSE" if s == 0 else None)
        ax2.step(h["epoch"], h["learning_rate"], color="#555", alpha=0.35 if s else 0.8, lw=1, ls="--",
                 where="post", label="Learning rate" if s == 0 else None)
        b = h["val_MSE"].idxmin(); ax.plot(h.loc[b, "epoch"], h.loc[b, "val_MSE"], "o", color="#D55E00", ms=4)
    ax.set_yscale("log"); ax2.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (" + ("K$^2$" if tgt == "raw" else "ln K units$^2$") + ")")
    ax2.set_ylabel("Learning rate")
    ax.set_title(f"({'ab'[tgt == 'log']}) ANN, {'original' if tgt == 'raw' else 'log-transformed'} $T_b$ "
                 f"({N_SEEDS} seeds; dots = restored best epoch)", loc="left", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.savefig(OUT + "fig_ann_training_curves_seeded.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT + "fig_ann_training_curves_seeded.pdf", bbox_inches="tight"); plt.show()
print("Saved per-epoch histories, ann_seed_summary.csv and figure to", OUT)
