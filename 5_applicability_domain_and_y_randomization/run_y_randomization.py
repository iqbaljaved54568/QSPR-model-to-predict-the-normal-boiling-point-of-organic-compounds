# =============================================================================
# Standalone Y-Randomization (Y-Scrambling) Validation — FAST, PARALLEL, RESUMABLE
#
# Re-runs the Y-randomization test from `4_parameter_optimization_and_model_
# development/final_model_pipeline.py` (Section 16) WITHOUT retraining the
# real model, by loading the training data + locked hyperparameters from the
# saved checkpoint (`checkpoint_optimal.pkl`). Works for whichever model
# actually won (Linear / RandomForest / XGBoost / LightGBM / ANN).
#
# Same engine as the inline pipeline version: one fit per iteration (single
# train/val split, not a full k-fold), early stopping where the model
# supports it (LightGBM, XGBoost), iterations parallelised across all cores
# via joblib, and results checkpointed to disk every batch so an interrupted
# run can resume rather than restart. Significance is assessed via a
# one-tailed z-test of the real R2 against the scrambled-R2 distribution.
#
# Usage:
#   python run_y_randomization.py --checkpoint /path/to/checkpoint_optimal.pkl \
#          --out_dir ./yrand_outputs/ --n_iter 100
# =============================================================================

import argparse
import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

DEFAULT_N_ITER = {"Linear": 200, "RandomForest": 100, "XGBoost": 100, "LightGBM": 100, "ANN": 20}


def _yrand_lgbm(seed, X_tr, y_tr, X_te, y_te, params, log_transform, val_frac, es_rounds):
    from lightgbm import early_stopping, log_evaluation
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_tr)
    y_fit = np.log(y_shuffled) if log_transform else y_shuffled
    n = len(y_tr); idx = rng.permutation(n)
    n_val = max(10, int(val_frac * n)); val_idx, fit_idx = idx[:n_val], idx[n_val:]
    m = LGBMRegressor(**{**params, "random_state": seed, "n_jobs": 1})
    m.fit(X_tr[fit_idx], y_fit[fit_idx], eval_set=[(X_tr[val_idx], y_fit[val_idx])],
          callbacks=[early_stopping(stopping_rounds=es_rounds, verbose=False), log_evaluation(period=0)])
    pred_val, pred_test = m.predict(X_tr[val_idx]), m.predict(X_te)
    if log_transform:
        pred_val, pred_test = np.exp(pred_val), np.exp(pred_test)
    return r2_score(y_shuffled[val_idx], pred_val), r2_score(y_te, pred_test)


def _yrand_xgb(seed, X_tr, y_tr, X_te, y_te, params, log_transform, val_frac, es_rounds):
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_tr)
    y_fit = np.log(y_shuffled) if log_transform else y_shuffled
    n = len(y_tr); idx = rng.permutation(n)
    n_val = max(10, int(val_frac * n)); val_idx, fit_idx = idx[:n_val], idx[n_val:]
    p = {**params, "random_state": seed, "n_jobs": 1, "early_stopping_rounds": es_rounds}
    m = XGBRegressor(**p)
    m.fit(X_tr[fit_idx], y_fit[fit_idx], eval_set=[(X_tr[val_idx], y_fit[val_idx])], verbose=False)
    pred_val, pred_test = m.predict(X_tr[val_idx]), m.predict(X_te)
    if log_transform:
        pred_val, pred_test = np.exp(pred_val), np.exp(pred_test)
    return r2_score(y_shuffled[val_idx], pred_val), r2_score(y_te, pred_test)


def _yrand_sklearn(seed, X_tr, y_tr, X_te, y_te, build_fn, log_transform, val_frac):
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_tr)
    y_fit = np.log(y_shuffled) if log_transform else y_shuffled
    n = len(y_tr); idx = rng.permutation(n)
    n_val = max(10, int(val_frac * n)); val_idx, fit_idx = idx[:n_val], idx[n_val:]
    m = build_fn()
    m.fit(X_tr[fit_idx], y_fit[fit_idx])
    pred_val, pred_test = m.predict(X_tr[val_idx]), m.predict(X_te)
    if log_transform:
        pred_val, pred_test = np.exp(pred_val), np.exp(pred_test)
    return r2_score(y_shuffled[val_idx], pred_val), r2_score(y_te, pred_test)


def _yrand_ann_iter(seed, X_tr, y_tr, X_te, y_te, params, log_transform, val_frac, device):
    """
    Only needed if the winning model was the ANN. Duplicates the minimal
    training/prediction logic from the main pipeline script -- if you're
    running this branch, make sure `torch` is installed.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    ACT_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "selu": nn.SELU}

    def build_ann(input_dim, hidden_layers, activation="tanh", dropout=0.2, batch_norm=False):
        act_fn = ACT_MAP.get(activation, nn.Tanh)
        layers, in_dim = [], input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_tr)
    y_fit = np.log(y_shuffled) if log_transform else y_shuffled
    n = len(y_tr); idx = rng.permutation(n)
    n_val = max(10, int(val_frac * n)); val_idx, fit_idx = idx[:n_val], idx[n_val:]

    model = build_ann(X_tr.shape[1], params["hidden_layers"], params["activation"],
                       params["dropout"], batch_norm=params.get("batch_norm", False)).to(device)
    X_fit_t = torch.tensor(X_tr[fit_idx], dtype=torch.float32).to(device)
    y_fit_t = torch.tensor(y_fit[fit_idx], dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_tr[val_idx], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_fit[val_idx], dtype=torch.float32).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(X_fit_t, y_fit_t), batch_size=params["batch_size"], shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    grad_clip = params.get("grad_clip", None)
    best_val, best_state, patience_cnt = np.inf, None, 0
    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, patience_cnt = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= params["patience"]:
                break
    model.load_state_dict(best_state)

    def _predict(X):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze().cpu().numpy()

    pred_val, pred_test = _predict(X_tr[val_idx]), _predict(X_te)
    if log_transform:
        pred_val, pred_test = np.exp(pred_val), np.exp(pred_test)
    return r2_score(y_shuffled[val_idx], pred_val), r2_score(y_te, pred_test)


def run_yrand_parallel(model_name, X_train_sc, y_train, X_test_sc, y_test, log_flag,
                        model_builders_simple, lgbm_params, xgb_params, ann_params,
                        n_random, batch_size, n_jobs, val_frac, es_rounds,
                        checkpoint_path, random_state, ann_device):
    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        cv_rand, test_rand = list(ckpt["cv_rand"]), list(ckpt["test_rand"])
        n_done = len(cv_rand)
        print(f"  Resuming from checkpoint: {n_done}/{n_random} iterations already done")
    else:
        cv_rand, test_rand, n_done = [], [], 0

    if n_done >= n_random:
        print(f"  Already have {n_done} iterations -- nothing more to run.")
        return np.array(cv_rand[:n_random]), np.array(test_rand[:n_random])

    remaining = n_random - n_done
    seeds_remaining = [random_state + n_done + i for i in range(remaining)]
    print(f"  Running {remaining} more scrambled-target refits for {model_name} "
          f"({'log' if log_flag else 'raw'} target) ...")
    t_start = time.time()

    for batch_start in range(0, remaining, batch_size):
        batch_seeds = seeds_remaining[batch_start: batch_start + batch_size]
        t0 = time.time()
        if model_name == "LightGBM":
            batch_results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_yrand_lgbm)(seed, X_train_sc, y_train, X_test_sc, y_test,
                                             lgbm_params, log_flag, val_frac, es_rounds)
                for seed in batch_seeds)
        elif model_name == "XGBoost":
            batch_results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_yrand_xgb)(seed, X_train_sc, y_train, X_test_sc, y_test,
                                            xgb_params, log_flag, val_frac, es_rounds)
                for seed in batch_seeds)
        elif model_name in model_builders_simple:
            batch_results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_yrand_sklearn)(seed, X_train_sc, y_train, X_test_sc, y_test,
                                                model_builders_simple[model_name], log_flag, val_frac)
                for seed in batch_seeds)
        else:  # ANN -- sequential; multi-process parallelism is unsafe with a shared CUDA/CPU context
            batch_results = [
                _yrand_ann_iter(seed, X_train_sc, y_train, X_test_sc, y_test,
                                 ann_params, log_flag, val_frac, ann_device)
                for seed in batch_seeds]

        for cv_r2_i, test_r2_i in batch_results:
            cv_rand.append(cv_r2_i); test_rand.append(test_r2_i)

        np.savez(checkpoint_path, cv_rand=np.array(cv_rand), test_rand=np.array(test_rand))
        n_completed = len(cv_rand)
        print(f"    {n_completed}/{n_random}  mean CV-like R2 = {np.mean(cv_rand):.4f}  "
              f"mean test R2 = {np.mean(test_rand):.4f}  "
              f"[batch {time.time()-t0:.1f}s, total {time.time()-t_start:.1f}s]")

    return np.array(cv_rand[:n_random]), np.array(test_rand[:n_random])


def y_rand_stats(real_r2, rand_r2, label="", split="CV"):
    n = len(rand_r2); mean_rand = rand_r2.mean(); std_rand = rand_r2.std(); max_rand = rand_r2.max()
    delta = real_r2 - mean_rand
    z_score = (real_r2 - mean_rand) / (std_rand / np.sqrt(n)) if std_rand > 0 else np.inf
    p_value = 1 - norm.cdf(z_score)
    print(f"\n  {label} | {split}")
    print(f"  Real R2 = {real_r2:.4f}   Scrambled R2 = {mean_rand:.4f} +/- {std_rand:.4f}  (max {max_rand:.4f})")
    print(f"  Delta R2 = {delta:.4f}   Z = {z_score:.2f}   p = {p_value:.2e}")
    verdict = "VALID -- genuine structure-property relationship" if (mean_rand < 0.3 and delta > 0.5) \
              else "CHECK -- scrambled performance too high"
    print(f"  Verdict: {verdict}")
    return {"n": n, "real_r2": real_r2, "mean_rand": mean_rand, "std_rand": std_rand,
            "max_rand": max_rand, "delta": delta, "z_score": z_score, "p_value": p_value}


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Standalone, parallel, resumable Y-randomization from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_optimal.pkl")
    parser.add_argument("--out_dir", default="./yrand_outputs/", help="Directory to save figures/CSVs/resume-checkpoint")
    parser.add_argument("--n_iter", type=int, default=None, help="Override number of scrambling iterations")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.15, help="Held-out fraction of TRAIN for the internal split")
    parser.add_argument("--es_rounds", type=int, default=30, help="Early-stopping patience (LightGBM/XGBoost only)")
    parser.add_argument("--batch_size", type=int, default=20, help="Iterations per checkpoint save")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers (ANN forced to 1 regardless)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cp = joblib.load(args.checkpoint)

    X_train_sc, y_train = cp["X_train_sc"], cp["y_train"]
    X_test_sc, y_test = cp["X_test_sc"], cp["y_test"]
    best_model_name = cp["best_model_name"]
    tag_str = cp["tag_str"]
    best_res = cp["best_res"]
    log_flag = best_res["log_transform"]
    real_test_r2 = best_res["test_r2"]
    real_cv_r2 = best_res["cv_r2_mean"]

    model_builders_simple = {
        "Linear": lambda: Ridge(**cp["LINEAR_PARAMS"]),
        "RandomForest": lambda: RandomForestRegressor(**{**cp["RF_PARAMS"], "n_jobs": 1}),
    }
    ann_device = "cuda" if _torch_cuda_available() else "cpu"
    ann_params = cp["ANN_PARAMS_LOG"] if log_flag else cp["ANN_PARAMS_RAW"]

    n_iter = args.n_iter if args.n_iter is not None else DEFAULT_N_ITER.get(best_model_name, 50)
    n_jobs = 1 if best_model_name == "ANN" else args.n_jobs
    checkpoint_path = os.path.join(args.out_dir, f"yrand_checkpoint_{best_model_name}.npz")

    print(f"Y-Randomisation -- {best_model_name}{tag_str}  (n_iter={n_iter})")
    t0 = time.time()
    cv_rand, test_rand = run_yrand_parallel(
        best_model_name, X_train_sc, y_train, X_test_sc, y_test, log_flag,
        model_builders_simple, cp["LGBM_PARAMS"], cp["XGB_PARAMS"], ann_params,
        n_iter, args.batch_size, n_jobs, args.val_frac, args.es_rounds,
        checkpoint_path, args.random_state, ann_device)
    print(f"done in {time.time()-t0:.1f}s")

    stats_cv = y_rand_stats(real_cv_r2, cv_rand, best_model_name, "CV-like (single split)")
    stats_test = y_rand_stats(real_test_r2, test_rand, best_model_name, "Test")

    summary = pd.DataFrame([
        {"model": best_model_name, "split": "CV-like", "n_iter": n_iter, "log_transform": log_flag, **stats_cv},
        {"model": best_model_name, "split": "Test", "n_iter": n_iter, "log_transform": log_flag, **stats_test},
    ])
    summary_csv = os.path.join(args.out_dir, "yrandomization_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved -> {summary_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Y-Randomisation Test - {best_model_name}{tag_str}\n"
                 "Scrambled R2 distribution vs real model performance", fontsize=13, fontweight="bold", y=1.03)
    for ax, real_r2, rand_r2, title in [
        (axes[0], real_cv_r2, cv_rand, f"{best_model_name} - CV-like R2 (single split)"),
        (axes[1], real_test_r2, test_rand, f"{best_model_name} - External Test R2"),
    ]:
        ax.hist(rand_r2, bins=25, color="#7F8C8D", alpha=0.65, edgecolor="white", linewidth=0.5,
                density=True, label=f"Scrambled (n={len(rand_r2)})")
        x_fit = np.linspace(rand_r2.min()-0.05, max(rand_r2.max(), real_r2)+0.05, 300)
        ax.plot(x_fit, norm.pdf(x_fit, rand_r2.mean(), rand_r2.std()), color="#7F8C8D",
                linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(real_r2, color="#2C6E8A", linewidth=2.2, label=f"Real R2 = {real_r2:.4f}", zorder=5)
        ax.axvline(rand_r2.mean(), color="#555", linewidth=1.2, linestyle=":", label=f"Mean scrambled = {rand_r2.mean():.4f}")
        ax.set_xlabel("R2"); ax.set_ylabel("Probability Density")
        ax.set_title(title, loc="left", fontsize=10, fontstyle="italic")
        ax.legend(fontsize=8); ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"fig_yrandomization_panel_{best_model_name}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {fig_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    iters = np.arange(1, len(cv_rand) + 1)
    run_mean = np.cumsum(cv_rand) / iters
    run_std = pd.Series(cv_rand).expanding().std().values
    ax.plot(iters, run_mean, color="#2C6E8A", linewidth=2, label=f"{best_model_name} running mean")
    ax.fill_between(iters, run_mean - run_std, run_mean + run_std, alpha=0.12, color="#2C6E8A")
    ax.axhline(0, color="#AAB7B8", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Number of Randomisation Iterations"); ax.set_ylabel("Running Mean Scrambled R2")
    ax.set_title(f"Y-Randomisation Convergence - {best_model_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    fig_path2 = os.path.join(args.out_dir, f"fig_yrandomization_convergence_{best_model_name}.png")
    fig.savefig(fig_path2, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {fig_path2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
