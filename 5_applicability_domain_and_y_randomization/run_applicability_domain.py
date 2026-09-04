# =============================================================================
# Standalone Applicability Domain (Williams Plot) Analysis
#
# Re-runs the AD analysis from `4_parameter_optimization_and_model_development/
# final_model_pipeline.py` (Section 14) WITHOUT retraining any model, by
# loading everything it needs from the saved checkpoint
# (`checkpoint_optimal.pkl`). Useful for re-generating the Williams plot,
# outlier tables, or tweaking SIGMA_LIMIT without re-running the full
# training pipeline.
#
# Usage:
#   python run_applicability_domain.py --checkpoint /path/to/checkpoint_optimal.pkl --out_dir ./ad_outputs/
# =============================================================================

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_leverage(X_ref, X_query):
    """Leverage of each row of X_query against the reference (training) set X_ref."""
    U, S, Vt = np.linalg.svd(X_ref, full_matrices=False)
    tol = S.max() * max(X_ref.shape) * np.finfo(float).eps * 100
    S_inv = np.where(S > tol, 1.0 / S**2, 0.0)
    return (X_query @ Vt.T) ** 2 @ S_inv


def main():
    parser = argparse.ArgumentParser(description="Standalone Applicability Domain analysis from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_optimal.pkl")
    parser.add_argument("--out_dir", default="./ad_outputs/", help="Directory to save figures/CSVs")
    parser.add_argument("--sigma_limit", type=float, default=None,
                         help="Override the sigma limit used at training time (default: use checkpoint's SIGMA_LIMIT)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cp = joblib.load(args.checkpoint)

    X_train_sc = cp["X_train_sc"]
    X_test_sc = cp["X_test_sc"]
    y_train = cp["y_train"]
    y_test = cp["y_test"]
    n_train = cp["n_train"]
    p = cp["p"]
    best_model_name = cp["best_model_name"]
    tag_str = cp["tag_str"]
    best_res = cp["best_res"]
    test_df = cp["test_df"]
    SIGMA_LIMIT = args.sigma_limit if args.sigma_limit is not None else cp["SIGMA_LIMIT"]

    h_train = compute_leverage(X_train_sc, X_train_sc)
    h_test = compute_leverage(X_train_sc, X_test_sc)
    h_star = 3.0 * (p + 1) / n_train
    s = best_res["cv_rmse_orig"]

    resid_train = y_train - best_res["cv_oof_pred"]
    resid_test = y_test - best_res["y_pred_test"]
    sr_train, sr_test = resid_train / s, resid_test / s

    in_ad_train = (h_train <= h_star) & (np.abs(sr_train) <= SIGMA_LIMIT)
    in_ad_test = (h_test <= h_star) & (np.abs(sr_test) <= SIGMA_LIMIT)

    print(f"AD: h*={h_star:.5f}  Best model: {best_model_name}{tag_str}  s(CV RMSE)={s:.3f} K  sigma_limit={SIGMA_LIMIT}")
    print(f"Train AD={in_ad_train.mean()*100:.1f}%  Test AD={in_ad_test.mean()*100:.1f}%")

    print(f"\n{'':22}  {'R2':>7}  {'MAE (K)':>9}  {'RMSE (K)':>10}  {'N':>6}")
    for mask, ad_label in [(in_ad_test, "Test - Within AD"), (~in_ad_test, "Test - Outside AD"),
                            (np.ones(len(y_test), dtype=bool), "Test - Overall")]:
        if mask.sum() < 2:
            continue
        _r2 = r2_score(y_test[mask], best_res["y_pred_test"][mask])
        _mae = mean_absolute_error(y_test[mask], best_res["y_pred_test"][mask])
        _rmse = np.sqrt(mean_squared_error(y_test[mask], best_res["y_pred_test"][mask]))
        print(f"{ad_label:22}  {_r2:7.4f}  {_mae:9.3f}  {_rmse:10.3f}  {mask.sum():6,}")

    mask_high_h, mask_high_sr = h_test > h_star, np.abs(sr_test) > SIGMA_LIMIT
    type_A = mask_high_h & ~mask_high_sr
    type_B = ~mask_high_h & mask_high_sr
    type_C = mask_high_h & mask_high_sr
    type_D = in_ad_test
    outlier_type = np.where(type_D, "D - Within AD", np.where(type_A, "A - High Leverage",
                    np.where(type_B, "B - High Residual", "C - Both")))
    abs_error = np.abs(resid_test)
    print(f"\nOutlier types: A={type_A.sum()}  B={type_B.sum()}  C={type_C.sum()}")

    # ---- Williams plot ----
    fig, ax = plt.subplots(figsize=(11, 8))
    sr_max = min(max(np.abs(sr_train).max(), np.abs(sr_test).max()) * 1.12, 12)
    x_max = max(h_train.max(), h_test.max()) * 1.18
    ax.fill_betweenx([-SIGMA_LIMIT, SIGMA_LIMIT], 0, h_star, color="#EBF5FB", alpha=0.55, zorder=0)
    ax.axvline(h_star, color="#2C3E50", linewidth=1.8, linestyle="--", label=f"h*={h_star:.5f}")
    ax.axhline(SIGMA_LIMIT, color="#2C3E50", linewidth=1.4, linestyle=":", label=f"+/-{SIGMA_LIMIT:.0f}sigma ({SIGMA_LIMIT*s:.1f} K)")
    ax.axhline(-SIGMA_LIMIT, color="#2C3E50", linewidth=1.4, linestyle=":")
    ax.scatter(h_train[in_ad_train], sr_train[in_ad_train], color="#2C6E8A", s=18, alpha=0.55, label=f"Train in AD ({in_ad_train.sum()})")
    ax.scatter(h_train[~in_ad_train], sr_train[~in_ad_train], color="#F39C12", marker="^", s=22, alpha=0.65, label=f"Train out ({(~in_ad_train).sum()})")
    ax.scatter(h_test[in_ad_test], sr_test[in_ad_test], color="#27AE60", s=18, alpha=0.55, label=f"Test in AD ({in_ad_test.sum()})")
    ax.scatter(h_test[~in_ad_test], sr_test[~in_ad_test], color="#C0392B", marker="s", s=22, alpha=0.75, label=f"Test out ({(~in_ad_test).sum()})")
    ax.set_xlabel("Leverage $h_i$")
    ax.set_ylabel("Standardised Residual")
    ax.set_title(f"Williams Plot - {best_model_name}{tag_str} (p={p})", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.002, x_max)
    ax.set_ylim(-sr_max, sr_max)
    ax.legend(fontsize=8.5)
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "fig_williams_plot.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {fig_path}")
    plt.close(fig)

    # ---- Export AD-annotated test set + outliers ----
    test_out = test_df.copy()
    test_out["y_actual"] = y_test
    test_out["y_predicted"] = best_res["y_pred_test"]
    test_out["Residual_K"] = resid_test
    test_out["Std_Residual"] = sr_test
    test_out["Abs_Error_K"] = abs_error
    test_out["Leverage_h"] = h_test
    test_out["Within_AD"] = in_ad_test
    test_out["Outlier_Type"] = outlier_type
    out_csv = os.path.join(args.out_dir, "test_with_AD.csv")
    test_out.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    outliers_df = test_out[~in_ad_test].copy().sort_values("Abs_Error_K", ascending=False)
    outliers_csv = os.path.join(args.out_dir, "outliers_outside_AD.csv")
    outliers_df.to_csv(outliers_csv, index=False)
    print(f"Saved -> {outliers_csv} ({len(outliers_df)} compounds)")

    print(f"\nDone. h_star = {h_star:.5f} | sigma (CV RMSE) = {s:.3f} K | 3*sigma = {3*s:.1f} K")


if __name__ == "__main__":
    main()
