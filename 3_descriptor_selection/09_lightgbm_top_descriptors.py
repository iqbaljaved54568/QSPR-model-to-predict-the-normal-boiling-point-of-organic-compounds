"""
Top descriptors of the LightGBM elimination checkpoint

Part of: A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of
Structurally Diverse Organic Compounds (Javed, Waters, Belton, Cooke; ACS Omega, submitted).
Originally run in Google Colab; set the file paths below to your own data folder.
"""

import shap

TOP_N_OVERALL = 20

lgbm_row_summary = summary_df[summary_df["model"] == "LightGBM"].iloc[0]
lgbm_log_transform = bool(results_df.loc[results_df["model"] == "LightGBM", "log_transform"].iloc[0])
n_opt_lgbm = int(lgbm_row_summary["optimal_n_99pct"])
lgbm_top10_features = lgbm_row_summary["features_at_optimal_n_99pct"].split("|")

print(f"\n{'='*64}")
print(f"(A) LightGBM TOP-{n_opt_lgbm} DESCRIPTORS (its own 99%-of-peak elimination optimum)")
print(f"{'='*64}")
print(f"  Target transform locked for LightGBM during the sweep: "
      f"{'log' if lgbm_log_transform else 'original'}")

lgbm_top10_idx = [feature_names.index(f) for f in lgbm_top10_features]
X_train_lgbm10 = X_train[:, lgbm_top10_idx]
X_test_lgbm10  = X_test[:, lgbm_top10_idx]
y_fit_lgbm10   = np.log(y_train) if lgbm_log_transform else y_train

lgbm_final10 = LGBMRegressor(**LGBM_PARAMS)
lgbm_final10.fit(X_train_lgbm10, y_fit_lgbm10)
imp10 = lgbm_final10.feature_importances_
order10 = np.argsort(imp10)[::-1]

lgbm_top10_table = pd.DataFrame({
    "Rank": range(1, len(order10) + 1),
    "Descriptor": [lgbm_top10_features[i] for i in order10],
    "Gain_Importance": imp10[order10],
})
lgbm_top10_table.to_csv(f"{OUT_DIR}lightgbm_top10_descriptors.csv", index=False)
print(lgbm_top10_table.to_string(index=False))
print(f"  Saved -> {OUT_DIR}lightgbm_top10_descriptors.csv")

fig, ax = plt.subplots(figsize=(9, 5))
plot_data = lgbm_top10_table.iloc[::-1]
norm = plot_data["Gain_Importance"] / plot_data["Gain_Importance"].max()
bar_c = plt.cm.RdYlBu_r(0.25 + 0.65 * norm.values)
bars = ax.barh(plot_data["Descriptor"], plot_data["Gain_Importance"],
               color=bar_c, edgecolor="white", linewidth=0.5, height=0.7)
for bar, val in zip(bars, plot_data["Gain_Importance"]):
    ax.text(bar.get_width() + plot_data["Gain_Importance"].max()*0.02,
            bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", fontsize=8.5)
ax.set_xlabel("Gain Importance", fontsize=10.5)
ax.set_title(f"LightGBM — Top {n_opt_lgbm} Descriptors (elimination optimum)",
             fontsize=12, fontweight="bold")
ax.xaxis.grid(True); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, f"{OUT_DIR}fig_lightgbm_top10_importance.png")
plt.show(); plt.close(fig)

# ---- (B) Top-20 overall: LightGBM refit on the full 35-descriptor set ----
print(f"\n{'='*64}")
print(f"(B) TOP-{TOP_N_OVERALL} DESCRIPTORS OVERALL (LightGBM, full 35-descriptor set)")
print(f"{'='*64}")

y_fit_full_lgbm = np.log(y_train) if lgbm_log_transform else y_train
lgbm_full = LGBMRegressor(**LGBM_PARAMS)
lgbm_full.fit(X_train, y_fit_full_lgbm)
imp_full = lgbm_full.feature_importances_

print("  Computing SHAP values on full 35-descriptor LightGBM model ...")
explainer = shap.TreeExplainer(lgbm_full)
shap_values_full = explainer.shap_values(X_train)
shap_df_full = pd.DataFrame(shap_values_full, columns=feature_names)
mean_shap_full = shap_df_full.abs().mean().sort_values(ascending=False)
top20_shap_feats = mean_shap_full.head(TOP_N_OVERALL).index.tolist()

top20_table = pd.DataFrame({
    "Rank": range(1, TOP_N_OVERALL + 1),
    "Descriptor": top20_shap_feats,
    "Mean_abs_SHAP": mean_shap_full.head(TOP_N_OVERALL).values,
    "Gain_Importance": [imp_full[feature_names.index(f)] for f in top20_shap_feats],
})
top20_table.to_csv(f"{OUT_DIR}lightgbm_top20_overall_shap_importance.csv", index=False)
print(top20_table.to_string(index=False))
print(f"  Saved -> {OUT_DIR}lightgbm_top20_overall_shap_importance.csv")

# Combined bar (SHAP) + beeswarm figure, matching the style of the main pipeline
fig = plt.figure(figsize=(16, 0.45*TOP_N_OVERALL + 3))
fig.suptitle(f"SHAP Analysis — LightGBM (Full 35-Descriptor Set) | Top {TOP_N_OVERALL}",
             fontsize=13, fontweight="bold", y=1.02)
gs = plt.GridSpec(1, 2, wspace=0.45, left=0.06, right=0.97, top=0.90, bottom=0.10)

ax_bar = fig.add_subplot(gs[0, 0])
data = mean_shap_full.head(TOP_N_OVERALL).iloc[::-1]
norm = (data - data.min()) / (data.max() - data.min())
bc = plt.cm.Oranges(0.35 + 0.60 * norm.values)
bars = ax_bar.barh(data.index, data.values, color=bc, edgecolor="white", linewidth=0.5, height=0.72)
for bar, val in zip(bars, data.values):
    ax_bar.text(bar.get_width() + data.max()*0.02, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7.8)
ax_bar.set_xlabel("Mean |SHAP| (K)", fontsize=10)
ax_bar.set_title("(A)  Global Feature Importance", loc="left", fontsize=10, fontstyle="italic")
ax_bar.tick_params(axis="y", length=0, labelsize=8.5)
ax_bar.xaxis.grid(True); ax_bar.set_axisbelow(True)

ax_bee = fig.add_subplot(gs[0, 1])
plt.sca(ax_bee)
top_idx = [feature_names.index(f) for f in top20_shap_feats]
shap.summary_plot(shap_values_full[:, top_idx], X_train[:, top_idx],
                   feature_names=top20_shap_feats, plot_type="dot",
                   max_display=TOP_N_OVERALL, show=False, color_bar=True, plot_size=None)
ax_bee = plt.gca()
ax_bee.set_title("(B)  Direction & Magnitude (Beeswarm)", loc="left", fontsize=10, fontstyle="italic")
ax_bee.set_xlabel("SHAP Value — Impact on BP (K)", fontsize=10)

save_fig(fig, f"{OUT_DIR}fig_lightgbm_top20_shap_combined.png")
plt.show(); plt.close(fig)

print(f"\n{'='*64}")
print("FINAL DESCRIPTOR ANALYSIS COMPLETE")
print(f"{'='*64}")
print(f"  LightGBM top-{n_opt_lgbm} (own elimination optimum) -> lightgbm_top10_descriptors.csv")
print(f"  Top-{TOP_N_OVERALL} overall (SHAP + gain, full 35-descriptor set) -> lightgbm_top20_overall_shap_importance.csv")

# =============================================================================
# 13. CHECKPOINT — everything the NEXT pipeline cell needs
#     Saves selected descriptor sets, transform decisions, and locked
#     hyperparameters so the full tuning -> AD -> Y-randomization pipeline
#     can be re-run on the chosen optimal descriptor set WITHOUT re-running
#     this entire elimination sweep. Load this in the next cell with:
#         import joblib
#         ckpt = joblib.load(".../checkpoint_backward_elimination_35desc.pkl")
# =============================================================================

import joblib

per_model_transform = dict(zip(results_df["model"], results_df["log_transform"]))
per_model_selected_features_99pct = {
    row["model"]: row["features_at_optimal_n_99pct"].split("|") for _, row in summary_df.iterrows()
}
per_model_selected_features_stat = {
    row["model"]: (row["features_at_optimal_n_stat"].split("|") if row["features_at_optimal_n_stat"] else None)
    for _, row in summary_df.iterrows()
}

checkpoint = {
    "source": "backward_elimination_35desc.py",
    "base_descriptor_set": "35desc",
    "feature_names_35": feature_names,
    "results_df": results_df.drop(columns=["cv_r2_folds"], errors="ignore"),
    "summary_df": summary_df,
    "anchor_df": anchor_df,
    "per_model_transform": per_model_transform,                       # e.g. {"LightGBM": False, ...}
    "per_model_selected_features_99pct": per_model_selected_features_99pct,
    "per_model_selected_features_stat": per_model_selected_features_stat,
    "lightgbm_top10_features": lgbm_top10_features,
    "lightgbm_top10_optimal_n": n_opt_lgbm,
    "lightgbm_top20_overall_features": top20_shap_feats,
    # locked hyperparameters used throughout this sweep (35-descriptor tuned)
    "LINEAR_PARAMS": LINEAR_PARAMS, "RF_PARAMS": RF_PARAMS,
    "XGB_PARAMS": XGB_PARAMS, "LGBM_PARAMS": LGBM_PARAMS,
    "ANN_PARAMS_RAW": ANN_PARAMS_RAW, "ANN_PARAMS_LOG": ANN_PARAMS_LOG,
    "TRAIN_FILE": TRAIN_FILE, "TEST_FILE": TEST_FILE,
    "TARGET_COL_INDEX": TARGET_COL_INDEX, "DESCRIPTOR_START_INDEX": DESCRIPTOR_START_INDEX,
}

checkpoint_path = f"{OUT_DIR}checkpoint_backward_elimination_35desc.pkl"
joblib.dump(checkpoint, checkpoint_path)
print(f"\n  Checkpoint saved -> {checkpoint_path}")
