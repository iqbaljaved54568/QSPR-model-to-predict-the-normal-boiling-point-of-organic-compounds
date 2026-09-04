# NEXT-CELL LOADER — build the optimal-descriptor train/test CSVs from the
# backward-elimination checkpoint, ready to feed straight into the full
# tuning -> AD -> Y-randomization pipeline (just point that pipeline's
# ACTIVE_SET / FILES config at the CSVs this produces).
#
# WHY A SEPARATE CELL: the backward-elimination sweep is the expensive part.
# This loader does none of that work over again -- it just reads the
# checkpoint's decisions and slices the original 35-descriptor CSVs down to
# the chosen columns.
# =============================================================================

import os
import joblib
import pandas as pd

CHECKPOINT_PATH = "/content/drive/MyDrive/BP_FINAL_MODEL/figures/backward_elimination_35desc/checkpoint_backward_elimination_35desc.pkl"
OUT_DIR_OPTIMAL = "/content/drive/MyDrive/BP_FINAL_MODEL/descriptors/train_test_split/"
os.makedirs(OUT_DIR_OPTIMAL, exist_ok=True)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found at {CHECKPOINT_PATH}\n"
        f"This means backward_elimination_35desc.py (Sections 12-13 / the resume "
        f"cell) hasn't successfully completed and saved the checkpoint yet. "
        f"Run that first, confirm you see 'Checkpoint saved -> ...' printed, "
        f"THEN run this script."
    )
print(f"Checkpoint found: {CHECKPOINT_PATH} ({os.path.getsize(CHECKPOINT_PATH):,} bytes)")

ckpt = joblib.load(CHECKPOINT_PATH)
print(f"Loaded checkpoint from: {ckpt['source']}  (base set: {ckpt['base_descriptor_set']})")

# ── Choose which descriptor set to finalize on ───────────────────────────────
# Options:
#   "lightgbm_top10"   -> LightGBM's own elimination optimum (n=10) -- use this
#                          if LightGBM is the model you're deploying/reporting.
#   "lightgbm_top20"   -> broader top-20 overall (SHAP-ranked, full 35-set) --
#                          use this if you want a slightly larger, more
#                          conservative set.
#   "per_model_99pct"  -> each model keeps its OWN 99%-of-peak set (different
#                          descriptor count/list per model) -- use this if
#                          you're re-tuning all 5 models separately rather
#                          than committing to one shared descriptor set.
#   "common_17"        -> the n=17 set that Linear/XGBoost/ANN_raw
#                          independently converged to (only valid if all
#                          three actually agree on the same 17 descriptors --
#                          checked below, not assumed).
SELECTION_MODE = "lightgbm_top10"

print(f"\nSelection mode: {SELECTION_MODE}")

if SELECTION_MODE == "lightgbm_top10":
    chosen_features = {"__ALL_MODELS__": ckpt["lightgbm_top10_features"]}
    print(f"  LightGBM's own top-{ckpt['lightgbm_top10_optimal_n']} descriptors:")
    print(f"  {ckpt['lightgbm_top10_features']}")

elif SELECTION_MODE == "lightgbm_top20":
    chosen_features = {"__ALL_MODELS__": ckpt["lightgbm_top20_overall_features"]}
    print(f"  Top-20 overall (SHAP-ranked): {ckpt['lightgbm_top20_overall_features']}")

elif SELECTION_MODE == "per_model_99pct":
    chosen_features = ckpt["per_model_selected_features_99pct"]
    for m, feats in chosen_features.items():
        print(f"  {m}: n={len(feats)}  {feats}")

elif SELECTION_MODE == "common_17":
    per_model = ckpt["per_model_selected_features_99pct"]
    sets_17 = {m: set(feats) for m, feats in per_model.items() if len(feats) == 17}
    if len(sets_17) < 2:
        raise ValueError("Fewer than two models actually landed on n=17 -- "
                          "check per_model_selected_features_99pct in the checkpoint.")
    common = set.intersection(*sets_17.values())
    print(f"  Models with n=17: {list(sets_17.keys())}")
    print(f"  Descriptors common to ALL of them: {len(common)} -> {sorted(common)}")
    if len(common) < 17:
        print(f"  NOTE: not all 17 descriptors are identical across these models -- "
              f"only {len(common)} are shared. Using the shared subset.")
    chosen_features = {"__ALL_MODELS__": sorted(common)}

else:
    raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")

# ── Slice the original 35-descriptor CSVs down to the chosen columns ────────
train_df = pd.read_csv(ckpt["TRAIN_FILE"])
test_df  = pd.read_csv(ckpt["TEST_FILE"])

meta_cols = list(train_df.columns[:ckpt["DESCRIPTOR_START_INDEX"]])  # id/smiles/target/etc columns kept as-is

def build_subset(df, features, tag):
    keep_cols = meta_cols + features
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in source CSV: {missing}")
    out = df[keep_cols].copy()
    print(f"  {tag}: {out.shape[0]} rows x {len(features)} descriptors (+{len(meta_cols)} metadata cols)")
    return out

if "__ALL_MODELS__" in chosen_features:
    feats = chosen_features["__ALL_MODELS__"]
    train_out = build_subset(train_df, feats, "train")
    test_out  = build_subset(test_df,  feats, "test")

    train_path = f"{OUT_DIR_OPTIMAL}train_optimal.csv"
    test_path  = f"{OUT_DIR_OPTIMAL}test_optimal.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Hard verification -- don't just trust the print statement below, actually
    # check the files landed on disk with real content before declaring success.
    for path, expected_rows in [(train_path, len(train_out)), (test_path, len(test_out))]:
        if not os.path.exists(path):
            raise RuntimeError(f"Save appeared to succeed but file is missing: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(f"File was created but is empty (0 bytes): {path}")
        print(f"  VERIFIED: {path} exists, {size:,} bytes, {expected_rows} rows written")

    print(f"\nSaved -> {train_path}")
    print(f"Saved -> {test_path}")
    print(f"\nIn the full pipeline script, confirm these paths are set:")
    print(f'  TRAIN_FILE = "{train_path}"')
    print(f'  TEST_FILE  = "{test_path}"')

else:
    # per-model mode: write one train/test pair per model
    print()
    for model_name, feats in chosen_features.items():
        train_out = build_subset(train_df, feats, f"train ({model_name})")
        test_out  = build_subset(test_df,  feats, f"test ({model_name})")
        train_path = f"{OUT_DIR_OPTIMAL}train_optimal_{model_name}.csv"
        test_path  = f"{OUT_DIR_OPTIMAL}test_optimal_{model_name}.csv"
        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)
        for path in [train_path, test_path]:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise RuntimeError(f"Save failed or produced an empty file: {path}")
        print(f"  Saved -> {train_path}")
        print(f"  Saved -> {test_path}")
    print(f"\nNote: with per-model descriptor sets, the full pipeline needs to be run\n"
          f"once per model (each pointed at its own train_optimal_<model>.csv), since\n"
          f"they no longer share a common feature space.")

# ── Carry forward the locked hyperparameters + transform decisions too ─────
print(f"\n{'='*64}")
print("Locked hyperparameters (35-descriptor tuned) available from checkpoint:")
print(f"{'='*64}")
for name in ["LINEAR_PARAMS", "RF_PARAMS", "XGB_PARAMS", "LGBM_PARAMS", "ANN_PARAMS_RAW", "ANN_PARAMS_LOG"]:
    print(f"  {name} = {ckpt[name]}")

print(f"\nPer-model transform locked during the sweep (raw vs. log):")
for m, is_log in ckpt["per_model_transform"].items():
    print(f"  {m}: {'log' if is_log else 'original'}")

print(f"\nNOTE: these hyperparameters were tuned for the FULL 35-descriptor set.")
print(f"Once you're training on the reduced optimal set, you should still set")
print(f"AUTO_TUNE = True in the full pipeline and let Optuna re-tune from scratch --")
print(f"a smaller feature space changes the optimal hyperparameters (e.g. shallower")
print(f"trees, different regularization), so these values are a reasonable starting")
print(f"point / warm-start reference, not a final answer for the reduced set.")

# !ls -la /content/drive/MyDrive/BP_FINAL_MODEL/figures/backward_elimination_35desc/

# =============================================================================
