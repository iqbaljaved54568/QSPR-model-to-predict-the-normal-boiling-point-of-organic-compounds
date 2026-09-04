# Capture exact package versions for the SI "Software, Versions, and
# Reproducibility" section (Table S8).
# =============================================================================

import sys
import platform
from importlib.metadata import version, PackageNotFoundError

def get_version(pkg_name, import_name=None):
    """Try importlib.metadata first (works even for packages without __version__),
    fall back to the module's own __version__ attribute."""
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        pass
    try:
        mod = __import__(import_name or pkg_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "NOT INSTALLED / NOT FOUND"

packages = [
    ("Python",       None,          None),   # handled specially below
    ("numpy",        "numpy",       "numpy"),
    ("pandas",       "pandas",      "pandas"),
    ("scikit-learn", "scikit-learn","sklearn"),
    ("scipy",        "scipy",       "scipy"),
    ("matplotlib",   "matplotlib",  "matplotlib"),
    ("seaborn",      "seaborn",     "seaborn"),
    ("torch",        "torch",       "torch"),
    ("xgboost",      "xgboost",     "xgboost"),
    ("lightgbm",     "lightgbm",    "lightgbm"),
    ("optuna",       "optuna",      "optuna"),
    ("shap",         "shap",        "shap"),
    ("rdkit",        "rdkit",       "rdkit"),
    ("joblib",       "joblib",      "joblib"),
]

print(f"{'Package':<15} {'Version'}")
print("-" * 40)
print(f"{'Python':<15} {platform.python_version()}")

rows_for_table = [("Python", platform.python_version())]

for display_name, pip_name, import_name in packages[1:]:
    v = get_version(pip_name, import_name)
    print(f"{display_name:<15} {v}")
    rows_for_table.append((display_name, v))

# Also flag CUDA/GPU info since it can affect ANN reproducibility
try:
    import torch
    print(f"\nCUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version      : {torch.version.cuda}")
        print(f"GPU device        : {torch.cuda.get_device_name(0)}")
except Exception:
    pass

# Print ready-to-paste rows for Table S8 in the SI
print("\n" + "=" * 40)
print("Copy-paste rows for Table S8:")
print("=" * 40)
for name, v in rows_for_table:
    print(f'["{name}", "{v}"],')

# =============================================================================
