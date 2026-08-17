import pandas as pd
import glob
import os

feature_selection_dir = "../feature_selection"
folder = os.path.join(feature_selection_dir, "DispersionRatio")

files = sorted(glob.glob(os.path.join(folder, "DispersionRatio_*.csv")))
n_seeds = len(files)

combined = pd.concat(pd.read_csv(f)[['Feature', 'DispersionRatio']] for f in files)

# Unlike the other four methods, Dispersion Ratio doesn't select a subset of
# features - it computes a value for every feature on every run, so there's
# no "times selected" to count here. Instead: average dispersion ratio (higher
# = more variable relative to its own scale) and how consistent that value
# was across the 20 runs (lower std = more stable estimate).
summary = combined.groupby('Feature').agg(
    mean_dispersion_ratio=('DispersionRatio', 'mean'),
    std_dispersion_ratio=('DispersionRatio', 'std')
).reset_index()

summary = summary.sort_values('mean_dispersion_ratio', ascending=False).reset_index(drop=True)

summary.to_csv(os.path.join(feature_selection_dir, "DispersionRatio", "DispersionRatio_summary.csv"), index=False)

print(f"Dispersion Ratio: averaged across {n_seeds} runs, {len(summary)} features total")
print("Top 15 by average dispersion ratio:")
print(summary.head(15).to_string(index=False))