import pandas as pd
import glob
import os

feature_selection_dir = "../feature_selection"
folder = os.path.join(feature_selection_dir, "Lasso")
 
files = sorted(glob.glob(os.path.join(folder, "Lasso*.csv")))
n_seeds = len(files)
 
combined = pd.concat(pd.read_csv(f)[['Feature', 'Coefficient']] for f in files)
 
# Lasso coefficients can be positive or negative (direction matters: positive
# ~ associated with hotspot, negative ~ associated with non-hotspot). A feature
# selected often but flipping sign between runs is weaker evidence than one
# selected often with a consistent sign.
summary = combined.groupby('Feature').agg(
    times_selected=('Coefficient', 'count'),
    mean_coefficient=('Coefficient', 'mean'),
    pct_positive=('Coefficient', lambda x: (x > 0).mean() * 100)
).reset_index()
 
summary['pct_of_runs'] = summary['times_selected'] / n_seeds * 100
 
# Sort by frequency first, then by |mean coefficient| for ties
summary = summary.reindex(
    summary['mean_coefficient'].abs().sort_values(ascending=False).index
).sort_values('times_selected', ascending=False, kind='stable').reset_index(drop=True)
 
summary.to_csv(os.path.join(feature_selection_dir, "Lasso_summary.csv"), index=False)
 
print(f"Lasso: {len(summary)} distinct features selected at least once across {n_seeds} runs")
print("Top 15 by how often they were selected:")
print(summary.head(15).to_string(index=False))