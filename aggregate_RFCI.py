import pandas as pd
import glob
import os

feature_selection_dir = "../feature_selection/rf"
folder = os.path.join(feature_selection_dir, "RFCI")

files = sorted(glob.glob(os.path.join(folder, "RFCI*.csv")))
n_seeds = len(files)

combined = pd.concat(pd.read_csv(f)[['Feature', 'Importance']] for f in files)

# RF impurity-based importance is always non-negative, so no sign to track -
# just frequency (how often a feature made the top 40) and average importance
summary = combined.groupby('Feature').agg(
    times_selected=('Importance', 'count'),
    mean_importance=('Importance', 'mean')
).reset_index()

summary['pct_of_runs'] = summary['times_selected'] / n_seeds * 100
summary = summary.sort_values(['times_selected', 'mean_importance'], ascending=[False, False]).reset_index(drop=True)

summary.to_csv(os.path.join(feature_selection_dir, "RFCI", "RFCI_summary.csv"), index=False)

print(f"RF Importance: {len(summary)} distinct features in the top 40 at least once across {n_seeds} runs")
print("Top 15 by how often they were selected:")
print(summary.head(15).to_string(index=False))