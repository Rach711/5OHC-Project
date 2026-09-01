import pandas as pd
import glob
import os

feature_selection_dir = "../feature_selection/rf"
folder = os.path.join(feature_selection_dir, "RFECV")

files = sorted(glob.glob(os.path.join(folder, "RF_RFECV_[0-9]*.csv")))
n_seeds = len(files)

combined = pd.concat(pd.read_csv(f)[['Feature', 'Importance']] for f in files)

# Unlike RFCI's fixed top-40 cutoff, RFECV's list is whatever features its
# own cross-validated elimination decided to keep - so appearing here at all
# already reflects a formal decision, not just a ranking cutoff. Frequency
# across runs is still the main signal for stability.
summary = combined.groupby('Feature').agg(
    times_selected=('Importance', 'count'),
    mean_importance=('Importance', 'mean')
).reset_index()

summary['pct_of_runs'] = summary['times_selected'] / n_seeds * 100
summary = summary.sort_values(['times_selected', 'mean_importance'], ascending=[False, False]).reset_index(drop=True)

summary.to_csv(os.path.join(feature_selection_dir, "RFECV", "RF_RFECV_summary.csv"), index=False)

print(f"RFECV: {len(summary)} distinct features kept at least once across {n_seeds} runs")
print("Top 15 by how often they were selected:")
print(summary.head(15).to_string(index=False))