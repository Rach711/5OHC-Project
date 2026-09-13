import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# ============================================================
# DATA LOADING - same merge logic as RandomForest.py/svm.py/xgb.py, since
# Lasso needs the same X/y as every other feature-selection method even
# though it doesn't depend on which model (RF/SVM/XGB) is being evaluated.
# This is why it now lives in its own standalone script instead of inside
# RandomForest.py - it only needs to run once, ever.
# ============================================================
csv_dir = "../parameters_csv"

params = ["tbend", "shear", "stretch", "stagger", "buckle", "propel", "opening",
          "xdisp", "ydisp", "inclin", "tip", "axbend", "shift", "slide", "rise",
          "tilt", "roll", "twist", "hris", "htwi", "phaseW", "ampW", "gammaW",
          "gammaC", "phaseC", "ampC", "minw", "mind", "majw", "majd"]

dfs = []
for p in params:
    df = pd.read_csv(f"{csv_dir}/{p}.csv")
    data_cols = df.columns.drop("sequence")
    df = df.rename(columns={c: f"{c}_{p}" for c in data_cols})
    dfs.append(df.set_index("sequence"))

merged_df = pd.concat(dfs, axis="columns").reset_index()

hotspot_map = {
    'APC_637': 1, 'APC_641': 0, 'APC_3335': 0, 'APC_3340': 1,
    'APC_4099': 1, 'APC_4103': 0, 'APC_4343': 0, 'APC_4348': 1,
    'TP53_632': 0, 'TP53_637': 1, 'TP53_844': 1, 'TP53_849': 0,
}
merged_df['base_sequence'] = merged_df['sequence'].str.rsplit('_', n=1).str[0]
merged_df['Outcome'] = merged_df['base_sequence'].map(hotspot_map)
assert merged_df['Outcome'].isnull().sum() == 0, "Some sequences didn't match hotspot_map - check naming"

total_data = merged_df.drop(columns=['sequence', 'base_sequence'])
X = total_data.drop(['Outcome'], axis=1)
y = total_data['Outcome']

# Same 20 seeds as RandomForest.py/svm.py/xgb.py, so results stay comparable
random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173, 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657, 101777, 517844, 969889, 159625]

# Lasso and Dispersion Ratio don't depend on the model, so their output
# lives under "shared" rather than under rf/svm/xgb/kmeans - one set of 20
# runs, reused wherever feature selection is discussed.
shared_dir = "../feature_selection/shared"
os.makedirs(os.path.join(shared_dir, "Lasso"), exist_ok=True)


# ============================================================
# GENERATION - one Lasso fit per seed (moved here from RandomForest.py's
# old "FEATURE SELECTION #1" block - logic is unchanged)
# ============================================================
for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    lasso = Lasso(alpha=1, random_state=random_seed)
    lasso.fit(X_train, y_train)

    coefs = lasso.coef_
    nonzero_idx = np.nonzero(coefs)[0]

    coeff_df = pd.DataFrame({'Feature': X_train.columns[nonzero_idx], 'Coefficient': coefs[nonzero_idx]})
    filename = os.path.join(shared_dir, "Lasso", f"Lasso{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("Lasso generation done")


# ============================================================
# AGGREGATION - combine the 20 per-seed files into one summary
# ============================================================
folder = os.path.join(shared_dir, "Lasso")

# "Lasso[0-9]*.csv" (digit right after "Lasso") rather than "Lasso*.csv" -
# the wider pattern would also match Lasso_summary.csv itself on a rerun
# and fold last run's summary back into the next one's average
files = sorted(glob.glob(os.path.join(folder, "Lasso[0-9]*.csv")))
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

summary.to_csv(os.path.join(folder, "Lasso_summary.csv"), index=False)

print(f"Lasso: {len(summary)} distinct features selected at least once across {n_seeds} runs")
print("Top 15 by how often they were selected:")
print(summary.head(15).to_string(index=False))