import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================================
# DATA LOADING - same merge logic as RandomForest.py/svm.py/xgb.py, since
# Dispersion Ratio needs the same X as every other feature-selection method
# even though it doesn't depend on which model (RF/SVM/XGB) is being
# evaluated. This is why it now lives in its own standalone script instead
# of inside RandomForest.py - it only needs to run once, ever.
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
os.makedirs(os.path.join(shared_dir, "DispersionRatio"), exist_ok=True)


# ============================================================
# GENERATION - one dispersion ratio per seed, plus a bar-chart PNG of it
# (moved here from RandomForest.py's old "FEATURE SELECTION #4" block -
# logic, including the matplotlib plot, is unchanged)
# ============================================================
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    am = np.mean(X_train, axis=0)
    gm = np.power(np.prod(X_train, axis=0), 1 / X_train.shape[0])
    disp_ratio = am / gm

    plt.bar(np.arange(X_train.shape[1]), disp_ratio, color='teal')
    plt.savefig(os.path.join(shared_dir, "DispersionRatio", f"DispersionRatio_{i}.png"))
    plt.clf()

    disp_df = pd.DataFrame({'Feature': X_train.columns, 'DispersionRatio': disp_ratio})
    csv_filename = os.path.join(shared_dir, "DispersionRatio", f"DispersionRatio_{i}.csv")
    disp_df.to_csv(csv_filename, index=False)

print("Dispersion Ratio generation done")


# ============================================================
# AGGREGATION - combine the 20 per-seed CSVs into one summary. Unlike the
# other methods, Dispersion Ratio doesn't select a subset of features - it
# computes a value for every feature on every run, so there's no "times
# selected" to count here. Instead: average dispersion ratio (higher = more
# variable relative to its own scale) and how consistent that value was
# across the 20 runs (lower std = more stable estimate).
# ============================================================
folder = os.path.join(shared_dir, "DispersionRatio")

# "DispersionRatio_[0-9]*.csv" (digit right after the underscore) rather
# than "DispersionRatio_*.csv" - the wider pattern would also match
# DispersionRatio_summary.csv itself on a rerun and fold last run's summary
# back into the next one's average
files = sorted(glob.glob(os.path.join(folder, "DispersionRatio_[0-9]*.csv")))
n_seeds = len(files)

combined = pd.concat(pd.read_csv(f)[['Feature', 'DispersionRatio']] for f in files)

summary = combined.groupby('Feature').agg(
    mean_dispersion_ratio=('DispersionRatio', 'mean'),
    std_dispersion_ratio=('DispersionRatio', 'std')
).reset_index()

summary = summary.sort_values('mean_dispersion_ratio', ascending=False).reset_index(drop=True)

summary.to_csv(os.path.join(folder, "DispersionRatio_summary.csv"), index=False)

print(f"Dispersion Ratio: averaged across {n_seeds} runs, {len(summary)} features total")
print("Top 15 by average dispersion ratio:")
print(summary.head(15).to_string(index=False))