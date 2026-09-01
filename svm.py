import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

# Importing the datasets
csv_dir = "../parameters_csv"

# Lasso and Dispersion Ratio are generated separately by
# SharedFeatureSelection.py (run that once - it doesn't depend on the model,
# so there's no reason to duplicate it here). SVMCoef, RFECV, and
# PermImportance are genuinely SVM-specific and stay in this script.
svm_feature_selection_dir = "../feature_selection/svm"
for subfolder in ["SVMCoef", "RFECV", "PermImportance"]:
    os.makedirs(os.path.join(svm_feature_selection_dir, subfolder), exist_ok=True)

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

random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173, 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657, 101777, 517844, 969889, 159625]


# ============================================================
# MODEL - grid search + averaged evaluation across all 20 seeds
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=159625, stratify=y)

svm_param_grid = [
    {'svm__kernel': ['linear'], 'svm__C': [0.01, 0.1, 1, 10, 100]},
    {'svm__kernel': ['rbf'], 'svm__C': [0.01, 0.1, 1, 10, 100], 'svm__gamma': ['scale', 'auto']}
]

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=159625))
])

svm_grid_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(X_train, y_train)
print("Best SVM params:", svm_grid_search.best_params_)
print("Best SVM CV accuracy:", svm_grid_search.best_score_)

best_svm_params = svm_grid_search.best_params_

svm_accuracies, svm_precisions, svm_f1s, svm_recalls = [], [], [], []

for random_seed in random_seeds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**{k.replace('svm__', ''): v for k, v in best_svm_params.items()},
                    random_state=random_seed))
    ])
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    svm_accuracies.append(accuracy_score(y_test, y_pred))
    svm_precisions.append(precision_score(y_test, y_pred, average='weighted'))
    svm_f1s.append(f1_score(y_test, y_pred, average='weighted'))
    svm_recalls.append(recall_score(y_test, y_pred, average='weighted'))

print(f"SVM Accuracy: {np.mean(svm_accuracies):.3f} +/- {np.std(svm_accuracies):.3f}")
print(f"SVM Precision: {np.mean(svm_precisions):.3f} +/- {np.std(svm_precisions):.3f}")
print(f"SVM f1: {np.mean(svm_f1s):.3f} +/- {np.std(svm_f1s):.3f}")
print(f"SVM Recall: {np.mean(svm_recalls):.3f} +/- {np.std(svm_recalls):.3f}")


# ============================================================
# FEATURE SELECTION #2 - SVM Coefficient Magnitude (linear-kernel only)
# ============================================================

for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    svm_coef_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=0.1, random_state=random_seed))
    ])
    svm_coef_model.fit(X_train, y_train)

    coefs = svm_coef_model.named_steps['svm'].coef_[0]
    sorted_idx = np.abs(coefs).argsort()[::-1]
    top_idx = sorted_idx[:40]

    coeff_df = pd.DataFrame({'Feature': X_train.columns[top_idx], 'Coefficient': coefs[top_idx]})
    filename = os.path.join(svm_feature_selection_dir, "SVMCoef", f"SVMCoef_{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("SVMCoef done")


# ============================================================
# FEATURE SELECTION #3 - RFECV with linear SVM
# ============================================================

for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    svm_rfecv_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=0.1, random_state=random_seed))
    ])

    rfecv = RFECV(estimator=svm_rfecv_pipeline, cv=5, step=1, n_jobs=-1,
                  importance_getter='named_steps.svm.coef_')
    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.get_support()]
    coefs = rfecv.estimator_.named_steps['svm'].coef_[0]

    coeff_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': coefs})
    filename = os.path.join(svm_feature_selection_dir, "RFECV", f"RFECV_{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("RFECV done")


# ============================================================
# FEATURE SELECTION #5 - Permutation Importance (cross-validated, any kernel)
# ============================================================

for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    fold_importances = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        svm_perm_model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=0.1, random_state=random_seed))
        ])
        svm_perm_model.fit(X_train, y_train)

        result = permutation_importance(svm_perm_model, X_test, y_test, n_repeats=10, random_state=random_seed, n_jobs=-1)
        fold_importances.append(result.importances_mean)

    mean_importances = np.mean(fold_importances, axis=0)
    sorted_idx = mean_importances.argsort()[::-1]
    top_idx = sorted_idx[:40]

    coeff_df = pd.DataFrame({'Feature': X.columns[top_idx], 'Importance': mean_importances[top_idx]})
    filename = os.path.join(svm_feature_selection_dir, "PermImportance", f"PermImportance_{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("Permutation Importance done")