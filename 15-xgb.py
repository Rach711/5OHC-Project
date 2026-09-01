import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

# Importing the datasets - same logic as RandomForest.py
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

# Same 20 seeds as RandomForest.py and SVM.py, so results are directly comparable
random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173, 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657, 101777, 517844, 969889, 159625]

# Lasso and Dispersion Ratio are generated separately by
# SharedFeatureSelection.py (run that once - it doesn't depend on the model,
# so there's no reason to duplicate it here). XGBImportance, RFECV, and
# PermImportance are genuinely XGBoost-specific and stay in this script.
xgb_feature_selection_dir = "../feature_selection/xgb"
for subfolder in ["XGBImportance", "RFECV", "PermImportance"]:
    os.makedirs(os.path.join(xgb_feature_selection_dir, subfolder), exist_ok=True)


# Splitting the dataset for the hyperparameter search
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=159625, stratify=y)

# XGBoost hyperparameters. Unlike Random Forest, boosting fits trees
# sequentially to correct previous errors, which makes it more prone to
# overfitting on a dataset this small (~29 training rows, 360 features) if
# left at default settings - so max_depth is kept shallow, and subsample /
# colsample_bytree (randomly limiting rows and features per tree, similar
# in spirit to Random Forest's bootstrap and max_features) are searched
# specifically to guard against that.
xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [2, 3],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.3, 0.6]
}

xgb_grid_search = GridSearchCV(XGBClassifier(random_state=159625), xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
print("Best XGBoost params:", xgb_grid_search.best_params_)
print("Best XGBoost CV accuracy:", xgb_grid_search.best_score_)

best_xgb_params = xgb_grid_search.best_params_


# XGBoost - evaluated across all 20 seeds and averaged, same approach as
# the Random Forest and SVM scripts, rather than trusting a single split
xgb_accuracies, xgb_precisions, xgb_f1s, xgb_recalls = [], [], [], []

for random_seed in random_seeds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    xgb_model = XGBClassifier(**best_xgb_params, random_state=random_seed)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    xgb_accuracies.append(accuracy_score(y_test, y_pred))
    xgb_precisions.append(precision_score(y_test, y_pred, average='weighted'))
    xgb_f1s.append(f1_score(y_test, y_pred, average='weighted'))
    xgb_recalls.append(recall_score(y_test, y_pred, average='weighted'))

# Note: no OOB-equivalent here either, same reasoning as SVM - out-of-bag
# scoring is specific to bagging ensembles like Random Forest. XGBoost builds
# trees sequentially rather than independently on bootstrap samples, so
# there's no analogous free validation signal to report.
print(f"XGBoost Accuracy: {np.mean(xgb_accuracies):.3f} +/- {np.std(xgb_accuracies):.3f}")
print(f"XGBoost Precision: {np.mean(xgb_precisions):.3f} +/- {np.std(xgb_precisions):.3f}")
print(f"XGBoost f1: {np.mean(xgb_f1s):.3f} +/- {np.std(xgb_f1s):.3f}")
print(f"XGBoost Recall: {np.mean(xgb_recalls):.3f} +/- {np.std(xgb_recalls):.3f}")


# ============================================================
# FEATURE SELECTION #2 - XGBoost Feature Importance (gain-based)
# ============================================================

for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    xgb_imp_model = XGBClassifier(**best_xgb_params, random_state=random_seed)
    xgb_imp_model.fit(X_train, y_train)

    importances = xgb_imp_model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    top_idx = sorted_idx[:40]

    coeff_df = pd.DataFrame({'Feature': X_train.columns[top_idx], 'Importance': importances[top_idx]})
    filename = os.path.join(xgb_feature_selection_dir, "XGBImportance", f"XGBImportance_{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("XGBImportance done")


# ============================================================
# FEATURE SELECTION #3 - RFECV with XGBoost
# ============================================================

for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    xgb_rfecv_model = XGBClassifier(**best_xgb_params, random_state=random_seed)

    rfecv = RFECV(estimator=xgb_rfecv_model, cv=5, step=1, n_jobs=-1)
    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.get_support()]
    importances = rfecv.estimator_.feature_importances_

    coeff_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
    filename = os.path.join(xgb_feature_selection_dir, "RFECV", f"RFECV_{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("RFECV done")

# ============================================================
# FEATURE SELECTION #5 - Permutation Importance (cross-validated)
# ============================================================

for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    fold_importances = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        xgb_perm_model = XGBClassifier(**best_xgb_params, random_state=random_seed)
        xgb_perm_model.fit(X_train, y_train)

        result = permutation_importance(xgb_perm_model, X_test, y_test, n_repeats=10, random_state=random_seed, n_jobs=-1)
        fold_importances.append(result.importances_mean)

    mean_importances = np.mean(fold_importances, axis=0)
    sorted_idx = mean_importances.argsort()[::-1]
    top_idx = sorted_idx[:40]

    coeff_df = pd.DataFrame({'Feature': X.columns[top_idx], 'Importance': mean_importances[top_idx]})
    filename = os.path.join(xgb_feature_selection_dir, "PermImportance", f"PermImportance_{i}.csv")
    coeff_df.to_csv(filename, index=False)

print("Permutation Importance done")