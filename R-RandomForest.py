import sklearn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# Importing the datasets
csv_dir = "../parameters_csv"

# Output folders for feature selection results - one per method,
feature_selection_dir = "../feature_selection"
for subfolder in ["Lasso", "RFCI", "RFECV", "DispersionRatio", "PermImportance"]:
    os.makedirs(os.path.join(feature_selection_dir, subfolder), exist_ok=True)

params = ["tbend", "shear", "stretch", "stagger", "buckle", "propel", "opening",
          "xdisp", "ydisp", "inclin", "tip", "axbend", "shift", "slide", "rise",
          "tilt", "roll", "twist", "hris", "htwi", "phaseW", "ampW", "gammaW",
          "gammaC", "phaseC", "ampC", "minw", "mind", "majw", "majd"]

dfs = []
# read in each csv, 
# appends parameter suffix to every column but sequence column,
# sets sequence column as index and appends it to dfs
print(">>> loading parameter csvs <<<")
for p in params:
    df = pd.read_csv(f"{csv_dir}/{p}.csv")
    data_cols = df.columns.drop("sequence")
    df = df.rename(columns={c: f"{c}_{p}" for c in data_cols})
    dfs.append(df.set_index("sequence"))

# concatenates all dataframes by sequence
# reset.index changes sequences back from an index to a normal column
print(">>> creating dataframe of all parameters <<<")
merged_df = pd.concat(dfs, axis="columns").reset_index()
# makes sure not to truncate the columns
pd.set_option('display.max_columns', None)
merged_df.to_csv("merged_output.csv", index=False)
print(merged_df.shape)

# Assign hotspot (1) / non-hotspot (0) labels by matching each row's base sequence name
hotspot_map = {
    'APC_637': 1, 'APC_641': 0, 'APC_3335': 0, 'APC_3340': 1,
    'APC_4099': 1, 'APC_4103': 0, 'APC_4343': 0, 'APC_4348': 1,
    'TP53_632': 0, 'TP53_637': 1, 'TP53_844': 1, 'TP53_849': 0,
}

# extract APC_637 from the APC_637_1 sequence name
print(">>> creating hotspot and non-hotspot column <<<")
merged_df['base_sequence'] = merged_df['sequence'].str.rsplit('_', n=1).str[0]
# looks up the sequence in the dictionary and adds the number to a new column called outcome
merged_df['Outcome'] = merged_df['base_sequence'].map(hotspot_map)
# sanity check to ensure no sequence was left unmapped
assert merged_df['Outcome'].isnull().sum() == 0, "Some sequences didn't match hotspot_map - check naming"

# removes sequence and base_sequence columns,
# leaves a dataframe of only feature matrix and the outcome
print(">>> creating feature matrix <<<")
total_data = merged_df.drop(columns=['sequence', 'base_sequence'])
total_data

# Establish X and y variables
# everything but the label
X=total_data.drop(['Outcome'], axis=1)
# only the label
y=total_data['Outcome']

# List of 20 predetermined random seeds
random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173, 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657, 101777, 517844, 969889, 159625]


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=159625)


# Define hyperparamaters for Random Forest
# Number of trees (10) in random forest
print(">>> assigning hyperparameters <<<")
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]

# Create the parameter grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(param_grid)

# Search the grid with 5-fold cross-validation on the training set only,
# so the held-out test set stays untouched for the final evaluation below
grid_search = GridSearchCV(RandomForestClassifier(bootstrap=True, random_state=159625),
                            param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# best_rf_params is reused below for every other Random Forest in this script,
# so all five methods are built on the same evidence-based configuration
# instead of five separately-guessed ones
best_rf_params = grid_search.best_params_

# Random Forest - evaluated across all 20 seeds and averaged, rather than
# trusting a single train/test split (with only 8 test rows, one split's
# result can vary a lot from another purely by chance)
accuracies, precisions, f1s, recalls, oob_scores = [], [], [], [], []

print(">>> Running Random Forest <<<")
for random_seed in random_seeds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    rf_Model = RandomForestClassifier(**best_rf_params,
                                      bootstrap=True,
                                      oob_score=True,
                                      random_state=random_seed)
    rf_Model.fit(X_train, y_train)

    y_pred = rf_Model.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    f1s.append(f1_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    oob_scores.append(rf_Model.oob_score_)

print(f"Accuracy: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
print(f"Precision: {np.mean(precisions):.3f} +/- {np.std(precisions):.3f}")
print(f"f1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
print(f"Recall: {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}")
print(f"OOB Score: {np.mean(oob_scores):.3f} +/- {np.std(oob_scores):.3f}")



# FEATURE SELECTION #1 - Lasso (L1)

# Iterate through the random seeds
print(">>> feature selection 1 - Lasso <<<")
for i, random_seed in enumerate(random_seeds, start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    lasso = Lasso(alpha=1)

    # Fit the model on the training data
    lasso.fit(X_train, y_train)

    # Get the selected features and their coefficients
    selected_features = X_train.columns[lasso.coef_ != 0]
    feature_coefficients = lasso.coef_[lasso.coef_ != 0]

    # Create a DataFrame
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': feature_coefficients})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = os.path.join(feature_selection_dir, "Lasso", f"Lasso{i}.csv")
    coeff_df.to_csv(filename, index=False)

# FEATURE SELECTION #2 - Random Forest Importance Values

# Iterate through the random seeds
print(">>> feature selection 2 - RF Importance Values <<<")
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


    # Create a Random Forest Classifier
    rf = RandomForestClassifier(bootstrap=True,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=2,
                                min_samples_split=2,
                                n_estimators=100,
                                random_state=random_seed)

    # Train the Random Forest model
    rf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Sort features by importance in descending order
    sorted_indices = feature_importances.argsort()[::-1]

    # Select the top features based on importance
    top_feature_indices = sorted_indices[:40]  # Selecting top 10 features, adjust as needed

    # Subset the original feature names using the selected indices
    selected_features = X_train.columns[top_feature_indices]
    selected_importances = feature_importances[top_feature_indices]

    # Create a DataFrame with selected features and importances
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Importance': selected_importances})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = os.path.join(feature_selection_dir, "RFCI", f"RFCI{i}.csv")
    coeff_df.to_csv(filename, index=False)


# FEATURE SELECTION #3 - Recursive Feature Elimination with Cross-Validation

# Iterate through the random seeds
print(">>> feature selection 3 - Recursive Feature Elimination with CV <<<")
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Create a Random Forest Classifier
    rf_Model = RandomForestClassifier(n_estimators=100,
                                      max_features='sqrt',
                                      max_depth=4,
                                      min_samples_split=2,
                                      min_samples_leaf=2,
                                      bootstrap=True,
                                      random_state=random_seed)

    # Perform Recursive Feature Elimination with Cross-Validation
    rfecv = RFECV(estimator=rf_Model, cv=5, step=1)  # Set the step value to 1 for forward selection
    X_train_selected = rfecv.fit_transform(X_train, y_train)

    # Get the selected features' indices
    selected_feature_indices = rfecv.get_support(indices=True)

    # Subset the original feature names using the selected indices
    selected_features = X_train.columns[selected_feature_indices]

    # Get feature importances from RFECV
    feature_importances = rfecv.estimator_.feature_importances_

    # Create a DataFrame with selected features and importances
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})

    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = os.path.join(feature_selection_dir, "RFECV", f"RFECV_{i}.csv")
    coeff_df.to_csv(filename, index=False)


# FEATURE SELECTION #4 - Dispersion Ratio

# Iterate through the random seeds
print(">>> Dispersion ratio <<<")
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    am = np.mean(X_train, axis=0)
    gm = np.power(np.prod(X_train, axis=0), 1 / X_train.shape[0])
    disp_ratio = am / gm

    plt.bar(np.arange(X_train.shape[1]), disp_ratio, color='teal')

    # Save the plot as an image file with a different name for each random seed
    filename = os.path.join(feature_selection_dir, "DispersionRatio", f"DispersionRatio_{i}.png")
    plt.savefig(filename)
    plt.clf()  # Clear the plot for the next iteration


# FEATURE SELECTION #5 - Permutation Importance (cross-validated)

# Iterate through the random seeds
print(">>> feature selection 5 - Permutation Importance <<<")
for i, random_seed in enumerate(random_seeds, start=1):
    np.random.seed(random_seed)  # Set the random seed
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    fold_importances = []
 
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
        # Create a Random Forest Classifier
        rf = RandomForestClassifier(bootstrap=True,
                                    max_depth=4,
                                    max_features='sqrt',
                                    min_samples_leaf=2,
                                    min_samples_split=2,
                                    n_estimators=100,
                                    random_state=random_seed)
 
        # Train the Random Forest model
        rf.fit(X_train, y_train)
 
        # Compute permutation importance on this fold's held-out rows
        # n_repeats=10 keeps runtime reasonable; fold-averaging (not repeats alone)
        # is what stabilises the estimate here. n_jobs=-1 parallelises across cores.
        result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=random_seed, n_jobs=-1)
        fold_importances.append(result.importances_mean)
 
    # Average importance across the 5 folds
    mean_importances = np.mean(fold_importances, axis=0)
 
    # Sort features by importance in descending order
    sorted_indices = mean_importances.argsort()[::-1]
 
    # Select the top features based on importance
    top_feature_indices = sorted_indices[:40]
 
    # Subset the original feature names using the selected indices
    selected_features = X.columns[top_feature_indices]
    selected_importances = mean_importances[top_feature_indices]
 
    # Create a DataFrame with selected features and importances
    coeff_df = pd.DataFrame({'Feature': selected_features, 'Importance': selected_importances})
 
    # Save the DataFrame as a CSV file with a different name for each random seed
    filename = os.path.join(feature_selection_dir, "PermImportance", f"PermImportance_{i}.csv")
    coeff_df.to_csv(filename, index=False)
