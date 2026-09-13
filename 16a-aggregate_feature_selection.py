import pandas as pd
import glob
import os

# ============================================================
# One consolidated aggregator for every model-specific feature-selection
# method: RFCI, RFECV, and PermImportance for RF/SVM/XGB, plus SVMCoef and
# XGBImportance. Each method's 20 per-seed CSVs get combined into one
# summary CSV, saved back into that method's own folder - replaces the
# separate aggregate_RFCI.py/aggregate_RFECV.py/aggregate_perm.py scripts
# (which only ever covered RF) with one script that covers all three
# models, since the aggregation logic itself doesn't depend on which model
# produced the numbers - only the folder and filename prefix do.
#
# Two aggregation "shapes" are needed, exactly matching what the original
# per-method scripts already did:
#  - Importance-style (column named "Importance"): frequency + mean
#    importance. Used by RFCI, RFECV (RF/XGB), PermImportance (all three
#    models), and XGBImportance - these come from feature_importances_ or
#    permutation importance, which don't carry a supervised "direction".
#  - Coefficient-style (column named "Coefficient"): frequency + mean
#    coefficient + % of runs it was positive, same as lasso_aggregate.py.
#    Used by SVMCoef and SVM's RFECV specifically, since a linear SVM's
#    .coef_ is signed (positive ~ associated with hotspot, negative ~
#    associated with non-hotspot) - unlike RF/XGB's RFECV, which reports
#    impurity/gain-based feature_importances_ instead, so SVM's RFECV is
#    the one case where the same method name needs the other shape.
# ============================================================

# (folder, filename prefix used by the generation script, value column)
methods = [
    ("../feature_selection/rf/RFCI",            "RF_RFCI",            "Importance"),
    ("../feature_selection/rf/RFECV",            "RF_RFECV",           "Importance"),
    ("../feature_selection/rf/PermImportance",   "RF_PermImportance",  "Importance"),
    ("../feature_selection/svm/SVMCoef",         "SVM_SVMCoef",        "Coefficient"),
    ("../feature_selection/svm/RFECV",           "SVM_RFECV",          "Coefficient"),
    ("../feature_selection/svm/PermImportance",  "SVM_PermImportance", "Importance"),
    ("../feature_selection/xgb/XGBImportance",   "XGB_XGBImportance",  "Importance"),
    ("../feature_selection/xgb/RFECV",           "XGB_RFECV",          "Importance"),
    ("../feature_selection/xgb/PermImportance",  "XGB_PermImportance", "Importance"),
]

for folder, prefix, value_col in methods:
    # "{prefix}_[0-9]*.csv" (digit right after the prefix) rather than
    # "{prefix}_*.csv" - the wider pattern would also match
    # {prefix}_summary.csv itself on a rerun and fold last run's summary
    # back into the next one's average
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_[0-9]*.csv")))
    n_seeds = len(files)

    if n_seeds == 0:
        print(f"{prefix}: no files found in {folder} - skipping")
        print()
        continue

    combined = pd.concat(pd.read_csv(f)[['Feature', value_col]] for f in files)

    if value_col == "Coefficient":
        summary = combined.groupby('Feature').agg(
            times_selected=(value_col, 'count'),
            mean_coefficient=(value_col, 'mean'),
            pct_positive=(value_col, lambda x: (x > 0).mean() * 100)
        ).reset_index()
        summary['pct_of_runs'] = summary['times_selected'] / n_seeds * 100
        # Sort by frequency first, then by |mean coefficient| for ties
        summary = summary.reindex(
            summary['mean_coefficient'].abs().sort_values(ascending=False).index
        ).sort_values('times_selected', ascending=False, kind='stable').reset_index(drop=True)
    else:
        summary = combined.groupby('Feature').agg(
            times_selected=(value_col, 'count'),
            mean_importance=(value_col, 'mean')
        ).reset_index()
        summary['pct_of_runs'] = summary['times_selected'] / n_seeds * 100
        summary = summary.sort_values(['times_selected', 'mean_importance'], ascending=[False, False]).reset_index(drop=True)

    summary.to_csv(os.path.join(folder, f"{prefix}_summary.csv"), index=False)

    print(f"{prefix}: {len(summary)} distinct features selected at least once across {n_seeds} runs")
    print("Top 15 by how often they were selected:")
    print(summary.head(15).to_string(index=False))
    print()