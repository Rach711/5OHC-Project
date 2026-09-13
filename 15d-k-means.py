import pandas as pd
import numpy as np
import os
from itertools import permutations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

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
y = total_data['Outcome'].to_numpy()

# Same 20 seeds as the other scripts - here they test K-means' sensitivity to
# random centroid initialisation, since (unlike a train/test split) K-means
# is unsupervised and has no natural train/test framing of its own
random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173, 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657, 101777, 517844, 969889, 159625]

# Dispersion Ratio is generated separately by SharedFeatureSelection.py (run
# that once - it doesn't depend on the model, so there's no reason to
# duplicate it here). PCA loadings and silhouette-based permutation
# importance are specific to this clustering pipeline and stay in this script.
kmeans_feature_selection_dir = "../feature_selection/kmeans"
for subfolder in ["PCALoadings", "SilhouettePermImportance"]:
    os.makedirs(os.path.join(kmeans_feature_selection_dir, subfolder), exist_ok=True)


# Scaling is essential here - K-means assigns points by Euclidean distance,
# so without it, whichever features happen to have the largest raw numeric
# range would dominate that distance regardless of actual relevance. Fit on
# the full dataset (not a train/test split) since clustering has no
# prediction-on-unseen-data framing the way classification does.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 12 components - chosen because it captures 90% of the variance
# across all 360 original features (checked directly against this data, not
# an arbitrary round number). With 36 points spread across 360 dimensions,
# distance-based clustering suffers badly from the curse of dimensionality;
# reducing dimensionality first is the standard fix, and also makes the
# first 2 components usable for an actual 2D visualisation of the clusters.
pca = PCA(n_components=12, random_state=159625)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_pca.shape[1]} components capture {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")


def align_labels(true_labels, cluster_labels):
    """K-means' cluster numbers (0/1) are arbitrary - nothing ties 'cluster 0'
    to 'hotspot' specifically. This finds whichever 0/1 -> hotspot/non-hotspot
    mapping gives the best agreement with the known labels, so accuracy isn't
    unfairly penalised just because the clusters came out numbered the
    other way around."""
    best_acc = 0
    for perm in permutations(set(cluster_labels)):
        mapping = dict(zip(sorted(set(cluster_labels)), perm))
        mapped = np.array([mapping[c] for c in cluster_labels])
        acc = (mapped == true_labels).mean()
        best_acc = max(best_acc, acc)
    return best_acc


# K-means (k=2, matching the known hotspot/non-hotspot split) - run across
# all 20 seeds to check how much the result depends on random initialisation,
# then averaged. No GridSearchCV here: k is fixed at 2 by the known number of
# classes rather than searched, and n_init already handles picking the best
# of several initialisations internally within each individual fit.
aligned_accuracies, aris, nmis, silhouettes = [], [], [], []

for random_seed in random_seeds:
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(X_pca)

    aligned_accuracies.append(align_labels(y, cluster_labels))
    # Adjusted Rand Index and Normalized Mutual Information don't need the
    # alignment step above - they're mathematically invariant to how the
    # cluster numbers happen to be labelled, and are the standard metrics
    # for comparing a clustering against known ground truth.
    aris.append(adjusted_rand_score(y, cluster_labels))
    nmis.append(normalized_mutual_info_score(y, cluster_labels))
    # Silhouette score uses no label information at all - it measures how
    # well-separated the clusters are geometrically, on their own terms
    silhouettes.append(silhouette_score(X_pca, cluster_labels))

print(f"Aligned accuracy vs known labels: {np.mean(aligned_accuracies):.3f} +/- {np.std(aligned_accuracies):.3f}")
print(f"Adjusted Rand Index: {np.mean(aris):.3f} +/- {np.std(aris):.3f}")
print(f"Normalized Mutual Information: {np.mean(nmis):.3f} +/- {np.std(nmis):.3f}")
print(f"Silhouette score: {np.mean(silhouettes):.3f} +/- {np.std(silhouettes):.3f}")

# ============================================================
# SAVE CLUSTERING METRICS - previously print-only, so every rerun
# silently overwrote the last one. Written alongside the feature
# selection output rather than into it, since this is a summary
# table, not a per-seed file like PCALoadings/SilhouettePermImportance.
# ============================================================
performance_dir = "../performance"
os.makedirs(performance_dir, exist_ok=True)

with open(os.path.join(performance_dir, "KMeans_pca_summary.txt"), "w") as f:
    f.write(f"PCA components: {X_pca.shape[1]}\n")
    f.write(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%\n")

kmeans_performance = pd.DataFrame({
    "Metric": ["Aligned_Accuracy", "Adjusted_Rand_Index", "Normalized_Mutual_Info", "Silhouette_Score"],
    "Mean": [np.mean(aligned_accuracies), np.mean(aris), np.mean(nmis), np.mean(silhouettes)],
    "Std": [np.std(aligned_accuracies), np.std(aris), np.std(nmis), np.std(silhouettes)],
})
kmeans_performance.to_csv(os.path.join(performance_dir, "KMeans_performance.csv"), index=False)
print(f"Saved clustering metrics to {performance_dir}/KMeans_performance.csv")


# ============================================================
# FEATURE SELECTION #2 - PCA Loadings
# Fully unsupervised: shows which of the 360 original features contribute
# most to the 12 components K-means actually clusters on. Each feature's
# loading on every component is weighted by that component's explained
# variance ratio, then summed - so a feature that loads heavily onto PC1
# (which explains more variance) counts for more than one that only loads
# onto a minor, low-variance component. Computed once - PCA is deterministic
# given the same data, nothing here depends on random seed.
# ============================================================

loadings = pca.components_  # shape (12 components, 360 features)
variance_weights = pca.explained_variance_ratio_  # shape (12,)

weighted_loading_scores = np.sum(np.abs(loadings) * variance_weights[:, np.newaxis], axis=0)

loadings_df = pd.DataFrame({'Feature': X.columns, 'WeightedLoading': weighted_loading_scores})
loadings_df = loadings_df.sort_values('WeightedLoading', ascending=False).reset_index(drop=True)
loadings_df.to_csv(os.path.join(kmeans_feature_selection_dir, "PCALoadings", "KMeans_PCALoadings.csv"), index=False)

print("PCA Loadings done")
print("Top 10 features by weighted PCA loading:")
print(loadings_df.head(10).to_string(index=False))


# ============================================================
# FEATURE SELECTION #3 - Silhouette-based Permutation Importance
# Fully unsupervised analog of Permutation Importance: shuffle one original
# feature at a time, run it through the SAME fixed scaler -> PCA -> K-means
# pipeline already fit above, and measure how much the silhouette score
# (cluster separation quality) drops - never touches the known labels.
# Computed once rather than per seed, since the clustering itself was
# identical across all 20 seeds above (zero variance) - repeating this 20
# times would just recompute the same answer 20 times.
# ============================================================

n_repeats = 10
rng = np.random.RandomState(159625)

baseline_score = silhouette_score(X_pca, cluster_labels)  # cluster_labels from the last seed's fit above

importances = np.zeros(X.shape[1])
for feat_idx in range(X.shape[1]):
    drops = []
    for rep in range(n_repeats):
        X_permuted = X.copy()
        X_permuted.iloc[:, feat_idx] = rng.permutation(X_permuted.iloc[:, feat_idx].to_numpy())

        X_permuted_scaled = scaler.transform(X_permuted)
        X_permuted_pca = pca.transform(X_permuted_scaled)
        permuted_labels = kmeans.predict(X_permuted_pca)

        permuted_score = silhouette_score(X_permuted_pca, permuted_labels)
        drops.append(baseline_score - permuted_score)
    importances[feat_idx] = np.mean(drops)

silhouette_perm_df = pd.DataFrame({'Feature': X.columns, 'SilhouetteDrop': importances})
silhouette_perm_df = silhouette_perm_df.sort_values('SilhouetteDrop', ascending=False).reset_index(drop=True)
silhouette_perm_df.to_csv(os.path.join(kmeans_feature_selection_dir, "SilhouettePermImportance", "KMeans_SilhouettePermImportance.csv"), index=False)

print("Silhouette-based Permutation Importance done")
print("Top 10 features by silhouette score drop when shuffled:")
print(silhouette_perm_df.head(10).to_string(index=False))