import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ============================================================
# Same data loading as 15-k-means.py - kept identical so this
# diagnostic is checking the real pipeline, not a simplified stand-in.
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
merged_df['gene'] = merged_df['base_sequence'].str.split('_', n=1).str[0]

total_data = merged_df.drop(columns=['sequence', 'base_sequence', 'gene'])
X = total_data.drop(['Outcome'], axis=1)
y = total_data['Outcome'].to_numpy()

random_seeds = [685641, 249077, 18533, 426353, 622463, 103321, 396546, 427173,
                 286636, 335318, 785535, 231325, 405031, 390995, 37176, 755657,
                 101777, 517844, 969889, 159625]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=12, random_state=159625)
X_pca = pca.fit_transform(X_scaled)


def canonicalize(labels):
    """Cluster numbers 0/1 are arbitrary - fix a reference point (row 0's
    cluster is always called '0') so identical partitions compare equal
    regardless of which number k-means happened to assign each cluster."""
    labels = np.asarray(labels)
    return tuple(labels) if labels[0] == 0 else tuple(1 - labels)


# ============================================================
# CHECK 1 - Are the 20 seeds genuinely producing the identical partition,
# or is something upstream (e.g. std computed on the wrong array) making
# real variation look like zero?
# ============================================================
print("=" * 70)
print("CHECK 1: per-seed partition, inertia, and iterations to converge")
print("=" * 70)

partitions, inertias = [], []
for seed in random_seeds:
    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    labels = km.fit_predict(X_pca)
    partitions.append(canonicalize(labels))
    inertias.append(km.inertia_)
    print(f"seed={seed:>7}  inertia={km.inertia_:.4f}  n_iter={km.n_iter_}  "
          f"labels={list(labels)}")

n_distinct = len(set(partitions))
print(f"\nDistinct partitions across 20 seeds: {n_distinct} "
      f"({'all identical' if n_distinct == 1 else 'genuine variation found'})")
print(f"Distinct inertia values: {len(set(np.round(inertias, 6)))}")

# ============================================================
# CHECK 2 - Does n_init=10 (10 internal restarts per seed, best kept) mask
# seed-to-seed variability that would show up with a single restart?
# If n_init=1 ALSO gives zero variance, the data itself has one dominant
# optimum - that's a real result, not an n_init artefact.
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: same seeds, but n_init=1 (no internal best-of-10 restart)")
print("=" * 70)

partitions_ninit1 = []
for seed in random_seeds:
    km1 = KMeans(n_clusters=2, n_init=1, random_state=seed)
    labels1 = km1.fit_predict(X_pca)
    partitions_ninit1.append(canonicalize(labels1))

n_distinct_1 = len(set(partitions_ninit1))
print(f"Distinct partitions with n_init=1: {n_distinct_1} "
      f"({'still identical - real global optimum' if n_distinct_1 == 1 else 'variation appears once n_init drops - n_init=10 was masking it'})")

# ============================================================
# CHECK 3 - What is the clustering actually splitting on? Compare the
# partition against hotspot label, gene, and sequence identity. If every
# replicate of a sequence lands in the same cluster, the split may be
# tracking sequence/gene identity rather than hotspot status - both could
# get a similar-looking accuracy number by coincidence.
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: what does the (canonical) partition actually track?")
print("=" * 70)

ref_labels = np.array(partitions[0])  # identical across seeds per Check 1
check_df = pd.DataFrame({
    'sequence': merged_df['sequence'],
    'base_sequence': merged_df['base_sequence'],
    'gene': merged_df['gene'],
    'hotspot': y,
    'cluster': ref_labels,
})

print("\nDo all 3 replicates of each sequence fall in the same cluster?")
per_seq_cluster_nunique = check_df.groupby('base_sequence')['cluster'].nunique()
print(per_seq_cluster_nunique.to_string())
print(f"-> {(per_seq_cluster_nunique == 1).sum()} / {len(per_seq_cluster_nunique)} "
      f"sequences have all replicates in the same cluster")

print("\nCluster composition by hotspot label (mean hotspot rate per cluster):")
print(check_df.groupby('cluster')['hotspot'].mean().to_string())

print("\nCluster composition by gene (row counts):")
print(pd.crosstab(check_df['cluster'], check_df['gene']).to_string())

print("\nFull cross-tab: cluster vs base_sequence (row counts, max 3 per cell):")
print(pd.crosstab(check_df['cluster'], check_df['base_sequence']).to_string())

print("\n" + "=" * 70)
print("READ THIS: if Check 1 shows 1 distinct partition, Check 2 also shows")
print("1, and Check 3 shows replicates always grouping together AND cluster")
print("composition split cleanly by gene (not hotspot) - the clustering is")
print("almost certainly tracking gene/sequence identity, not hotspot status,")
print("and the ~72% aligned accuracy is likely coincidental overlap between")
print("gene identity and your hotspot labelling, not a real hotspot signal.")
print("=" * 70)