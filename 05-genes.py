import pandas as pd

# Load data
df = pd.read_csv("../Cancers/combined_hotspot_pairs_deduplicated_raw.csv")

# Group by gene and collect cancer types
gene_cancer = (
    df.groupby("Gene")["Cancer_Type"]
      .apply(lambda x: "; ".join(sorted(set(x))))
      .reset_index()
)

# Number of cancer types per gene
gene_cancer["N_Cancer_Types"] = (
    gene_cancer["Cancer_Type"]
    .str.split("; ")
    .str.len()
)

# Sort so multi-cancer genes appear first
gene_cancer = gene_cancer.sort_values(
    ["N_Cancer_Types", "Gene"],
    ascending=[False, True]
)

# Save
gene_cancer.to_csv(
    "gene_cancer_mapping.csv",
    index=False
)

print(gene_cancer.head(20))
print(f"\nSaved {len(gene_cancer)} genes")