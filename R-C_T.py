import os
import re
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/counts_csvs"

OUTPUT_FILE = os.path.join(
    ROOT_DIR,
    "coding_CT_counts_by_CDS_mutation.csv"
)

# ============================================================
# HELPERS
# ============================================================

def parse_gene_cancer_from_filename(filename):
    """
    Expected examples:
    KRAS_Breast.csv
    TP53_Colorectal_Adenocarcinoma.csv

    Gene = first field before _
    Cancer = remaining filename
    """

    base = os.path.basename(filename)
    base = re.sub(r"\.csv$", "", base)

    parts = base.split("_")

    gene = parts[0].upper()
    cancer = "_".join(parts[1:])

    return gene, cancer


def is_ct_mutation(cds):
    """
    Keep only coding C>T substitutions.
    Examples kept:
        c.34C>T
        c.244C>T

    Examples excluded:
        c.4093-1C>T
        c.3712+355C>T
        c.*1830C>T
        c.34C>A
        c.38del
    """

    cds = str(cds)

    return bool(
        re.fullmatch(r"c\.\d+C>T", cds)
    )


# ============================================================
# MAIN
# ============================================================

all_results = []

csv_files = [
    os.path.join(ROOT_DIR, f)
    for f in os.listdir(ROOT_DIR)
    if f.endswith(".csv")
]

print(f"Found {len(csv_files)} CSV files")

for file in csv_files:

    gene, cancer = parse_gene_cancer_from_filename(file)

    print(f"\nProcessing {gene} / {cancer}")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"  Skipping: could not read file: {e}")
        continue

    required_cols = [
        "Position",
        "CDS Mutation",
        "AA Mutation",
        "Count",
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        print(f"  Skipping: missing columns {missing}")
        continue

    # --------------------------------------------------------
    # Keep only coding G>T substitutions
    # --------------------------------------------------------

    ct = df[
        df["CDS Mutation"]
        .astype(str)
        .apply(is_ct_mutation)
    ].copy()

    print(f"  C>T rows before Position filtering: {len(ct)}")

    # --------------------------------------------------------
    # Remove rows without coding/protein position
    # --------------------------------------------------------

    ct = ct.dropna(subset=["Position"])

    print(f"  C>T rows after removing NaN Position: {len(ct)}")

    if ct.empty:
        print("  No coding C>T mutations retained")
        continue

    ct["Position"] = ct["Position"].astype(int)
    ct["Count"] = pd.to_numeric(ct["Count"], errors="coerce")
    ct = ct.dropna(subset=["Count"])
    ct["Count"] = ct["Count"].astype(int)

    # --------------------------------------------------------
    # Aggregate by CDS mutation, not amino-acid position
    # --------------------------------------------------------

    summary = (
        ct.groupby(
            [
                "CDS Mutation",
                "AA Mutation",
                "Position",
            ],
            as_index=False
        )
        .agg(
            CT_Count=("Count", "sum")
        )
    )

    total_ct = summary["CT_Count"].sum()

    summary["Gene"] = gene
    summary["Cancer_Type"] = cancer
    summary["Gene_Total_CT_Count"] = total_ct

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    summary = summary[
        [
            "Gene",
            "Cancer_Type",
            "Position",
            "CDS Mutation",
            "AA Mutation",
            "CT_Count",
            "Gene_Total_CT_Count",
        ]
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    raw_sum = ct["Count"].sum()
    summary_sum = summary["CT_Count"].sum()

    if raw_sum != summary_sum:
        print("  WARNING: count mismatch")
        print(f"  Raw sum:     {raw_sum}")
        print(f"  Summary sum: {summary_sum}")
    else:
        print(f"  Counts OK: {summary_sum}")

    print(f"  Unique coding C>T CDS mutations: {len(summary)}")

    all_results.append(summary)

# ============================================================
# SAVE
# ============================================================

if not all_results:
    raise SystemExit("No results generated.")

combined = pd.concat(
    all_results,
    ignore_index=True
)

combined = combined.sort_values(
    [
        "Cancer_Type",
        "Gene",
        "CT_Count",
    ],
    ascending=[
        True,
        True,
        False,
    ]
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================================")
print("FINISHED")
print("===================================")
print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(combined):,}")
print(f"Gene/cancer pairs: {combined[['Gene', 'Cancer_Type']].drop_duplicates().shape[0]:,}")
