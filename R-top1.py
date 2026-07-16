import os
import pandas as pd

# pip install pandas scipy statsmodels

# ============================================================
# CONFIG
# ============================================================

root_dir = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers"

all_results = []

# ============================================================
# PROCESS EACH CANCER TYPE
# ============================================================

for cancer_type in os.listdir(root_dir):

    cancer_dir = os.path.join(root_dir, cancer_type)

    if not os.path.isdir(cancer_dir):
        continue

    mutation_file = next(
        (
            os.path.join(cancer_dir, f)
            for f in os.listdir(cancer_dir)
            if f.startswith("Gene") and f.endswith(".csv")
        ),
        None,
    )

    if mutation_file is None:
        print(f"Skipping {cancer_type}: no Gene*.csv found")
        continue

    print(f"\nProcessing {cancer_type}")

    try:
        df = pd.read_csv(mutation_file)
    except Exception as e:
        print(f"Failed to read {mutation_file}: {e}")
        continue

    # ============================================================
    # REQUIRED COLUMNS
    # ============================================================

    required_cols = [
        "Gene Name",
        "Transcript",
        "Sample ID",
        "AA Mutation",
        "CDS Mutation",
        "Genomic Co-ordinates",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"Missing columns in {cancer_type}: {missing}")
        continue

    df = df[required_cols].dropna()

    # ============================================================
    # KEEP CANONICAL TRANSCRIPTS ONLY
    #
    # Keep:
    #   TP53
    #   KRAS
    #
    # Remove:
    #   TP53_ENST00000269305
    #   KRAS_ENST00000311936
    # ============================================================

    df = df[
        ~df["Gene Name"]
        .astype(str)
        .str.contains("_", regex=False)
    ]

    if df.empty:
        print(f"No canonical transcript mutations in {cancer_type}")
        continue

    # ============================================================
    # COUNT RECURRENCE
    # ============================================================

    summary = (
        df.groupby(
            [
                "Gene Name",
                "Transcript",
                "CDS Mutation",
                "AA Mutation",
            ]
        )
        .agg(
            Mutation_Records=("Sample ID", "size"),
            Unique_Samples=("Sample ID", "nunique"),
            Example_Genomic_Coordinate=("Genomic Co-ordinates", "first"),
        )
        .reset_index()
    )

    summary["Cancer_Type"] = cancer_type

    summary = summary.sort_values(
        ["Unique_Samples", "Mutation_Records"],
        ascending=[False, False],
    )

    # ============================================================
    # SAVE PER-CANCER RESULTS
    # ============================================================

    output_file = os.path.join(
        cancer_dir,
        "most_frequent_mutation_sites.csv"
    )

    summary.to_csv(output_file, index=False)

    print(
        f"Saved {output_file} "
        f"({len(summary)} unique mutations)"
    )

    all_results.append(summary)

# ============================================================
# COMBINED RESULTS
# ============================================================

if not all_results:
    print("\nNo valid mutation data found.")
    raise SystemExit

combined = pd.concat(all_results, ignore_index=True)

# ============================================================
# REMOVE DUPLICATES ACROSS CANCERS
# ============================================================

combined = combined.sort_values(
    ["Unique_Samples", "Mutation_Records"],
    ascending=[False, False],
)

combined = combined.drop_duplicates(
    subset=[
        "Gene Name",
        "Transcript",
        "CDS Mutation",
        "AA Mutation",
    ],
    keep="first",
)

# ============================================================
# FINAL SORT
# ============================================================

combined = combined.sort_values(
    ["Unique_Samples", "Mutation_Records"],
    ascending=[False, False],
)

# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

combined_output = os.path.join(
    root_dir,
    "combined_most_frequent_mutation_sites.csv"
)

combined.to_csv(combined_output, index=False)

print(f"\nSaved combined results:")
print(combined_output)

print(f"\nTotal unique mutations: {len(combined):,}")

# ============================================================
# CREATE UNIQUE GENE / TRANSCRIPT TABLE
# FOR G-COUNT SCRIPT
# ============================================================

transcript_table = (
    combined[
        [
            "Gene Name",
            "Transcript",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Gene Name",
            "Transcript",
        ]
    )
)

transcript_output = os.path.join(
    root_dir,
    "genes_with_transcripts.csv"
)

transcript_table.to_csv(
    transcript_output,
    index=False,
)

print(
    f"\nSaved transcript list:\n{transcript_output}"
)

print(
    f"Unique gene/transcript pairs: "
    f"{len(transcript_table):,}"
)

# ============================================================
# TOP RECURRENT MUTATIONS
# ============================================================

print("\nTop 50 recurrent mutations:\n")

print(
    combined[
        [
            "Cancer_Type",
            "Gene Name",
            "Transcript",
            "CDS Mutation",
            "AA Mutation",
            "Unique_Samples",
            "Mutation_Records",
            "Example_Genomic_Coordinate",
        ]
    ]
    .head(50)
    .to_string(index=False)
)