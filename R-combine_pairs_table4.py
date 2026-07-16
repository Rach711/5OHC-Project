import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

root_dir = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers"

output_file = os.path.join(
    root_dir,
    "combined_gene_hotspots_deduplicated.csv"
)

pairs_output_file = os.path.join(
    root_dir,
    "combined_hotspot_pairs_deduplicated.csv"
)

hotspot_filename = "gene_hotspot_analysis.csv"
pairs_filename = "hotspot_pairs.csv"

# ============================================================
# LOAD ALL CANCER RESULTS (main hotspot/non-hotspot table)
# ============================================================

all_results = []
all_pairs = []

for cancer_type in os.listdir(root_dir):

    cancer_dir = os.path.join(
        root_dir,
        cancer_type
    )

    if not os.path.isdir(cancer_dir):
        continue

    hotspot_file = os.path.join(
        cancer_dir,
        hotspot_filename
    )

    if os.path.exists(hotspot_file):

        print(f"Loading {cancer_type}")

        try:

            df = pd.read_csv(hotspot_file)

            df["Cancer_Type"] = cancer_type

            all_results.append(df)

        except Exception as e:

            print(
                f"Failed loading "
                f"{hotspot_file}\n{e}"
            )

    pairs_file = os.path.join(
        cancer_dir,
        pairs_filename
    )

    if os.path.exists(pairs_file):

        try:

            pdf = pd.read_csv(pairs_file)

            if not pdf.empty:
                all_pairs.append(pdf)

        except Exception as e:

            print(
                f"Failed loading "
                f"{pairs_file}\n{e}"
            )

# ============================================================
# COMBINE MAIN RESULTS
# ============================================================

if len(all_results) == 0:

    print("No hotspot files found.")
    raise SystemExit

combined = pd.concat(
    all_results,
    ignore_index=True
)

print(
    f"\nCombined rows before deduplication: "
    f"{len(combined):,}"
)

# ============================================================
# KEEP MOST SIGNIFICANT DUPLICATE
# (Bonferroni_P can legitimately underflow to exactly 0.0 for very
# strong hits, so Raw_P_Value is used as a tiebreaker before falling
# back to Observed_Mutations)
#
# Dedup is keyed on Gene + Site + CDS_Mutation + AA_Mutation across
# ALL cancer types: if the same exact mutation shows up as a result
# in multiple cancers, keep only the single most significant one.
# ============================================================

has_raw_pval = "Raw_P_Value" in combined.columns

sort_cols = ["Bonferroni_P"]
sort_order = [True]

if has_raw_pval:
    sort_cols.append("Raw_P_Value")
    sort_order.append(True)

sort_cols.append("Observed_Mutations")
sort_order.append(False)

combined = combined.sort_values(
    sort_cols,
    ascending=sort_order
)

combined = combined.drop_duplicates(
    subset=[
        "Gene",
        "Site",
        "CDS_Mutation",
        "AA_Mutation",
    ],
    keep="first"
)

# ============================================================
# DROP ROWS THAT ARE NEITHER A HOTSPOT NOR A NEAR-HOTSPOT
# NON-HOTSPOT (i.e. keep only Is_Hotspot=True or Near_Hotspot=True)
# ============================================================

rows_before_filter = len(combined)

if "Near_Hotspot" in combined.columns:

    combined["Near_Hotspot"] = combined["Near_Hotspot"].fillna(False)

    combined = combined[
        combined["Is_Hotspot"] | combined["Near_Hotspot"]
    ]

else:

    combined = combined[combined["Is_Hotspot"]]

print(
    f"\nDropped {rows_before_filter - len(combined):,} rows that were "
    f"neither a hotspot nor near a hotspot"
)

# ============================================================
# FINAL SORT
# ============================================================

combined = combined.sort_values(
    [
        "Is_Hotspot",
        "Bonferroni_P",
        "Observed_Mutations",
    ],
    ascending=[
        False,
        True,
        False,
    ]
)

# ============================================================
# SAVE MAIN RESULTS
# ============================================================

combined.to_csv(
    output_file,
    index=False
)

print(
    f"\nRows kept (hotspots + near-hotspot sites): "
    f"{len(combined):,}"
)

print(
    f"\nSignificant hotspots: "
    f"{combined['Is_Hotspot'].sum():,}"
)

if "Near_Hotspot" in combined.columns:
    print(
        f"\nNon-hotspot sites near a hotspot (kept after dedup): "
        f"{combined['Near_Hotspot'].sum():,}"
    )

print(
    f"\nSaved:\n{output_file}"
)

# ============================================================
# COMBINE AND SAVE PAIRS TABLE
# (one row per hotspot/non-hotspot relationship; this is the
# explicit, unambiguous record of which non-hotspot sites sit
# near which hotspot sites, including cases where one non-hotspot
# is near multiple hotspots or vice versa)
#
# Dedup here is keyed on the exact relationship -- Cancer_Type +
# Gene + Hotspot_Site + NonHotspot_Site -- since the same hotspot
# site in different cancers is a different relationship, and the
# same non-hotspot paired with two different hotspots is also two
# distinct, both-valid relationships that should NOT be collapsed.
# ============================================================

if all_pairs:

    combined_pairs = pd.concat(
        all_pairs,
        ignore_index=True
    )

    combined_pairs = combined_pairs.drop_duplicates(
        subset=[
            "Cancer_Type",
            "Gene",
            "Hotspot_Site",
            "NonHotspot_Site",
        ],
        keep="first"
    )

    combined_pairs = combined_pairs.sort_values(
        [
            "Hotspot_Bonferroni_P",
            "Distance",
        ],
        ascending=[
            True,
            True,
        ]
    )

else:

    combined_pairs = pd.DataFrame()

combined_pairs.to_csv(
    pairs_output_file,
    index=False
)

print(
    f"\nHotspot/non-hotspot pair relationships kept: "
    f"{len(combined_pairs):,}"
)

print(
    f"\nSaved:\n{pairs_output_file}"
)

