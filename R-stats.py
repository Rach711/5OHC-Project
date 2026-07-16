import os
import re
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/counts_csvs"

CT_COUNTS_FILE = os.path.join(
    ROOT_DIR,
    "coding_CT_counts_by_CDS_mutation.csv"
)

C_COUNTS_FILE = (
    "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers/c_count_results/c_counts_per_gene_transcript.csv"
)

OUTPUT_FILE = os.path.join(
    ROOT_DIR,
    "coding_CT_hotspot_analysis_by_CDS_mutation.csv"
)

PAIRS_OUTPUT_FILE = os.path.join(
    ROOT_DIR,
    "coding_CT_hotspot_nonhotspot_pairs_by_CDS_mutation.csv"
)

NEAR_HOTSPOT_WINDOW = 5
NEAR_HOTSPOT_MIN_MUTATIONS = 1

# ============================================================
# FUNCTIONS
# ============================================================

def extract_cds_position(cds_mutation):
    """
    Extract CDS nucleotide coordinate from strings such as:
        c.844C>T
        c.843_844del
        c.375+1G>T
        c.672-2A>G

    Returns the first CDS base coordinate.
    """

    if pd.isna(cds_mutation):
        return None

    cds_mutation = str(cds_mutation)

    match = re.search(r"c\.(\d+)", cds_mutation)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# LOAD DATA
# ============================================================

counts = pd.read_csv(CT_COUNTS_FILE)
c_df = pd.read_csv(C_COUNTS_FILE)

c_df["GeneName"] = (
    c_df["GeneName"]
    .astype(str)
    .str.upper()
)

gene_to_ccount = dict(
    zip(
        c_df["GeneName"],
        c_df["C_count"]
    )
)

# ============================================================
# BINOMIAL TEST
# ============================================================

all_results = []

for (cancer, gene), sub in counts.groupby(
    ["Cancer_Type", "Gene"]
):

    gene = str(gene).upper()

    c_count = gene_to_ccount.get(gene)

    if c_count is None:
        print(f"Skipping {gene} / {cancer}: no C_count")
        continue

    c_count = float(c_count)

    if c_count <= 0:
        print(f"Skipping {gene} / {cancer}: invalid C_count")
        continue

    gene_total = int(
        sub["Gene_Total_CT_Count"].iloc[0]
    )

    if gene_total <= 0:
        continue

    expected_p = 1.0 / c_count

    for _, row in sub.iterrows():

        observed = int(row["CT_Count"])

        pval = binomtest(
            observed,
            gene_total,
            expected_p,
            alternative="greater"
        ).pvalue

        cds_mutation = row["CDS Mutation"]
        cds_position = extract_cds_position(cds_mutation)

        if cds_position is None:
            continue

        all_results.append(
            {
                "Cancer_Type": cancer,
                "Gene": gene,
                "Position": int(row["Position"]),
                "CDS_Position": int(cds_position),
                "CDS_Mutation": cds_mutation,
                "AA_Mutation": row["AA Mutation"],
                "CT_Count": observed,
                "Gene_Total_CT_Count": gene_total,
                "C_Count": c_count,
                "Expected_Probability": expected_p,
                "Raw_P_Value": pval,
            }
        )

# ============================================================
# COMBINE
# ============================================================

if not all_results:
    raise SystemExit("No hotspot results generated.")

result_df = pd.DataFrame(all_results)

# ============================================================
# BONFERRONI CORRECTION
# per cancer type
# ============================================================

result_df["Bonferroni_P"] = pd.NA
result_df["Is_Hotspot"] = False

for cancer, idx in result_df.groupby("Cancer_Type").groups.items():

    pvals = result_df.loc[idx, "Raw_P_Value"].astype(float)

    reject, p_corr, _, _ = multipletests(
        pvals,
        method="bonferroni"
    )

    result_df.loc[idx, "Bonferroni_P"] = p_corr
    result_df.loc[idx, "Is_Hotspot"] = reject

result_df["Bonferroni_P"] = result_df["Bonferroni_P"].astype(float)

# ============================================================
# FIND NEARBY NON-HOTSPOTS
# same gene + same cancer + within +/- 5 CDS nucleotides
# ============================================================

result_df["Near_Hotspot"] = False
result_df["Near_Hotspot_Of_CDS"] = ""
result_df["Near_Hotspot_Of_CDS_Position"] = ""

pairs_rows = []

for (cancer, gene), sub in result_df.groupby(
    ["Cancer_Type", "Gene"]
):

    hotspot_rows = sub[sub["Is_Hotspot"]]

    if hotspot_rows.empty:
        continue

    for idx, row in sub.iterrows():

        if row["Is_Hotspot"]:
            continue

        # Include mutations with at least NEAR_HOTSPOT_MIN_MUTATIONS
        if row["CT_Count"] < NEAR_HOTSPOT_MIN_MUTATIONS:
            continue

        nearby_hotspots = []

        for _, hs in hotspot_rows.iterrows():

            distance = abs(
                int(row["CDS_Position"]) -
                int(hs["CDS_Position"])
            )

            if distance <= NEAR_HOTSPOT_WINDOW:
                nearby_hotspots.append(
                    (hs, distance)
                )

        if not nearby_hotspots:
            continue

        result_df.at[idx, "Near_Hotspot"] = True

        result_df.at[idx, "Near_Hotspot_Of_CDS"] = ";".join(
            str(hs["CDS_Mutation"])
            for hs, _ in nearby_hotspots
        )

        result_df.at[idx, "Near_Hotspot_Of_CDS_Position"] = ";".join(
            str(int(hs["CDS_Position"]))
            for hs, _ in nearby_hotspots
        )

        for hs, distance in nearby_hotspots:

            pairs_rows.append(
                {
                    "Cancer_Type": cancer,
                    "Gene": gene,

                    "Hotspot_CDS_Position": int(hs["CDS_Position"]),
                    "Hotspot_CDS_Mutation": hs["CDS_Mutation"],
                    "Hotspot_AA_Mutation": hs["AA_Mutation"],
                    "Hotspot_CT_Count": int(hs["CT_Count"]),
                    "Hotspot_Bonferroni_P": hs["Bonferroni_P"],

                    "NonHotspot_CDS_Position": int(row["CDS_Position"]),
                    "NonHotspot_CDS_Mutation": row["CDS_Mutation"],
                    "NonHotspot_AA_Mutation": row["AA_Mutation"],
                    "NonHotspot_CT_Count": int(row["CT_Count"]),
                    "NonHotspot_Bonferroni_P": row["Bonferroni_P"],

                    "Distance_CDS_bp": int(distance),
                }
            )

pairs_df = pd.DataFrame(pairs_rows)

# ============================================================
# SORT
# ============================================================

result_df = result_df.sort_values(
    [
        "Is_Hotspot",
        "Bonferroni_P",
        "CT_Count",
    ],
    ascending=[
        False,
        True,
        False,
    ]
)

if not pairs_df.empty:
    pairs_df = pairs_df.sort_values(
        [
            "Cancer_Type",
            "Gene",
            "Hotspot_CDS_Position",
            "Distance_CDS_bp",
            "NonHotspot_CDS_Position",
        ]
    )

# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

pairs_df.to_csv(
    PAIRS_OUTPUT_FILE,
    index=False
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n===================================")
print("FINISHED")
print("===================================")
print(f"Total CDS mutations tested: {len(result_df):,}")
print(f"Significant hotspots: {result_df['Is_Hotspot'].sum():,}")
print(f"Nearby non-hotspot CDS mutations: {result_df['Near_Hotspot'].sum():,}")
print(f"Hotspot/non-hotspot pair relationships: {len(pairs_df):,}")
print(f"\nSaved hotspot analysis:\n{OUTPUT_FILE}")
print(f"\nSaved pairs file:\n{PAIRS_OUTPUT_FILE}")

print("\nTop hotspots:")
print(
    result_df[
        result_df["Is_Hotspot"]
    ][
        [
            "Cancer_Type",
            "Gene",
            "Position",
            "CDS_Position",
            "CDS_Mutation",
            "AA_Mutation",
            "CT_Count",
            "Gene_Total_CT_Count",
            "Bonferroni_P",
        ]
    ]
    .head(30)
    .to_string(index=False)
)