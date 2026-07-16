import os
import re
import sys
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIGURATION
# ============================================================

root_dir = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers"

c_counts_path = (
    "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers/c_count_results/c_counts_per_gene_transcript.csv"
)

combined_output = os.path.join(
    root_dir,
    "combined_gene_hotspot_analysis.csv"
)

combined_pairs_output = os.path.join(
    root_dir,
    "combined_hotspot_pairs.csv"
)

# How many CDS bases away a non-hotspot can be from a hotspot
# to still count as "near" it
NEAR_HOTSPOT_WINDOW = 5

# Minimum Observed_Mutations a non-hotspot site needs to qualify
# as "near" a hotspot (excludes singleton mutations)
NEAR_HOTSPOT_MIN_MUTATIONS = 1

# ============================================================
# LOAD G COUNTS
# ============================================================

try:
    c_df = pd.read_csv(c_counts_path)

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

    print(f"Loaded G-counts for {len(gene_to_ccount):,} genes")

except Exception as e:
    print(f"Failed to load C-count file:\n{e}")
    sys.exit(1)

# ============================================================
# REPLACED PRIORITY ORDER WITH NORMAL
# ============================================================

cancer_order = sorted([
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])

# ============================================================
# STORE RESULTS
# ============================================================

all_results = []
all_pairs = []

# ============================================================
# PROCESS EACH CANCER TYPE
# ============================================================

for cancer_type in cancer_order:

    cancer_dir = os.path.join(root_dir, cancer_type)

    if not os.path.isdir(cancer_dir):
        print(f"Skipping {cancer_type}: folder not found")
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
        print(f"Failed reading {mutation_file}\n{e}")
        continue

    required_cols = [
        "Gene Name",
        "Transcript",
        "CDS Mutation",
        "AA Mutation",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"Missing columns in {cancer_type}: {missing}")
        continue

    df = df[required_cols].dropna()

    if df.empty:
        print(f"No usable rows in {cancer_type}")
        continue

    # ========================================================
    # EXTRACT SITE
    # ========================================================

    processed = []

    for _, row in df.iterrows():

        gene = str(row["Gene Name"]).upper()
        transcript = str(row["Transcript"])
        cds = str(row["CDS Mutation"])
        aa = str(row["AA Mutation"])

        match = re.match(r"c\.(\d+)", cds)

        if not match:
            continue

        site = int(match.group(1))

        processed.append(
            {
                "Gene": gene,
                "Transcript": transcript,
                "Site": site,
                "CDS_Mutation": cds,
                "AA_Mutation": aa,
            }
        )

    if not processed:
        print(f"No valid mutations in {cancer_type}")
        continue

    processed_df = pd.DataFrame(processed)

    # ========================================================
    # TOTAL MUTATIONS PER GENE
    # ========================================================

    gene_totals = (
        processed_df
        .groupby("Gene")
        .size()
        .to_dict()
    )

    # ========================================================
    # COUNT MUTATIONS PER SITE
    # ========================================================

    site_counts = (
        processed_df
        .groupby(["Gene", "Site"])
        .size()
        .reset_index(name="Observed_Mutations")
    )

    # ========================================================
    # REPRESENTATIVE ROW PER GENE + SITE
    # ========================================================

    representatives = (
        processed_df
        .drop_duplicates(subset=["Gene", "Site"])
        .set_index(["Gene", "Site"])
    )

    results = []

    # ========================================================
    # HOTSPOT TESTING
    # (unchanged: binomial test against 1/C_count, then Bonferroni
    # correction across all sites tested for this cancer)
    # ========================================================

    for _, row in site_counts.iterrows():

        gene = row["Gene"]
        site = row["Site"]

        c_count = gene_to_ccount.get(gene)

        if c_count is None:
            continue

        try:
            c_count = float(c_count)
        except Exception:
            continue

        if c_count <= 0:
            continue

        gene_mut_total = gene_totals.get(gene)

        if gene_mut_total is None or gene_mut_total <= 0:
            continue

        observed = int(row["Observed_Mutations"])

        p = 1.0 / c_count

        pval = binomtest(
            observed,
            gene_mut_total,
            p,
            alternative="greater"
        ).pvalue

        representative = representatives.loc[(gene, site)]

        results.append(
            {
                "Cancer_Type": cancer_type,
                "Gene": gene,
                "Transcript": representative["Transcript"],
                "Site": site,
                "Observed_Mutations": observed,
                "Gene_Total_Mutations": gene_mut_total,
                "C_Count": c_count,
                "Expected_Probability": p,
                "CDS_Mutation": representative["CDS_Mutation"],
                "AA_Mutation": representative["AA_Mutation"],
                "Raw_P_Value": pval,
            }
        )

    if len(results) == 0:
        print(f"No sites tested in {cancer_type}")
        continue

    result_df = pd.DataFrame(results)

    # ========================================================
    # BONFERRONI CORRECTION (unchanged)
    # ========================================================

    reject, p_corr, _, _ = multipletests(
        result_df["Raw_P_Value"],
        method="bonferroni"
    )

    result_df["Bonferroni_P"] = p_corr
    result_df["Is_Hotspot"] = reject

    # ========================================================
    # FIND NON-HOTSPOT SITES NEAR A HOTSPOT
    # (same Gene + same Cancer_Type only, no generalisation)
    #
    # IMPORTANT: a single non-hotspot can legitimately sit within
    # the window of MULTIPLE hotspots, and a single hotspot can
    # have MULTIPLE nearby non-hotspots. This is a many-to-many
    # relationship, so instead of squashing it into one "Pair_Group"
    # label per row (which can only point at one partner), every
    # relationship is recorded as its own row in a separate pairs
    # table (pairs_rows below). The main result_df still gets simple
    # per-row flags (Near_Hotspot, Near_Hotspot_Of_Site) for
    # filtering, but the pairs table is the source of truth for
    # "which hotspot goes with which non-hotspot".
    # ========================================================

    result_df["Near_Hotspot"] = False
    result_df["Near_Hotspot_Of_Site"] = ""

    hotspot_rows = result_df[result_df["Is_Hotspot"]]

    # Gene -> list of hotspot Site values, for this cancer
    gene_to_hotspot_sites = (
        hotspot_rows
        .groupby("Gene")["Site"]
        .apply(list)
        .to_dict()
    )

    # Gene -> Site -> full hotspot row, for quick lookup when building pairs
    hotspot_lookup = {
        (r["Gene"], r["Site"]): r
        for _, r in hotspot_rows.iterrows()
    }

    pairs_rows = []

    for idx, row in result_df.iterrows():

        if row["Is_Hotspot"]:
            # Hotspots themselves are not "near hotspot" non-hotspots
            continue

        if row["Observed_Mutations"] <= NEAR_HOTSPOT_MIN_MUTATIONS:
            # Non-hotspot must have more than this many observed
            # mutations to qualify (excludes singleton mutations)
            continue

        gene = row["Gene"]
        site = row["Site"]

        hotspot_sites_for_gene = gene_to_hotspot_sites.get(gene)

        if not hotspot_sites_for_gene:
            continue

        nearby_hotspot_sites = sorted(
            hs for hs in hotspot_sites_for_gene
            if hs != site and abs(site - hs) <= NEAR_HOTSPOT_WINDOW
        )

        if not nearby_hotspot_sites:
            continue

        result_df.at[idx, "Near_Hotspot"] = True
        result_df.at[idx, "Near_Hotspot_Of_Site"] = ";".join(
            str(s) for s in nearby_hotspot_sites
        )

        for hs_site in nearby_hotspot_sites:

            hotspot_row = hotspot_lookup[(gene, hs_site)]

            pairs_rows.append(
                {
                    "Cancer_Type": cancer_type,
                    "Gene": gene,
                    "Hotspot_Site": hs_site,
                    "Hotspot_CDS_Mutation": hotspot_row["CDS_Mutation"],
                    "Hotspot_AA_Mutation": hotspot_row["AA_Mutation"],
                    "Hotspot_Observed_Mutations": hotspot_row["Observed_Mutations"],
                    "Hotspot_Bonferroni_P": hotspot_row["Bonferroni_P"],
                    "NonHotspot_Site": site,
                    "NonHotspot_CDS_Mutation": row["CDS_Mutation"],
                    "NonHotspot_AA_Mutation": row["AA_Mutation"],
                    "NonHotspot_Observed_Mutations": row["Observed_Mutations"],
                    "NonHotspot_Bonferroni_P": row["Bonferroni_P"],
                    "Distance": abs(site - hs_site),
                }
            )

    pairs_df = pd.DataFrame(pairs_rows)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            ["Hotspot_Site", "Distance", "NonHotspot_Site"],
            ascending=[True, True, True]
        )

    # ========================================================
    # PRINT HOTSPOT / NON-HOTSPOT PAIRS TO CONSOLE
    # ========================================================

    if not pairs_df.empty:
        print(f"Hotspot / nearby non-hotspot pairs in {cancer_type}:")
        for hs_site, group in pairs_df.groupby("Hotspot_Site"):
            print(f"  Hotspot: {group.iloc[0]['Gene']} Site {hs_site}")
            for _, prow in group.iterrows():
                print(
                    f"    -> Near non-hotspot: Site {prow['NonHotspot_Site']} "
                    f"(distance={prow['Distance']}, "
                    f"Observed_Mutations={prow['NonHotspot_Observed_Mutations']})"
                )
    else:
        print(f"No hotspot/non-hotspot pairs found in {cancer_type}")

    # ========================================================
    # SORT MAIN RESULTS (no more Pair_Group grouping attempt --
    # the pairs_df above is the correct place to see relationships)
    # ========================================================

    result_df = result_df.sort_values(
        ["Is_Hotspot", "Bonferroni_P", "Observed_Mutations"],
        ascending=[False, True, False]
    )

    # ========================================================
    # SAVE PER-CANCER RESULTS IMMEDIATELY
    # ========================================================

    cancer_output = os.path.join(
        cancer_dir,
        "gene_hotspot_analysis.csv"
    )

    result_df.to_csv(cancer_output, index=False)

    cancer_pairs_output = os.path.join(
        cancer_dir,
        "hotspot_pairs.csv"
    )

    pairs_df.to_csv(cancer_pairs_output, index=False)

    print(f"Sites tested: {len(result_df):,}")
    print(f"Significant hotspots: {result_df['Is_Hotspot'].sum():,}")
    print(f"Non-hotspot sites within {NEAR_HOTSPOT_WINDOW}bp of a hotspot: {result_df['Near_Hotspot'].sum():,}")
    print(f"Hotspot/non-hotspot pair relationships: {len(pairs_df):,}")
    print(f"Saved: {cancer_output}")
    print(f"Saved: {cancer_pairs_output}")

    all_results.append(result_df)
    all_pairs.append(pairs_df)

# ============================================================
# COMBINE ALL COMPLETED CANCERS
# ============================================================

if len(all_results) == 0:
    print("No results generated.")
    sys.exit()

combined = pd.concat(all_results, ignore_index=True)

combined = combined.sort_values(
    ["Is_Hotspot", "Bonferroni_P", "Observed_Mutations"],
    ascending=[False, True, False]
)

combined.to_csv(combined_output, index=False)

non_empty_pairs = [p for p in all_pairs if not p.empty]

if non_empty_pairs:
    combined_pairs = pd.concat(non_empty_pairs, ignore_index=True)
else:
    combined_pairs = pd.DataFrame()

combined_pairs.to_csv(combined_pairs_output, index=False)

print("\n===================================")
print("FINISHED")
print("===================================")
print(f"Total sites tested: {len(combined):,}")
print(f"Significant hotspots: {combined['Is_Hotspot'].sum():,}")
print(f"Non-hotspot sites within {NEAR_HOTSPOT_WINDOW}bp of a hotspot (same gene/cancer): {combined['Near_Hotspot'].sum():,}")
print(f"Total hotspot/non-hotspot pair relationships: {len(combined_pairs):,}")
print(f"\nSaved combined file:\n{combined_output}")
print(f"Saved combined pairs file:\n{combined_pairs_output}")
