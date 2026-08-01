import os
import time
import requests
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

input_file = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers/genes_with_transcripts.csv"

output_dir = "/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/Cancers/c_count_results"

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "c_counts_per_gene_transcript.csv"
)

missing_file = os.path.join(
    output_dir,
    "missing_transcripts.csv"
)

BATCH_SIZE = 50

# ============================================================
# LOAD INPUT
# ============================================================

df = pd.read_csv(input_file)

if "Gene Name" not in df.columns:
    raise ValueError("Missing 'Gene Name' column")

if "Transcript" not in df.columns:
    raise ValueError("Missing 'Transcript' column")

df = df.drop_duplicates(subset=["Gene Name", "Transcript"])

print(f"Loaded {len(df):,} unique gene/transcript pairs")

# ============================================================
# PREPARE LOOKUP TABLE
# ============================================================

transcript_map = {}

for _, row in df.iterrows():

    gene = str(row["Gene Name"]).strip()
    transcript = str(row["Transcript"]).strip()

    transcript_id = transcript.split(".")[0]

    transcript_map[transcript_id] = (
        gene,
        transcript
    )

transcript_ids = list(transcript_map.keys())

# ============================================================
# RESULTS
# ============================================================

results = []
missing = []

# ============================================================
# ENSEMBL BATCH QUERY
# ============================================================

server = "https://rest.ensembl.org"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

n_batches = (len(transcript_ids) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(n_batches):

    start = batch_num * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(transcript_ids))

    batch_ids = transcript_ids[start:end]

    print(
        f"\nBatch {batch_num+1}/{n_batches}"
        f" ({len(batch_ids)} transcripts)"
    )

    endpoint = f"{server}/sequence/id"

    payload = {
        "ids": batch_ids,
        "type": "cds"
    }

    try:

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=120
        )

        if not response.ok:

            print(
                f"Batch failed: HTTP "
                f"{response.status_code}"
            )

            for tid in batch_ids:

                gene, transcript = transcript_map[tid]

                missing.append(
                    (
                        gene,
                        transcript,
                        f"HTTP_{response.status_code}"
                    )
                )

            continue

        returned = response.json()

        found_ids = set()

        for record in returned:

            tid = record.get("id")

            if tid is None:
                continue

            found_ids.add(tid)

            gene, transcript = transcript_map[tid]

            seq = record.get("seq", "")

            if not seq:

                missing.append(
                    (
                        gene,
                        transcript,
                        "EMPTY_SEQUENCE"
                    )
                )

                continue

            c_count = seq.upper().count("C")

            results.append(
                (
                    gene,
                    transcript,
                    c_count
                )
            )

        # Anything not returned by Ensembl
        for tid in batch_ids:

            if tid not in found_ids:

                gene, transcript = transcript_map[tid]

                missing.append(
                    (
                        gene,
                        transcript,
                        "NOT_RETURNED"
                    )
                )

        print(
            f"Retrieved "
            f"{len(found_ids)}/{len(batch_ids)}"
        )

        # tiny pause between batches
        time.sleep(0.2)

    except Exception as e:

        print(f"Batch exception: {e}")

        for tid in batch_ids:

            gene, transcript = transcript_map[tid]

            missing.append(
                (
                    gene,
                    transcript,
                    str(e)
                )
            )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "GeneName",
        "TranscriptID",
        "C_count"
    ]
)

results_df = results_df.sort_values(
    "C_count",
    ascending=False
)

results_df.to_csv(
    output_file,
    index=False
)

missing_df = pd.DataFrame(
    missing,
    columns=[
        "GeneName",
        "TranscriptID",
        "Reason"
    ]
)

missing_df.to_csv(
    missing_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("Finished")
print("===================================")

print(
    f"Successful: "
    f"{len(results_df):,}"
)

print(
    f"Missing: "
    f"{len(missing_df):,}"
)

print(f"\nResults:")
print(output_file)

print(f"\nMissing:")
print(missing_file)
