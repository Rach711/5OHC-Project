#!/bin/bash
set -euo pipefail

# Absolute paths to Curves+ binaries and standard library
CURVES_BIN="/mnt/c/Users/rache/downloads/curves+/Cur+"
CANAL_BIN="/mnt/c/Users/rache/downloads/curves+/canal"
LIB_PATH="/mnt/c/Users/rache/downloads/curves+/standard"

# Get the directory where this script lives (your 'scripts' folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Target the 'output' folder located at the same level as 'scripts'
BASE_OUTPUT_DIR="${SCRIPT_DIR}/../output"

# List of all 12 sequence constructs
SEQUENCES=(
  "APC_637_hot"
  "APC_641_non"
  "APC_3335_non"
  "APC_3340_hot"
  "APC_4099_hot"
  "APC_4103_non"
  "APC_4343_non"
  "APC_4348_hot"
  "TP53_632_non"
  "TP53_637_hot"
  "TP53_844_hot"
  "TP53_849_non"
)

# Loop over each sequence directory
for SEQ in "${SEQUENCES[@]}"; do

  # Extract prefix for trj files (e.g., "APC_637_hot" -> "APC_637")
  SEQ_PREFIX=$(echo "$SEQ" | awk -F'_' '{print $1"_"$2}')

  # Loop over the 3 replicate folders (1, 2, 3)
  for REP in 1 2 3; do

    # Target the 'curves' subfolder inside each replicate
    TARGET_DIR="${BASE_OUTPUT_DIR}/${SEQ}/${REP}/curves"

    # Verify the curves directory exists
    if [ ! -d "${TARGET_DIR}" ]; then
      echo "Warning: Directory ${TARGET_DIR} does not exist. Skipping..."
      continue
    fi

    # 1. Navigate into the replicate's 'curves' folder
    cd "${TARGET_DIR}" || exit

    LIS_NAME="${SEQ_PREFIX}_${REP}"                # e.g., APC_637_1
    TRJ_FILE="${SEQ_PREFIX}_${REP}dna.trj"         # e.g., APC_637_1dna.trj
    TOP_FILE="clean_${SEQ}_hyd.top"               # e.g., clean_APC_637_hot_hyd.top

    echo "=================================================="
    echo " Processing folder: ${TARGET_DIR}"
    echo "=================================================="

    # Check that input files exist inside this curves folder before running
    if [ ! -f "${TRJ_FILE}" ] || [ ! -f "${TOP_FILE}" ]; then
      echo "Error: Missing ${TRJ_FILE} or ${TOP_FILE} in ${TARGET_DIR}. Skipping..."
      cd "${SCRIPT_DIR}" || exit
      continue
    fi

    # 2. Run Curves+ inside the curves folder
    "${CURVES_BIN}" <<!
 &inp file=${TRJ_FILE}, ftop=${TOP_FILE}, lis=${LIS_NAME}, lib=${LIB_PATH}, &end
2 1 -1 0 0
1:15
30:16
!

    # 3. Extract exact sequence strings from the local .lis file
    if [ -f "${LIS_NAME}.lis" ]; then
      S1=$(grep "Strand  1" "${LIS_NAME}.lis" | awk -F': ' '{print $2}' | tr -d ' ')
      S2=$(grep "Strand  2" "${LIS_NAME}.lis" | awk -F': ' '{print $2}' | tr -d ' ')

      # 4. Run Canal across all 14 individual base steps
      for (( i=1; i<=14; i++ )); do
        j=$((i+1))
        OUT_LIS="${LIS_NAME}-${i}-${j}"

        "${CANAL_BIN}" <<!
 &inp lis=${OUT_LIS}, seq=*, lev1=${i}, lev2=${j}, histo=.f., &end
${LIS_NAME} ${S1}
${LIS_NAME} ${S2}
!
      done
    else
      echo "Error: ${LIS_NAME}.lis was not created. Skipping Canal."
    fi

    # Return to the scripts folder
    cd "${SCRIPT_DIR}" || exit

  done
done

echo "=================================================="
echo " All 36 runs completed successfully!"
echo "=================================================="