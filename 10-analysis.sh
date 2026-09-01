#!/bin/bash
set -euo pipefail

# Global Paths & Configurations
GMX_BIN="/home/admin/Documents/gromacs-2025.2/build/bin/gmx"
SCRIPT_DIR=$(pwd)
BASE_DIR="$SCRIPT_DIR/.."

# ==============================================================================
# COMMAND LINE ARGUMENT HANDLING
# ==============================================================================
if [ $# -lt 1 ]; then
  echo "ERROR: No sequence name provided!" >&2
  echo "Usage: ./analysis.sh <sequence_name>" >&2
  echo "Example: ./analysis.sh APC_4103_non" >&2
  exit 1
fi

SEQUENCE="$1"

# ==============================================================================
# MAIN ANALYSIS LOOP (Runs only the 3 replicates for the selected sequence)
# ==============================================================================
for rep in 1 2 3; do
  echo "========= STARTING ANALYSIS: $SEQUENCE | REPLICATE: $rep ========="
  
  # Define directory paths matching your MD output structure
  MD_DIR="$BASE_DIR/output/${SEQUENCE}/${rep}/md"
  NPT_DIR="$BASE_DIR/output/${SEQUENCE}/${rep}/npt"
  ANALYSIS_DIR="$BASE_DIR/output/${SEQUENCE}/${rep}/analysis"
  
  if [ ! -d "$MD_DIR" ]; then
    echo "!!! CRITICAL: Source directory $MD_DIR not found. Halting execution! !!!" >&2
    exit 1
  fi
  
  # Ensure the analysis directory exists and move into it
  mkdir -p "$ANALYSIS_DIR"
  cd "$ANALYSIS_DIR"

  # --- STEP 1: TRAJECTORY POST-PROCESSING (REMOVE WATER) ---
  echo ">>> Stripping water and fitting trajectory... <<<"
  # Fit on Protein (1), output non-Water (17)
  printf "1\n17\n" | $GMX_BIN trjconv -s "$MD_DIR/md.tpr" -f "$MD_DIR/md.xtc" -o md_nowater.xtc -pbc nojump -fit translation 

  # Strip water from NPT reference structure
  printf "1\n17\n" | $GMX_BIN trjconv -f "$NPT_DIR/npt.gro" -o no-water.gro -pbc nojump -fit translation -s "$NPT_DIR/npt.tpr"

  # Generate matching water-free .tpr topology file for analysis
  # Select non-Water (Group 17) to match the atom count of md_nowater.xtc
  echo ">>> Generating water-free topology file... <<<"
  printf "17\n" | $GMX_BIN convert-tpr -s "$MD_DIR/md.tpr" -o no-water.tpr

  # --- STEP 2: ROOT MEAN SQUARE DEVIATION (RMSD) ---
  echo ">>> Calculating RMSD... <<<"
  printf "4\n4\n" | $GMX_BIN rms -f md_nowater.xtc -s no-water.tpr -tu ns -o rmsd_backbone.xvg
  printf "12\n12\n" | $GMX_BIN rms -f md_nowater.xtc -s no-water.tpr -tu ns -o rmsd_dna.xvg

  # --- STEP 3: ROOT MEAN SQUARE FLUCTUATION (RMSF) ---
  echo ">>> Calculating RMSF... <<<"
  printf "4\n" | $GMX_BIN rmsf -f md_nowater.xtc -s no-water.tpr -res -o rmsf_backbone.xvg
  printf "12\n" | $GMX_BIN rmsf -f md_nowater.xtc -s no-water.tpr -res -o rmsf_dna.xvg

  # --- STEP 4: RADIUS OF GYRATION (Rg) ---
  echo ">>> Calculating Radius of Gyration... <<<"
  printf "4\n" | $GMX_BIN gyrate -f md_nowater.xtc -s no-water.tpr -o rg_backbone.xvg
  printf "12\n" | $GMX_BIN gyrate -f md_nowater.xtc -s no-water.tpr -o rg_dna.xvg

  # --- STEP 5: SOLVATION ACCESSIBILITY (SASA) ---
  echo ">>> Calculating SASA... <<<"
  printf "1\n" | $GMX_BIN sasa -f md_nowater.xtc -s no-water.tpr -o sasa_protein.xvg
  printf "4\n" | $GMX_BIN sasa -f md_nowater.xtc -s no-water.tpr -o sasa_backbone.xvg
  printf "12\n" | $GMX_BIN sasa -f md_nowater.xtc -s no-water.tpr -o sasa_dna.xvg

  echo "========= COMPLETED ANALYSIS: $SEQUENCE | REPLICATE: $rep ========="
done

echo "=== ALL 3 REPLICATES FOR $SEQUENCE ANALYSED SUCCESSFULLY ==="