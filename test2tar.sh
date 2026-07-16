#!/bin/bash
set -euo pipefail

# Default settings
ARCHIVE_CLEANUP=false

# Global Paths & Configurations
GMX_BIN="/home/admin/Documents/gromacs-2025.2/build/bin/gmx"
SCRIPT_DIR=$(pwd)
BASE_DIR="$SCRIPT_DIR/.."
RESOURCES="$BASE_DIR/resources"

# Lock environment to GPU 0 and allocate half the CPU
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12

sequences=(
  "APC_4103_non"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -a|--archive) ARCHIVE_CLEANUP=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ==============================================================================
# MAIN WORKLOAD QUEUE
# ==============================================================================
for seq in "${sequences[@]}"; do
  
  for rep in 1 2 3; do
    echo "========= STARTING: $seq | REPLICATE: $rep ========="
    
    OUT_DIR="$BASE_DIR/output/${seq}/${rep}"
    INPUT_PDB="$BASE_DIR/input/clean/clean_${seq}_hyd.pdb"
    
    TOPOL="$OUT_DIR/topologies"
    EM="$OUT_DIR/em"
    NVT="$OUT_DIR/nvt"
    NPT="$OUT_DIR/npt"
    MD="$OUT_DIR/md"
    
    mkdir -p "$TOPOL" "$EM" "$NVT" "$NPT" "$MD"

    # Execution block with try/catch layout
    {
      # --- STEP 1: TOPOLOGY ---
      cd "$TOPOL"
      rm -rf ./charmm36.ff ./residuetypes.dat
      ln -sf "$SCRIPT_DIR/charmm36.ff" ./charmm36.ff
      ln -sf "$SCRIPT_DIR/residuetypes.dat" ./residuetypes.dat
      
      printf "1\n1\n0\n0\n4\n6\n4\n6\n" | $GMX_BIN pdb2gmx -f "$INPUT_PDB" -o processed.gro -p topol.top -i posre.itp -ter
      
      compgen -G "*.itp" >/dev/null && sed -i 's|../output/topologies/||g' *.itp || true
      compgen -G "*.top" >/dev/null && sed -i 's|../output/topologies/||g' *.top || true

      # --- STEP 2: SOLVATION ---
      $GMX_BIN editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic
      $GMX_BIN solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

      # --- STEP 3: IONS ---
      $GMX_BIN grompp -f "$RESOURCES/ions.mdp" -c solv.gro -p topol.top -o ions.tpr
      echo "SOL" | $GMX_BIN genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

      # --- STEP 4: ENERGY MINIMIZATION ---
      cd "$EM"
      rm -rf ./charmm36.ff; ln -sf "$SCRIPT_DIR/charmm36.ff" ./charmm36.ff
      $GMX_BIN grompp -f "$RESOURCES/em.mdp" -c "$TOPOL/solv_ions.gro" -p "$TOPOL/topol.top" -o em.tpr
      $GMX_BIN mdrun -v -deffnm em -ntmpi 1

      # --- STEP 5: NVT ---
      cd "$NVT"
      rm -rf ./charmm36.ff; ln -sf "$SCRIPT_DIR/charmm36.ff" ./charmm36.ff
      $GMX_BIN grompp -f "$RESOURCES/nvt.mdp" -c "$EM/em.gro" -r "$EM/em.gro" -p "$TOPOL/topol.top" -o nvt.tpr
      $GMX_BIN mdrun -ntomp 12 -v -deffnm nvt -ntmpi 1

      # --- STEP 6: NPT ---
      cd "$NPT"
      rm -rf ./charmm36.ff; ln -sf "$SCRIPT_DIR/charmm36.ff" ./charmm36.ff
      $GMX_BIN grompp -f "$RESOURCES/npt.mdp" -c "$NVT/nvt.gro" -r "$NVT/nvt.gro" -t "$NVT/nvt.cpt" -p "$TOPOL/topol.top" -o npt.tpr
      $GMX_BIN mdrun -ntomp 12 -v -deffnm npt -ntmpi 1

      # --- STEP 7: PRODUCTION MD ---
      cd "$MD"
      rm -rf ./charmm36.ff; ln -sf "$SCRIPT_DIR/charmm36.ff" ./charmm36.ff
      if [ -f "md.cpt" ]; then
        echo ">>> [RESUME] Found checkpoint. Appending to trajectory... <<<"
        $GMX_BIN mdrun -v -deffnm md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -ntmpi 1 -cpi md.cpt -append
      else
        $GMX_BIN grompp -f "$RESOURCES/md.mdp" -c "$NPT/npt.gro" -t "$NPT/npt.cpt" -p "$TOPOL/topol.top" -o md.tpr
        $GMX_BIN mdrun -v -deffnm md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -ntmpi 1
      fi

      echo "========= COMPLETED: $seq | REPLICATE: $rep ========="

    } || {
      echo "!!! CRITICAL ERROR: $seq Replicate $rep failed. Logging and skipping... !!!" >&2
      echo "$seq | Replicate $rep | Crashed at $(date)" >> "$BASE_DIR/failed_runs.log"
    }
  done

  # ==============================================================================
  # POST-PROCESSING: OPTIONAL ARCHIVE AND CLEANUP (Hard Physical Check)
  # ==============================================================================
  if [ "$ARCHIVE_CLEANUP" = true ]; then
    SEQ_OUT_DIR="$BASE_DIR/output/${seq}"
    TAR_FILE="$BASE_DIR/output/${seq}.tar.gz"

    # Verify all 3 replicates physically produced a final log structure
    all_replicates_exist=true
    for r in 1 2 3; do
      if [ ! -f "$SEQ_OUT_DIR/$r/md/md.log" ]; then
        echo "!!! WARNING: Replicate $r did not finish successfully (missing md.log)! !!!" >&2
        all_replicates_exist=false
      fi
    done

    if [ "$all_replicates_exist" = false ]; then
      echo ">>> CRITICAL: Skipping archive/cleanup for $seq. Some replicates are missing or crashed! <<<"
      echo "$seq | Archive Skipped - Missing Run Data | $(date)" >> "$BASE_DIR/failed_archives.log"
    elif [ -d "$SEQ_OUT_DIR" ]; then
      echo ">>> [--archive flag active] Archiving completed sequence: $seq <<<"
      
      # Protect strict 'set -e' environment during execution
      if tar -czf "$TAR_FILE" -C "$BASE_DIR/output" "${seq}" && tar -tzf "$TAR_FILE" >/dev/null 2>&1; then
        echo ">>> Archive integrity verified successfully. Deleting original directory. <<<"
        rm -rf "$SEQ_OUT_DIR"
      else
        echo "!!! CRITICAL ERROR: Archive validation failed for $seq! Directory kept safe. !!!" >&2
        echo "$seq | Archive Failed | $(date)" >> "$BASE_DIR/failed_archives.log"
      fi
    fi
  else
    echo ">>> Skipping archive and cleanup stage for $seq (run with --archive to enable) <<<"
  fi

done

echo ">>> ALL QUEUED RUNS FINISHED PROCESSING ON GPU 0 <<<"