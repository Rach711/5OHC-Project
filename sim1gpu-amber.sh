#!/bin/bash
set -euo pipefail

# Global Paths & Configurations
GMX_BIN="/home/admin/Documents/gromacs-2025.2/build/bin/gmx"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES="$BASE_DIR/resources"

# Lock your environment to GPU 0 and allocate half the CPU
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
CPU_MASK="0-5,12-17"

sequences=(
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

# ==============================================================================
# MAIN TRACK WORKLOAD QUEUE
# ==============================================================================
for seq in "${sequences[@]}"; do
  for rep in 1 2 3; do
    
    echo "========= STARTING: $seq | REPLICATE: $rep ========="
    
    # 1. Setup Folder Structure Pathing
    OUT_DIR="$BASE_DIR/output/${seq}/${rep}"
    INPUT_PDB="$BASE_DIR/input/clean/clean_${seq}_hyd.pdb"
    
    TOPOL="$OUT_DIR/topologies"
    EM="$OUT_DIR/em"
    NVT="$OUT_DIR/nvt"
    NPT="$OUT_DIR/npt"
    MD="$OUT_DIR/md"
    
    mkdir -p "$TOPOL" "$EM" "$NVT" "$NPT" "$MD"

    # Wrap the actual execution in a try/catch block for robust fault tolerance
    {
      # --- STEP 1: TOPOLOGY ---
      cd "$TOPOL"
      rm -rf ./amber99bsc1.ff ./residuetypes.dat
      ln -sf "$SCRIPT_DIR/amber99bsc1.ff" ./amber99bsc1.ff
      ln -sf "$SCRIPT_DIR/residuetypes.dat" ./residuetypes.dat
      printf "1\n1\n" | $GMX_BIN pdb2gmx -f "$INPUT_PDB" -o processed.gro -p topol.top -i posre.itp -ter
      
      sed -i 's|../output/topologies/||g' *.itp 2>/dev/null || true
      sed -i 's|../output/topologies/||g' *.top 2>/dev/null || true

      # --- STEP 2: SOLVATION ---
      $GMX_BIN editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic
      $GMX_BIN solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

      # --- STEP 3: IONS ---
      $GMX_BIN grompp -f "$RESOURCES/ions.mdp" -c solv.gro -p topol.top -o ions.tpr
      echo "SOL" | $GMX_BIN genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

      # --- STEP 4: ENERGY MINIMIZATION ---
      cd "$EM"
      rm -rf ./amber99bsc1.ff; ln -sf "$SCRIPT_DIR/amber99bsc1.ff" ./amber99bsc1.ff
      $GMX_BIN grompp -f "$RESOURCES/em.mdp" -c "$TOPOL/solv_ions.gro" -p "$TOPOL/topol.top" -o em.tpr
      taskset -c $CPU_MASK $GMX_BIN mdrun -v -deffnm em -ntmpi 1 -ntomp 12 -nb gpu -pin off

      # --- STEP 5: NVT ---
      cd "$NVT"
      rm -rf ./amber99bsc1.ff; ln -sf "$SCRIPT_DIR/amber99bsc1.ff" ./amber99bsc1.ff
      $GMX_BIN grompp -f "$RESOURCES/nvt.mdp" -c "$EM/em.gro" -r "$EM/em.gro" -p "$TOPOL/topol.top" -o nvt.tpr
      taskset -c $CPU_MASK $GMX_BIN mdrun -v -deffnm nvt -ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pin off

      # --- STEP 6: NPT ---
      cd "$NPT"
      rm -rf ./amber99bsc1.ff; ln -sf "$SCRIPT_DIR/amber99bsc1.ff" ./amber99bsc1.ff
      $GMX_BIN grompp -f "$RESOURCES/npt.mdp" -c "$NVT/nvt.gro" -r "$NVT/nvt.gro" -t "$NVT/nvt.cpt" -p "$TOPOL/topol.top" -o npt.tpr
      taskset -c $CPU_MASK $GMX_BIN mdrun -v -deffnm npt -ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pin off

      # --- STEP 7: PRODUCTION MD ---
      cd "$MD"
      rm -rf ./amber99bsc1.ff; ln -sf "$SCRIPT_DIR/amber99bsc1.ff" ./amber99bsc1.ff
      
      $GMX_BIN grompp -f "$RESOURCES/md.mdp" -c "$NPT/npt.gro" -t "$NPT/npt.cpt" -p "$TOPOL/topol.top" -o md.tpr
      
      taskset -c $CPU_MASK $GMX_BIN mdrun -v -deffnm md \
        -nb gpu -pme gpu -bonded gpu -update gpu \
        -ntmpi 1 -ntomp 12 -pin off -dlb yes

    } || {
      # This block triggers ONLY if anything inside the curly braces crashes
      echo "!!! CRITICAL ERROR: $seq Replicate $rep failed. Logging and skipping... !!!" >&2
      echo "$seq | Replicate $rep | Crashed at $(date)" >> "$BASE_DIR/failed_runs.log"
    }

  done
done

echo ">>> ALL QUEUED RUNS FINISHED PROCESSING ON GPU 0 <<<"