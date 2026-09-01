#!/bin/bash
set -eou pipefail

# Define your verified local GROMACS 2025.2 absolute binary path
GMX_BIN="/home/admin/Documents/gromacs-2025.2/build/bin/gmx"

# --- CONFIGURATION: LIST YOUR 12 SEQUENCES HERE ---
# Add all 12 of your exact sequence names to this array
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
# --------------------------------------------------

# Background worker function to process a single simulation trajectory
run_on_gpu() {
  local gpu_id=$1
  local seq=$2
  local rep=$3

  # Isolate this specific process to exactly one RTX 5090 card
  export CUDA_VISIBLE_DEVICES=$gpu_id
  
  # Allocate 12 OpenMP threads so two concurrent runs perfectly fill your 24 logical units
  export OMP_NUM_THREADS=12 

  # Establish absolute paths mapping to your local storage layout
  BASE_DIR=$(pwd)/..
  OUT_DIR="$BASE_DIR/output/${seq}/${rep}"
  INPUT_PDB="$BASE_DIR/input/clean/clean_${seq}_hyd.pdb"
  RESOURCES="$BASE_DIR/resources"

  TOPOL=$OUT_DIR/topologies
  EM=$OUT_DIR/em
  NVT=$OUT_DIR/nvt
  NPT=$OUT_DIR/npt
  MD=$OUT_DIR/md

  # Generate the folder tree for this specific replicate if it doesn't exist
  mkdir -p $TOPOL $EM $NVT $NPT $MD

  echo ">>> [GPU $gpu_id] Starting Simulation: $seq | Replicate: $rep <<<"

  # --- STEP 1: TOPOLOGY GENERATION ---
  cd $TOPOL
  rm -rf ./amber99bsc1.ff ./residuetypes.dat
  ln -sf $BASE_DIR/scripts/amber99bsc1.ff ./amber99bsc1.ff
  ln -sf $BASE_DIR/scripts/residuetypes.dat ./residuetypes.dat

  # Uses your exact working interactive menu choices and the -ter flag
  printf "1\n1\n0\n0\n4\n6\n4\n6\n" | $GMX_BIN pdb2gmx -f $INPUT_PDB -o processed.gro -p topol.top -i posre.itp -ter
  
  # Clean up relative path formatting anomalies inside the topology output
  sed -i 's|../output/topologies/||g' *.itp 2>/dev/null || true
  sed -i 's|../output/topologies/||g' *.top 2>/dev/null || true

  # --- STEP 2: SOLVATION & BOX GENERATION ---
  $GMX_BIN editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic
  $GMX_BIN solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

  # --- STEP 3: NEUTRALIZATION & IONS ---
  $GMX_BIN grompp -f $RESOURCES/ions.mdp -c solv.gro -p topol.top -o ions.tpr
  echo "SOL" | $GMX_BIN genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

  # --- STEP 4: ENERGY MINIMIZATION ---
  cd $EM
  rm -rf ./amber99bsc1.ff
  ln -sf $BASE_DIR/scripts/amber99bsc1.ff ./amber99bsc1.ff
  $GMX_BIN grompp -f $RESOURCES/em.mdp -c $TOPOL/solv_ions.gro -p $TOPOL/topol.top -o em.tpr
  $GMX_BIN mdrun -v -deffnm em -ntmpi 1

  # --- STEP 5: NVT EQUILIBRATION ---
  cd $NVT
  rm -rf ./amber99bsc1.ff
  ln -sf $BASE_DIR/scripts/amber99bsc1.ff ./amber99bsc1.ff
  $GMX_BIN grompp -f $RESOURCES/nvt.mdp -c $EM/em.gro -r $EM/em.gro -p $TOPOL/topol.top -o nvt.tpr
  $GMX_BIN mdrun -ntomp 12 -v -deffnm nvt -ntmpi 1

  # --- STEP 6: NPT EQUILIBRATION ---
  cd $NPT
  rm -rf ./amber99bsc1.ff
  ln -sf $BASE_DIR/scripts/amber99bsc1.ff ./amber99bsc1.ff
  $GMX_BIN grompp -f $RESOURCES/npt.mdp -c $NVT/nvt.gro -r $NVT/nvt.gro -t $NVT/nvt.cpt -p $TOPOL/topol.top -o npt.tpr
  $GMX_BIN mdrun -ntomp 12 -v -deffnm npt -ntmpi 1

  # --- STEP 7: PRODUCTION MD FULL RUN ---
  cd $MD
  rm -rf ./amber99bsc1.ff
  ln -sf $BASE_DIR/scripts/amber99bsc1.ff ./amber99bsc1.ff
  $GMX_BIN grompp -f $RESOURCES/md.mdp -c $NPT/npt.gro -t $NPT/npt.cpt -p $TOPOL/topol.top -o md.tpr

  # Full GPU offloading execution (Notice: No artificial -nsteps limit here)
  $GMX_BIN mdrun -v -deffnm md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -ntmpi 1

  echo ">>> [GPU $gpu_id] Finished Simulation: $seq | Replicate: $rep <<<"
}

# --- MAIN ORCHESTRATION LOOP ---
# Sweeps sequentially through your array lists and manages your dual 5090 cards
gpu=0
for seq in "${sequences[@]}"; do
  for rep in 1 2 3; do
    
    # Fire off the simulation worker block into the background
    run_on_gpu $gpu "$seq" "$rep" &
    
    # Alternating GPU core assignment mapping
    if [ $gpu -eq 0 ]; then
      gpu=1
    else
      gpu=0
      # Wait right here until BOTH card processes finish their ~14-hour wave
      wait 
    fi

  done
done

# Catch any remaining trailing single job if applicable
wait
echo ">>> SIMULATIONS COMPLETED SUCCESSFULLY OUT ACROSS BOTH RTX 5090s <<<"