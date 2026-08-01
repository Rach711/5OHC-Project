#!/bin/bash
set -eou pipefail

# Path to local GROMACS 2025.2 binary
GMX_BIN="/home/admin/Documents/gromacs-2025.2/build/bin/gmx"

# Remaining 7 sequences
sequences=(
  "APC_4103_non"
  "APC_4343_non"
  "APC_4348_hot"
  "TP53_632_non"
  "TP53_637_hot"
  "TP53_844_hot"
  "TP53_849_non"
)

REPLICATES=(1 2 3)

run_single_job() {
  local gpu_id=$1
  local seq=$2
  local rep=$3

  # CCD Binding for AMD Ryzen 9 9900X (6 cores / 12 threads per CCD)
  local cpu_mask=""
  if [ "$gpu_id" -eq 0 ]; then
    cpu_mask="0-5,12-17"
  else
    cpu_mask="6-11,18-23"
  fi

  export CUDA_VISIBLE_DEVICES=$gpu_id
  export OMP_NUM_THREADS=12

  BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  OUT_DIR="$BASE_DIR/output/${seq}/${rep}"
  INPUT_PDB="$BASE_DIR/input/clean/clean_${seq}_hyd.pdb"
  RESOURCES="$BASE_DIR/resources"

  TOPOL="$OUT_DIR/topologies"
  EM="$OUT_DIR/em"
  NVT="$OUT_DIR/nvt"
  NPT="$OUT_DIR/npt"
  MD="$OUT_DIR/md"

  mkdir -p "$TOPOL" "$EM" "$NVT" "$NPT" "$MD"

  echo "======================================================================"
  echo ">>> [GPU $gpu_id | CPU Cores $cpu_mask] STARTING: $seq (Rep $rep) <<<"
  echo "======================================================================"

  # --- STEP 1: TOPOLOGY GENERATION ---
  cd "$TOPOL"
  rm -rf ./amber99bsc1.ff ./residuetypes.dat
  ln -sf "$BASE_DIR/scripts/amber99bsc1.ff" ./amber99bsc1.ff
  ln -sf "$BASE_DIR/scripts/residuetypes.dat" ./residuetypes.dat

  printf "1\n1\n" | $GMX_BIN pdb2gmx -f "$INPUT_PDB" -o processed.gro -p topol.top -i posre.itp -ter

  sed -i 's|../output/topologies/||g' *.itp 2>/dev/null || true
  sed -i 's|../output/topologies/||g' *.top 2>/dev/null || true

  # --- STEP 2: SOLVATION & BOX GENERATION ---
  $GMX_BIN editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic
  $GMX_BIN solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

  # --- STEP 3: NEUTRALIZATION & IONS ---
  $GMX_BIN grompp -f "$RESOURCES/ions.mdp" -c solv.gro -p topol.top -o ions.tpr
  echo "SOL" | $GMX_BIN genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

  # --- STEP 4: ENERGY MINIMIZATION ---
  cd "$EM"
  ln -sf "$BASE_DIR/scripts/amber99bsc1.ff" ./amber99bsc1.ff
  $GMX_BIN grompp -f "$RESOURCES/em.mdp" -c "$TOPOL/solv_ions.gro" -p "$TOPOL/topol.top" -o em.tpr
  taskset -c $cpu_mask $GMX_BIN mdrun -v -deffnm em -ntmpi 1 -ntomp 12 -nb gpu -pin off

  # --- STEP 5: NVT EQUILIBRATION ---
  cd "$NVT"
  ln -sf "$BASE_DIR/scripts/amber99bsc1.ff" ./amber99bsc1.ff
  $GMX_BIN grompp -f "$RESOURCES/nvt.mdp" -c "$EM/em.gro" -r "$EM/em.gro" -p "$TOPOL/topol.top" -o nvt.tpr
  taskset -c $cpu_mask $GMX_BIN mdrun -v -deffnm nvt -ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pin off

  # --- STEP 6: NPT EQUILIBRATION ---
  cd "$NPT"
  ln -sf "$BASE_DIR/scripts/amber99bsc1.ff" ./amber99bsc1.ff
  $GMX_BIN grompp -f "$RESOURCES/npt.mdp" -c "$NVT/nvt.gro" -r "$NVT/nvt.gro" -t "$NVT/nvt.cpt" -p "$TOPOL/topol.top" -o npt.tpr
  taskset -c $cpu_mask $GMX_BIN mdrun -v -deffnm npt -ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pin off

  # --- STEP 7: PRODUCTION MD FULL RUN ---
  cd "$MD"
  ln -sf "$BASE_DIR/scripts/amber99bsc1.ff" ./amber99bsc1.ff
  $GMX_BIN grompp -f "$RESOURCES/md.mdp" -c "$NPT/npt.gro" -t "$NPT/npt.cpt" -p "$TOPOL/topol.top" -o md.tpr

  taskset -c $cpu_mask $GMX_BIN mdrun -v -deffnm md \
    -nb gpu -pme gpu -bonded gpu -update gpu \
    -ntmpi 1 -ntomp 12 -pin off

  echo ">>> [GPU $gpu_id] FINISHED: $seq (Rep $rep) <<<"
}

export -f run_single_job
export GMX_BIN

# Dynamic FIFO queue to allocate GPU tokens asynchronously
WORK_QUEUE=$(mktemp -u)
mkfifo "$WORK_QUEUE"
exec 3<>"$WORK_QUEUE"
rm "$WORK_QUEUE"

# Seed GPU IDs into queue
echo "0" >&3
echo "1" >&3

for seq in "${sequences[@]}"; do
  for rep in "${REPLICATES[@]}"; do
    read -u 3 gpu_id
    (
      # Guaranteed token release even if a step encounters an error
      trap 'echo "$gpu_id" >&3' EXIT
      run_single_job "$gpu_id" "$seq" "$rep"
    ) &
  done
done

wait
exec 3>&-
echo ">>> ALL SIMULATIONS COMPLETED SUCCESSFULLY <<<"