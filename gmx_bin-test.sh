#!/bin/bash

# Stop immediately if any step fails
set -e

# Force the shell environment to find your local GROMACS installation
source /home/admin/Documents/gromacs-2025.2/build/scripts/GMXRC

# Define the path to your working GROMACS binary
GMX_BIN="/home/admin/Documents/gromacs-2025.2/build/bin/gmx"

# --- CONFIGURATION FOR THE TEST RUN ---
GENE="APC"
POS="637"
STATUS="hot"
REP_NAME="3" 
# --------------------------------------

# 1. Target exactly one specific 5090 GPU (GPU 0)
export CUDA_VISIBLE_DEVICES=0

# 2. Hardened OpenMP configuration for a local workstation
# Adjust '8' to match the number of physical CPU cores you want to allocate to this single test run
export OMP_NUM_THREADS=8

# Construct paths matching your local directory structure
INPUT_FILE="clean_${GENE}_${POS}_${STATUS}_hyd.pdb"
SEQ_NAME="${GENE}_${POS}_${STATUS}"

BASE_DIR=$(pwd)/..
INPUT_PDB=$BASE_DIR/input/clean/${INPUT_FILE}
RESOURCES=$BASE_DIR/resources
REP_DIR=$BASE_DIR/output/${SEQ_NAME}/${REP_NAME}

TOPOL=$REP_DIR/topologies; EM=$REP_DIR/em; NVT=$REP_DIR/nvt; NPT=$REP_DIR/npt; MD=$REP_DIR/md

echo ">>> Starting Fresh Local GPU Test Run for ${SEQ_NAME} <<<"
mkdir -p $TOPOL $EM $NVT $NPT $MD

# --- STEP 1: TOPOLOGY ---
echo ">>> Running pdb2gmx..."
cd $TOPOL

# FIX: Forcefully clear out any pre-existing links from previous attempts so 'ln' won't nest folders
rm -rf ./charmm36.ff ./residuetypes.dat

ln -sf $BASE_DIR/scripts/charmm36.ff ./charmm36.ff

# Forcefully link your residuetypes.dat file so the custom flipped 'DO' base is recognized
ln -sf $BASE_DIR/scripts/residuetypes.dat ./residuetypes.dat

# Use automated default termini selection (No -ter flag)
printf "1\n1\n0\n0\n4\n6\n4\n6\n" | $GMX_BIN pdb2gmx -f $INPUT_PDB -o processed.gro -p topol.top -i posre.itp

# Clean up path modifications if necessary
sed -i 's|../output/topologies/||g' *.itp 2>/dev/null || true
sed -i 's|../output/topologies/||g' *.top 2>/dev/null || true

# --- STEP 2: SOLVATION & IONS ---
echo ">>> Running Solvation..."
$GMX_BIN editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic
$GMX_BIN solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

echo ">>> Neutralizing System with Ions..."
$GMX_BIN grompp -f $RESOURCES/ions.mdp -c solv.gro -p topol.top -o ions.tpr
echo "SOL" | $GMX_BIN genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral

# --- STEP 3: ENERGY MINIMIZATION ---
echo ">>> Running Energy Minimization..."
cd $EM; rm -rf ./charmm36.ff; ln -sf $BASE_DIR/scripts/charmm36.ff ./charmm36.ff
$GMX_BIN grompp -f $RESOURCES/em.mdp -c $TOPOL/solv_ions.gro -p $TOPOL/topol.top -o em.tpr
$GMX_BIN mdrun -v -deffnm em

# --- STEP 4: NVT EQUILIBRATION ---
echo ">>> Running NVT Equilibration..."
cd $NVT; rm -rf ./charmm36.ff; ln -sf $BASE_DIR/scripts/charmm36.ff ./charmm36.ff
$GMX_BIN grompp -f $RESOURCES/nvt.mdp -c $EM/em.gro -r $EM/em.gro -p $TOPOL/topol.top -o nvt.tpr
$GMX_BIN mdrun -v -deffnm nvt

# --- STEP 5: NPT EQUILIBRATION ---
echo ">>> Running NPT Equilibration..."
cd $NPT; rm -rf ./charmm36.ff; ln -sf $BASE_DIR/scripts/charmm36.ff ./charmm36.ff
$GMX_BIN grompp -f $RESOURCES/npt.mdp -c $NVT/nvt.gro -r $NVT/nvt.gro -t $NVT/nvt.cpt -p $TOPOL/topol.top -o npt.tpr
$GMX_BIN mdrun -v -deffnm npt

# --- STEP 6: PRODUCTION MD TEST (Run briefly to verify GPU offloading) ---
echo ">>> Testing Production MD Offloading..."
cd $MD; rm -rf ./charmm36.ff; ln -sf $BASE_DIR/scripts/charmm36.ff ./charmm36.ff
$GMX_BIN grompp -f $RESOURCES/md.mdp -c $NPT/npt.gro -t $NPT/npt.cpt -p $TOPOL/topol.top -o md.tpr

# Run with full GPU offloading constraints natively on your local card
$GMX_BIN mdrun -v -deffnm md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -nsteps 5000

echo ">>> Local 5090 GPU Single Sequence Test Completed Successfully! <<<"