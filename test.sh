#!/bin/bash
set -e

# --- CORRECTED OUTPUT DIR VARIABLE ---
OUT_DIR="../output/APC_637_hot/3"
mkdir -p $OUT_DIR/topologies $OUT_DIR/em $OUT_DIR/nvt $OUT_DIR/npt $OUT_DIR/md

# --- STEP 1: TOPOLOGY ---
echo ">>> Running pdb2gmx..."

# Use -residuetypes to explicitly point to your custom file right here in scripts/
printf "1\n1\n0\n0\n4\n6\n4\n6\n" | gmx_mpi pdb2gmx \
  -f ../input/clean/clean_APC_637_hot_hyd.pdb \
  -o $OUT_DIR/topologies/processed.gro \
  -p $OUT_DIR/topologies/topol.top \
  -i $OUT_DIR/topologies/posre.itp \
  -ter

# --- STEP 2: SOLVATION & BOX ---
echo ">>> Defining Box Volume..."
gmx_mpi editconf \
  -f $OUT_DIR/topologies/processed.gro \
  -o $OUT_DIR/topologies/box.gro \
  -c \
  -d 1.0 \
  -bt cubic

echo ">>> Filling Box with Water..."
gmx_mpi solvate \
  -cp $OUT_DIR/topologies/box.gro \
  -cs spc216.gro \
  -o $OUT_DIR/topologies/solv.gro \
  -p $OUT_DIR/topologies/topol.top

# --- STEP 3: IONS ---
echo ">>> Generating Ions..."
gmx_mpi grompp \
  -f ../resources/ions.mdp \
  -c $OUT_DIR/topologies/solv.gro \
  -p $OUT_DIR/topologies/topol.top \
  -o $OUT_DIR/topologies/ions.tpr

echo "SOL" | gmx_mpi genion \
  -s $OUT_DIR/topologies/ions.tpr \
  -o $OUT_DIR/topologies/solv_ions.gro \
  -p $OUT_DIR/topologies/topol.top \
  -pname NA \
  -nname CL \
  -neutral

# --- STEP 4: ENERGY MINIMIZATION ---
echo ">>> Running Energy Minimization..."
gmx_mpi grompp \
  -f ../resources/em.mdp \
  -c $OUT_DIR/topologies/solv_ions.gro \
  -p $OUT_DIR/topologies/topol.top \
  -o $OUT_DIR/em/em.tpr

gmx_mpi mdrun -v -deffnm $OUT_DIR/em/em

# Clean up topology file paths if necessary
sed -i "s|$OUT_DIR/topologies/||g" $OUT_DIR/topologies/*.itp 2>/dev/null || true

# --- STEP 5: NVT EQUILIBRATION ---
echo ">>> Running NVT Equilibration..."
gmx_mpi grompp \
  -f ../resources/nvt.mdp \
  -c $OUT_DIR/em/em.gro \
  -r $OUT_DIR/em/em.gro \
  -p $OUT_DIR/topologies/topol.top \
  -o $OUT_DIR/nvt/nvt.tpr

gmx_mpi mdrun -ntomp 16 -v -deffnm $OUT_DIR/nvt/nvt

# --- STEP 6: NPT EQUILIBRATION ---
echo ">>> Running NPT Equilibration..."
gmx_mpi grompp \
  -f ../resources/npt.mdp \
  -c $OUT_DIR/nvt/nvt.gro \
  -r $OUT_DIR/nvt/nvt.gro \
  -t $OUT_DIR/nvt/nvt.cpt \
  -p $OUT_DIR/topologies/topol.top \
  -o $OUT_DIR/npt/npt.tpr

gmx_mpi mdrun -ntomp 16 -v -deffnm $OUT_DIR/npt/npt

# --- STEP 7: PRODUCTION MD TEST RUN ---
echo ">>> Testing Production MD Offloading..."
gmx_mpi grompp \
  -f ../resources/md.mdp \
  -c $OUT_DIR/npt/npt.gro \
  -t $OUT_DIR/npt/npt.cpt \
  -p $OUT_DIR/topologies/topol.top \
  -o $OUT_DIR/md/md.tpr

# Note: Removed 'srun' because you are running natively on the local workstation
gmx_mpi mdrun -v -deffnm $OUT_DIR/md/md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -nsteps 5000

$GMX_BIN mdrun -v -deffnm $OUT_DIR/md/md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -nsteps 5000
