## TO FINISH RUNS 5 AND 6 THAT WERE INTERRUPTED IN MD STAGE
# one gpu only, Robbie on the other

cd /home/admin/Rachel/output/APC_641_non/2/md/
or
cd /home/admin/Rachel/output/APC_4099_hot/3/md/

# Lock it to GPU 0, allocate your threads, and append to the checkpoint
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12

/home/admin/Documents/gromacs-2025.2/build/bin/gmx mdrun -v -deffnm md -nb gpu -pme gpu -bonded gpu -update gpu -dlb yes -ntmpi 1 -cpi md.cpt -append


/home/admin/Documents/gromacs-2025.2/build/bin/gmx editconf -f no-water.gro -o APC_637_hyd1.pdb