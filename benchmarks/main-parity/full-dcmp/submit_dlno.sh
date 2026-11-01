#!/usr/bin/env bash
#SBATCH --job-name=mp2-parity-dlno
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=debug
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=dlno-slurm-%j.out
#SBATCH --error=dlno-slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this calculation directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK" OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
srun --cpu-bind=cores "$PYTHON" -u checks_driver.py candidate-precond dlno-unfrozen > dlno-unfrozen.log 2>&1
srun --cpu-bind=cores "$PYTHON" -u checks_driver.py candidate-precond dlno-frozen > dlno-frozen.log 2>&1
