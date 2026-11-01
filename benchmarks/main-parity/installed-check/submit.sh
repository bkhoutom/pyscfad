#!/usr/bin/env bash
#SBATCH --job-name=fork-parity-installed
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=debug
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this calculation directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
srun --cpu-bind=cores "$PYTHON" -u driver.py > validation.log 2>&1
srun --cpu-bind=cores "$PYTHON" /resnick/groups/sharmagroup/bkhou/src/pyscfad/verify_native.py > linkage.json
"$PYTHON" -m pip check
