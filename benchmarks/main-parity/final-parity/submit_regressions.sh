#!/usr/bin/env bash
#SBATCH --job-name=fork-parity-checks
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=checks-slurm-%j.out
#SBATCH --error=checks-slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this calculation directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK" OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
mkdir -p scratch-tests
export TMPDIR="$PWD/scratch-tests" PYSCF_TMPDIR="$PWD/scratch-tests"
export PYSCF_CONFIG_FILE="$PWD/pyscf_config.py"
srun --cpu-bind=cores "$PYTHON" -u checks_driver.py candidate-precond regressions > regressions-final.log 2>&1
srun --cpu-bind=cores "$PYTHON" -u checks_driver.py candidate-precond dlno-unfrozen > dlno-unfrozen.log 2>&1
srun --cpu-bind=cores "$PYTHON" -u checks_driver.py candidate-precond dlno-frozen > dlno-frozen.log 2>&1
