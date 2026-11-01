#!/usr/bin/env bash
#SBATCH --job-name=dcmp-native-parity
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this calculation directory}"
FORK_PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
MAIN_PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_main_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK" OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p scratch
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch"
export PYSCF_CONFIG_FILE="$PWD/pyscf_config.py"

srun --cpu-bind=cores "$FORK_PYTHON" -u driver.py baseline > baseline.out 2> baseline.err
srun --cpu-bind=cores "$FORK_PYTHON" -u driver.py candidate > candidate.out 2> candidate.err
srun --cpu-bind=cores "$FORK_PYTHON" -u driver.py candidate-precond > candidate-precond.out 2> candidate-precond.err
srun --cpu-bind=cores "$MAIN_PYTHON" -u driver.py main > main.out 2> main.err
