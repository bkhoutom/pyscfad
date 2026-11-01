#!/usr/bin/env bash
#SBATCH --job-name=dcmp-mkl-parity
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=debug
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this calculation directory}"
FORK_PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
MAIN_PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_main_env/bin/python
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYSCFAD_INTEGRAL_VJP_SCHEDULE=dynamic
mkdir -p scratch
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch" PYSCF_CONFIG_FILE="$PWD/pyscf_config.py"

# Each case starts fresh and uses the same allocated 16 cores.
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$MAIN_PYTHON" -u driver.py main benchmark main-a > main-a.out 2> main-a.err
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$FORK_PYTHON" -u driver.py candidate-precond benchmark fork-a > fork-a.out 2> fork-a.err
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$FORK_PYTHON" -u driver.py candidate-precond benchmark fork-b > fork-b.out 2> fork-b.err
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores "$MAIN_PYTHON" -u driver.py main benchmark main-b > main-b.out 2> main-b.err
"$FORK_PYTHON" compare.py > comparison.log

# Run numerical regressions after the timings to avoid competing for bandwidth.
export OMP_NUM_THREADS=1
srun --exact --ntasks=1 --cpus-per-task=16 --cpu-bind=cores "$FORK_PYTHON" -u checks_driver.py candidate-precond regressions > regressions-final.log 2>&1
srun --exact --ntasks=1 --cpus-per-task=16 --cpu-bind=cores "$FORK_PYTHON" -u checks_driver.py candidate-precond dlno-unfrozen > dlno-unfrozen.log 2>&1
srun --exact --ntasks=1 --cpus-per-task=16 --cpu-bind=cores "$FORK_PYTHON" -u checks_driver.py candidate-precond dlno-frozen > dlno-frozen.log 2>&1
