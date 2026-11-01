#!/usr/bin/env bash
#SBATCH --job-name=fork-native-blas
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this directory}"
PYTHON=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python
export PATH="${PYTHON%/*}:/usr/bin:/bin:$PATH"
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_THREAD_LIMIT="$OMP_NUM_THREADS" OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
mkdir -p scratch
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch"

bash build.sh > build.log 2>&1
"$PYTHON" run_variant.py original linkage > original-linkage.log
"$PYTHON" run_variant.py candidate linkage > candidate-linkage.log
OMP_NUM_THREADS=1 "$PYTHON" run_variant.py candidate tests > tests.log 2>&1
# These tests set max_memory=1000 MB; isolate them from prior JAX caches.
OMP_NUM_THREADS=1 "$PYTHON" run_variant.py original gradient-test None > original-gradient.log 2>&1
OMP_NUM_THREADS=1 "$PYTHON" run_variant.py candidate gradient-test None > gradient-unfrozen.log 2>&1
OMP_NUM_THREADS=1 "$PYTHON" run_variant.py candidate gradient-test 1 > gradient-frozen.log 2>&1
OMP_NUM_THREADS=2 "$PYTHON" run_variant.py candidate smoke > smoke.log 2>&1
srun --cpu-bind=cores "$PYTHON" -u run_variant.py original benchmark > original.out 2> original.err
srun --cpu-bind=cores "$PYTHON" -u run_variant.py candidate benchmark > candidate.out 2> candidate.err
