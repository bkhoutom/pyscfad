#!/usr/bin/env bash
#SBATCH --job-name=xing-pyscf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=pyscf_scf.out.16.%j
#SBATCH --error=pyscf_scf.err.16.%j

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from caltech-cluster}"
PYTHON=${PYTHON_BIN:-/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env/bin/python}
module purge
module load openmpi/4.1.6-gcc-11.3.1-phbhsc7
export LD_LIBRARY_PATH="${PYTHON%/*}/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SCRATCH_DIR="/resnick/scratch/$USER/xing/${SLURM_JOB_ID:?Run via sbatch}"
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR" PYSCF_TMPDIR="$SCRATCH_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_THREAD_LIMIT="$OMP_NUM_THREADS" OMP_MAX_ACTIVE_LEVELS=1
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE MKL_DYNAMIC=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
# Keep BLAS single-threaded inside PySCF OpenMP regions.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

srun --cpu-bind=cores "$PYTHON" -u pyscf_scf.py
