#!/usr/bin/env bash
#SBATCH --job-name=dcmp-int3c-schedule
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from this diagnostic directory}"
env_prefix=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env
native=/resnick/groups/sharmagroup/bkhou/src/pyscfad-perf-mp2/benchmarks/native-mp2/candidate/pyscfadlib
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK"
export OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE
unset OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_DYNAMIC=FALSE
mkdir -p scratch
export TMPDIR="$PWD/scratch" PYSCF_TMPDIR="$PWD/scratch"

for schedule in static dynamic; do
    /usr/bin/gcc -shared -fPIC -Wall -O3 -DNDEBUG -march=x86-64-v3 \
        -mtune=generic -msse3 -std=gnu99 -fopenmp -Iinclude \
        "$schedule.c" -o "libschedule_$schedule.so" \
        -L"$native" -lcgto_ad -lvjp_util -L"$native/thirdparty/lib" -lcintad -lm \
        -Wl,-z,defs -Wl,-rpath,"$native" -Wl,-rpath,"$native/thirdparty/lib" \
        -Wl,-rpath,"$env_prefix/lib"
done
srun --exact --ntasks=1 --cpus-per-task=16 --distribution=block:block --cpu-bind=cores \
    "$env_prefix/bin/python" -u driver.py > results.out 2> results.err
