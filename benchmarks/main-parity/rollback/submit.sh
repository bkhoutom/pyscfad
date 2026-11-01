#!/usr/bin/env bash
#SBATCH --job-name=pyscfad-native-build
#SBATCH --partition=expansion
#SBATCH --account=sharmagroup
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pyscfad source directory}"
env_prefix=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env
export PATH="$env_prefix/bin:/usr/bin:/bin:$PATH"
unset PYTHONPATH LD_LIBRARY_PATH OPENBLAS_CORETYPE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK"
export OMP_MAX_ACTIVE_LEVELS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export LDFLAGS="-Wl,-rpath,$env_prefix/lib"

# Keep this fork's existing patched libcintad; rebuild its AD support libraries.
/usr/bin/cmake -S pyscfadlib/pyscfadlib -B build/pyscfadlib-openblas \
  -DPython_EXECUTABLE="$env_prefix/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DBLAS_LIBRARIES="$env_prefix/lib/libopenblas.so.0" \
  -DLAPACK_LIBRARIES="$env_prefix/lib/libopenblas.so.0" \
  -DENABLE_OPENMP=ON -DBUILD_MARCH_NATIVE=OFF -DBUILD_LIBCINT=OFF \
  > native-build.log 2>&1
/usr/bin/cmake --build build/pyscfadlib-openblas -j "$SLURM_CPUS_PER_TASK" \
  >> native-build.log 2>&1
python verify_native.py > native-linkage.json
python -m pytest -q pyscfadlib/test \
  pyscfad/scipy/test/test_linalg.py::test_eigh \
  tests/test_scf.py::test_rhf_nuc_grad \
  tests/test_scf.py::test_df_rhf_nuc_grad > native-tests.log 2>&1
