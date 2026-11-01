#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside submit.sh}"
env_prefix=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env
export LDFLAGS="-Wl,-rpath,$env_prefix/lib"
/usr/bin/cmake -S candidate/pyscfadlib -B build \
  -DPython_EXECUTABLE="$env_prefix/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DBLAS_LIBRARIES="$env_prefix/lib/libopenblas.so.0" \
  -DLAPACK_LIBRARIES="$env_prefix/lib/libopenblas.so.0" \
  -DENABLE_OPENMP=ON -DBUILD_MARCH_NATIVE=OFF -DBUILD_LIBCINT=OFF
/usr/bin/cmake --build build -j "$SLURM_CPUS_PER_TASK"
