#!/usr/bin/env bash
set -euo pipefail
env_prefix=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env
mkl_lib=/resnick/software9/spack/opt/spack/linux-rhel9-skylake_avx512/none-none/intel-oneapi-mkl-2024.2.2-vmnjeo2f64yeg54h3dt3bvwmd2pt7m54/mkl/2024.2/lib
blas_libraries="$mkl_lib/libmkl_intel_lp64.so;$mkl_lib/libmkl_sequential.so;$mkl_lib/libmkl_core.so;-lpthread;-lm;-ldl"
export LDFLAGS="-Wl,--no-as-needed -Wl,-rpath,$mkl_lib -Wl,-rpath,$env_prefix/lib"
/usr/bin/cmake -S pyscfadlib -B build \
  -DPython_EXECUTABLE="$env_prefix/bin/python" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DBLAS_LIBRARIES="$blas_libraries" -DLAPACK_LIBRARIES="$blas_libraries" \
  -DENABLE_OPENMP=ON -DBUILD_MARCH_NATIVE=OFF -DBUILD_LIBCINT=OFF
/usr/bin/cmake --build build -j "${SLURM_CPUS_PER_TASK:-4}"
