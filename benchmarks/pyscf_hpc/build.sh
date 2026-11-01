#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this build inside the supplied SLURM job}"
cd "$(dirname "$0")"
env_prefix=/resnick/groups/sharmagroup/bkhou/envs/pyscfad_env
cmake="$env_prefix/bin/cmake"

# Explicit CPU target works on the Intel and AMD nodes used by this project.
export LDFLAGS="-Wl,-rpath,$env_prefix/lib"
"$cmake" -S native/pyscf/lib -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic -fno-math-errno" \
  -DCMAKE_Fortran_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DBLAS_LIBRARIES="$env_prefix/lib/libopenblas.so" \
  -DENABLE_OPENMP=ON -DBUILD_MARCH_NATIVE=OFF \
  -DBUILD_LIBCINT=OFF -DBUILD_LIBXC=OFF -DBUILD_XCFUN=OFF \
  -DENABLE_SMD=OFF
"$cmake" --build build/native -j "${SLURM_CPUS_PER_TASK}"

mkdir -p integrals
cp -a native/pyscf integrals/
libcint_source=(sources/sunqm-libcint-*)
"$cmake" -S "${libcint_source[0]}" -B build/libcint \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -fno-fast-math" \
  -DCMAKE_INSTALL_PREFIX="$PWD/integrals/pyscf/lib/deps" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_MARCH_NATIVE=OFF -DWITH_F12=ON -DWITH_RANGE_COULOMB=ON \
  -DWITH_FORTRAN=OFF -DWITH_CINT2_INTERFACE=OFF -DWITH_4C1E=ON \
  -DMIN_EXPCUTOFF=20 -DKEEP_GOING=ON
"$cmake" --build build/libcint -j "${SLURM_CPUS_PER_TASK}"
"$cmake" --install build/libcint

ldd native/pyscf/lib/libcvhf.so
ldd integrals/pyscf/lib/libcgto.so
cp "${libcint_source[0]}/LICENSE" integrals/pyscf/lib/deps/LIBCINT_LICENSE
