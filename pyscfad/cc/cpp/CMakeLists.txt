cmake_minimum_required(VERSION 3.18)
project(pyscfad_ccsd_kernels LANGUAGES CXX)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
pybind11_add_module(_ccsd_kernels NO_EXTRAS kernels.cpp)
target_compile_features(_ccsd_kernels PRIVATE cxx_std_14)
target_link_libraries(_ccsd_kernels PRIVATE OpenMP::OpenMP_CXX)
# BLAS is obtained through SciPy's Cython C API at import time, so this
# extension uses the same BLAS pool as SciPy, without another BLAS runtime.
install(TARGETS _ccsd_kernels LIBRARY DESTINATION pyscfad/cc)
