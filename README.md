<div align="left">
  <img src="https://fishjojo.github.io/pyscfad/_static/pyscfad_logo.svg" height="80px"/>
</div>

PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

* [Documentation](https://fishjojo.github.io/pyscfad/index.html)
* [Domain MP2 and DLNO-CCSD/CCSD(T): energies, forces and usage](pyscfad/dlno_stc/README.md)

DLNO-STC branch
--------------

This checkout provides domain MP2 and DLNO-CCSD/CCSD(T) energies and forces.
Standalone MP2 computes scalar energies without W; CC uses one W calculation
per physical fragment with smooth LIS frames. Solvers default to deterministic,
with stochastic solvers available explicitly.

Start with the [DLNO-STC user guide](pyscfad/dlno_stc/README.md) for the pipeline,
controls and examples. See [smooth LIS and triples theory](doc/source/user_guide/smooth_lis_triples.md)
for derivations and [SCF/DF performance notes](README_SERIAL_DF_SCF.md) for
shared buffers, streamed pullbacks and response caches.

For this implementation, install the current checkout (`pip install -e .`) and
follow the [native STC build instructions](pyscfad/dlno_stc/README.md#build-and-verification).

The CPU source build requires a C/C++17 toolchain with a shared OpenMP runtime.
Installation builds the deterministic CCSD/MP2/triples extension automatically;
it obtains BLAS from SciPy at runtime. Armadillo is needed only for the optional
legacy stochastic extension. Binary wheels may bundle the OpenMP runtime used
by their build toolchain; separate runtime copies are allowed.

With BLAS development libraries, Git, and the compiler available, install from
this checkout using:

```bash
python -m pip install 'pyscf>=2.14'
python -m pip install ./pyscfadlib
python -m pip install .
```

On macOS, install a compiler-compatible OpenMP runtime such as Homebrew `libomp`;
on Linux, use the runtime supplied with the selected compiler. Installed wheel
runtimes are left intact. Thread-budget scopes manage the loaded runtimes;
DF-J and SGX use fixed teams during their kernels and restore caller settings.
PySCFADlib retains its pinned, patched `libcintad`; its legacy C support kernels
use an external LP64 BLAS, while the deterministic C++ kernels and generalized
eigensolver use SciPy's BLAS/LAPACK interfaces.

MPI is optional. Install or load the system/cluster MPI implementation, then
use `python -m pip install '.[mpi]'` to add `mpi4py`. This extra does not install
MPICH, Open MPI, or Intel MPI runtime wheels. Serial calculations need no MPI.
The release and upstream installation commands below install upstream PySCFAD.

Installation
------------

* To install the latest release, run:
```
pip install pyscfad
```

* To install the CUDA compatible version, run:
```
pip install pyscfad[cuda12]
```

* To install the development version, run:
```
pip install git+https://github.com/fishjojo/pyscfad.git
```
The dependent C/C++ library `pyscfadlib` can be compiled from source following the instruction
[here](https://fishjojo.github.io/pyscfad/getting_started/install.html#installing-pyscfadlib).


`pyscfad` depends on
`jax>=0.7.1,<0.11`,
`pyscfadlib>=0.3`,
`pyscf>=2.14`,
`scipy>=1.11`, and
`pyscf-properties`.

Citing PySCFAD
--------------
The following paper should be cited in publications utilizing the PySCFAD program package:

[Differentiable quantum chemistry with PySCF for molecules and materials at the mean-field level and beyond](https://doi.org/10.1063/5.0118200), 
X. Zhang, G. K.-L. Chan, *J. Chem. Phys.*, **157**, 204801 (2022)
