<div align="left">
  <img src="https://fishjojo.github.io/pyscfad/_static/pyscfad_logo.svg" height="80px"/>
</div>

PySCF with Auto-differentiation
===============================

[![Build Status](https://github.com/fishjojo/pyscfad/workflows/CI/badge.svg)](https://github.com/fishjojo/pyscfad/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/fishjojo/pyscfad/branch/main/graph/badge.svg?token=NLSWGI0PLE)](https://codecov.io/gh/fishjojo/pyscfad)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6960749.svg)](https://doi.org/10.5281/zenodo.6960749)

* [Documentation](https://fishjojo.github.io/pyscfad/index.html)
* [Smooth DLNO-CCSD(T): modes, defaults, theory, and force examples](README_SMOOTH_CCSD_T.md)

DLNO-STC branch
--------------

This checkout includes D/W-selected LIS CCSD(T) with deterministic reference
forces and native asymmetric stochastic triples. Multiple physical fragments
and truncated LISs use anchored occupied/virtual frames with full Fock response.
See the [branch README](README_SMOOTH_CCSD_T.md) for supported modes and defaults,
32-thread examples, the 0.05 mEh triples SE target, and force-error reporting.
Library defaults and run-script overrides are listed separately.

For this implementation, install the current checkout (`pip install -e .`) and
follow the [native STC build instructions](pyscfad/dlno_stc/README.md#build-and-checks).
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
`pyscf>=2.3`, and
`pyscf-properties`.

Citing PySCFAD
--------------
The following paper should be cited in publications utilizing the PySCFAD program package:

[Differentiable quantum chemistry with PySCF for molecules and materials at the mean-field level and beyond](https://doi.org/10.1063/5.0118200), 
X. Zhang, G. K.-L. Chan, *J. Chem. Phys.*, **157**, 204801 (2022)
