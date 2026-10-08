# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import jax
import numpy
from pyscf.data.nist import BOHR
from pyscfad import numpy as np
from pyscfad import gto, scf
from pyscfad.lo import boys
from pyscfad.lo.boys import dipole_integral
from pyscfad import config_update


@pytest.mark.parametrize("gradient_tolerance", [1e-7, 1e-9])
def test_boys_replay_reaches_requested_tight_stationarity(gradient_tolerance):
    from scipy.linalg import expm

    mol = gto.Mole(
        atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
        basis="sto-3g", verbose=0,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    # Build a stationary reference, then mimic reference-frame replay with a
    # tiny displacement. The original CIAH cutoffs discard the small steps
    # needed to satisfy the explicitly requested outer gradient tolerance.
    reference = boys.Boys(mol, numpy.asarray(mf.mo_coeff[:, 1:5]))
    reference.conv_tol = 1e-14
    reference.conv_tol_grad = 1e-11
    reference.ah_conv_tol = 1e-16
    reference.ah_lindep = 1e-24
    coeff = reference.kernel()
    perturbation = numpy.random.default_rng(713).normal(size=(4, 4))
    perturbation -= perturbation.T.copy()
    displaced = coeff @ expm(1e-8 * perturbation)

    localized = boys.boys(
        mol, displaced, init_guess=numpy.eye(4), conv_tol=1e-12,
        conv_tol_grad=gradient_tolerance,
    )
    check = boys.Boys(mol, numpy.asarray(localized))
    actual_gradient = numpy.linalg.norm(check.get_grad(numpy.eye(4)))
    assert actual_gradient < gradient_tolerance

def cost_function(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
    mo_coeff = boys.boys(mol, orbocc, init_guess='atomic')

    dip = dipole_integral(mol, mo_coeff)
    r2 = mol.intor_symmetric('int1e_r2')
    r2 = np.einsum('pi,pi->', mo_coeff, np.dot(r2, mo_coeff))
    val = r2 - np.einsum('xii,xii->', dip, dip)
    return val

@pytest.fixture
def get_mol():
    with config_update('pyscfad_scf_implicit_diff', True):
        mol = gto.Mole()
        mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
        mol.basis = '631G'
        mol.verbose = 0
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        yield mol

def test_boys_cost_nuc_grad(get_mol):
    mol = get_mol
    g0 = jax.grad(cost_function)(mol).coords

    mol = mol.set_geom_('O 0. 0.  0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol)
    mol = mol.set_geom_('O 0. 0. -0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[0,2]-g1) < 1e-4

    mol.set_geom_('O 0. 0. 0.; H 0. , -0.756 , 0.587; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol)
    mol.set_geom_('O 0. 0. 0.; H 0. , -0.758 , 0.587; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[1,1]-g1) < 1e-4

    mol.set_geom_('O 0. 0. 0.; H 0. , -0.757 , 0.588; H 0. , 0.757 , 0.587')
    f1 = cost_function(mol)
    mol.set_geom_('O 0. 0. 0.; H 0. , -0.757 , 0.586; H 0. , 0.757 , 0.587')
    f2 = cost_function(mol)
    g1 = (f1 - f2) / (0.002 / BOHR)
    assert abs(g0[1,2]-g1) < 1e-4
