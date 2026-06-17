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
"""
Restricted Hartree-Fock
"""
from functools import partial
import os
import time
import numpy
import jax
from jax import custom_vjp
from jax.interpreters import ad as jax_ad

from pyscf.data import nist
from pyscf.lib import alias, module_method, with_doc
from pyscf.scf import hf as pyscf_hf
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL

from pyscfad import config, config_update
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad import util
from pyscfad import ops
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.scf import _vhf
from pyscfad.scf import chkfile
from pyscfad.scf.diis import SCF_DIIS
from pyscfad.scipy.linalg import eigh
from pyscfad.tools.linear_solver import gen_gmres


def _profile_enabled():
    value = os.environ.get('PYSCFAD_PROFILE_BACKWARD_PHASES')
    return value is not None and value.strip().lower() in ('1', 'true', 'yes', 'on')


def _profile_msg(msg):
    if _profile_enabled():
        print(f'[profile][scf.hf] {msg}', flush=True)


def mo_response_eq78(dfock, ds1e, mo_energy_ref, mo_coeff_ref, deg_thresh=1e-9):
    eps = mo_energy_ref
    c = mo_coeff_ref
    cdc = c.T.conj() @ dfock @ c
    csc = c.T.conj() @ ds1e @ c
    x = cdc - csc * eps[None, :]

    eji = eps[None, :] - eps[:, None]
    safe_eji = np.where(np.abs(eji) > deg_thresh, eji, 1.0)
    wmat = np.where(np.abs(eji) > deg_thresh, 1.0 / safe_eji, 0.0)
    ident_mask = np.where(np.abs(eji) > deg_thresh, 0.0, 1.0)

    deps = np.diag(ident_mask * x)
    rhs = wmat * x - ident_mask * (0.5 * csc)
    dcoeff = c @ rhs
    return deps, dcoeff, rhs


@ops.custom_jvp
def mo_from_fock_eq78(fock, s1e, mo_energy_ref, mo_coeff_ref):
    del fock, s1e
    return mo_energy_ref, mo_coeff_ref


@mo_from_fock_eq78.defjvp
def _mo_from_fock_eq78_jvp(primals, tangents):
    fock, s1e, mo_energy_ref, mo_coeff_ref = primals
    dfock, ds1e, _, _ = tangents

    deps, dcoeff, _ = mo_response_eq78(dfock, ds1e, mo_energy_ref, mo_coeff_ref)
    return (mo_energy_ref, mo_coeff_ref), (deps, dcoeff)


def _stash_dfjk_mo_data(mf, mo_coeff, mo_occ):
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        return
    from pyscfad.df import _df_jk_opt
    _df_jk_opt.set_fast_exchange_dm_data(
        with_df,
        ops.to_numpy(mo_coeff),
        ops.to_numpy(mo_occ),
    )


def _scf_fixed_point(dm, mf, s1e, h1e):
    vhf = mf.get_veff(mf.mol, dm, s1e=s1e)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    _stash_dfjk_mo_data(mf, mo_coeff, mo_occ)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    del mo_energy, mo_occ
    return dm


def _scf(dm, mf, s1e, h1e, *,
         conv_tol, conv_tol_grad, diis=None,
         dump_chk=False, callback=None, log,
         e_tot, vhf, cput1):
    scf_conv = False
    fock_last = None
    mol = mf.mol

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, diis, fock_last=fock_last)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        _stash_dfjk_mo_data(mf, mo_coeff, mo_occ)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm-dm_last)
        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = log.timer(f'cycle = {cycle+1:d}', *cput1)

        if scf_conv:
            break

    return dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def _kernel_explicit_trace(mf, conv_tol=1e-10, conv_tol_grad=None,
                           dump_chk=False, dm0=None, callback=None,
                           conv_check=True):
    """Run the JAX-traceable SCF loop without implicit-diff wrapping."""
    if not config.scf_implicit_diff:
        raise RuntimeError(
            "Explicit-trace SCF requires pyscfad_scf_implicit_diff=True "
            "so the low-level SCF ops stay JAX-traceable."
        )

    log = logger.new_logger(mf)
    cput0 = log.get_t0()
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol, s1e=s1e)
    vhf = mf.get_veff(mol, dm, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)

    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        _stash_dfjk_mo_data(mf, mo_coeff, mo_occ)
        mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)
        return False, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    mf.pre_kernel(locals())
    cput1 = log.timer('initialize scf', *cput0)
    dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = _scf(
        dm, mf, s1e, h1e,
        conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis=mf_diis, dump_chk=dump_chk, callback=callback,
        log=log, e_tot=e_tot, vhf=vhf, cput1=cput1,
    )

    if scf_conv and conv_check:
        cput1 = log.get_t0()
        vhf = mf.get_veff(mol, dm, s1e=s1e)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        _stash_dfjk_mo_data(mf, mo_coeff, mo_occ)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        log.timer('extra cycle', *cput1)

    mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)
    log.timer('scf_cycle', *cput0)
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def _first_order_settings(mf_ref):
    with_df = getattr(mf_ref, 'with_df', None)
    return {
        'with_df': with_df is not None,
        'auxbasis': getattr(with_df, 'auxbasis', None) if with_df is not None else None,
        'df_incore': getattr(with_df, 'incore', True) if with_df is not None else True,
        'df_cderi_to_save': getattr(with_df, '_cderi_to_save', None) if with_df is not None else None,
        'df_has_outcore_cderi': (
            with_df._has_outcore_cderi_placeholder()
            if with_df is not None and hasattr(with_df, '_has_outcore_cderi_placeholder')
            else False
        ),
        'conv_tol': mf_ref.conv_tol,
        'conv_tol_grad': mf_ref.conv_tol_grad,
        'max_cycle': mf_ref.max_cycle,
        'init_guess': mf_ref.init_guess,
        'chkfile': mf_ref.chkfile,
        'verbose': mf_ref.verbose,
        'max_memory': mf_ref.max_memory,
        'direct_scf': mf_ref.direct_scf,
        'direct_scf_tol': mf_ref.direct_scf_tol,
        'conv_check': mf_ref.conv_check,
        'level_shift': mf_ref.level_shift,
        'damp': mf_ref.damp,
        'diis': bool(mf_ref.diis),
        'diis_space': mf_ref.diis_space,
        'diis_space_rollback': mf_ref.diis_space_rollback,
        'diis_damp': mf_ref.diis_damp,
        'diis_start_cycle': mf_ref.diis_start_cycle,
    }


def _build_mf_for_first_order(mol, settings):
    mf = RHF(mol)
    if settings['with_df']:
        from pyscfad import df as df_mod
        with_df = df_mod.DF(
            mol,
            auxbasis=settings['auxbasis'],
            incore=settings.get('df_incore', True),
        )
        with_df.max_memory = settings['max_memory']
        with_df.auxmol = df_mod.addons.make_auxmol(mol, settings['auxbasis'])
        cderi_to_save = settings.get('df_cderi_to_save', None)
        if cderi_to_save is not None:
            with_df._cderi_to_save = cderi_to_save
        if settings.get('df_has_outcore_cderi', False):
            with_df._cderi = numpy.zeros((0, 0))
            with_df._prefer_cderi_to_save = True
        mf = mf.density_fit(with_df=with_df)
    mf.conv_tol = settings['conv_tol']
    mf.conv_tol_grad = settings['conv_tol_grad']
    mf.max_cycle = settings['max_cycle']
    mf.init_guess = settings['init_guess']
    mf.chkfile = settings['chkfile']
    mf.verbose = settings['verbose']
    mf.max_memory = settings['max_memory']
    mf.direct_scf = settings['direct_scf']
    mf.direct_scf_tol = settings['direct_scf_tol']
    mf.conv_check = settings['conv_check']
    mf.level_shift = settings['level_shift']
    mf.damp = settings['damp']
    mf.diis = settings['diis']
    mf.diis_space = settings['diis_space']
    mf.diis_space_rollback = settings['diis_space_rollback']
    mf.diis_damp = settings['diis_damp']
    mf.diis_start_cycle = settings['diis_start_cycle']
    return mf


def _zero_cotangent_like(x):
    if x is None:
        return None
    return np.zeros_like(x)


def _is_zero_cotangent(x):
    return x is None or isinstance(x, jax_ad.Zero)


def _has_jvp_tracer(*xs):
    leaves = []
    for x in xs:
        if x is None:
            continue
        leaves.extend(jax.tree_util.tree_leaves(x))
    return any(_contains_jvp_tracer(x) for x in leaves)


def _has_tracer(*xs):
    leaves = []
    for x in xs:
        if x is None:
            continue
        leaves.extend(jax.tree_util.tree_leaves(x))
    return any(_contains_tracer(x) for x in leaves)


def _contains_tracer(x):
    if isinstance(x, jax.core.Tracer):
        return True
    for attr in ('primal', 'val'):
        if hasattr(x, attr):
            try:
                if _contains_tracer(getattr(x, attr)):
                    return True
            except AttributeError:
                pass
    return False


def _contains_jvp_tracer(x):
    if isinstance(x, jax_ad.JVPTracer):
        return True
    for attr in ('primal', 'val'):
        if hasattr(x, attr):
            try:
                if _contains_jvp_tracer(getattr(x, attr)):
                    return True
            except AttributeError:
                pass
    return False


def _static_bool(x, default):
    if isinstance(x, jax.core.Tracer):
        return default
    return bool(x)


def _scf_outputs_first_order_impl(mol, settings, dm0):
    mf = _build_mf_for_first_order(mol, settings)
    with config_update('pyscfad_scf_first_order_custom', False):
        if _has_jvp_tracer(mol, dm0):
            with config_update('pyscfad_scf_implicit_diff', False):
                e_tot = mf.kernel(dm0=dm0)
        else:
            e_tot = mf.kernel(dm0=dm0)
    return mf.converged, e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ


@partial(custom_vjp, nondiff_argnums=(1,))
def _scf_outputs_first_order(mol, settings, dm0):
    return _scf_outputs_first_order_impl(mol, settings, dm0)


def _scf_outputs_first_order_fwd(mol, settings, dm0):
    out = _scf_outputs_first_order_impl(mol, settings, dm0)
    res = (
        mol,
        dm0,
        np.asarray(out[2]),
        np.asarray(out[3]),
    )
    return out, res


def _scf_outputs_first_order_bwd(settings, res, cotangent):
    t_bwd = time.perf_counter()
    _profile_msg('first_order_bwd start')
    mol, dm0, mo_energy_fwd, mo_coeff_fwd = res
    _, bar_e_tot, bar_mo_energy, bar_mo_coeff, _ = cotangent

    if (
        _is_zero_cotangent(bar_e_tot)
        and _is_zero_cotangent(bar_mo_energy)
        and _is_zero_cotangent(bar_mo_coeff)
    ):
        _profile_msg(
            'first_order_bwd zero cotangent '
            f'{time.perf_counter() - t_bwd:.2f} s'
        )
        return _zero_cotangent_like(mol), _zero_cotangent_like(dm0)

    def _contract_outputs(mol_replay, mf, e_tot, mo_energy_raw, mo_coeff_raw, mo_occ_raw):
        dm = mf.make_rdm1(mo_coeff_raw, mo_occ_raw)
        s1e = mf.get_ovlp(mol_replay)
        h1e = mf.get_hcore(mol_replay, s1e=s1e)
        vhf = mf.get_veff(mol_replay, dm, s1e=s1e)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mo_from_fock_eq78(
            fock,
            s1e,
            mo_energy_fwd,
            mo_coeff_fwd,
        )

        out = 0.0
        if not _is_zero_cotangent(bar_e_tot):
            out = out + np.real(bar_e_tot * e_tot)
        if not _is_zero_cotangent(bar_mo_energy):
            out = out + np.real(np.vdot(bar_mo_energy, mo_energy))
        if not _is_zero_cotangent(bar_mo_coeff):
            out = out + np.real(np.vdot(bar_mo_coeff, mo_coeff))
        return out

    def weighted_outputs_explicit(mol):
        t = time.perf_counter()
        _profile_msg('first_order_bwd explicit SCF replay start')
        mf = _build_mf_for_first_order(mol, settings)
        with config_update('pyscfad_scf_first_order_custom', False):
            _, e_tot, mo_energy_raw, mo_coeff_raw, mo_occ_raw = _kernel_explicit_trace(
                mf,
                mf.conv_tol,
                mf.conv_tol_grad,
                dump_chk=False,
                dm0=dm0,
                callback=None,
                conv_check=mf.conv_check,
            )
        _profile_msg(
            'first_order_bwd explicit SCF replay returned '
            f'{time.perf_counter() - t:.2f} s'
        )
        t = time.perf_counter()
        out = _contract_outputs(
            mol, mf, e_tot, mo_energy_raw, mo_coeff_raw, mo_occ_raw
        )
        _profile_msg(
            'first_order_bwd contract_outputs traced '
            f'{time.perf_counter() - t:.2f} s'
        )
        return out

    _profile_msg('first_order_bwd jax.grad(weighted_outputs_explicit) start')
    mol_bar = jax.grad(weighted_outputs_explicit)(mol)
    _profile_msg(
        'first_order_bwd jax.grad(weighted_outputs_explicit) done '
        f'{time.perf_counter() - t_bwd:.2f} s'
    )
    return mol_bar, _zero_cotangent_like(dm0)


_scf_outputs_first_order.defvjp(
    _scf_outputs_first_order_fwd,
    _scf_outputs_first_order_bwd,
)


@with_doc(pyscf_hf.kernel.__doc__)
def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    log = logger.new_logger(mf)
    cput0 = log.get_t0()
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol, s1e=s1e)
    vhf = mf.get_veff(mol, dm, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        _stash_dfjk_mo_data(mf, mo_coeff, mo_occ)
        # hack for ROHF
        mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = log.timer('initialize scf', *cput0)
    # wrapped function for SCF iteration that is implicitly differentiable
    _scf_wrapped = make_implicit_diff(
        _scf,
        config.scf_implicit_diff,
        optimality_cond=_scf_fixed_point,
        solver=gen_gmres(),
        has_aux=True,
    )
    if config.scf_implicit_diff:
        e_tot = ops.stop_grad(e_tot)
        vhf = ops.stop_grad(vhf)
        if mf_diis is not None:
            mf_diis.Corth = ops.stop_grad(mf_diis.Corth)
    # NOTE if use implicit differentiation, only dm will have gradient.
    dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = _scf_wrapped(
        dm, mf, s1e, h1e,
        conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis=mf_diis, dump_chk=dump_chk, callback=callback,
        log=log, e_tot=e_tot, vhf=vhf, cput1=cput1,
    )

    _extra_cycle = False
    scf_conv_static = _static_bool(scf_conv, default=True)
    if config.scf_implicit_diff and (not conv_check or not scf_conv_static):
        log.warn('\tAn extra scf cycle is going to be run\n'
                 '\tin order to restore the mo_energy derivatives\n'
                 '\tmissing in implicit differentiation.')
        _extra_cycle = True

    if (scf_conv_static and conv_check) or _extra_cycle:
        cput1 = log.get_t0()
        vhf = mf.get_veff(mol, dm, s1e=s1e)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        _stash_dfjk_mo_data(mf, mo_coeff, mo_occ)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        log.info('Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

        log.timer('extra cycle', *cput1)

    # hack for ROHF
    mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)

    log.timer('scf_cycle', *cput0)
    del log
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


@partial(ops.jit, static_argnums=(2,3))
def _dot_eri_dm_s1(eri, dm, with_j, with_k):
    nao = dm.shape[-1]
    eri = eri.reshape((nao,)*4)
    dms = dm.reshape(-1,nao,nao)
    vj = vk = None
    if with_j:
        vj = np.einsum('ijkl,xji->xkl', eri, dms)
        vj = vj.reshape(dm.shape)
    if with_k:
        vk = np.einsum('ijkl,xjk->xil', eri, dms)
        vk = vk.reshape(dm.shape)
    return vj, vk


def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    if np.iscomplexobj(eri) or eri.size == nao**4:
        vj, vk = _dot_eri_dm_s1(eri, dm, with_j, with_k)
    else:
        if np.iscomplexobj(eri):
            raise NotImplementedError
        vj, vk = _vhf.incore(eri, dm, hermi, with_j, with_k)
    return vj, vk


@with_doc(pyscf_hf.energy_elec.__doc__)
def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm).real
    e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
    mf.scf_summary['e1'] = e1
    mf.scf_summary['e2'] = e_coul
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    return e1+e_coul, e_coul


@with_doc(pyscf_hf.make_rdm1.__doc__)
def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:,mo_occ>0]
    dm = (mocc*mo_occ[mo_occ>0]) @ mocc.conj().T
    return dm


@with_doc(pyscf_hf.level_shift.__doc__)
def level_shift(s, d, f, factor):
    dm_vir = s - s @ d @ s
    return f + dm_vir * factor


@with_doc(pyscf_hf.dip_moment.__doc__)
def dip_moment(mol, dm, unit='Debye', verbose=logger.NOTE, **kwargs):
    log = logger.new_logger(mol, verbose)

    if 'unit_symbol' in kwargs:
        log.warn('Kwarg "unit_symbol" was deprecated. It was replaced by kwarg '
                 'unit since PySCF-1.5.')
        unit = kwargs['unit_symbol']

    if getattr(dm, 'ndim', None) != 2:
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', np.asarray(ao_dip), dm).real

    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords  = np.asarray(mol.atom_coords())
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    mol_dip = nucl_dip - el_dip

    if unit.upper() == 'DEBYE':
        mol_dip *= nist.AU2DEBYE
        log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
    else:
        log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
    del log
    return mol_dip


@with_doc(pyscf_hf.get_fock.__doc__)
def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm, s1e=s1e)

    # hack for DFT
    vhf = getattr(vhf, 'vxc', vhf)

    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None:
        s1e = mf.get_ovlp()
    if dm is None:
        dm = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
        f = pyscf_hf.damping(f, fock_last, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf, f_prev=fock_last)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    nuc = mf.energy_nuc()
    mf.scf_summary['nuc'] = nuc.real

    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    return e_tot


class SCF(pytree.PytreeNode, pyscf_hf.SCF):
    """Subclass of :class:`pyscf.scf.hf.SCF` with traceable attributes.

    Attributes
    ----------
    mol : :class:`pyscfad.gto.Mole`
        :class:`pyscfad.gto.Mole` instance.
    mo_coeff : array
        MO coefficients.
    mo_energy : array
        MO energies.
    _eri : array
        Two-electron repulsion integrals.
    """
    DIIS = SCF_DIIS
    _dynamic_attr = ['mol', '_eri', 'mo_coeff', 'mo_energy']

    def get_hcore(self, mol=None, **kwargs):
        return super().get_hcore(mol)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        aosym = 's4' if config.moleintor_opt else 's1'
        if omega:
            with mol.with_range_coulomb(omega):
                _eri = mol.intor('int2e', aosym=aosym)
        elif _has_tracer(mol):
            _eri = mol.intor('int2e', aosym=aosym)
        else:
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym=aosym)
            _eri = self._eri

        vj, vk = dot_eri_dm(_eri, dm, hermi, with_j, with_k)
        return vj, vk

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        if mol is None:
            mol = self.mol
        dm0 = pyscf_hf.SCF.get_init_guess(self, mol.to_pyscf(), key, **kwargs)
        dm0 = np.asarray(dm0) #remove tags
        return dm0

    def scf(self, dm0=None, **kwargs):
        self.dump_flags()
        self.build(self.mol)

        use_first_order_custom = (
            config.scf_first_order_custom
            and dm0 is None
            and isinstance(self, RHF)
        )
        if config.scf_first_order_custom and dm0 is not None:
            logger.warn(
                self,
                'Custom first-order SCF response does not yet support '
                'explicit dm0 robustly; falling back to the standard implicit '
                'SCF derivative for this call.',
            )

        if use_first_order_custom:
            if not config.scf_implicit_diff:
                raise NotImplementedError(
                    "The experimental first-order SCF backward requires "
                    "pyscfad_scf_implicit_diff=True."
                )
            if self.max_cycle <= 0 and self.mo_coeff is not None:
                raise NotImplementedError(
                    "The experimental first-order SCF backward does not "
                    "support the skip-SCF / frozen-orbitals code path."
                )
            settings = _first_order_settings(self)
            scf_conv, self.e_tot, self.mo_energy, self.mo_coeff, self.mo_occ = \
                _scf_outputs_first_order(self.mol, settings, dm0)
            self.converged = _static_bool(scf_conv, default=True)
            nocc = self.mol.nelectron // 2
            mo_occ = numpy.zeros(self.mo_energy.shape[-1], dtype=float)
            mo_occ[:nocc] = 2.0
            self.mo_occ = mo_occ
        elif self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        self._finalize()
        return self.e_tot

    kernel = alias(scf, alias_name='kernel')

    def _eigh(self, h, s, *args, **kwargs):
        del args, kwargs
        return eigh(h, s)

    def energy_grad(self, dm0=None, mode='rev'):
        """Computing energy gradients w.r.t AO parameters.

        In principle, MO response is not needed, and it is sufficient to
        compute the gradient of the eigen decomposition with the converged
        density matrix. But this function is implemented as to trace the SCF iterations
        to show the difference between unrolling for loops and implicit differentiation.

        Parameters
        ----------
        dm0 : array, optional
            Input density matrix.
        mode : string, default='rev'
            Differentiating using the ``forward`` or ``reverse`` mode.

        Returns
        -------
        mol : :class:`pyscfad.gto.Mole`
            :class:`Mole` object that contains the gradients.

        Notes
        -----
        The attributes of the :class:`SCF` instance will not be modified.
        This function only works with the JAX backend.

        .. deprecated:: 0.2.0
            This function is deprecated since PySCFAD v0.2.0.
        """
        import jax
        import warnings
        warnings.warn(f'{self.__class__.__name__}.energy_grad is deprecated, '
                      'and will be removed in the future.',
                      FutureWarning, stacklevel=2)
        if dm0 is None:
            try:
                dm0 = self.make_rdm1()
            except TypeError:
                pass

        def hf_energy(self, dm0=None):
            self.reset()
            e_tot = self.kernel(dm0=dm0)
            return e_tot

        if mode.lower().startswith('rev'):
            jac = jax.grad(hf_energy)(self, dm0=dm0)
        else:
            if config.scf_implicit_diff:
                msg = """Forward mode differentiation is not available
                         when applying the implicit function differentiation."""
                raise KeyError(msg)
            jac = jax.jacfwd(hf_energy)(self, dm0=dm0)
        if hasattr(jac, 'cell'):
            return jac.cell
        else:
            return jac.mol

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        from pyscfad.df import df_jk # pylint: disable=cyclic-import
        return df_jk.density_fit(self, auxbasis, with_df, only_dfj)

    @with_doc(pyscf_hf.SCF.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.direct_scf:
            ddm = np.asarray(dm) - dm_last
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return vhf_last + vj - vk * .5
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5

    @with_doc(pyscf_hf.SCF.dip_moment.__doc__)
    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm =self.make_rdm1()
        return dip_moment(mol, dm, unit, verbose=verbose, **kwargs)

    def dump_chk(self, envs):
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['mo_energy'],
                             envs['mo_coeff'], envs['mo_occ'],
                             overwrite_mol=False)
        return self

    def energy_nuc(self):
        # recompute nuclear energy to trace it
        return self.mol.energy_nuc()

    def check_sanity(self):
        pass

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        return pyscf_hf.SCF.get_occ(self, ops.to_numpy(mo_energy))

    make_rdm1 = module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])
    energy_elec = energy_elec
    energy_tot = energy_tot
    get_fock = get_fock
    to_pyscf = util.to_pyscf


class RHF(SCF, pyscf_hf.RHF):
    def check_sanity(self):
        mol = self.mol
        if mol.nelectron != 1 and mol.spin != 0:
            logger.warn(self, 'Invalid number of electrons %d for RHF method.',
                        mol.nelectron)
        return SCF.check_sanity(self)

    @with_doc(pyscf_hf.RHF.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * .5
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * .5
            vhf += np.asarray(vhf_last)
        return vhf
