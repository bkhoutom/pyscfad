"""Bounded numerical handoff from selection to same-reference energy calls.

Only the latest selection is retained, with weak owners and no response tape.
Nothing is attached to selections, restart metadata, or the SCF pytree. Mutable
in-memory integral buffers cannot attest unchanged contents cheaply and stay
on the ordinary reconstruction path.
"""
from dataclasses import dataclass
import os
import weakref

import jax
import numpy

from ._restart import scientific_digest, RestartSerializationError


_latest = None
_STANDARD_OPERATOR_MODULES = frozenset((
    'pyscf.scf.hf', 'pyscfad.scf.hf', 'pyscf.df.df', 'pyscfad.df.df',
    'pyscfad.df.df_jk', 'pyscf.gto.mole', 'pyscfad.gto.mole',
))


def _molecular_state(mol):
    if mol is None:
        return None
    return ({name: getattr(mol, name, None) for name in (
        '_atm', '_bas', '_env', '_basis', 'basis', 'cart', 'charge', 'spin',
        'ecp', '_ecp', '_ecpbas', 'pseudo', '_pseudo', 'coords', 'exp',
        'ctr_coeff', 'r0', 'nucmod', 'nucprop')}, mol.atom_coords())


def _source_state(source):
    if source is None:
        return ('none',), None
    if isinstance(source, jax.Array):
        return ('array', id(source), source.shape, str(source.dtype)), weakref.ref(source)
    if isinstance(source, (str, bytes, os.PathLike)):
        path = os.path.realpath(source)
        stat = os.stat(path)
        return ('file', path, stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns), None
    # Sampling a writable NumPy CDERI would silently miss in-place edits.
    # Read-only views also have potentially writable aliases. Never scan or
    # copy a multi-gigabyte integral buffer just to enable this optimization.
    raise TypeError('mutable or unsupported integral source')


def _state(mf, static):
    fitting = mf.with_df
    if getattr(mf, '_sgx_backend', None) is not None:
        raise TypeError('SGX preparation has a separate mutable backend')
    # Changes to decorated SCF operators can change Fock even when orbitals,
    # basis descriptors and the fitting source remain numerically identical.
    methods, method_refs = [], []
    auxiliary = getattr(fitting, 'auxmol', None)
    integral_methods = ('intor', 'intor_symmetric', 'atom_coords')
    for owner, names in ((mf, ('get_fock', 'get_ovlp', 'get_hcore', 'get_veff',
                              'make_rdm1', 'get_jk', 'get_j', 'get_k')),
                         (fitting, ('get_jk', 'loop', 'build', '_get_cderi_source')),
                         (mf.mol, integral_methods), (auxiliary, integral_methods)):
        if owner is None:
            continue
        for name in names:
            method = getattr(owner, name, None)
            if method is None:
                continue
            function = getattr(method, '__func__', method)
            if getattr(function, '__module__', None) not in _STANDARD_OPERATOR_MODULES:
                # Decorated references (QMMM, external fields, etc.) can depend
                # on extra mutable state beyond the ordinary RHF/DF inputs.
                raise TypeError('unsupported decorated scientific operator')
            if (name in vars(owner)
                    and function is not getattr(type(owner), name, None)):
                # Arbitrary closures can change captured numerical inputs
                # without replacing themselves. Their state is not attestable.
                raise TypeError('custom instance SCF/DF operator')
            methods.append((name, id(function)))
            method_refs.append(weakref.ref(function))
    get_source = getattr(fitting, '_get_cderi_source', None)
    source = get_source() if get_source is not None else fitting._cderi
    source_state, source_ref = _source_state(source)
    eri_state, eri_ref = _source_state(getattr(mf, '_eri', None))
    identities = (id(mf.mol), id(fitting), id(auxiliary))
    state = (
        static, identities, tuple(methods),
        {name: getattr(mf, name, None) for name in (
            'mo_coeff', 'mo_energy', 'mo_occ', 'level_shift', 'damp',
            'direct_scf', 'direct_scf_tol', 'only_dfj')},
        _molecular_state(mf.mol),
        {name: getattr(fitting, name, None) for name in (
            'auxbasis', 'incore', 'linear_dep_threshold', '_dataname',
            '_prefer_cderi_to_save')},
        _molecular_state(auxiliary),
        source_state, eri_state,
    )
    references = (source_ref, eri_ref, weakref.ref(mf.mol), weakref.ref(fitting),
                  None if auxiliary is None else weakref.ref(auxiliary), *method_refs)
    return scientific_digest(state), references


@dataclass
class _Preparation:
    mf: object
    static: object
    digest: str
    sources: tuple
    common: object
    weak: tuple


def _discard(owner):
    global _latest
    if _latest is not None and (owner is _latest.mf or owner is _latest.static):
        _latest = None


def remember(mf, static, common, *, verify_stationary=False):
    """Keep one immutable numerical preparation for its exact living owners."""
    global _latest
    _latest = None
    # Detached selection callbacks already have a driver-owned response path.
    if hasattr(mf, '_stc_selection_response'):
        return
    try:
        digest, sources = _state(mf, static)
        owners = weakref.ref(mf, _discard), weakref.ref(static, _discard)
    except (TypeError, ValueError, OSError, RestartSerializationError):
        return
    if verify_stationary and common.iao_coeff.shape[1] > 1:
        # Supplied targets can pass a loose stationarity tolerance yet still
        # move on legacy replay. Certify the actual replay frame with existing
        # S/F; supplied targets still pay for only their one required refinement.
        from .domain import _rebuild_boys_targets
        try:
            refined = _rebuild_boys_targets(
                mf, common.occupied_coeff, static, fock=common.fock)
        except (RuntimeError, ValueError):
            # Optional reuse must not move a replay-only error into selection.
            return
        if not numpy.allclose(refined, common.iao_coeff, atol=1e-12, rtol=0.):
            return
    from ._pairs import weak_pair_indices, weak_pair_weight
    pairs = tuple(weak_pair_indices(static))
    weak = sum((weak_pair_weight(static, left, right)
                * static.weak_pair_energy[left, right] for left, right in pairs), 0.)
    # Copies detach supplied target frames and NumPy-backed reference arrays.
    # Read-only leaves prevent accidental in-place mutation between calls.
    def snapshot(value):
        result = numpy.array(jax.device_get(value), copy=True)
        result.flags.writeable = False
        return result
    common = jax.tree_util.tree_map(snapshot, common)
    _latest = _Preparation(*owners, digest, sources, common, (pairs, weak))


def lookup(mf, static):
    """Reuse only exact identities with unchanged small scientific inputs."""
    receipt = _latest
    if receipt is None or receipt.mf() is not mf or receipt.static() is not static:
        return None
    if any(source is not None and source() is None for source in receipt.sources):
        return None
    try:
        digest, _ = _state(mf, static)
    except (TypeError, ValueError, OSError, RestartSerializationError):
        return None
    if digest != receipt.digest:
        return None
    return receipt.common, receipt.weak
