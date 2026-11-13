"""Optional external STC-CCSD amplitude solver for prepared DLNO fragments.

Only the private sampling frame is localized. By default the callback keeps
traced averaged amplitudes, integrals and the physical projector in one shared
joint occupied/virtual/auxiliary DF frame for projected CCSD and smooth triples.
``shared_local_frame=False``
restores amplitude backrotation to the incoming LIS frame. STC's total-energy
orbital cotangent filter is never used: a projected fragment functional need
not have that invariance.
"""
from collections.abc import Mapping
from functools import cached_property
import copy
import contextlib
import io
import math
import numbers
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy

from pyscfad import lib


_DEFAULTS = dict(
    localize=True, df_rotations=True, df_rotation_tol=5e-3, shared_local_frame=True,
    target_error=4e-4, error_tol=None, max_cycle=50, average=4,
    seed=0, diis_space=4, diis_shift=4., init_damp=None, gamma=.5,
    batch_size=1 << 20, tail_batch_size=None, nrun=8, init_nsamples=10000,
    std_tol=[.02, .05], max_nsamples_var=1 << 24, min_nsamples=100,
    vjp_nsamples=None, share_tables=True, iid_root=False, sampled_terms=None,
    unroll=None, flat=None, sampler=None, block_size=16,
    block_min_entries=1 << 25, block_store_P=False,
    grad_max_cycle=None, grad_average=None, grad_target_error=None, grad_seed=None,
    energy_tolerance=None, precision_pilot_replicas=4,
    precision_min_replicas=4, precision_max_replicas=32,
)

_MEH_ALIASES = {name + '_mEh': name for name in (
    'target_error', 'error_tol', 'grad_target_error', 'energy_tolerance')}
_PRECISION_CONTROLS = frozenset((
    'energy_tolerance', 'precision_pilot_replicas',
    'precision_min_replicas', 'precision_max_replicas'))
_FRAME_CONTROLS = frozenset(('localize', 'df_rotations', 'df_rotation_tol',
                             'shared_local_frame'))


def normalize_controls(controls):
    """Validate controls, converting explicit ``*_mEh`` inputs to internal Eh.

    ``energy_tolerance_mEh`` requests a molecular sampling standard error via
    independent projected-fragment replicas. The other mEh inputs control the
    local solver. Legacy unsuffixed inputs keep Eh units; never supply both
    unit forms of one setting. Returned controls use Eh and are idempotent.
    ``error_tol`` controls the external *total active-space* energy stopping
    rule.  It is not a projected-fragment or force accuracy guarantee.
    ``average=0`` selects the final iterate, as in the external driver.
    """
    if controls is None:
        controls = {}
    if not isinstance(controls, Mapping):
        raise TypeError('stc_cc_controls must be a mapping or None')
    controls = copy.deepcopy(dict(controls))
    for alias, canonical in _MEH_ALIASES.items():
        if alias not in controls:
            continue
        if canonical in controls:
            raise ValueError(f'Do not specify both {alias} and {canonical}')
        value = controls.pop(alias)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise TypeError(f'{alias} must be finite and real (mEh)')
            value = float(value) * 1e-3
        controls[canonical] = value
    unknown = set(controls) - set(_DEFAULTS)
    if unknown:
        raise ValueError('Unknown STC CC controls: ' + ', '.join(sorted(unknown)))
    out = copy.deepcopy(_DEFAULTS)
    out.update(copy.deepcopy(dict(controls)))
    optional_ints = {'tail_batch_size', 'vjp_nsamples', 'grad_max_cycle', 'grad_average',
                     'grad_seed', 'diis_space'}
    nonnegative = {'average', 'grad_average', 'seed', 'grad_seed', 'block_min_entries'}
    ints = optional_ints | {'max_cycle', 'average', 'seed', 'batch_size', 'nrun',
                           'init_nsamples', 'max_nsamples_var', 'min_nsamples',
                           'block_size', 'block_min_entries', 'precision_pilot_replicas',
                           'precision_min_replicas', 'precision_max_replicas'}
    for name in ints:
        value = out[name]
        if value is None and name in optional_ints:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f'{name} must be an integer')
        if value < (0 if name in nonnegative else 1):
            raise ValueError(f'{name} is outside its allowed range')
        if name in {'seed', 'grad_seed'} and value > 2**32 - 1:
            raise ValueError(f'{name} must fit in an unsigned 32-bit integer')
        out[name] = int(value)
    for name in {'target_error', 'error_tol', 'grad_target_error', 'diis_shift',
                 'init_damp', 'gamma', 'energy_tolerance', 'df_rotation_tol'}:
        value = out[name]
        if value is None and name in {'error_tol', 'grad_target_error', 'diis_shift', 'init_damp', 'energy_tolerance'}:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise TypeError(f'{name} must be finite and real')
        if name in {'target_error', 'error_tol', 'grad_target_error', 'energy_tolerance'} and value <= 0:
            raise ValueError(f'{name} must be positive (Hartree)')
        if name == 'df_rotation_tol' and value <= 0:
            raise ValueError('df_rotation_tol must be positive (dimensionless)')
        if name == 'diis_shift' and value < 0:
            raise ValueError('diis_shift must be nonnegative')
        if name in {'gamma', 'init_damp'} and not 0 <= value <= 1:
            raise ValueError(f'{name} must be between zero and one')
        out[name] = float(value)
    for name in {'localize', 'df_rotations', 'shared_local_frame', 'share_tables', 'iid_root', 'block_store_P', 'unroll', 'flat'}:
        if out[name] is None and name in {'unroll', 'flat'}:
            continue
        if not isinstance(out[name], bool):
            raise TypeError(f'{name} must be boolean')
    if out['diis_space'] is not None and out['diis_space'] < 2:
        raise ValueError('diis_space must be at least two, or None to disable DIIS')
    if out['average'] > out['max_cycle']:
        raise ValueError('average must not exceed max_cycle')
    if out['grad_average'] is not None and out['grad_average'] > (out['grad_max_cycle'] or out['max_cycle']):
        raise ValueError('grad_average must not exceed the adjoint iteration count')
    if out['sampler'] not in {None, 'alias', 'cdf', 'block'}:
        raise ValueError('sampler must be alias, cdf, block or None')
    terms = out['sampled_terms']
    if terms is not None:
        if not isinstance(terms, (list, tuple)) or not terms:
            raise ValueError('sampled_terms must select at least one of the 23 external path terms')
        if any(isinstance(t, bool) or not isinstance(t, numbers.Integral) or not 0 <= t < 23 for t in terms):
            raise ValueError('sampled_terms entries must be integers in [0, 23)')
        if len(set(terms)) != len(terms):
            raise ValueError('sampled_terms entries must be distinct')
        out['sampled_terms'] = list(map(int, terms))
    tol = out['std_tol']
    if not isinstance(tol, (list, tuple)) or len(tol) != 2 or any(
            isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(t) or t <= 0 for t in tol):
        raise ValueError('std_tol must contain two finite positive relative tolerances')
    out['std_tol'] = list(map(float, tol))
    if out['nrun'] < 2:
        raise ValueError('nrun must be at least two for a variance estimate')
    if out['init_nsamples'] < 2:
        raise ValueError('init_nsamples must be at least two for an unbiased sample variance')
    if out['tail_batch_size'] is not None and out['tail_batch_size'] > out['batch_size']:
        raise ValueError('tail_batch_size must not exceed batch_size')
    for name in ('precision_pilot_replicas', 'precision_min_replicas', 'precision_max_replicas'):
        if out[name] < 2:
            raise ValueError(f'{name} must be at least two to estimate sampling variance')
    if out['precision_max_replicas'] < out['precision_min_replicas']:
        raise ValueError('precision_max_replicas must be at least precision_min_replicas')
    return out



def _package_file_hashes(root):
    """Fingerprint source and native files, excluding generated runtime caches."""
    import hashlib
    from pathlib import Path
    root = Path(root)
    hashes = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part == '__pycache__' or part.startswith('torch_cache') for part in relative.parts):
            continue
        if path.is_file() and path.suffix in {'.py', '.so', '.pyd', '.dll', '.dylib'}:
            with path.open('rb') as stream:
                hashes[relative.as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashes


def external_implementation_identity():
    """Identify optional STC code for restart; called only for STC checkpoints.

    Content hashes capture dirty/editable Python installations and the actual
    installed native extension, without depending on git or installation paths.
    No identity is cached: changed source or a rebuilt binary invalidates resume.
    """
    import hashlib
    import importlib
    from importlib import metadata, util
    from pathlib import Path
    try:
        packages = {name: importlib.import_module(name) for name in ('stc_cc_jax', 'stc_cc')}
        native = importlib.import_module('stc_cc_jax.stc.native')
    except ImportError as exc:
        raise ImportError('STC checkpoints require stc_cc_jax and stc_cc; install the optional '
                          'stc-cc-jax CC dependencies in the active environment') from exc
    identity = {}
    for package, distribution in (('stc_cc_jax', 'stc-cc-jax'), ('stc_cc', 'stc_cc')):
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = 'uninstalled-source'
        identity[package] = dict(version=version,
            files=_package_file_hashes(Path(packages[package].__file__).resolve().parent))
    spec = util.find_spec('stc_cc_jax.stc._stc_tables')
    extension = None
    if spec is not None and spec.origin is not None:
        path = Path(spec.origin)
        with path.open('rb') as stream:
            extension = dict(filename=path.name, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
    identity['native_extension'] = extension
    identity['native_enabled'] = bool(native.available())
    identity['jax_version'] = jax.__version__
    identity['jax_backend'] = jax.default_backend()
    return identity


def _fragment_seed(seed, fragment_index):
    # SeedSequence ownership avoids reusing a sampling stream across fragments;
    # the external adjoint separately folds in its dedicated gradient stream.
    return int(numpy.random.SeedSequence([seed, fragment_index, 0x43435344]).generate_state(1)[0])


def _rotate_amplitudes(t1, t2, qo, qv):
    return (qo.T @ t1 @ qv,
            jnp.einsum('ijab,iI,jJ,aA,bB->IJAB', t2, qo, qo, qv, qv, optimize=True))


def _backrotate_amplitudes(t1, t2, qo, qv):
    return (qo @ t1 @ qv.T,
            jnp.einsum('IJAB,iI,jJ,aA,bB->ijab', t2, qo, qo, qv, qv, optimize=True))


def _working_rotations(mcc, eris, localize):
    no, nv = eris.Lov.shape[1:]
    if not localize:
        return jnp.eye(no), jnp.eye(nv)
    from pyscf import lo
    from pyscfad.ops import to_numpy
    coeff = numpy.asarray(to_numpy(eris.mo_coeff))
    overlap = numpy.asarray(to_numpy(mcc._scf.get_ovlp()))
    mol = mcc.mol.to_pyscf() if hasattr(mcc.mol, 'to_pyscf') else mcc.mol
    def rotation(c):
        if c.shape[1] < 2:
            return jnp.eye(c.shape[1])
        localized = lo.Boys(mol, c).kernel()
        return jax.lax.stop_gradient(jnp.asarray(c.T @ overlap @ localized))
    return rotation(coeff[:, :no]), rotation(coeff[:, no:])


def _sampling_rotations(mcc, eris, config):
    """Optimize a numerical gauge inside the *retained* LIS spaces.

    The upstream public wrapper reads molecular AO DF factors; its low-level
    DFAll entry point instead accepts exactly the compressed LIS Lov we need.
    Only the gauge is stopped, never the physical tensor transformed by it.
    """
    qo, qv = _working_rotations(mcc, eris, config['localize'])
    qa = jnp.eye(eris.Lov.shape[0], dtype=eris.Lov.dtype)
    if config['localize'] and config['df_rotations']:
        from stc_cc_jax.lo import _df_rotations
        from pyscfad.lib._threading import dense_blas_threads
        lov = numpy.asarray(jax.lax.stop_gradient(eris.Lov))
        if not numpy.isfinite(lov).all():
            raise ValueError('DF rotations require finite LIS factors')
        # DFAll's relative-cost stopping rule is undefined for zero factors.
        if numpy.any(lov):
            rov = numpy.einsum('xia,iI,aA->IAx', lov, numpy.asarray(qo),
                               numpy.asarray(qv), optimize=True)
            with dense_blas_threads(), contextlib.redirect_stdout(io.StringIO()):
                uo, uv, ua = _df_rotations(rov, config['df_rotation_tol'])
            qo, qv, qa = qo @ jnp.asarray(uo), qv @ jnp.asarray(uv), jnp.asarray(ua)
    return tuple(jax.lax.stop_gradient(rotation) for rotation in (qo, qv, qa))


class SamplingFrame:
    """Numerical rotations owned by one physical LIS evaluation.

    The driver shares this object only across pilots/replicas with identical
    preparation inputs, and discards it before the next fragment or geometry.
    Never save live ERIs or a response tape here: each replica transforms its
    own differentiable inputs. The upstream amplitude VJP reuses those ERIs.
    """
    def __init__(self):
        self._rotations = None

    def rotations(self, mcc, eris, config):
        if self._rotations is None:
            self._rotations = _sampling_rotations(mcc, eris, config)
        return self._rotations


def _make_local_eris(mcc, eris, qo, qv, rccsd, *, qa=None):
    """Transform the prepared domain-aware DF factors; never rebuild AO ERIs."""
    from pyscfad.scipy.linalg import eigh
    no, nv = eris.Lov.shape[1:]
    local = rccsd._LocalDFERIs()
    local.mol, local.nocc, local.nvir = mcc.mol, no, nv
    local.naux = eris.Lov.shape[0]
    local.Uaux = None  # already applied below; do not rotate twice upstream
    full_vv = lib.unpack_tril(eris.Lvv) if eris.Lvv.ndim == 2 else eris.Lvv
    local.Loo = jnp.einsum('xij,iI,jJ->xIJ', eris.Loo, qo, qo)
    local.Lov = jnp.einsum('xia,iI,aA->xIA', eris.Lov, qo, qv)
    local.Lvv = jnp.einsum('xab,aA,bB->xAB', full_vv, qv, qv)
    if qa is not None:
        local.Loo = jnp.einsum('xY,xIJ->YIJ', qa, local.Loo)
        local.Lov = jnp.einsum('xY,xIA->YIA', qa, local.Lov)
        local.Lvv = jnp.einsum('xY,xAB->YAB', qa, local.Lvv)
    q = jnp.zeros((no + nv, no + nv)).at[:no, :no].set(qo).at[no:, no:].set(qv)
    local.mo_coeff = jax.lax.stop_gradient(jnp.asarray(eris.mo_coeff) @ q)
    local.fock = q.T @ eris.fock @ q
    local.oooo = jnp.einsum('xij,xkl->ijkl', local.Loo, local.Loo)
    local.ovoo = jnp.einsum('xia,xjk->iajk', local.Lov, local.Loo)
    local.ovov = jnp.einsum('xia,xjb->iajb', local.Lov, local.Lov)
    local.oovv = jnp.einsum('xij,xab->ijab', local.Loo, local.Lvv)
    eo, local.Uocc = eigh(local.fock[:no, :no])
    ev, local.Uvir = eigh(local.fock[no:, no:])
    local.mo_energy = jnp.concatenate((eo, ev))
    local.t2_mp2 = rccsd._mp2_amplitudes(local.Lov, local.Uocc, local.Uvir,
                                       rccsd._eia(local.mo_energy, no))
    mp2 = jax.lax.stop_gradient(local.t2_mp2)
    local.O2 = 2 * mp2 - mp2.transpose(1, 0, 2, 3)
    return local


def _numeric(value):
    return float(numpy.asarray(jax.lax.stop_gradient(value)))


class _SharedFrameERIs:
    """Downstream view; leave the external amplitude solver's pytree intact.

    Smooth triples expects packed symmetric DF factors. Its cached-integral
    path also needs ovvv, constructed only on first use, in this same frame.
    All arrays retain their physical integral/Fock derivatives.
    """
    def __init__(self, local):
        for name in ('fock', 'Loo', 'Lov', 'ovov', 'ovoo', 'mo_coeff', 'nocc', 'nvir', 'naux'):
            setattr(self, name, getattr(local, name))
        self._dense_lvv = local.Lvv

    @cached_property
    def Lvv(self):
        return lib.pack_tril(self._dense_lvv)

    @cached_property
    def ovvv(self):
        full = jnp.einsum('Pia,Pbc->iabc', self.Lov, self._dense_lvv)
        return lib.pack_tril(full.reshape(self.nocc * self.nvir, self.nvir, self.nvir)).reshape(
            self.nocc, self.nvir, -1)


def make_solver(controls, fragment_index=0, *, sampling_frame=None):
    """Create an independent fragment callback; optional STC imports are lazy."""
    config = normalize_controls(controls)
    if isinstance(fragment_index, bool) or not isinstance(fragment_index, numbers.Integral) or fragment_index < 0:
        raise ValueError('fragment_index must be a nonnegative integer')
    seed = _fragment_seed(config['seed'], int(fragment_index))
    grad_seed = None if config['grad_seed'] is None else _fragment_seed(config['grad_seed'], int(fragment_index))

    def solver(mcc, eris, t1, t2, *, prjlo):
        try:
            from stc_cc_jax.cc import grad, rccsd
            # Term lists live in the separate optional stc_cc distribution.
            # Force its lazy loading here so dependency failures get context.
            rccsd.terms.opt_terms()
        except ImportError as exc:
            raise ImportError('STC CCSD requires the optional stc_cc_jax package and its dependencies; '
                              'install stc-cc-jax in the active environment before selecting cc_solver="stc"') from exc
        from pyscfad.lno.ccsd import ccsd_fragment_energy, FragmentAmplitudeResult
        # A directly used callback is safe to call with new physical inputs;
        # only the replica driver supplies an evaluation-scoped shared frame.
        frame = SamplingFrame() if sampling_frame is None else sampling_frame
        qo, qv, qa = frame.rotations(mcc, eris, config)
        use_aux = config['localize'] and config['df_rotations']
        local = _make_local_eris(mcc, eris, qo, qv, rccsd, qa=qa if use_aux else None)
        stc_config = {k: v for k, v in config.items()
                      if k not in _FRAME_CONTROLS and k not in _PRECISION_CONTROLS}
        stc_config.update(seed=seed, grad_seed=grad_seed, semicanonical=True,
                          mo_response_no_oo_vv_rotation=False)
        cc = rccsd.RCCSD(mcc._scf, frozen=mcc.frozen, mo_coeff=mcc.mo_coeff, **stc_config)
        cc.nocc, cc.nmo = local.nocc, local.nocc + local.nvir
        guess = _rotate_amplitudes(t1, t2, qo, qv)
        # Numerical callback diagnostics must not enter the differentiated graph.
        physical = SimpleNamespace(fock=jax.lax.stop_gradient(eris.fock),
                                   ovov=jax.lax.stop_gradient(eris.ovov))
        projector = jax.lax.stop_gradient(prjlo)
        projected = [_numeric(ccsd_fragment_energy(physical, jax.lax.stop_gradient(t1),
                                                   jax.lax.stop_gradient(t2), projector))]
        sampling_physical = SimpleNamespace(fock=jax.lax.stop_gradient(local.fock),
                                            ovov=jax.lax.stop_gradient(local.ovov))
        sampling_projector = jax.lax.stop_gradient(prjlo @ qo)
        def record(state):
            # Covariant projected functional: avoid a full t2 backrotation at
            # every iteration; the legacy return path alone needs it.
            projected.append(_numeric(ccsd_fragment_energy(sampling_physical,
                state['t1'], state['t2'], sampling_projector)))
        cc.callback = record
        vec, result = grad.amplitudes(cc, local, *guess)
        a, b = rccsd.vector_to_amplitudes(vec, cc.nmo, cc.nocc)
        shared = config['shared_local_frame']
        if shared:
            returned = (a, b)
            # This is the physical projector, not the stopped diagnostic copy.
            returned_projector = prjlo @ qo
            returned_eris = _SharedFrameERIs(local)
            diagnostic_eris, diagnostic_projector = sampling_physical, sampling_projector
        else:
            returned = _backrotate_amplitudes(a, b, qo, qv)
            diagnostic_eris, diagnostic_projector = physical, projector
        window = projected[cc.burnin:]
        count = len(window)
        # Descriptive independent-iterate SE only: correlated iterates and finite
        # burn-in may make it optimistic.  This is not the total local error bar.
        se = float(numpy.std(window, ddof=1) / math.sqrt(count)) if count > 1 else None
        diagnostic = dict(
            backend='stc', fragment_index=int(fragment_index), seed=seed,
            grad_seed=seed if grad_seed is None else grad_seed,
            localize=config['localize'],
            localization_method=('Boys+DF' if use_aux else 'Boys') if config['localize'] else 'identity',
            auxiliary_localized=use_aux, sampling_naux=int(local.naux),
            shared_local_frame=shared,
            energy_frame='stc_local' if shared else 'lis',
            controls=copy.deepcopy(config), converged=bool(result[0]),
            termination_mode='fixed_iterations' if config['error_tol'] is None else 'energy_stderr',
            convergence_note=('Fixed iteration budget completed; converged does not certify CC residual convergence.'
                              if config['error_tol'] is None else
                              'converged denotes the total local mean-energy stderr stopping rule, not CC residual convergence.'),
            iterations=len(cc.cycles), burnin=int(cc.burnin), averaged_count=count,
            local_mean_iterate_energy=float(result[1]), local_mean_iterate_stderr=float(cc.e_corr_err),
            projected_iteration_energies=projected,
            projected_mean_iterate_energy=float(numpy.mean(window)),
            projected_mean_iterate_stderr_iid=se,
            projected_energy_at_mean_amplitudes=_numeric(ccsd_fragment_energy(diagnostic_eris,
                jax.lax.stop_gradient(returned[0]), jax.lax.stop_gradient(returned[1]), diagnostic_projector)),
            projected_stderr_note='Observed IID-iterate SE; ignores autocorrelation and burn-in bias; not a force bound.',
            nsamples_forward_production=int(sum(numpy.asarray(c.get('nsamples', 0)).sum() for c in cc.cycles)),
            sample_count_note='Forward production draws only; excludes variance pilots and adjoint sampling.',
        )
        if shared:
            return FragmentAmplitudeResult(*returned, returned_eris, returned_projector, diagnostic)
        return *returned, diagnostic

    solver.seed = seed
    solver.controls = copy.deepcopy(config)
    return solver
