"""Final SCF matrices share their original tape with domain correlation."""

from collections import Counter
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, df, gto, scf
from pyscfad.dlno_stc import domain, driver
from pyscfad.scf import _response_data
from pyscfad.tools import linear_solver
from pyscfad.dlno_stc.test.test_fragment_df_inputs import molecular_case


@pytest.mark.parametrize('outcore', [False, True])
def test_joint_scf_outputs_preserve_values_response_and_avoid_rebuild(
        tmp_path, monkeypatch, outcore):
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61',
                   basis='sto-3g', verbose=0, max_memory=10000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    path = str(tmp_path / 'scf.h5')
    counts = Counter()
    for owner, name in ((scf.hf.SCF, 'get_fock'), (scf.hf.SCF, 'get_hcore'),
                        (scf.hf.SCF, 'get_ovlp')):
        original = getattr(owner, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    def source(m):
        obj = df.DF(m, auxbasis='weigend', incore=not outcore)
        obj.auxmol = df.addons.make_auxmol(m, obj.auxbasis)
        # An independent auxiliary coordinate map must keep its own response.
        obj.auxmol.coords = obj.auxmol.coords * 1.03
        return obj

    if outcore:
        obj = source(mol)
        obj._cderi_to_save = path
        obj.build()

    def build(m):
        obj = source(m)
        if outcore:
            obj.attach_outcore_cderi(path)
        mf = scf.RHF(m).density_fit(with_df=obj)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with linear_solver.eager_gmres():
            mf.kernel()
        return mf

    seeds = jnp.asarray(np.random.default_rng(81).normal(size=(mol.nao, mol.nao)))

    def objective(mf, data):
        overlap, fock = data
        return (mf.e_tot + .01 * jnp.sum(fock * seeds)
                + .007 * jnp.sum(overlap * seeds)
                + .02 * jnp.sum(mf.mo_energy) + .003 * jnp.sum(mf.mo_coeff))

    values, work, data_values = [], [], []
    with config_update('pyscfad_moleintor_opt', True):
        for shared in (False, True):
            counts.clear()
            mf, pullback = driver._initialize_scf(
                mol, build, combine_df=True, share_scf_data=shared)
            work.append(counts.copy())
            if shared:
                data = pullback.scf_data
                assert data is not None
                assert counts == work[-1], 'joint outputs must not rebuild S, hcore, or F'
            else:
                data, data_pull = jax.vjp(domain.build_scf_data, mf)
                assert counts['get_fock'] == work[-1]['get_fock'] + 1
            data_values.append(data)
            value, objective_pull = jax.vjp(objective, mf, data)
            with driver._global_df_response({'mf': mf}):
                mf_bar, data_bar = objective_pull(jnp.ones_like(value))
                if shared:
                    mol_bar, = pullback(mf_bar, data_bar)
                else:
                    extra, = data_pull(data_bar)
                    mf_bar = jax.tree.map(driver._add_cotangent, mf_bar, extra)
                    mol_bar, = pullback(mf_bar)
                jax.block_until_ready(mol_bar)
            values.append((value, mol_bar.coords))
            assert _response_data._capture.get() is None
        assert work[0] == work[1]
        for actual, expected in zip(data_values[1], data_values[0]):
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
        np.testing.assert_allclose(values[1][0], values[0][0], atol=2e-12, rtol=0)
        np.testing.assert_allclose(values[1][1], values[0][1], atol=3e-9, rtol=0)
        # No seed accumulator or consumed cache: the tape accepts another seed.
        with driver._global_df_response({'mf': mf}):
            mf_bar, data_bar = objective_pull(jnp.asarray(-.4))
            repeated, = pullback(mf_bar, data_bar)
        np.testing.assert_allclose(repeated.coords, -.4 * values[0][1], atol=3e-9, rtol=0)
        print('SCF_DATA_REUSE', dict(outcore=outcore, scf_calls=dict(work[1]),
              gradient_error=float(np.max(np.abs(values[1][1] - values[0][1])))))


def test_capture_releases_records_after_failure_and_restores_outer_scope():
    owner = object()
    with _response_data.capture() as outer:
        _response_data.record(owner, 'S', 'F')
        with pytest.raises(RuntimeError, match='builder failed'):
            with _response_data.capture() as inner:
                _response_data.record(owner, 'inner S', 'inner F')
                raise RuntimeError('builder failed')
        assert not inner
        assert _response_data._capture.get() is outer
        assert outer[id(owner)][1] == ('S', 'F')
    assert not outer
    assert _response_data._capture.get() is None


def test_joint_builder_without_final_matrices_preserves_ordinary_pullback():
    # Some restart builders adopt an existing SCF without invoking its kernel.
    mol = gto.Mole(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    def restart(m):
        mf = scf.RHF(m).density_fit()
        mf.e_tot = jnp.sum(m.coords ** 2)
        return mf

    mf, pullback = driver._initialize_scf(
        mol, restart, fix_phases=False, share_scf_data=True)
    assert pullback.scf_data is None
    _, seed = jax.vjp(lambda current: current.e_tot, mf)
    bar, = pullback(seed(jnp.array(3.))[0])
    np.testing.assert_allclose(bar.coords, 6 * mol.coords, rtol=1e-14)


def test_public_ccsd_t_joint_scf_matches_separate_scf_data(molecular_case, monkeypatch):
    from pyscfad.dlno_stc import ccsd
    case = molecular_case
    path = case.mf.with_df._cderi_to_save

    def build(m):
        mf = scf.RHF(m).density_fit()
        mf.with_df.attach_outcore_cderi(path)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with linear_solver.eager_gmres():
            mf.kernel()
        return mf

    static = ccsd.build_static_selections(case.mf, case.domains, matrix_kind='W',
        thresh_occ=1e-4, thresh_vir=3e-5, baseline_controls=case.controls,
        matrix_controls=case.controls, single_pass=True, lis_space='domain')

    def evaluate():
        return ccsd.value_and_grad(case.mf.mol, build, static,
            baseline_controls=case.controls, matrix_controls=case.controls,
            fragment_indices=case.indices[:1], include_baseline=True, include_hf=True,
            ccsd_t=True, return_components=True,
            triples_controls=dict(mode='stochastic', integral_transform='rotate',
                nsamples=16, pilot_samples=16, energy_tolerance=1e-3, global_seed=71))

    initialize = driver._initialize_scf

    def separate(*args, **kwargs):
        kwargs['share_scf_data'] = False
        return initialize(*args, **kwargs)

    with monkeypatch.context() as legacy:
        legacy.setattr(driver, '_initialize_scf', separate)
        expected = evaluate()

    def forbidden(*args, **kwargs):
        raise AssertionError('public CCSD(T) rebuilt the final SCF matrices')

    monkeypatch.setattr(driver, 'build_scf_data', forbidden)
    monkeypatch.setattr(domain, 'build_scf_data', forbidden)
    actual = evaluate()
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-11, rtol=0)
    np.testing.assert_allclose(actual[1].coords, expected[1].coords, atol=2e-8, rtol=0)
    print('CCSD_T_SCF_DATA_REUSE', dict(
        energy_error=float(abs(actual[0] - expected[0])),
        gradient_error=float(np.max(np.abs(actual[1].coords - expected[1].coords)))))
