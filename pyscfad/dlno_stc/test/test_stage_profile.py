"""Stage timing boundaries preserve numerical derivatives and forward work."""
from collections import Counter, namedtuple
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd, lis
from pyscfad.dlno_stc._profile import response
from pyscfad.dlno_stc.domain import BoysEDFrame


def _assert_same(left, right):
    for got, want in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(got, want, atol=2e-12, rtol=2e-12)


def test_live_lis_profile_preserves_layout_gradient_and_one_construction(monkeypatch, capsys):
    calls = []
    frozen = np.array([0, 3], dtype=np.int32)
    def construct(mf, common, static, index, *, matrices, frame, frame_fragment_index):
        calls.append(index)
        coeff = frame.occupied_coeff * (mf + matrices['occupied'])
        coeff = coeff + frame.virtual_coeff * (common + matrices['virtual'])
        full = jnp.concatenate((coeff, 2 * coeff, 3 * coeff, 4 * coeff), axis=1)
        return SimpleNamespace(mo_coeff=full, fragment_iao_coeff=coeff * common,
            active_occupied_coeff=full[:, 1:2], active_virtual_coeff=full[:, 2:3],
            frozen=frozen)
    monkeypatch.setattr(lis, 'build_fragment_lis', construct)
    def energy(mf, common, blocks, occupied, virtual):
        frame = BoysEDFrame(occupied, virtual, np.array([0, 1]), np.array([0]), 0, (0,))
        result = ccsd._build_solver_lis(mf, common, None, 0, blocks, frame, 0)
        np.testing.assert_array_equal(result.frozen, frozen)
        return jnp.sum(result.mo_coeff**2) + jnp.sum(result.fragment_iao_coeff**2)
    args = (.3, .7, {'occupied': jnp.asarray(.2), 'virtual': jnp.asarray(.4)},
            jnp.array([[.1], [.2]]), jnp.array([[.3], [.5]]))
    results = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', enabled)
        calls.clear()
        results.append(jax.value_and_grad(energy, argnums=tuple(range(5)))(*args))
        assert calls == [0]
        output = capsys.readouterr().out
        if enabled == '0':
            assert '[DLNO-RESOURCE]' not in output
        else:
            assert output.count('phase=stc.lis.build ') == 1
            assert output.count('phase=stc.lis.pullback ') == 1
    _assert_same(*results)


def test_mp2_profile_preserves_full_fock_and_integral_response(monkeypatch, capsys):
    from pyscfad.dlno_stc.smooth import noncanonical_mp2
    ERIs = namedtuple('ERIs', 'fock ovov')
    fock = jnp.array([[-.8, .03, .02], [.03, .3, .04], [.02, .04, .6]])
    integrals = jnp.array([[[[.07, .02]], [[.02, .09]]]])
    projection = jnp.array([[.8]])
    def energy(f, g, p):
        return response('solver.mp2_reference_pullback', noncanonical_mp2,
                        ERIs(f, g), p, frag=1)[0]
    results = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', enabled)
        results.append(jax.value_and_grad(energy, argnums=(0, 1, 2))(
            fock, integrals, projection))
        output = capsys.readouterr().out
        assert ('phase=solver.mp2_reference_pullback ' in output) == (enabled == '1')
    _assert_same(*results)


def test_triples_profile_preserves_saved_gradient_and_native_work(monkeypatch, capsys):
    from pyscfad.dlno_stc import smooth_triples
    from pyscfad.dlno_stc.test.test_smooth import inputs_and_reference, evaluate
    x, foo, fvv, _, grid = inputs_and_reference()
    calls = []
    native = smooth_triples._native
    def counted(*args, **kwargs):
        calls.append(kwargs.get('with_grad'))
        return native(*args, **kwargs)
    monkeypatch.setattr(smooth_triples, '_native', counted)
    results, work = [], []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', enabled)
        solver = smooth_triples.make_solver({'mode': 'deterministic'}, grid, with_grad=True)
        calls.clear()
        results.append(jax.value_and_grad(lambda x, fo, fv: evaluate(solver, x, fo, fv),
            argnums=(0, 1, 2))(x, foo, fvv))
        work.append(calls.copy())
        output = capsys.readouterr().out
        if enabled == '0':
            assert '[DLNO-RESOURCE]' not in output
        else:
            assert output.count('phase=stc.triples.native_timings ') == 1
            assert output.count('phase=stc.triples.pullback_scale ') == 1
    assert work[0] == work[1]
    _assert_same(*results)


@pytest.mark.parametrize('with_grad', [False, True])
def test_weak_profile_preserves_pair_and_endpoint_work(monkeypatch, capsys, with_grad):
    from pyscfad.dlno_stc import _pairs
    from pyscfad.dlno_stc.test.test_weak_endpoints import _fixture, _install_split
    mf, common, static, endpoint = _fixture(4)
    screens, moments, pair_calls = Counter(), Counter(), []
    _install_split(monkeypatch, endpoint, screens, moments)
    kernel_name = 'pair_energy_and_grad' if with_grad else '_multipole_pair_energy'
    kernel = getattr(_pairs, kernel_name)
    def counted(*args):
        pair_calls.append(1)
        return kernel(*args)
    monkeypatch.setattr(_pairs, kernel_name, counted)
    run = _pairs.weak_pair_energy_and_grad if with_grad else _pairs.weak_pair_energy_sum
    results, work = [], []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PYSCFAD_DLNO_RESOURCE_PROFILE', enabled)
        screens.clear(); moments.clear(); pair_calls.clear()
        results.append(run(mf, common, static))
        work.append((screens.copy(), moments.copy(), len(pair_calls)))
        output = capsys.readouterr().out
        if enabled == '0':
            assert '[DLNO-RESOURCE]' not in output
        else:
            assert output.count('phase=stc.domain.weak_prepare ') == 1
            phase = 'stc.cc.weak_pair_value_and_grad' if with_grad else 'stc.cc.weak_forward'
            assert output.count(f'phase={phase} ') == 1
            assert output.count('phase=stc.domain.weak_pullback ') == int(with_grad)
    assert work[0] == work[1]
    assert work[0][0] == Counter({0: 1, 1: 1, 2: 1})
    assert work[0][2] == 3
    _assert_same(*results)
