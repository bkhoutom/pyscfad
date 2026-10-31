"""Tiny native DF-MP2 contractions, seed corrections, and public AD routing."""
from types import SimpleNamespace

import jax
import numpy
import pytest

from pyscfad import config_update
from pyscfad import numpy as np
from pyscfad.mp import dfmp2


def _inputs(dtype=numpy.float64):
    rng = numpy.random.default_rng(180)
    return (np.asarray(rng.normal(size=(4, 6)).astype(dtype)),
            np.asarray(numpy.array([-1.3, -0.7, 0.2, 0.6, 1.0], dtype=dtype)))


def _reference(lov, energies, with_t2=False):
    # Independent complete (ijab) contraction, rather than pair reduction.
    factors = lov.reshape((-1, 2, 3))
    eri = np.einsum("Lia,Ljb->ijab", factors, factors)
    denominator = (energies[:2, None, None, None]
                   + energies[None, :2, None, None]
                   - energies[None, None, 2:, None]
                   - energies[None, None, None, 2:])
    t2 = eri / denominator
    result = (np.sum(eri * t2), -np.sum(eri.swapaxes(2, 3) * t2))
    return result + (t2,) if with_t2 else result


def _public(lov, energies, with_t2=False):
    # The custom transform is real, small and differentiable; it must always run.
    eris = SimpleNamespace(mo_energy=energies, mo_coeff=np.eye(5))
    mp = SimpleNamespace(nocc=2, nmo=5, frozen=1,
                         loop_ao2mo=lambda coeff, nocc, keep: lov * 1.2)
    return dfmp2.kernel(mp, eris=eris, with_t2=with_t2)


def _assert_tree(actual, expected, tol=1e-10):
    for a, e in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected)):
        numpy.testing.assert_allclose(a, e, atol=tol, rtol=tol)


def test_native_lov_entry_exists():
    assert callable(getattr(dfmp2, "_contract_d_lov", None))


@pytest.mark.parametrize("with_t2", [False, True])
@pytest.mark.parametrize("weights", [(2., 1.), (-0.3, 0.8), (0., 0.), (0.7, 0.)])
def test_native_lov_values_and_repeated_pullbacks(with_t2, weights):
    lov, energies = _inputs()
    expected, expected_pullback = jax.vjp(
        lambda x, e: _reference(x, e, with_t2), lov, energies)
    actual, pullback = jax.vjp(
        lambda x, e: dfmp2._contract_d_lov(x, e, 2, 3, with_t2),
        lov, energies)
    _assert_tree(actual, expected)
    # A nonsymmetric seed exercises the T_ijab + T_jiba reduction.
    t2_seed = np.asarray(numpy.random.default_rng(3).normal(size=(2, 2, 3, 3)))
    seed = tuple(np.asarray(w) for w in weights)
    if with_t2:
        seed += (t2_seed,)
    _assert_tree(pullback(seed), expected_pullback(seed))
    scaled = jax.tree_util.tree_map(lambda x: -0.4 * x, seed)
    _assert_tree(pullback(scaled), expected_pullback(scaled))
    _assert_tree(pullback(seed), expected_pullback(seed))


def test_total_energy_fusion_avoids_second_native_pass(monkeypatch):
    lov, energies = _inputs()
    native = dfmp2.libmp_vjp
    calls = []
    def record(name):
        original = getattr(native, name)
        def call(*args):
            calls.append(name)
            return original(*args)
        return call
    proxy = SimpleNamespace(
        MP2_contract_d_vjp=record("MP2_contract_d_vjp"),
        MP2_contract_d_energy_vjp=record("MP2_contract_d_energy_vjp"))
    monkeypatch.setattr(dfmp2, "libmp_vjp", proxy)
    # One invocation feeds both energy components.
    def fn(x):
        ed, ex = dfmp2._contract_d_lov(x, energies, 2, 3)
        return 2 * ed + ex
    with dfmp2._native_reverse_mode():
        jax.value_and_grad(fn)(lov)
    assert calls == ["MP2_contract_d_energy_vjp"]


@pytest.mark.parametrize("with_t2", [False, True])
def test_public_kernel_uses_transform_and_native_reverse_scope(with_t2, monkeypatch):
    lov, energies = _inputs()
    original = dfmp2._contract_d_lov
    calls = []
    def recorded(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(dfmp2, "_contract_d_lov", recorded)
    with config_update("pyscfad_moleintor_opt", True):
        actual = _public(lov, energies, with_t2)
        assert len(calls) == 1
        def objective(x, e):
            energy, t2 = _public(x, e, with_t2)
            return energy if t2 is None else energy + 0.01 * np.sum(t2 * t2)
        def reference(x, e):
            result = _reference(x * 1.2, e, with_t2)
            energy = 2 * result[0] + result[1]
            return energy if not with_t2 else energy + 0.01 * np.sum(result[2] ** 2)
        with dfmp2._native_reverse_mode():
            result = jax.value_and_grad(objective, argnums=(0, 1))(lov, energies)
        assert len(calls) == 2
        _assert_tree(result, jax.value_and_grad(reference, argnums=(0, 1))(lov, energies))
        ref = _reference(lov * 1.2, energies, with_t2)
        _assert_tree(actual, (2 * ref[0] + ref[1], ref[2] if with_t2 else None))


@pytest.mark.parametrize("transform", ["grad", "jvp", "linearize", "jit", "nested"])
def test_public_kernel_preserves_generic_ad_without_native_scope(transform, monkeypatch):
    lov, energies = _inputs()
    def forbidden(*args):
        raise AssertionError("generic AD entered a native custom VJP")
    monkeypatch.setattr(dfmp2, "_contract_d_lov", forbidden)
    fn = lambda e: _public(lov, e)[0]
    ref = lambda e: 2 * _reference(lov * 1.2, e)[0] + _reference(lov * 1.2, e)[1]
    direction = np.arange(5.) * 0.1
    def evaluate(f):
        if transform == "grad":
            return jax.grad(f)(energies)
        if transform == "jvp":
            return jax.jvp(f, (energies,), (direction,))
        if transform == "linearize":
            value, tangent = jax.linearize(f, energies)
            return value, tangent(direction)
        if transform == "jit":
            return jax.jit(jax.value_and_grad(f))(energies)
        return jax.jacrev(jax.grad(f))(energies)
    with config_update("pyscfad_moleintor_opt", True):
        _assert_tree(evaluate(fn), evaluate(ref), 1e-9)


@pytest.mark.parametrize("transform", ["jvp", "jit", "nested"])
def test_native_scope_rejects_other_or_nested_traces(transform, monkeypatch):
    lov, energies = _inputs()
    monkeypatch.setattr(dfmp2, "_contract_d_lov",
                        lambda *args: (_ for _ in ()).throw(AssertionError("native trace")))
    fn = lambda e: _public(lov, e)[0]
    with config_update("pyscfad_moleintor_opt", True), dfmp2._native_reverse_mode():
        if transform == "jvp":
            jax.jvp(fn, (energies,), (np.ones_like(energies),))
        elif transform == "jit":
            jax.jit(jax.value_and_grad(fn))(energies)
        else:
            jax.jacrev(jax.grad(fn))(energies)


@pytest.mark.parametrize("missing", ["library", "fused_symbol"])
def test_missing_optional_native_falls_back(monkeypatch, missing):
    lov, energies = _inputs()
    old = SimpleNamespace(MP2_contract_d_vjp=lambda *a: None)
    monkeypatch.setattr(dfmp2, "libmp_vjp", None if missing == "library" else old)
    with config_update("pyscfad_moleintor_opt", True), dfmp2._native_reverse_mode():
        result = jax.value_and_grad(lambda x: _public(x, energies)[0])(lov)
    ref = lambda x: 2 * _reference(x * 1.2, energies)[0] + _reference(x * 1.2, energies)[1]
    _assert_tree(result, jax.value_and_grad(ref)(lov))


def test_unsupported_dtype_keeps_generic_path(monkeypatch):
    lov, energies = _inputs(numpy.float32)
    monkeypatch.setattr(dfmp2, "_contract_d_lov",
                        lambda *args: (_ for _ in ()).throw(AssertionError("float32 native")))
    with config_update("pyscfad_moleintor_opt", True):
        result = _public(lov, energies)
    ref = _reference(lov * 1.2, energies)
    _assert_tree(result[0], 2 * ref[0] + ref[1], 2e-5)


def test_native_scope_resets_after_exception():
    with pytest.raises(RuntimeError):
        with dfmp2._native_reverse_mode():
            with dfmp2._native_reverse_mode():
                raise RuntimeError("scope reset")
    assert not dfmp2._NATIVE_REVERSE_MODE.get()


def test_public_transform_memory_rejection_is_not_bypassed():
    lov, energies = _inputs()
    eris = SimpleNamespace(mo_energy=energies, mo_coeff=np.eye(5))
    def limited_transform(*args):
        raise RuntimeError("insufficient transform memory")
    mp = SimpleNamespace(nocc=2, nmo=5, frozen=1, loop_ao2mo=limited_transform)
    with config_update("pyscfad_moleintor_opt", True):
        with pytest.raises(RuntimeError, match="insufficient transform memory"):
            dfmp2.kernel(mp, eris=eris, with_t2=False)


def test_fused_resource_profile_reports_one_completed_interval(monkeypatch, capsys):
    lov, energies = _inputs()
    monkeypatch.setenv("PYSCFAD_DLNO_RESOURCE_PROFILE", "1")
    with config_update("pyscfad_moleintor_opt", True), dfmp2._native_reverse_mode():
        jax.block_until_ready(jax.value_and_grad(lambda x: _public(x, energies)[0])(lov))
    phases = [line.split("phase=")[1].split()[0]
              for line in capsys.readouterr().out.splitlines() if "phase=" in line]
    assert phases == ["dfmp2.ao2mo", "dfmp2.energy_vjp_fused"]


def test_optional_loader_absent_library_keeps_other_native_imports(monkeypatch):
    import importlib.util
    import pyscfadlib.lib as native_lib
    original = numpy.ctypeslib.load_library
    def without_mp(name, directory):
        if name == "libmp_vjp":
            raise OSError("optional native library absent")
        return original(name, directory)
    monkeypatch.setattr(numpy.ctypeslib, "load_library", without_mp)
    spec = importlib.util.spec_from_file_location("mp_optional_loader_probe", native_lib.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.libmp_vjp is None
    assert module.libao2mo_vjp.AO2MOnr_e2_cderi_bar_pack_aux_block
    assert module.libcgto_vjp


@pytest.mark.parametrize("missing", ["library", "forward_symbol"])
def test_review_missing_pyscf_mp_forward_support_keeps_generic(monkeypatch, missing):
    lov, energies = _inputs()
    monkeypatch.setattr(dfmp2, "libmp", None if missing == "library" else SimpleNamespace())
    monkeypatch.setattr(dfmp2, "_contract_d_lov",
                        lambda *args: (_ for _ in ()).throw(AssertionError("missing forward support")))
    with config_update("pyscfad_moleintor_opt", True), dfmp2._native_reverse_mode():
        result = jax.value_and_grad(lambda x: _public(x, energies)[0])(lov)
    ref = lambda x: 2 * _reference(x * 1.2, energies)[0] + _reference(x * 1.2, energies)[1]
    _assert_tree(result, jax.value_and_grad(ref)(lov))


def test_review_missing_pyscf_mp_library_does_not_break_import(monkeypatch):
    import importlib.util
    original = dfmp2.pyscf_lib.load_library
    def without_mp(name):
        if name == "libmp":
            raise OSError("PySCF version has no native MP2 library")
        return original(name)
    monkeypatch.setattr(dfmp2.pyscf_lib, "load_library", without_mp)
    spec = importlib.util.spec_from_file_location("pyscfad.mp._optional_forward_probe", dfmp2.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.libmp is None
    lov, energies = _inputs()
    eris = SimpleNamespace(mo_energy=energies, mo_coeff=np.eye(5))
    mp = SimpleNamespace(nocc=2, nmo=5, frozen=1,
                         loop_ao2mo=lambda *args: lov)
    with config_update("pyscfad_moleintor_opt", True):
        actual = module.kernel(mp, eris=eris, with_t2=False)
    ref = _reference(lov, energies)
    _assert_tree(actual[0], 2 * ref[0] + ref[1])


@pytest.mark.parametrize("with_t2", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("incore_anyway", [False, True])
def test_review_standard_transform_native_added_memory_guard(
        monkeypatch, with_t2, reverse, incore_anyway):
    lov, energies = _inputs()
    # Real MP2.loop_ao2mo guard permits this tiny transform: 480 bytes,
    # or 768 bytes with T2. Its 1000-byte headroom cannot fit native extras.
    monkeypatch.setattr(dfmp2, "current_memory", lambda: (100., 0.))
    monkeypatch.setattr(dfmp2.pyscf_lib, "num_threads", lambda: 4)
    monkeypatch.setattr(dfmp2._ao2mo, "nr_e2", lambda *args, **kw: lov)
    source = numpy.zeros((4, 15))
    with_df = SimpleNamespace(get_naoaux=lambda: 4, _cderi=source)
    mp = SimpleNamespace(nocc=2, nmo=5, frozen=1, max_memory=100.001,
                         mol=SimpleNamespace(incore_anyway=incore_anyway),
                         with_df=with_df)
    mp.loop_ao2mo = lambda coeff, nocc, keep: dfmp2.MP2.loop_ao2mo(
        mp, coeff, nocc, keep)
    eris = SimpleNamespace(mo_energy=energies, mo_coeff=np.eye(5))
    if incore_anyway:
        monkeypatch.setattr(dfmp2, "_contract_scan",
                            lambda *args: (_ for _ in ()).throw(AssertionError("override ignored")))
    else:
        monkeypatch.setattr(dfmp2, "_contract_d_lov",
                            lambda *args: (_ for _ in ()).throw(AssertionError("native memory exceeded")))
    def objective(e):
        return dfmp2.kernel(mp, eris=SimpleNamespace(
            mo_energy=e, mo_coeff=eris.mo_coeff), with_t2=with_t2)[0]
    ref = lambda e: 2 * _reference(lov, e)[0] + _reference(lov, e)[1]
    with config_update("pyscfad_moleintor_opt", True):
        if reverse:
            with dfmp2._native_reverse_mode():
                _assert_tree(jax.value_and_grad(objective)(energies),
                             jax.value_and_grad(ref)(energies))
        else:
            _assert_tree(objective(energies), ref(energies))


def test_review_empty_auxiliary_transform_keeps_generic(monkeypatch):
    lov = np.zeros((0, 6))
    _, energies = _inputs()
    monkeypatch.setattr(dfmp2, "_contract_d_lov",
                        lambda *args: (_ for _ in ()).throw(AssertionError("empty aux native")))
    with config_update("pyscfad_moleintor_opt", True):
        energy, t2 = _public(lov, energies)
    assert energy == 0
    assert t2 is None
