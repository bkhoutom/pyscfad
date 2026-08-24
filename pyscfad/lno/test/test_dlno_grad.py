"""Low-level full-spin fragment identities used by IAO-DLNO-CCSD(T)."""

import jax
import jax.numpy as jnp
import numpy

from pyscfad.lno import ccsd as lnoccsd
from pyscfad.lno import lno_base


class _FakeEris:
    def __init__(self, ovov):
        self.ovov = ovov


def test_active_space_screening_report(capsys):
    fragment = {
        "extended_primary_domain": numpy.asarray([0, 1]),
        "occ_prescreen_coeff": numpy.zeros((5, 3)),
    }
    lno_base._print_active_space_screening(
        "Fragment 1/2",
        fragment,
        2,
        prescreen_nocc=8,
        prescreen_nvir=20,
        screened_nocc=6,
        screened_nvir=14,
    )
    output = capsys.readouterr().out
    assert "Fragment 1/2 active-space screening" in output
    assert "Domain       : 2 atoms / 5 AOs; fragment LOs = 2" in output
    assert "Prescreened  : 8 occ / 20 vir (28 MOs)" in output
    assert "PNO-screened : 6 occ / 14 vir (20 MOs)" in output


def test_projected_mp2_fragment_energy_matches_full_spin_term_and_fd():
    ovov = jnp.asarray(
        numpy.arange(3 * 2 * 3 * 2, dtype=float).reshape(3, 2, 3, 2)
        / 23.0
        - 0.4
    )
    t2 = jnp.asarray(
        numpy.arange(3 * 3 * 2 * 2, dtype=float).reshape(3, 3, 2, 2)
        / 19.0
        - 0.7
    )
    prj = jnp.asarray([[0.8, 0.3, -0.2], [0.1, 0.5, 0.4]])
    weight = prj.T @ prj
    direct = jnp.einsum("pq,pjab,qajb->", weight, t2, ovov)
    exchange = jnp.einsum("pq,pjab,qbja->", weight, t2, ovov)

    energy = lnoccsd.mp2_fragment_energy(_FakeEris(ovov), t2, prj)
    numpy.testing.assert_allclose(
        energy, 2.0 * direct - exchange, atol=1e-12, rtol=0.0
    )

    def energy_t2(t2_):
        return lnoccsd._mp2_fragment_energy_jax(ovov, t2_, prj)

    gradient = jax.grad(energy_t2)(t2)
    index = (1, 2, 0, 1)
    step = 1e-5
    finite_difference = (
        energy_t2(t2.at[index].add(step))
        - energy_t2(t2.at[index].add(-step))
    ) / (2.0 * step)
    numpy.testing.assert_allclose(
        gradient[index], finite_difference, atol=1e-7, rtol=0.0
    )
