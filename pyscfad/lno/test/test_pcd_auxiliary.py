"""Independent auxiliary supports preserve local DF values and full response."""
import jax
import numpy as np
import pytest

from pyscfad import numpy as jnp
from pyscfad.lno import df, _df_shared
from pyscfad.lno._df_half import HalfTransformedDFCache
from pyscfad.lno.test.test_shared_local_df_reverse import _problem, _mf


@pytest.mark.parametrize('half', [False, True])
def test_pcd_auxiliary_response_matches_separate_vjp_and_finite_difference(half):
    mol, coeff = _problem(4)
    # Distinct, permuted supports check the scatter as well as its shape.
    atoms, pcd = np.array([2, 0]), np.array([0])
    def objective(m, c):
        ed = df.transform_df_to_mo(_mf(m), c, (0, 2, 2, 4),
                                   atmlst=atoms, aux_atmlst=pcd)
        child = c[:, :3] * (1 + .03*jnp.sum(ed**2))
        lis = df.transform_df_to_mo(_mf(m), child, (0, 3, 0, 3),
                                    atmlst=atoms, aux_atmlst=pcd)
        return .1*jnp.sum(ed**2) + jnp.sum(lis**2)

    expected, expected_bars = jax.value_and_grad(objective, (0, 1))(mol, coeff)
    with _df_shared.reverse_scope(_mf(mol), atoms, aux_atoms=pcd) as scope:
        assert scope.auxmol.natm == 1 and scope.mol.natm == 2
        assert scope.auxmol.nao < df.make_local_auxmol(_mf(mol), atoms).nao
        if half:
            scope.half = HalfTransformedDFCache(scope)
        value, (mol_bar, coeff_bar) = jax.value_and_grad(objective, (0, 1))(mol, coeff)
        assert scope.receipt['consumers'] == 2
        if half:
            assert scope.receipt['half_builds'] == 1
            assert scope.receipt['half_rotations'] == 1
        grad = mol_bar.coords + scope.finish()
        np.testing.assert_allclose(value, expected, atol=2e-12, rtol=0)
        np.testing.assert_allclose(grad, expected_bars[0].coords, atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(coeff_bar, expected_bars[1], atol=2e-10, rtol=2e-9)
    # Include a center with auxiliary functions, an AO-only center, and an
    # omitted center whose orbital projection still contributes through S.
    step = 1e-5
    for atom, axis in ((0, 1), (2, 1), (1, 2)):
        plus = mol.set_geom_(np.asarray(mol.coords.at[atom, axis].add(step)), unit='Bohr', inplace=False)
        minus = mol.set_geom_(np.asarray(mol.coords.at[atom, axis].add(-step)), unit='Bohr', inplace=False)
        fd = (objective(plus, coeff) - objective(minus, coeff))/(2*step)
        np.testing.assert_allclose(grad[atom, axis], fd, atol=2e-8, rtol=3e-6)
    np.testing.assert_allclose(np.sum(grad, axis=0), 0, atol=2e-9)


def test_local_lov_and_lis_transform_use_the_same_pcd_metric():
    mol, coeff = _problem(4)
    atoms, pcd = np.array([2, 0]), np.array([0])
    ao = df.lno_tools.ao_index_by_atom(mol, atoms)
    local = coeff[ao]
    lov = df.get_local_Lov(_mf(mol), local, 2, atoms, aux_atmlst=pcd)
    full = df.transform_df_to_mo(_mf(mol), local, (0, 4, 0, 4),
        atmlst=atoms, aux_atmlst=pcd, coeff_is_local=True).reshape(-1, 4, 4)
    assert lov.shape == (df.make_local_auxmol(_mf(mol), pcd).nao, 2, 2)
    np.testing.assert_allclose(lov, full[:, :2, 2:], atol=2e-12, rtol=0)
