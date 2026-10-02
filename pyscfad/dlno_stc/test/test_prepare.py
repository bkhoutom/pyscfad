"""Full-support Boys packet checks for the external target-MP2 contract."""

import unittest
from dataclasses import replace

import jax
import numpy
from pyscfad import config_update
from pyscfad import numpy as jnp
from pyscfad import gto, scf
from pyscfad.dlno import _selection, dlno_base, domain as dlno_domain, mp2
from pyscfad.dlno_stc import domain as stc_domain
from pyscfad.dlno_stc.domain import build_stc_domain
from pyscfad.dlno_stc.prepare import prepare_finite_inputs, prepare_inputs
from pyscfad.dlno_stc.test.reference import target_mp2_energy
from pyscfad.lno import df as lno_df


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; H 0.05 0.76 0.59; H -0.03 -0.70 0.63"
        mol.basis = "sto-3g"
        mol.verbose = 0
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-11
        mf.kernel()
        cls.mf = mf
        topology = dlno_domain.build_domain_topology(
            mf, frozen=1, lo_type="boys", force_full_domains=True,
            thresholds=dlno_domain.DLNOThresholds(
                domain_pao=0.0, ed_pao=0.0, pao_norm=1e-10),
        )
        cls.static = _selection.build_domain_selections(mf, topology)
        cls.common = dlno_base.rebuild_domain_data(mf, cls.static)

    def test_packet_uses_one_complete_boys_frame(self):
        mf, common, static = self.mf, self.common, self.static
        domain = build_stc_domain(mf, common, static, 2)
        inputs = prepare_inputs(mf, common, static, 2)
        co = numpy.asarray(domain.occupied_coeff)
        cv = numpy.asarray(domain.virtual_coeff)
        overlap = numpy.asarray(common.s1e)
        fock = numpy.asarray(common.fock)
        self.assertEqual(domain.target_index, int(static.frag_lolist[2][0]))
        numpy.testing.assert_allclose(co, numpy.asarray(common.iao_coeff), atol=1e-10)
        numpy.testing.assert_allclose(co.T @ overlap @ co, numpy.eye(co.shape[1]), atol=1e-9)
        numpy.testing.assert_allclose(cv.T @ overlap @ cv, numpy.eye(cv.shape[1]), atol=1e-9)
        numpy.testing.assert_allclose(co.T @ overlap @ cv, 0, atol=1e-9)
        numpy.testing.assert_allclose(inputs["foo"], co.T @ fock @ co, atol=1e-9)
        numpy.testing.assert_allclose(inputs["fvv"], cv.T @ fock @ cv, atol=1e-9)
        reference = lno_df.get_local_Lov(
            mf, numpy.concatenate((co, cv), axis=1), co.shape[1],
            numpy.arange(mf.mol.natm), integral_direct=True,
        )
        numpy.testing.assert_allclose(inputs["B"], reference, atol=1e-9)
        self.assertEqual(inputs["B"].shape[1:], (co.shape[1], cv.shape[1]))

    def test_full_retention_finite_domain_is_full_packet_in_a_rotated_frame(self):
        """Full retention changes the orbital gauge, not the packet physics."""
        mf, common, static = self.mf, self.common, self.static
        overlap = numpy.asarray(common.s1e)
        for fragment_index in range(len(static.fragments)):
            full_domain = build_stc_domain(mf, common, static, fragment_index)
            finite_domain = stc_domain.build_local_strong_ed_domain(
                common, static, fragment_index
            )
            full = prepare_inputs(mf, common, static, fragment_index)
            finite = prepare_finite_inputs(mf, common, static, fragment_index)

            occupied_rotation = (
                numpy.asarray(full_domain.occupied_coeff).T @ overlap
                @ numpy.asarray(finite_domain.occupied_coeff)
            )
            virtual_rotation = (
                numpy.asarray(full_domain.virtual_coeff).T @ overlap
                @ numpy.asarray(finite_domain.virtual_coeff)
            )
            nocc = full["foo"].shape[0]
            nvir = full["fvv"].shape[0]
            self.assertEqual(finite["foo"].shape, (nocc, nocc))
            self.assertEqual(finite["fvv"].shape, (nvir, nvir))
            numpy.testing.assert_allclose(
                occupied_rotation.T @ occupied_rotation, numpy.eye(nocc),
                atol=1e-9,
            )
            numpy.testing.assert_allclose(
                virtual_rotation.T @ virtual_rotation, numpy.eye(nvir),
                atol=1e-9,
            )
            numpy.testing.assert_allclose(
                finite_domain.target_projection,
                occupied_rotation[full_domain.target_index:full_domain.target_index + 1],
                atol=1e-9,
            )
            numpy.testing.assert_allclose(
                finite_domain.partner_weight, numpy.eye(nocc), atol=1e-9,
            )
            selector = numpy.zeros((1, nocc))
            target = full_domain.target_index
            selector[0, target] = numpy.sign(
                numpy.asarray(finite_domain.target_projection)[0, target]
            )
            numpy.testing.assert_allclose(
                finite_domain.target_projection, selector, atol=1e-9,
            )
            numpy.testing.assert_allclose(
                finite["foo"], occupied_rotation.T @ full["foo"]
                @ occupied_rotation, atol=1e-9,
            )
            numpy.testing.assert_allclose(
                finite["fvv"], virtual_rotation.T @ full["fvv"]
                @ virtual_rotation, atol=1e-9,
            )
            numpy.testing.assert_allclose(
                finite["B"],
                numpy.einsum(
                    "Pia,ij,ab->Pjb", full["B"], occupied_rotation,
                    virtual_rotation,
                ), atol=1e-9,
            )

    def test_rejects_incomplete_scientific_support(self):
        mf, common, static = self.mf, self.common, self.static
        with self.assertRaisesRegex(ValueError, "Boys"):
            build_stc_domain(mf, common, replace(static, lo_type="iao"), 0)
        fragments = list(static.fragments)
        fragments[0] = replace(fragments[0], extended_atoms=numpy.array([0]))
        with self.assertRaisesRegex(ValueError, "full.*atom"):
            build_stc_domain(mf, common, replace(static, fragments=tuple(fragments)), 0)


    def test_fock_and_df_rotate_in_the_same_frame(self):
        mf, common, static = self.mf, self.common, self.static
        domain = build_stc_domain(mf, common, static, 0)
        inputs = prepare_inputs(mf, common, static, 0)
        rng = numpy.random.default_rng(41)
        u, _ = numpy.linalg.qr(rng.normal(size=(inputs["foo"].shape[0],) * 2))
        v, _ = numpy.linalg.qr(rng.normal(size=(inputs["fvv"].shape[0],) * 2))
        rotated = lno_df.get_local_Lov(
            mf,
            numpy.concatenate((domain.occupied_coeff @ u,
                               domain.virtual_coeff @ v), axis=1),
            u.shape[0], domain.extended_atoms, integral_direct=True,
        )
        numpy.testing.assert_allclose(
            rotated, numpy.einsum("Pia,ij,ab->Pjb", inputs["B"], u, v),
            atol=1e-9,
        )
        fock = numpy.asarray(common.fock)
        numpy.testing.assert_allclose(
            u.T @ inputs["foo"] @ u,
            (domain.occupied_coeff @ u).T @ fock @ (domain.occupied_coeff @ u),
            atol=1e-9,
        )
        numpy.testing.assert_allclose(
            v.T @ inputs["fvv"] @ v,
            (domain.virtual_coeff @ v).T @ fock @ (domain.virtual_coeff @ v),
            atol=1e-9,
        )


    def test_molecular_directional_derivative(self):
        static = self.static
        mol = self.mf.mol

        def objective(current_mol):
            current_mf = scf.RHF(current_mol).density_fit()
            current_mf.conv_tol = 1e-11
            current_mf.kernel()
            common = dlno_base.rebuild_domain_data(current_mf, static)
            packet = prepare_inputs(current_mf, common, static, 0)
            return (0.1 * jnp.sum(packet["foo"] ** 2)
                    + 0.2 * jnp.sum(packet["fvv"] ** 2)
                    + 0.01 * jnp.sum(packet["B"] ** 2))

        with (config_update("pyscfad_moleintor_opt", True),
              config_update("pyscfad_scf_implicit_diff", True),
              config_update("pyscfad_scf_first_order_custom", True)):
            gradient = jax.grad(objective)(mol).coords
            direction = numpy.zeros_like(numpy.asarray(mol.atom_coords()))
            direction[1, 2] = 1.0
            coords = numpy.asarray(mol.atom_coords())
            step = 2e-4
            def displaced(sign):
                moved = mol.set_geom_(coords + sign * step * direction,
                                      unit="Bohr", inplace=False)
                return objective(moved)
            finite_difference = (displaced(1) - displaced(-1)) / (2 * step)
        numpy.testing.assert_allclose(
            numpy.sum(numpy.asarray(gradient) * direction),
            finite_difference, rtol=2e-4, atol=2e-5,
        )


    def test_each_target_matches_deterministic_strong_fragment(self):
        mf, common, static = self.mf, self.common, self.static
        for fragment_index in range(len(static.fragments)):
            domain = build_stc_domain(mf, common, static, fragment_index)
            packet = prepare_inputs(mf, common, static, fragment_index)
            target = target_mp2_energy(packet, domain.target_index)
            deterministic = mp2.strong_fragment_energy(
                mf, common, static, fragment_index).total
            numpy.testing.assert_allclose(
                target, deterministic, rtol=1e-8, atol=1e-9,
                err_msg=f"fragment {fragment_index}",
            )


    def test_target_gradient_matches_deterministic_strong_fragment(self):
        static = self.static
        fragment_index = 0

        def objectives(current_mol):
            current_mf = scf.RHF(current_mol).density_fit()
            current_mf.conv_tol = 1e-11
            current_mf.kernel()
            common = dlno_base.rebuild_domain_data(current_mf, static)
            domain = build_stc_domain(current_mf, common, static, fragment_index)
            packet = prepare_inputs(current_mf, common, static, fragment_index)
            target = target_mp2_energy(packet, domain.target_index)
            deterministic = mp2.strong_fragment_energy(
                current_mf, common, static, fragment_index).total
            return target, deterministic

        with (config_update("pyscfad_moleintor_opt", True),
              config_update("pyscfad_scf_implicit_diff", True),
              config_update("pyscfad_scf_first_order_custom", True)):
            target_grad = jax.grad(lambda mol: objectives(mol)[0])(
                self.mf.mol).coords
            deterministic_grad = jax.grad(lambda mol: objectives(mol)[1])(
                self.mf.mol).coords
        numpy.testing.assert_allclose(
            target_grad, deterministic_grad, rtol=2e-5, atol=2e-7,
        )


if __name__ == "__main__":
    unittest.main()
