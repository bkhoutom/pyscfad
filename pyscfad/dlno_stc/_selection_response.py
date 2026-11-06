"""Short-lived, driver-owned response work shared with domain selection.

Never stored in DomainSelections or on a live SCF pytree. Numerical weak bars
outlive screening; endpoint tapes are consumed before strong-frame selection.
"""
from dataclasses import replace
from functools import partial

import jax
import numpy

from pyscfad.tools import resource_profile


class SelectionResponse:
    def __init__(self, mf, scf_data, scf_data_pullback):
        self.mf = mf
        self.scf_data = scf_data
        self.scf_data_pullback = scf_data_pullback
        self.reset()

    def reset(self):
        """Release work from a previous selection without detaching its owner."""
        self.common = self.common_pullback = self.pending = self.static = None
        self.products_preparation = None

    def matches_reference(self, reference):
        """A callback may deliberately select a different numerical reference."""
        for name in ('mo_coeff', 'mo_energy', 'mo_occ'):
            if not numpy.array_equal(getattr(reference, name), getattr(self.mf, name)):
                return False
        for name in ('_atm', '_bas', '_env'):
            if not numpy.array_equal(getattr(reference.mol, name), getattr(self.mf.mol, name)):
                return False
        return numpy.array_equal(reference.mol.atom_coords(), self.mf.mol.atom_coords())

    def prepare(self, static):
        from . import driver
        self.reset()
        with resource_profile.section('stc.domain.common_prepare'):
            self.common, self.common_pullback = driver._build_common_with_pullback(
                self.mf, static, self.scf_data, self.scf_data_pullback)
            jax.block_until_ready(self.common)
        return self.common

    def prepare_reference(self, **options):
        """Share one live Boys/PAO reference with concrete domain selection."""
        from . import domain
        from .adjoint import add_cotangent
        from ._pairs import _cached_weak_screen_products
        self.reset()
        with resource_profile.section('stc.domain.common_prepare'):
            common, construction_pullback, auxiliary = jax.vjp(
                partial(domain._build_selection_reference, **options),
                self.mf, self.scf_data, has_aux=True)
            parents, products = auxiliary
            jax.block_until_ready((common, products))
            # Capture this pullback, not the receipt: take() releases its owner.
            data_pullback = self.scf_data_pullback
            def common_pullback(common_bar):
                direct, data_bar = construction_pullback(common_bar)
                via_data, = data_pullback(data_bar)
                return (jax.tree_util.tree_map(add_cotangent, direct, via_data),)
            self.common, self.common_pullback = common, common_pullback
            self.products_preparation = _cached_weak_screen_products(common, products)
        return common, parents, products

    def complete_weak(self, static, strong, prepared_endpoints, pair_response=None):
        from ._pairs import weak_pair_indices, weak_pair_energy_and_grad
        selected = replace(static, strong_mask=strong)
        pairs = tuple(weak_pair_indices(selected))
        if pairs:
            with resource_profile.section('stc.domain.selection_weak_response'):
                result = weak_pair_energy_and_grad(
                    self.mf, self.common, selected, pairs=pairs,
                    prepared_endpoints=prepared_endpoints, _grouped_ao_response=True,
                    _prepared_pairs=pair_response)
                jax.block_until_ready(result)
            self.pending = (pairs, result)
        prepared_endpoints.clear()
        if pair_response is not None:
            pair_response.clear()
        self.products_preparation = None

    def bind(self, static):
        self.static = static

    def take(self, static):
        """Transfer work only for the exact final selection, at most once."""
        result = ((self.common, self.common_pullback, self.pending)
                  if static is self.static and self.common is not None else None)
        self.reset()
        self.mf = self.scf_data = self.scf_data_pullback = None
        return result
