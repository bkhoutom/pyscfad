# Copyright 2021-2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r"""IAO-DLNO-CCSD(T).

The solver uses the current IAO-DLNO-MP2 construction twice: strong-ED MP2
density matrices select each local interacting space (LIS), and the complete
strong-plus-weak local-MP2 energy supplies the unique PT2 correction.  The
implemented correlation energy is

.. math::

   \sum_F\left[E_{\mathrm{CCSD(T)},F}^{\mathrm{LIS}_F}
              -E_{\mathrm{MP2},F}^{\mathrm{LIS}_F}\right]
   +E_{\mathrm{IAO-DLNO-MP2}}^{\mathrm{strong+weak}}.

There is intentionally no SOS-MP2 option and no domain/full correction
switch.  The LIS subtraction is conventional full-spin MP2 and the complete
IAO-DLNO-MP2 correction is added once.
"""

from pyscfad.lno.ccsd import LNOCCSD

from .iao_lis import IAO_LIS_INTERNAL_RANK_THRESHOLD
from .iao_mp2 import IAOFragmentMP2Thresholds


__all__ = ["DLNOCCSD"]


class DLNOCCSD(LNOCCSD):
    """CCSD(T) in IAO-MP2-selected local interacting spaces."""

    def __init__(
        self,
        mf,
        thresh=None,
        frozen=None,
        *,
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        thresholds=None,
        pair_energy_model="multipole",
        force_full_domains=False,
        frag_lolist=None,
        frag_atmlist=None,
        internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
    ):
        if thresh is not None:
            if thresh_occ != 1e-4 or thresh_vir != 1e-5:
                raise ValueError(
                    "the legacy thresh alias cannot be combined with "
                    "thresh_occ or thresh_vir"
                )
            thresh_occ = thresh_vir = thresh
        for name, value in (
            ("thresh_occ", thresh_occ),
            ("thresh_vir", thresh_vir),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        super().__init__(
            mf,
            thresh=thresh_occ,
            frozen=frozen,
        )
        self.thresh_occ = float(thresh_occ)
        self.thresh_vir = float(thresh_vir)
        self.thresholds = (
            IAOFragmentMP2Thresholds()
            if thresholds is None else thresholds
        )
        self.pair_energy_model = pair_energy_model
        self.force_full_domains = bool(force_full_domains)
        self.frag_lolist = frag_lolist
        self.frag_atmlist = frag_atmlist
        self.internal_rank_threshold = float(internal_rank_threshold)
        if self.internal_rank_threshold < 0.0:
            raise ValueError("internal_rank_threshold must be non-negative")
        self.static_selections = None
        self.result = None

    def kernel(
        self,
        *,
        frag_lolist=None,
        frag_atmlist=None,
        thresholds=None,
        pair_energy_model=None,
        force_full_domains=None,
        thresh_occ=None,
        thresh_vir=None,
        internal_rank_threshold=None,
        static_selections=None,
    ):
        """Evaluate the energy and return the corrected correlation energy."""
        from .iao_ccsd import kernel

        if self.ccsd_t and self.dcsd:
            raise ValueError("perturbative triples are not defined for DCSD")

        if frag_lolist is None:
            frag_lolist = self.frag_lolist
        if frag_atmlist is None:
            frag_atmlist = self.frag_atmlist
        if thresholds is None:
            thresholds = self.thresholds
        if pair_energy_model is None:
            pair_energy_model = self.pair_energy_model
        if force_full_domains is None:
            force_full_domains = self.force_full_domains
        if thresh_occ is None:
            thresh_occ = self.thresh_occ
        if thresh_vir is None:
            thresh_vir = self.thresh_vir
        if internal_rank_threshold is None:
            internal_rank_threshold = self.internal_rank_threshold
        if static_selections is None:
            static_selections = self.static_selections

        self.result = kernel(
            self._scf,
            frag_lolist=frag_lolist,
            frag_atmlist=frag_atmlist,
            frozen=self.frozen,
            thresholds=thresholds,
            pair_energy_model=pair_energy_model,
            force_full_domains=force_full_domains,
            thresh_occ=thresh_occ,
            thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold,
            ccsd_t=self.ccsd_t,
            dcsd=self.dcsd,
            verbose_imp=self.verbose_imp,
            static_selections=static_selections,
        )
        return self.result.e_corr

    @property
    def e_corr(self):
        return None if self.result is None else self.result.e_corr

    @property
    def e_tot(self):
        return None if self.result is None else self.result.e_total

    @property
    def e_corr_ccsd(self):
        if self.result is None:
            return None
        return (
            self.result.e_ccsd
            - self.result.e_mp2_lis
            + self.result.e_iao_mp2
        )

    @property
    def e_corr_ccsd_t(self):
        return None if self.result is None else self.result.e_ccsd_t

    @property
    def e_corr_pt2(self):
        """Complete strong-plus-weak IAO-DLNO-MP2 correlation energy."""

        return None if self.result is None else self.result.e_iao_mp2

    @property
    def e_corr_mp2_lis(self):
        """Sum of the full-spin MP2 contributions subtracted in the LISs."""

        return None if self.result is None else self.result.e_mp2_lis

    @property
    def e_corr_pt2_correction(self):
        """Net LIS truncation correction, ``E_IAO-MP2 - sum_F E_MP2,LIS``."""

        if self.result is None:
            return None
        return self.result.e_iao_mp2 - self.result.e_mp2_lis

    @property
    def e_corr_pt2_domain(self):
        raise AttributeError(
            "IAO-DLNO-CCSD(T) has no domain-only PT2 correction"
        )

    @staticmethod
    def _reject_external_pt2_correction(*_args, **_kwargs):
        raise AttributeError(
            "IAO-DLNO-CCSD(T) already includes its unique full strong+weak "
            "local-MP2 correction"
        )

    e_corr_pt2corrected = _reject_external_pt2_correction
    e_tot_pt2corrected = _reject_external_pt2_correction
    e_corr_ccsd_pt2corrected = _reject_external_pt2_correction
    e_tot_ccsd_pt2corrected = _reject_external_pt2_correction
    e_corr_ccsd_t_pt2corrected = _reject_external_pt2_correction
    e_tot_ccsd_t_pt2corrected = _reject_external_pt2_correction

    @property
    def e_corr_iao_mp2(self):
        return None if self.result is None else self.result.e_iao_mp2

    @classmethod
    def value_and_grad(
        cls,
        mol,
        *,
        build_mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=None,
        pair_energy_model="multipole",
        force_full_domains=False,
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
        ccsd_t=False,
        dcsd=False,
        verbose_imp=0,
        static_selections=None,
        progress=False,
        checkpoint_dir=None,
        resume=False,
    ):
        """Return total energy and nuclear gradient with progressive AD.

        ``checkpoint_dir`` enables atomic restart records for the fixed
        topology, fragment forward/reverse work, the local-MP2 correction,
        and the final pre-SCF cotangent.  Set ``resume=True`` to reuse a
        scientifically compatible partial calculation from that directory.
        """
        from .iao_ccsd import value_and_grad

        if ccsd_t and dcsd:
            raise ValueError("perturbative triples are not defined for DCSD")

        return value_and_grad(
            mol,
            build_mf=build_mf,
            frag_lolist=frag_lolist,
            frag_atmlist=frag_atmlist,
            frozen=frozen,
            thresholds=thresholds,
            pair_energy_model=pair_energy_model,
            force_full_domains=force_full_domains,
            thresh_occ=thresh_occ,
            thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold,
            ccsd_t=ccsd_t,
            dcsd=dcsd,
            verbose_imp=verbose_imp,
            static_selections=static_selections,
            progress=progress,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )
