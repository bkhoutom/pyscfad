# Copyright 2023-2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from mpi4py import MPI
import numpy
import jax
import jax.numpy as jnp

from pyscfad import numpy as np
from pyscfad.ops import stop_trace
from pyscfad.lno import lno_base_mpi as lno_base_mpi_mod
from pyscfad.lno import lno_base
from pyscfad.lno.ccsd import LNOCCSD, _impurity_solve_core
from pyscfad.dlno import ccsd as dlno_ccsd
from pyscfad.dlno.ccsd import (
    DLNOCCSD as _DLNOCCSDSingle,
    _add_cotangent,
    _build_static_dlno_topology,
    _make_stub_eris,
    _VERBOSE_PROGRESS,
)


def _to_numpy_leaf(leaf):
    """Convert a JAX-array-ish leaf to a plain writable numpy array.

    None / float0 / non-array Python objects pass through unchanged so
    the gather / sum below treats them as no-ops.
    """
    if leaf is None:
        return None
    if hasattr(leaf, 'dtype') and leaf.dtype == jax.dtypes.float0:
        return leaf
    try:
        return numpy.array(numpy.asarray(leaf), copy=True, order='C')
    except (TypeError, ValueError):
        return leaf


def _path_key(path):
    """Stable, picklable string key for a jax tree-path.

    Uses JAX's own ``keystr`` so every distinct leaf path has a
    distinct key (essential when ranks built their pytrees from
    independent Python objects -- e.g., separate ``Mole`` and
    ``mf`` instances on different MPI ranks).
    """
    return jax.tree_util.keystr(path)


def _tree_allreduce_sum(comm, tree):
    """Sum a JAX pytree across MPI ranks by matching leaves on their
    structural path.

    Each rank flattens with ``tree_flatten_with_path`` to obtain
    ``(path, leaf)`` pairs.  Leaves are keyed by a string derived from
    the path so cross-rank alignment doesn't depend on the local
    iteration order or on object identity of registered pytree nodes
    (Mole, mf, with_df, ...).  Numeric leaves are summed via
    ``_add_cotangent``; None / float0 leaves pass through.  The final
    pytree is rebuilt with the *local* treedef so JAX aux mismatches
    across pickled objects never matter.
    """
    nproc = comm.Get_size()
    if nproc == 1:
        return tree
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: x is None,
    )
    local_paths = [_path_key(p) for p, _ in leaves_with_path]
    local_arrays = [_to_numpy_leaf(l) for _, l in leaves_with_path]
    # Sanity check: every path should be unique on this rank.  If not,
    # the dict-based reduce silently loses leaves and corrupts the sum.
    if len(set(local_paths)) != len(local_paths):
        from collections import Counter
        c = Counter(local_paths)
        dups = {k: n for k, n in c.items() if n > 1}
        rank = comm.Get_rank()
        raise RuntimeError(
            f'tree_allreduce: rank {rank} has {len(local_paths)} leaves '
            f'but only {len(set(local_paths))} unique path keys.  '
            f'Duplicates (first 5): {dict(list(dups.items())[:5])}'
        )
    local_dict = dict(zip(local_paths, local_arrays))

    # Sanity check across ranks: every rank should produce the same set
    # of keys; if not, fail loudly with a useful diagnostic.
    all_key_sets = comm.allgather(sorted(local_paths))
    if any(ks != all_key_sets[0] for ks in all_key_sets):
        # Print the symmetric difference between rank 0 and the first
        # disagreeing rank.
        for r, ks in enumerate(all_key_sets[1:], start=1):
            if ks != all_key_sets[0]:
                only_0 = sorted(set(all_key_sets[0]) - set(ks))[:5]
                only_r = sorted(set(ks) - set(all_key_sets[0]))[:5]
                raise RuntimeError(
                    f'tree_allreduce: path-set mismatch between rank 0 '
                    f'({len(all_key_sets[0])} keys) and rank {r} '
                    f'({len(ks)} keys).  Only on rank 0: {only_0}.  '
                    f'Only on rank {r}: {only_r}.'
                )

    # Allgather the dict {path_key -> array} from each rank.
    gathered = comm.allgather(local_dict)
    summed_dict = dict(gathered[0])
    for other in gathered[1:]:
        for k, v in other.items():
            summed_dict[k] = _add_cotangent(summed_dict.get(k), v)

    # Rebuild a leaf list in this rank's own flatten order.
    summed = [summed_dict[k] for k in local_paths]
    return jax.tree_util.tree_unflatten(treedef, summed)


class DLNOCCSD(lno_base_mpi_mod.LNO, _DLNOCCSDSingle):
    """MPI variant of :class:`pyscfad.dlno.ccsd.DLNOCCSD`.

    The MPI strategy mirrors :class:`pyscfad.lno.ccsd_mpi.LNOCCSD`:
    round-robin partition over fragments (rank ``r`` handles fragments
    where ``i % nproc == r``), then MPI Allreduce on the per-fragment
    energy and cotangent contributions before the LO and SCF close-out.

    Both :meth:`kernel` (inherited from ``lno_base_mpi.LNO`` for the
    plain-energy MPI flow used by example 12) and
    :meth:`value_and_grad` (defined here, MPI-parallel progressive AD)
    are exposed.
    """

    @classmethod
    def value_and_grad(cls, mol, *, build_mf, frag_lolist=None,
                       include_mp2_correction=True,
                       # CCSD solver config
                       frozen=0,
                       thresh_occ=1e-4,
                       thresh_vir=1e-5,
                       lo_type='iao',
                       no_type='ie',
                       ccsd_t=False,
                       dcsd=False,
                       verbose_imp=0,
                       # Prescreen build config
                       lmo_bp_domain_thr=None,
                       pao_bp_domain_thr=None,
                       domain_pao_thr=None,
                       pair_energy_thr=None,
                       multipole_order=None):
        """MPI-parallel value_and_grad: round-robin fragments + tree-allreduce.

        Same call signature and semantics as
        :meth:`pyscfad.dlno.ccsd.DLNOCCSD.value_and_grad`.  Run with
        ``mpirun -n N python script.py``; with ``N=1`` this reduces to
        the single-rank implementation.

        SCF, LO transform, and DLNO prescreen are built deterministically
        on every rank (no redundant work is saved here -- they're cheap
        relative to the fragment loop).  Each rank then processes its
        round-robin subset of fragments and accumulates its local
        ``grad_mf``, ``grad_lo``, ``e_corr_local``.  An MPI Allreduce
        sums these across ranks before the LO and SCF (CPHF) close-out,
        which is redundantly evaluated on every rank (it's deterministic
        and the result is identical, simpler than gathering to rank 0).

        On rank 0 the HF-energy cotangent is seeded with weight 1.0; on
        non-root ranks it is seeded with weight 0.0 so the allreduce sum
        does not over-count the single HF contribution.
        """
        # Resolve prescreen defaults from the class
        if lmo_bp_domain_thr is None: lmo_bp_domain_thr = cls.lmo_bp_domain_thr
        if pao_bp_domain_thr is None: pao_bp_domain_thr = cls.pao_bp_domain_thr
        if domain_pao_thr    is None: domain_pao_thr    = cls.domain_pao_thr
        if pair_energy_thr   is None: pair_energy_thr   = cls.pair_energy_thr
        if multipole_order   is None: multipole_order   = cls.multipole_order

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        rank = comm.Get_rank()

        verbose = int(getattr(mol, 'verbose', 0))
        log = ((lambda msg: print(msg, flush=True))
               if (rank == 0 and verbose >= _VERBOSE_PROGRESS)
               else (lambda msg: None))
        t_overall = time.perf_counter()
        log(f'DLNOCCSD.value_and_grad (MPI, nproc={nproc}): start')

        # ---- SCF + LO + prescreen: same on every rank (deterministic) ----
        t0 = time.perf_counter()
        mf, scf_vjp = jax.vjp(build_mf, mol)
        log(f'  SCF (build_mf + jax.vjp):   {time.perf_counter() - t0:8.2f} s')

        # HF-energy seed: every rank evaluates ``hf_vjp(1.0)`` so the
        # cotangent pytree has the *same* structure on every rank (JAX
        # prunes zero-cotangent paths, so ``hf_vjp(0.0)`` and
        # ``hf_vjp(1.0)`` can produce structurally different pytrees,
        # which would break the per-leaf Allreduce alignment below).  We
        # then scale by ``hf_weight`` (1.0 on rank 0, 0.0 elsewhere) so
        # the HF contribution appears exactly once after Allreduce.
        e_hf, hf_vjp = jax.vjp(lambda m: m.e_tot, mf)
        hf_cot = hf_vjp(1.0)[0]
        hf_weight = 1.0 if rank == 0 else 0.0
        def _scale_leaf(x):
            # Scale numeric leaves (JAX arrays AND Python scalars like the
            # 1.0 cotangent that lands on ``mf.e_tot``) by ``hf_weight``.
            # None / float0 / non-multipliable leaves pass through.
            if x is None:
                return None
            if hasattr(x, 'dtype') and x.dtype == jax.dtypes.float0:
                return x
            try:
                return x * hf_weight
            except TypeError:
                return x
        grad_mf = jax.tree_util.tree_map(_scale_leaf, hf_cot)

        def _build_lo(mf_):
            cc_local = LNOCCSD(mf_, frozen=frozen)
            cc_local.lo_type = lo_type
            return cc_local.get_lo(lo_type=cc_local.lo_type)

        t0 = time.perf_counter()
        lo_coeff, lo_vjp = jax.vjp(_build_lo, mf)
        grad_lo = jax.tree_util.tree_map(jnp.zeros_like, lo_coeff)
        log(f'  LO transform + jax.vjp:     {time.perf_counter() - t0:8.2f} s')

        def _topo_builder(mf_, frag_lolist_):
            return _build_static_dlno_topology(
                mf_, frag_lolist_, frozen, lo_type,
                lmo_bp_domain_thr, pao_bp_domain_thr,
                domain_pao_thr, pair_energy_thr, multipole_order,
            )
        t0 = time.perf_counter()
        frag_lolist_static, prescreen_data = stop_trace(_topo_builder)(mf, frag_lolist)
        log(f'  DLNO prescreen build:       {time.perf_counter() - t0:8.2f} s')

        nfrag = len(frag_lolist_static)
        sign_pt2 = -1.0 if include_mp2_correction else 0.0

        # Round-robin partition: rank r processes fragments i where i % nproc == r
        my_fragment_indices = [i for i in range(nfrag) if i % nproc == rank]

        if verbose >= _VERBOSE_PROGRESS:
            atom_counts = [
                int(np.asarray(prescreen_data['fragment_data'][i]
                               .get('extended_primary_domain')).size)
                for i in range(nfrag)
            ]
            atom_range = (f'{min(atom_counts)}-{max(atom_counts)}'
                          if min(atom_counts) != max(atom_counts)
                          else f'{atom_counts[0]}')
            log(f'  Fragments: {nfrag} (domain atoms/frag: {atom_range})')
            log(f'  MP2 correction: {"per-domain" if include_mp2_correction else "off"}'
                f',  CCSD(T): {ccsd_t}')
            log(f'  MPI partition: nproc={nproc}, '
                f'rank 0 handles {len([i for i in range(nfrag) if i % nproc == 0])} fragments')

        # ---- Per-fragment loop (rank-local subset) ----
        e_corr_local = jnp.float64(0.0)
        for ifrag in my_fragment_indices:
            fraglo_idx = frag_lolist_static[ifrag]
            frag_prescreen = prescreen_data['fragment_data'][ifrag]
            weight = 1.0

            def per_frag_fn(mf_, lo_coeff_,
                            _fraglo=fraglo_idx,
                            _frag_prescreen=frag_prescreen,
                            _weight=weight):
                cc_local = LNOCCSD(mf_, frozen=frozen)
                cc_local.thresh_occ = thresh_occ
                cc_local.thresh_vir = thresh_vir
                cc_local.lo_type = lo_type
                cc_local.no_type = no_type
                cc_local.ccsd_t = ccsd_t
                cc_local.dcsd = dcsd
                cc_local.verbose_imp = verbose_imp
                cc_local.compute_domain_pt2 = include_mp2_correction

                orbfragloc = lo_coeff_[:, _fraglo]
                stub_eris = _make_stub_eris(mf_)
                eris_fpno = lno_base.make_fragment_eris(
                    cc_local, stub_eris, _frag_prescreen,
                )
                frzfrag, orbfrag, domain_pt2 = lno_base.make_fpno1(
                    cc_local, eris_fpno, orbfragloc, no_type,
                    lno_base.THRESH_INTERNAL,
                    (cc_local.thresh_occ, cc_local.thresh_vir),
                    frag_prescreen=_frag_prescreen,
                    frozen_mask=cc_local.get_frozen_mask(),
                )

                if orbfrag is None:
                    if include_mp2_correction:
                        return _weight * domain_pt2
                    return jnp.float64(0.0)

                res = _impurity_solve_core(
                    mf_, orbfrag, orbfragloc,
                    eris_fpno.fock, eris_fpno.s1e,
                    frozen=frzfrag, frag_prescreen=_frag_prescreen,
                    verbose_imp=cc_local.verbose_imp,
                    ccsd_t=cc_local.ccsd_t,
                    dcsd=cc_local.dcsd,
                    profile_info=None,
                    profile_pass=getattr(cc_local, 'profile_pass', None),
                )
                e_pt2_frag, e_cc_frag, e_cc_t_frag = res

                contribution = e_cc_frag + e_cc_t_frag + sign_pt2 * e_pt2_frag
                if include_mp2_correction:
                    contribution = contribution + domain_pt2
                return _weight * contribution

            t_frag = time.perf_counter()
            if verbose >= _VERBOSE_PROGRESS:
                n_atoms = int(np.asarray(
                    frag_prescreen.get('extended_primary_domain')).size)
                print(f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] '
                      f'domain={n_atoms} atoms, lo_size={len(fraglo_idx)}: '
                      f'starting fwd+bwd...', flush=True)

            e_frag, vjp_fn = jax.vjp(per_frag_fn, mf, lo_coeff)
            e_corr_local = e_corr_local + e_frag
            g_mf_i, g_lo_i = vjp_fn(jnp.float64(1.0))
            grad_mf = jax.tree_util.tree_map(_add_cotangent, grad_mf, g_mf_i)
            grad_lo = jax.tree_util.tree_map(_add_cotangent, grad_lo, g_lo_i)

            if verbose >= _VERBOSE_PROGRESS:
                print(f'  [rank {rank}] [frag {ifrag+1}/{nfrag}] done in '
                      f'{time.perf_counter() - t_frag:.2f} s, '
                      f'contribution = {float(e_frag):+.8f}', flush=True)

        # ---- MPI Allreduce of energy and cotangents ----
        t0 = time.perf_counter()
        e_corr_total = comm.allreduce(float(e_corr_local), op=MPI.SUM)
        grad_mf = _tree_allreduce_sum(comm, grad_mf)
        grad_lo = _tree_allreduce_sum(comm, grad_lo)
        log(f'  MPI Allreduce (E + cotangents): {time.perf_counter() - t0:.2f} s')

        # ---- Close out lo and scf vjps (redundant on each rank; deterministic) ----
        t0 = time.perf_counter()
        grad_mf_via_lo, = lo_vjp(grad_lo)
        grad_mf = jax.tree_util.tree_map(_add_cotangent, grad_mf, grad_mf_via_lo)
        log(f'  LO vjp close-out:           {time.perf_counter() - t0:8.2f} s')

        t0 = time.perf_counter()
        grad_mol, = scf_vjp(grad_mf)
        log(f'  SCF (CPHF) vjp close-out:   {time.perf_counter() - t0:8.2f} s')

        e_total = e_hf + e_corr_total
        log(f'DLNOCCSD.value_and_grad (MPI, nproc={nproc}): done in '
            f'{time.perf_counter() - t_overall:.2f} s total, '
            f'e_total = {float(e_total):.10f}')
        return e_total, grad_mol
