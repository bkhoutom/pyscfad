"""RHF coordinate derivatives with integral-direct DF-J + SGX-K.

Requires PySCF 2.14+ with optimized SGX gradients. The two-electron backend
uses no global three-index DF tensor, either in SCF or in its response.
The auxiliary metric, AO matrices and grid screening data remain quadratic
in system size. DF-J derivatives contract shell integrals directly without
AO-pair derivative panels, and reuse the forward metric factorization.
SGX contracts the density/response cross terms in a shared grid traversal,
reusing the overlap-fit factorization. Grid panels grow to supply at least
four native grid jobs per thread when the workspace and current memory
headroom permit. Available memory is rechecked when entering each grid
traversal, so the final SCF response can use space released by correlation.
During eager GMRES, its actual vector size, dtype and restart length determine
a temporary solver reservation (including not-yet-resident basis pages).
Planning also reserves AO matrices and native per-thread buffers and leaves
20 percent of headroom for other allocations. The total ceiling comes from
mol.max_memory (or the enclosing SCF-response scope); df_memory_mb below is
the J/K workspace allowance. Metric derivative tiles remain at most 32 MiB.

The default uses symmetric overlap fitting, a fixed level-2 SGX grid, per-pair
screening and FP64-only arithmetic. Set mixed_precision=True for mixed FP32/FP64
forward, response and geometry kernels. The default requires libsgx_ad
version 4 and uses independent density-dependent skip thresholds:
energy_tol=1e-10 and potential_tol=1e-8 (also the per-pair auto defaults).
Use the same screening settings with mixed_precision=False to isolate the
precision error. Keep those cutoffs fixed across system sizes and while tuning
fp32_multiplier (default 1e5). Density-response K disables density-dependent
skipping in either precision mode; mixed arithmetic is approximately linear.
Modern native AO/grid contractions retain every density/output pair.

J/K builds, eager response solves and final integral pullbacks use the current
PySCF thread count with dynamic team shrinking disabled inside those stages.
Set the allocation with pyscf.lib.num_threads (or the usual launch settings);
no molecule-specific or fixed CPU count is imposed. Caller settings are restored
afterward, including on failure.

The STC driver automatically joins complete SCF geometry contributions at
their coordinate producer. Equal densities share summed seeds; different
densities share one ordinary/derivative SGX integral traversal and one DF-J
derivative pass, with their distinct contractions retained. The explicit
capture/scope below demonstrates the same workflow for a standalone SCF.

Only real closed-shell RHF and eager first-order reverse AD are supported.
Implicit differentiation and reverse differentiation through iterations are
both available. Experimental custom first-order RHF, the global-DF response
preconditioner, JIT/JVP, basis derivatives, and range separation are rejected.
For MPI correlated calculations, pass parallel_scf_jk=False.

The LNO benchmark defaults to DF-J/SGX-K; use --scf-jk df for conventional DF.
Optional SGX settings include --sgx-grid-level 2 and --no-sgx-fit-overlap.
Omit --scf-df-precond with SGX. The existing --df-memory-mb controls both J/K
and correlation DF workspace; --df-storage/--df-builder do not build global SCF
factors with this backend. The correlation fitting basis remains independent.
The water queue accepts scf_jk='dfj-sgx', sgx_grid_level=2, and
sgx_fit_overlap=True in its JSON config. Changing config does not launch it.
"""
import jax
from pyscfad import config_update, gto, scf
from pyscfad.sgx import sgx_fit
from pyscfad.sgx import _shared

mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
               basis='def2-svp', verbose=0)
mol.build(trace_exp=False, trace_ctr_coeff=False)


def build_mf(current):
    # with_df remains available for the local correlation code. The J-fitting
    # basis can differ from this correlation fitting basis.
    mf = scf.RHF(current).density_fit(auxbasis='def2-svp-ri')
    mf = sgx_fit(mf, auxbasis='def2-universal-jfit', grid_level=2,
                 df_memory_mb=2000)
    # For either precision mode with the validated per-pair skip policy, add:
    # screening='per_pair', mixed_precision=False (FP64) or True (mixed).
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-11, 1e-8, None
    mf.kernel()
    return mf


with config_update('pyscfad_scf_implicit_diff', True):
    with _shared.capture_producers():
        mf, pullback = jax.vjp(build_mf, mol)
    seed = jax.tree_util.tree_map(jax.numpy.zeros_like, mf)
    seed.e_tot = jax.numpy.ones_like(mf.e_tot)
    with _shared.producer_scope(mf):
        gradient, = pullback(seed)
    energy = mf.e_tot
print('RHF energy / Eh:', float(energy))
print('Nuclear gradient / Eh per Bohr:\n', gradient.coords)
