# OMCB/cc-pVTZ canonical DF-CCSD(T) gradient feasibility audit

Date: 2026-08-22

This is a static code-and-shape audit of the current working tree, not a
production benchmark.  No OMCB/cc-pVTZ CCSD, CCSD(T), or CCSD(T)-gradient
calculation was launched, and no canonical energy or gradient has been
invented to fill the missing result.  The OMCB/STO-3G pilot is explicitly
superseded and is not a substitute for the requested basis.

The assessment includes commit `941891b4`, which implements the recent
auxiliary-blocked CDERI/AO-to-MO reverse transformation.  It also includes the
new factor-native triples and lambda changes in the working tree.  Those
changes remove the previously dominant global `ovvv`, `wvvov`, `vvop`, and
`vvop_bar` allocations on the real, symmetry-free DF path.  The limiting
objects are now the canonical CC amplitudes, response intermediates, and
reverse-mode tape, all of which scale as occupied-squared times
virtual-squared and remain in core.

## Reference dimensions and tensor payloads

The original example 13 uses the 36-atom OMCB geometry, cc-pVTZ, no frozen
orbitals, and `density_fit()` without an explicit auxiliary basis.  PySCF
therefore selects cc-pVTZ-JKFIT.

| quantity | all-electron value | frozen-C(1s) orientation |
|---|---:|---:|
| electrons / active occupied spatial orbitals | 96 / 48 | 96 / 36 |
| orbital AOs / spatial MOs | 696 / 696 | 696 / 684 active |
| active virtual orbitals | 648 | 648 |
| cc-pVTZ-JKFIT auxiliary functions | 1668 | 1668 |
| virtual pairs, `v(v+1)/2` | 210,276 | 210,276 |
| AO pairs, `nao(nao+1)/2` | 242,556 | 242,556 |

The following numbers are exact real-float64 payloads implied by these
dimensions.  They are not measured RSS and exclude allocator padding,
JAX/XLA copies, cache blocks, BLAS workspaces, and filesystem metadata.

| object | shape | all electron (GiB) | frozen C(1s) (GiB) | current role |
|---|---:|---:|---:|---|
| packed AO CDERI | `1668 x 242556` | 3.014 | 3.014 | streamed source |
| transformed `Lpq` | `1668 x nmo x nmo` | 6.020 | 5.814 | current transient output |
| `Lov` | `1668 x o x 648` | 0.387 | 0.290 | retained DF factor |
| packed `Lvv` | `1668 x 210276` | 2.613 | 2.613 | retained DF factor |
| one `t2`-, `l2`-, `ovov`-, or `oovv`-sized object | `o x o x 648 x 648` | 7.208 | 4.055 | remaining blocker |
| packed CC/DIIS vector | `ov(ov+1)/2 + ov` | 3.604 | 2.028 | in-core DIIS unit |
| packed `ovvv` | `o x 648 x 210276` | 48.730 | 36.547 | eliminated on factor path |
| dense `ovvv` or `wvvov` | `o x 648^3` | 97.310 | 72.982 | eliminated on factor path |
| `vvop` or `vvop_bar` | `648 x 648 x o x nmo` | 104.518 | 77.037 | eliminated on factor path |

Freezing the twelve carbon 1s orbitals is shown only to diagnose the scaling.
It is not the requested all-electron example-13 calculation and is not used as
a replacement result.

## Improvements included in this audit

For real float64 C1 calculations, the perturbative-triples energy now builds
only the rectangular `vvop` cache requested by a virtual tile.  Its virtual
part is evaluated directly from `Lov` and packed `Lvv`; the occupied part is a
tile of `ovov`.  In the pullback, each cache cotangent is contracted
immediately into `ovov_bar`, `Lov_bar`, and `Lvv_bar` and then discarded.
Consequently neither global packed/dense `ovvv` nor global `vvop` and
`vvop_bar` is required.

The standard DF lambda equations now build complementary `ovvv` tiles from
the same factors and contract the `wvvov-l2` term in two virtual tiles.  They
do not retain a global `ovvv` or `wvvov`.  The complete differentiated driver
also keeps `ovvv` lazy while its generalized response is evaluated.  Targeted
small-system tests verify that (i) factor-native triples energy and factor
cotangents agree with the former dense path and a directional finite
difference and (ii) tiled factor-native lambda updates and converged
multipliers agree with the dense equations.  These are correctness tests, not
OMCB timings.

The recent CDERI transformation fix is likewise active: the packed AO CDERI
is read in auxiliary slabs and its custom reverse transformation is
auxiliary-blocked.  The present canonical DF-CCSD constructor nevertheless
allocates the complete transformed `Lpq` output before extracting `Loo`,
`Lov`, and `Lvv`, so the 6.020-GiB all-electron transient still exists.

## Post-patch memory floors

The table below inventories simultaneous float64 payloads from array shapes in
the current algorithms.  A parenthesized number is an optimistic lower value
if every transpose/view aliases its source.  These are conservative stage
floors rather than predictions of peak RSS.

| stage inventory | all electron (GiB) | frozen C(1s) (GiB) |
|---|---:|---:|
| saved DF/four-index ERI fields plus one `t2` | 32.434 (25.226) | 19.375 (15.321) |
| factor-native triples VJP before virtual cache blocks | 58.126 | 34.893 |
| generalized amplitude-response conservative floor | 68.480 | 39.660 |
| response with an additional resident `ovvo`/`tau`-sized object | about 75.7 | about 43.7 |
| six DIIS solution/error pairs, additional payload | 43.253 | 24.331 |

The response floor consists of the saved ERIs and amplitude plus at least five
additional `o^2 v^2` payloads: response input/output or cotangent arrays and
the two `wVOov`/`wvOOv`-class intermediates.  A sixth such object is commonly
live while `ovvo` and `tau`-like work arrays are evaluated.  This is why the
new virtual-integral tiling does not make the gradient in-core feasible.

The host has 48 GiB of physical RAM, the current calculation guard is 36 GiB,
and approximately 94 GiB of local scratch is free.  The all-electron response
floor exceeds physical RAM before allocator copies.  Even the frozen-core
response floor exceeds the 36-GiB guard.  The frozen-core triples VJP appears
to lie just below the guard only before caches: a single high-index eight-row
`vvop` cache and its cotangent add 1.902 GiB, already lifting that stage above
36 GiB.  The all-electron triples VJP exceeds physical RAM before any cache is
allocated.

DIIS is a separate obstruction.  With a six-vector space, storing both the
solution and error vector costs 43.253 GiB all electron.  The current PySCFAD
DIIS implementation explicitly rejects a large vector when it falls out of
the in-core buffer; setting `incore_complete=True` merely forces the payload
back into RAM.

## Why density fitting alone is insufficient

Density fitting factorizes the four-index electron-repulsion integral and
avoids storing a canonical `v^4` integral tensor.  It does not factorize the
CCSD doubles amplitude, multiplier, or residual.  Each of those remains an
`o^2 v^2` object: 7.208 GiB for the requested active space.  It also does not
make the nonlinear CC iterations, DIIS history, or the transpose-Jacobian
amplitude response blocked or out of core.

The production generalized response currently forms a JAX VJP of one
`update_amps` map at the converged amplitudes and repeatedly applies that
transposed map.  Although the DF `ovvv` contractions inside the map are now
factor-native, the saved reverse tape and its `o^2 v^2` primal/cotangent
arrays are still resident.  Thus "DF integrals" and "out-of-core CC
response" are distinct requirements; only the former is present.

The storage improvement also does not reduce the formal arithmetic of the
canonical contractions.  The DF `vvvv-t2` application and its reverse pass
still repeat large virtual-space matrix products, and the canonical (T)
kernel still traverses 45,559,800 unique virtual triples for `v=648`, with
occupied contractions for each tile.  Factor-native caches change where the
integrals live, not the polynomial order of these contractions.  No wall-time
extrapolation is reported here because no production OMCB CC calculation was
run.

The installed conventional PySCF analytic (T) gradient is not an out-of-core
fallback for this case.  Its response/RDM path converts `ovvv` to a dense
array and constructs a full `v^4` density before compression.  For `v=648`,
the full `dvvvv` payload alone is 1,313.682 GiB, the compressed virtual-pair
density is 329.435 GiB, and the AO-pair density is 438.343 GiB.  Moreover, the
generic four-center gradient is not the derivative of the factorized DF
energy used by the PySCFAD calculation.

## Required remaining implementation

The next scale-enabling change is not another `ovvv` optimization.  It is an
out-of-core or occupied-pair-blocked amplitude and adjoint backend:

1. write `Loo`, `Lov`, and packed `Lvv` directly from auxiliary slabs so the
   full transformed `Lpq` output is never present;
2. keep `t2`, `l2`, residuals, `wVOov`/`wvOOv`, and their cotangents as
   occupied-pair blocks or restartable memmaps, with contractions scheduled so
   only a bounded number of blocks is resident;
3. replace the traced whole-update response with explicit blocked
   transpose-Jacobian actions that accept the generalized triples amplitude
   source;
4. implement truly out-of-core DIIS with streamed inner products and
   extrapolation;
5. bound triples caches and native workspaces against measured current RSS,
   and checkpoint the SCF gauge, amplitudes, triples sources, response, and DF
   factor bars in separate processes.

The 94-GiB local scratch limit also requires a preflight disk gate; a useful
blocked implementation must not rely on the much larger conventional
four-index ERI/RDM files.  Each milestone should first reproduce the existing
small-system energy, factor cotangents, generalized response, and nuclear
finite differences.  Only after those checks pass should the exact
OMCB/cc-pVTZ canonical reference be launched.

## Result-reporting consequence

The exact-basis MPI local-MP2 comparison is independent of this CC storage
barrier and is reported separately against canonical DF-MP2.  A
DLNO-CCSD(T) threshold sweep, however, requires a genuine canonical
DF-CCSD(T) energy and gradient reference.  The results notes therefore state
the present resource gate rather than inserting a value from a smaller basis,
a frozen-core variant, or an incomplete calculation.
