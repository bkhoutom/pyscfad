# Make OMol 3TDX test1 complete under Caltech's existing mapping limit

User scope: `TEST_general/dlno_stc_test/cluster_test/dlno_ccsdt/omol_3dtx_mp2/test1`.
Baseline: `11c00a34c68e3f08a99e989de234cb825999b3e9`; failed job 6952944.

## Acceptance

Run the unchanged scientific inputs through complete deterministic domain-MP2
energy and nuclear gradient on Caltech, with `vm.max_map_count=65530`, automatic
JAX cache clearing disabled, all domains and weak pairs, and the existing
RunRecorder completion checks. Reuse a copy of the SCF density checkpoint, but
reconverge SCF and retain live orbital and geometry response. Save mapping,
memory, timing, source hashes, job receipts, energy, and gradient artifacts.
Use isolated source snapshots; leave the active remote production checkout alone.

## Tasks

1. Instrument the baseline and replay test1 from its checkpoint on an Ice Lake
   compute node. Measure maps at phase boundaries and weak endpoint operations,
   and sample externally through process failure. Expected: identify the
   accumulating executable families and confirm or reject mapping exhaustion.
2. Add a small failing regression for the measured cause and implement the
   narrowest structural fix. Preserve ambient derivatives, independent target
   leaves, complex conjugation, and cached primal reuse. Run numerical oracles
   and a fresh-process mapping probe. Expected: correct response with materially
   fewer compiled variants; no budget-driven cache clearing.
3. Run full test1 with the fix on Caltech. Expected: `status=complete`, finite
   192-by-3 gradient, every MP2/DF group, single shared SGX geometry response,
   correct energy accounting, and mapping headroom. Inspect performance and
   validate relevant small-case gradients before integrating the fix locally.

## Review focus

No frozen response, altered selection thresholds, or omitted weak pairs. Check
real/asymmetric/complex adjoints, empty and multiple occupied targets, and JAX
transform compatibility. Distinguish cold compilation and SCF-restart timing
from numerical speedup. A baseline failure alone is not acceptance.
