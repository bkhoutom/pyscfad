# Configurable weak-pair bucket size

**Goal / specification:** Make the weak-pair padding bucket an input, with
128 as the default, as requested by the user.

**Architecture:** Resolve the shared `pad_endpoint` bucket from an explicit
Python argument, otherwise `PYSCFAD_WEAK_PAIR_BUCKET_SIZE`, otherwise 128.
All serial/MPI callers already use this helper. The C50 MP2 job exposes
`--weak-pair-bucket-size` through its existing command-line/@input.par parser
and passes the selected value through the environment before calculations.
The option is a numerical execution setting; domain selections and physical
thresholds are unchanged.

**Constraints:** Positive integers only. Preserve explicit Python overrides,
zero-moment padding, positive dummy gaps, and cotangent cropping. Record the
effective input and padding source identity. Reject stale installed libraries
that cannot honor the job input. Leave archived results unchanged.

## Tasks

- [x] Add failing shared-helper tests for default/override/invalid inputs,
  bucket boundaries, energy/gradient preservation, and compiled-shape reuse.
- [x] Implement the library resolver and document its environment input.
- [x] Add and test the C50 CLI/input.par setting and runtime metadata.
- [x] Run focused multipole, weak-endpoint, serial/MPI integration checks;
  review the diff and leave the changes in the user's working checkout.

**Validation scope:** Small numerical fixtures and input/preflight checks.
The expensive C50 cluster calculation is not needed to validate this input;
its memory-mapping failure still requires a separately monitored cluster run.

**Verification:** 118 focused library/job tests passed, covering energies,
endpoint gradients, compilation reuse, validation and input precedence. The
four-rank MPI wrapper passed with the environment override set to 64 (12
checks per rank). The C50 `--check-only` preflight passed with no bucket input
and reported 128 consistently in settings, runtime policy and environment;
its source identity included `_multipole.py`. Independent code review found
no correctness issues. No production calculation was submitted.
