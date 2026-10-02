#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${1:-./stc_demo}"
RANKS="${RANKS:-2}"
THREADS="${THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Set threading before Python, JAX, BLAS, or the external backend imports.
export OMP_NUM_THREADS="$THREADS"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON_BIN" "$SCRIPT_DIR/workflow.py" export --workdir "$WORKDIR"
mpiexec -n "$RANKS" "$PYTHON_BIN" "$SCRIPT_DIR/workflow.py" reference-solve --workdir "$WORKDIR"
"$PYTHON_BIN" "$SCRIPT_DIR/workflow.py" import --workdir "$WORKDIR"
