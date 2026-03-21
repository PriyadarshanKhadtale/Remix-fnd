#!/bin/sh
# Stable launch on macOS: thread env + absolute path (works from any cwd).
set -e
BACKEND_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$BACKEND_DIR/.." && pwd)
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
exec python3 "$BACKEND_DIR/run.py" "$@"
