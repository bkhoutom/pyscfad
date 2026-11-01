"""Lazy persistent JAX executable caching for local-correlation drivers.

An explicitly configured JAX cache (or disabled caching) belongs to the caller.
Otherwise use the user's cache directory and retain even the small kernels
which dominate first-use weak-pair response. Configuration is process-wide;
later JAX compilations can reuse this cache as well.
"""
import os
from pathlib import Path
import warnings

import jax


_warned_paths = set()


def configure_compilation_cache():
    """Initialize an automatic cache without replacing a caller's policy.

    Defaults to ``$XDG_CACHE_HOME/pyscfad/jax`` or ``~/.cache/pyscfad/jax``.
    MPI launcher ranks use separate ``mpi-rank-N`` subdirectories to avoid
    concurrent writes by independent JAX processes within the MPI job.
    ``JAX_ENABLE_COMPILATION_CACHE=false`` opts out. Set a JAX cache directory
    explicitly to own its complete policy, including kernel-size thresholds.
    No configuration or filesystem work happens merely by importing this
    module, and a failed automatic cache never prevents a calculation.
    """
    if not jax.config.jax_enable_compilation_cache:
        return
    if jax.config.jax_compilation_cache_dir is not None:
        return

    root = os.environ.get('XDG_CACHE_HOME') or str(Path.home()/'.cache')
    directory = Path(root).expanduser()/'pyscfad'/'jax'
    # JAX's unbounded file cache writes entries without locking. MPI ranks
    # are independent JAX processes, so isolate their automatic caches before
    # the first compilation, including topology built through serial code.
    for name in ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK', 'SLURM_PROCID'):
        try:
            rank = int(os.environ.get(name, ''))
        except ValueError:
            continue
        if rank >= 0:
            directory /= f'mpi-rank-{rank}'
            break
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        if directory not in _warned_paths:
            warnings.warn(
                f'Cannot create automatic JAX compilation cache at {directory}: '
                f'{error}; continuing without persistent caching.',
                RuntimeWarning, stacklevel=2,
            )
            _warned_paths.add(directory)
        return

    # Only tune untouched JAX defaults for the automatic cache. Environment
    # settings and nondefault programmatic thresholds remain caller-owned.
    if ('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS' not in os.environ
            and jax.config.jax_persistent_cache_min_compile_time_secs == 1.0):
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    if ('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES' not in os.environ
            and jax.config.jax_persistent_cache_min_entry_size_bytes == 0):
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_compilation_cache_dir', str(directory))
