"""Select direct fragment integrals or resident H reuse."""
import os


def cache_mode(*, default='half'):
    mode = os.environ.get('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', default).lower()
    if mode not in ('direct', 'half'):
        raise ValueError('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE must be direct or half')
    return mode


def configure_cache(scope, *, default='half'):
    """Choose the forward cache, preserving explicit user configuration."""
    mode = cache_mode(default=default)
    scope.receipt['cache_mode'] = mode
    if mode == 'half':
        from pyscfad.lno._df_half import HalfTransformedDFCache
        scope.half = HalfTransformedDFCache(scope)
