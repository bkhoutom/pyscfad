"""Select direct fragment integrals or resident H reuse."""
import os


def cache_mode():
    mode = os.environ.get('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'half').lower()
    if mode not in ('direct', 'half'):
        raise ValueError('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE must be direct or half')
    return mode


def configure_cache(scope, static):
    """Reuse H_ED for every LIS in the group unless direct mode is requested."""
    from .lis import _validate_lis_space
    _validate_lis_space(static.lis_static.lis_space)
    mode = cache_mode()
    scope.receipt['cache_mode'] = mode
    if mode == 'half':
        from pyscfad.lno._df_half import HalfTransformedDFCache
        scope.half = HalfTransformedDFCache(scope)
