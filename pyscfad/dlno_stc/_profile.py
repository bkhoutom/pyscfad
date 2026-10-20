"""Small eager reverse boundaries for opt-in STC timing estimates."""

import jax
from pyscfad.tools import resource_profile


def response(phase, fun, *args, **details):
    """Retain the ordinary linearization and time its single concrete VJP.

    Callers pass differentiated inputs explicitly and close over static choices
    only. JIT tracing emits no runtime estimate; the production STC boundary is
    eager. Parent sections remain inclusive of these component estimates.
    """
    if not resource_profile.enabled():
        return fun(*args)

    @jax.custom_vjp
    def wrapped(*args):
        return fun(*args)

    def forward(*args):
        return jax.vjp(fun, *args)

    def backward(pullback, seed):
        concrete = not any(isinstance(x, jax.core.Tracer)
                           for x in jax.tree_util.tree_leaves((pullback, seed)))
        before = None
        if concrete:
            jax.block_until_ready(seed)
            before = resource_profile.start()
        bars = pullback(seed)
        if concrete:
            jax.block_until_ready(bars)
            resource_profile.finish(phase, before, **details)
        return bars

    wrapped.defvjp(forward, backward)
    return wrapped(*args)
