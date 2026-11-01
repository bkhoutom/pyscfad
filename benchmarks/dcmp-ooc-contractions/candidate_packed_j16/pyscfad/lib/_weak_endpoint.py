"""Reuse an endpoint's orbital linearization without retaining AO tapes."""

import jax


def prepare_endpoint(build_screen, build_moments, mf, common):
    """Build moments in the same orbital frame as their saved reverse rule.

    The screen and its pullback come from one evaluation. Only orbital
    algebra residuals persist; no AO-moment derivative tape is constructed
    during this forward phase.
    """
    screen, screen_pullback = jax.vjp(build_screen, common)
    endpoint = build_moments(mf, screen)
    return endpoint, (screen, screen_pullback)


def replay_endpoint(build_moments, mf, linearization):
    """Replay moments using saved orbitals and compose the two reverse rules.

    Differentiate the screen input as well as the molecular input: the
    saved orbitals are not frozen with respect to the common data. Callers
    release this AO tape and the saved orbital pullback after applying it.
    """
    screen, screen_pullback = linearization
    endpoint, moment_pullback = jax.vjp(build_moments, mf, screen)

    def pullback(endpoint_bar):
        mf_bar, screen_bar = moment_pullback(endpoint_bar)
        common_bar, = screen_pullback(screen_bar)
        return mf_bar, common_bar

    return endpoint, pullback
