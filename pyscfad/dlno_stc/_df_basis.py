"""Static correlation fitting configuration, separate from the SCF DF tree."""


def basis_kwargs(correlation_auxbasis):
    """Keep the existing call path unchanged when no override is requested."""
    return ({} if correlation_auxbasis is None else
            {'correlation_auxbasis': correlation_auxbasis})


def selection_auxbasis(mf, correlation_auxbasis):
    """Inherit the run configuration on a detached selection callback only."""
    if correlation_auxbasis is None:
        return getattr(mf, '_stc_correlation_auxbasis', None)
    return correlation_auxbasis


def basis_identity(mf, correlation_auxbasis):
    from ._restart import scientific_digest
    basis = mf.with_df.auxbasis if correlation_auxbasis is None else correlation_auxbasis
    return scientific_digest(basis)


def validate_selection_basis(mf, static, correlation_auxbasis):
    saved = getattr(static, 'correlation_basis_identity', None)
    if saved is None and correlation_auxbasis is not None:
        # Existing exact-screening/LIS selections used the SCF fit. Multipole
        # domains depend only on orbitals/Fock and may be reused with any fit.
        from .ccsd import CCSDSelections
        if (isinstance(static, CCSDSelections)
                or getattr(static, 'pair_energy_model', None) == 'exact'):
            saved = basis_identity(mf, None)
    if saved is not None and saved != basis_identity(mf, correlation_auxbasis):
        raise ValueError('correlation_auxbasis differs from the saved selection; '
                         'rebuild selections with the requested fitting basis')


def can_reuse_scf_df(mf, correlation_auxbasis):
    """Whether the cached SCF source safely represents this correlation fit.

    Explicit overrides require identical specifications and a matching built
    auxiliary molecule. Automatic SCF basis selection and unattested sources
    stay on the separate direct path. This changes execution, not metadata.
    """
    fitting = mf.with_df
    get_source = getattr(fitting, '_get_cderi_source', None)
    source = get_source() if get_source is not None else fitting._cderi
    if source is None:
        return False
    if correlation_auxbasis is None:
        return True
    from ._restart import scientific_digest
    if (fitting.auxbasis is None or
            scientific_digest(fitting.auxbasis) != scientific_digest(correlation_auxbasis)):
        return False
    auxiliary = getattr(fitting, 'auxmol', None)
    if auxiliary is None:
        return False
    # Auxiliary centers must follow the same geometry leaves. Independently
    # edited/traced fitting exponents or coefficients are not represented by
    # the static basis specification and cannot attest an identical fit.
    if (getattr(auxiliary, 'coords', None) is not getattr(mf.mol, 'coords', None)
            or getattr(auxiliary, 'exp', None) is not None
            or getattr(auxiliary, 'ctr_coeff', None) is not None):
        return False
    from pyscfad.df.addons import make_auxmol
    import numpy
    expected = make_auxmol(mf.mol, correlation_auxbasis)
    # These are fixed integral descriptors, not differentiable geometry leaves.
    # Checking the actual descriptors also rejects manually supplied auxmols
    # whose public basis label happens to match the SCF configuration.
    return (auxiliary.cart == expected.cart
            and scientific_digest(auxiliary._basis) == scientific_digest(expected._basis)
            and numpy.array_equal(auxiliary._atm, expected._atm)
            and numpy.array_equal(auxiliary._bas, expected._bas)
            and numpy.array_equal(auxiliary._env, expected._env))
