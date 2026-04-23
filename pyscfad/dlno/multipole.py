import jax.numpy as jnp
from .util import fake_mol_by_atom

__all__ = ['dipole_op', 'quadrupole_op', 'octupole_op']

def dipole_op(mol, R=jnp.zeros((3,)), atmlst=None):
    fake_mol = fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        r = jnp.asarray(fake_mol.intor('int1e_r')).reshape(3,nao,nao)
    return r

def quadrupole_op(mol, R=jnp.zeros((3,)), atmlst=None):
    fake_mol = fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        rr = jnp.asarray(fake_mol.intor('int1e_rr')).reshape(3,3,nao,nao)
    r2 = jnp.trace(rr)

    rr = rr * 3
    for x in range(3):
        rr = rr.at[x,x].add(-r2)
    rr = rr * 0.5
    return rr

def octupole_op(mol, R=jnp.zeros((3,)), atmlst=None):
    fake_mol = fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        rrr = jnp.asarray(fake_mol.intor('int1e_rrr')).reshape(3,3,3,nao,nao)

    r2r_0 = jnp.trace(rrr, axis1=1, axis2=2)
    r2r_1 = jnp.trace(rrr, axis1=2, axis2=0)
    r2r_2 = jnp.trace(rrr, axis1=0, axis2=1)

    rrr = rrr * 5
    for x in range(3):
        rrr = rrr.at[:,x,x].add(-r2r_0)
        rrr = rrr.at[x,:,x].add(-r2r_1)
        rrr = rrr.at[x,x,:].add(-r2r_2)
    rrr = rrr * 0.5
    return rrr
