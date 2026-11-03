"""Large three-center integrals must not overflow native component strides."""
import numpy
import pytest

from pyscf import gto
from pyscfad.gto import _pyscf_moleintor as mi


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('aosym', ['s1', 's2ij'])
@pytest.mark.parametrize('intor,comp', [('int3c2e', 1), ('int3c2e_ip1', 3),
                                       ('int3c2e_ipip1', 9)])
@pytest.mark.parametrize('sliced', [False, True])
def test_blocked_integrals_match_native_and_reuse_output(monkeypatch, cart, aosym,
                                                        intor, comp, sliced):
    mol = gto.M(atom='O 0 0 0; H 0 -0.8 0.6; H 0 0.7 0.5', basis='cc-pvdz',
                cart=cart, verbose=0)
    intor = mol._add_suffix(intor)
    ao_loc = mi.make_loc(mol._bas, intor)
    shls = ((1, mol.nbas-1, 0, mol.nbas-1, 2, mol.nbas-1) if sliced
            else (0, mol.nbas, 0, mol.nbas, 0, mol.nbas))
    expected = mi._getints3c(intor, mol._atm, mol._bas, mol._env,
                            shls, comp, aosym)
    plane = int(numpy.prod(expected.shape[-3:-1]) if aosym == 's1'
                else expected.shape[-2])
    limit = plane * int(max(numpy.diff(ao_loc[shls[4]:shls[5]+1])))
    monkeypatch.setattr(mi, '_INT3C_MAX_COMPONENT_SIZE', limit)
    native = mi._getints3c
    calls = []

    def checked(*args, **kwargs):
        value = native(*args, **kwargs)
        assert value.size // comp <= limit
        calls.append(value.shape)
        return value

    monkeypatch.setattr(mi, '_getints3c', checked)
    output = numpy.empty(expected.size)
    actual = mi.getints3c(intor, mol._atm, mol._bas, mol._env,
                          shls if sliced else None, comp, aosym, out=output)
    assert len(calls) > 1
    assert numpy.shares_memory(actual, output)
    numpy.testing.assert_array_equal(actual, expected)


def test_oversized_shell_fails_before_native_call(monkeypatch):
    mol = gto.M(atom='He 0 0 0', basis='cc-pvdz', verbose=0)
    monkeypatch.setattr(mi, '_INT3C_MAX_COMPONENT_SIZE', 1)
    with pytest.raises(OverflowError, match='native indexing limit'):
        mi.getints3c('int3c2e_ip1_sph', mol._atm, mol._bas, mol._env, comp=3)
