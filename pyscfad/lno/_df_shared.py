"""One explicitly scoped integral response for a local DF domain.

Each consumer still reverses its own MO transformation immediately: the LIS
orbital response is needed before the W response can be seeded. Only the
three-center integral and auxiliary-metric responses are delayed. Their
cotangents are summed in the common domain AO/auxiliary frame before one
coordinate contraction. Factored MO bars are spooled, so the scope never
retains a full auxiliary-by-AO-pair cotangent in RAM.

This is an opt-in eager first-order orchestration facility. Its caller must
add ``finish()``'s coordinate contribution exactly once. Ordinary standalone
VJPs are unchanged; scopes cannot be nested, reused, or left unfinished.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import tempfile

import h5py
import jax
import jax.scipy.linalg as jsp_linalg
import numpy
from pyscf import lib as pyscf_lib

from pyscfad.df import _cderi_vjp


_ACTIVE = ContextVar('pyscfad_local_df_integral_reverse', default=None)


def enabled():
    return os.environ.get('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', '0').lower() in (
        '1', 'true', 'yes', 'on')


def _same_domain(left, right):
    return (left.nao == right.nao and left.natm == right.natm
            and numpy.array_equal(left._bas, right._bas)
            and numpy.array_equal(left._atm, right._atm)
            and numpy.array_equal(numpy.asarray(left.atom_coords()),
                                  numpy.asarray(right.atom_coords()))
            and numpy.array_equal(left._env, right._env))


class LocalDFReverse:
    def __init__(self, mf, atoms):
        from .df import make_local_mol
        from pyscfad.df.addons import make_auxmol
        self.atoms = numpy.asarray(atoms, dtype=numpy.int32)
        self.global_natm = mf.mol.natm
        self.mol = make_local_mol(mf.mol, self.atoms)
        self.auxmol = make_auxmol(self.mol, mf.with_df.auxbasis)
        self.max_memory_mb = float(mf.mol.max_memory)
        self.workspace_mb = float(mf.with_df.max_memory)
        self.terms = []
        self.metric_bar = None
        self._file = self._handle = None
        self.finished = False
        self.receipt = dict(consumers=0, spooled_bytes=0, geometry_passes=0,
                            metric_passes=0, max_density_rows=0)

    def accepts(self, mol, auxmol):
        coordinate_only = all(x.coords is not None and x.exp is None
                              and x.ctr_coeff is None and x.r0 is None
                              for x in (mol, auxmol))
        return (coordinate_only and _same_domain(mol, self.mol)
                and _same_domain(auxmol, self.auxmol))

    def _rows(self, pair_count, orbital_width=1):
        # Include the temporary density, symmetrization, accumulator, and
        # derivative buffers. One indivisible auxiliary shell may be larger;
        # the native driver already partitions reads within that shell.
        from ._df_direct import _local_direct_int3c_block_mb
        remaining = self.max_memory_mb - pyscf_lib.current_memory()[0]
        budget = min(self.workspace_mb, .5 * max(0., remaining),
                     _local_direct_int3c_block_mb()) * 1e6
        row_bytes = 8 * (6*self.mol.nao**2 + pair_count
                         + self.mol.nao*orbital_width + 3*self.mol.nao)
        if budget < row_bytes:
            raise MemoryError('Insufficient user memory budget for one local DF reverse row')
        return min(self.auxmol.nao, max(1, int(budget // row_bytes)))

    def add(self, coeff, z, low_bar, slices):
        if self.finished:
            raise RuntimeError('Cannot add to a finished local DF reverse')
        z = numpy.asarray(z)
        coeff = numpy.asarray(jax.device_get(coeff))
        if z.dtype != numpy.float64 or coeff.dtype != numpy.float64:
            raise TypeError('Shared local DF reverse requires real float64 inputs')
        k0, k1, l0, l1 = map(int, slices)
        pair_count = (k1-k0)*(l1-l0)
        if z.shape != (self.auxmol.nao, pair_count):
            raise ValueError('Shared local DF cotangent shape mismatch')
        rows = self._rows(pair_count, min(k1-k0, l1-l0))
        if self._file is None:
            self._handle = tempfile.TemporaryFile(dir=pyscf_lib.param.TMPDIR)
            self._file = h5py.File(self._handle, 'w', rdcc_nbytes=0)
        dataset = self._file.create_dataset(str(len(self.terms)), shape=z.shape,
                                            dtype='f8')
        for p0 in range(0, z.shape[0], rows):
            dataset[p0:p0+rows] = z[p0:p0+rows]
        self.terms.append((dataset, numpy.array(coeff, copy=True), tuple(slices)))
        if self.metric_bar is None:
            self.metric_bar = numpy.array(low_bar, copy=True)
        else:
            self.metric_bar += low_bar
        self.receipt['consumers'] += 1
        self.receipt['spooled_bytes'] += z.nbytes

    def _density(self, p0, p1):
        density = None
        for dataset, coeff, slices in self.terms:
            k0, k1, l0, l1 = slices
            z = numpy.asarray(dataset[p0:p1]).reshape(p1-p0, k1-k0, l1-l0)
            contribution = _cderi_vjp._int3c_mo_physical_density(
                coeff[:, k0:k1], coeff[:, l0:l1], z)
            if density is None:
                density = contribution
            else:
                density += contribution
            del z, contribution
        self.receipt['max_density_rows'] = max(
            self.receipt['max_density_rows'], p1-p0)
        return density

    def finish(self):
        if self.finished:
            raise RuntimeError('Local DF reverse already finished')
        self.finished = True
        coords = numpy.zeros((self.global_natm, 3))
        if not self.terms:
            return coords
        max_pairs = max(dataset.shape[1] for dataset, _, _ in self.terms)
        width = max(min(s[1]-s[0], s[3]-s[2]) for _, _, s in self.terms)
        rows = self._rows(max_pairs, width)
        self.receipt['tile_rows'] = rows
        # This reader returns the sum in the common domain AO frame. Integral
        # derivatives are generated/contracted once, after all consumers.
        mol_bar, aux_bar = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(
            self.mol, self.auxmol, numpy.zeros((self.mol.nao, 1)), None,
            (0, 1, 0, 1), int3c=self.mol._add_suffix('int3c2e'), aosym='s2ij',
            block_memory_mb=self.workspace_mb, z_dtype=numpy.float64,
            z_aux_block_max_rows=rows, read_density_aux_block=self._density)
        self.receipt['geometry_passes'] += 1
        _, pullback = jax.vjp(lambda a: jsp_linalg.cholesky(
            a.intor(a._add_suffix('int2c2e'), hermi=1), lower=True), self.auxmol)
        metric_bar, = pullback(self.metric_bar)
        self.receipt['metric_passes'] += 1
        local = (numpy.asarray(mol_bar.coords) + numpy.asarray(aux_bar.coords)
                 + numpy.asarray(metric_bar.coords))
        numpy.add.at(coords, self.atoms, local)
        self.terms.clear()
        self.metric_bar = None
        return coords

    def close(self):
        self.terms.clear()
        self.metric_bar = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@contextmanager
def reverse_scope(mf, atoms):
    if _ACTIVE.get() is not None:
        raise RuntimeError('Nested local DF reverse scopes are unsupported')
    scope = LocalDFReverse(mf, atoms)
    token = _ACTIVE.set(scope)
    try:
        yield scope
        if not scope.finished:
            raise RuntimeError('Local DF reverse must be finished and added by its owner')
    finally:
        _ACTIVE.reset(token)
        scope.close()


def defer_geometry(mol, auxmol, coeff, z, low_bar, slices):
    scope = _ACTIVE.get()
    if scope is None or not scope.accepts(mol, auxmol):
        return False
    scope.add(coeff, z, low_bar, slices)
    return True
