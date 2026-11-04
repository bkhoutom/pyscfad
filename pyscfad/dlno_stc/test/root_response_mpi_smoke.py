"""Numerical regression for optimized root-owned response with MPI owners."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch


def run(method, output):
    import jax
    import numpy as np
    from mpi4py import MPI
    from pyscfad import config_update, df, dlno_stc, gto, scf
    from pyscfad.df import _ooc_cotangent, mpi_df_jk
    from pyscfad.dlno_stc import ccsd
    from pyscfad.lib import _scf_response_cache
    from pyscfad.tools.linear_solver import eager_gmres

    comm = MPI.COMM_WORLD
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=10000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    path = str(output.with_suffix('.h5'))
    if comm.rank == 0:
        obj = df.DF(mol, auxbasis='weigend', incore=False)
        obj._cderi_to_save = path
        obj.build()
    comm.Barrier()
    builders = dict(scf=0, worker=0)

    def build(current, **initial):
        mf = scf.RHF(current).density_fit(auxbasis='weigend')
        mf.with_df.attach_outcore_cderi(path)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        if initial.get('mo_coeff_init') is not None:
            builders['worker'] += 1
            for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot'):
                setattr(mf, name, initial[name + '_init'])
            mf.converged = True
        else:
            builders['scf'] += 1
            with eager_gmres():
                mf.kernel()
        return mf

    roots, weights = np.polynomial.legendre.leggauss(32)
    controls = dict(mode='deterministic', threads=1,
                    laplace_roots=20*(roots+1), laplace_weights=20*weights)

    def select(mf):
        domains = dlno_stc.build_domain_selections(
            mf, frozen=1, force_full_domains=True,
            lo_kwargs=dict(conv_tol=1e-12, conv_tol_grad=1e-9))
        if method == 'mp2':
            return domains
        return ccsd.build_static_selections(mf, domains, matrix_kind='W',
            thresh_occ=1e-5, thresh_vir=1e-5, baseline_controls=controls,
            matrix_controls=controls, single_pass=True)

    finish = _ooc_cotangent.GeometryScope.finish
    receipts = []

    def finished(scope, bar):
        result = finish(scope, bar)
        receipts.append(dict(scope.stats))
        return result

    values = []
    for enabled in ('0', '1'):
        os.environ['PYSCFAD_STC_COMBINE_SCF_DF'] = enabled
        receipts.clear()
        with ExitStack() as stack:
            stack.enter_context(config_update('pyscfad_moleintor_opt', True))
            stack.enter_context(config_update('pyscfad_scf_df_precond', True))
            stack.enter_context(patch.object(_ooc_cotangent.GeometryScope, 'finish', finished))
            caches = stack.enter_context(_scf_response_cache.record_response_caches())
            # A one-rank communicator never activates the distributed
            # executor, even when requested; it keeps the same optimizations.
            options = dict(build_static=select, include_hf=True,
                           parallel_scf_jk=comm.size == 1, comm=comm)
            if method == 'mp2':
                result = dlno_stc.value_and_grad(mol, build, controls=controls, **options)
            else:
                result = ccsd.value_and_grad(mol, build, baseline_controls=controls,
                                             matrix_controls=controls, **options)
            jax.block_until_ready(result)
        assert _ooc_cotangent._scope.get() is None
        assert _scf_response_cache._current.get() is None
        assert _scf_response_cache.total_memory_mb() is None
        assert mpi_df_jk.get_active_executor() is None
        if comm.rank == 0:
            values.append(dict(energy=float(result[0]), gradient=np.asarray(result[1].coords).tolist(),
                               geometry=list(receipts), caches=caches))
        else:
            assert result is None and receipts == [] and caches == []
    counts = comm.gather(builders, root=0)
    if comm.rank == 0:
        np.testing.assert_allclose(values[1]['energy'], values[0]['energy'], atol=2e-11, rtol=0)
        np.testing.assert_allclose(values[1]['gradient'], values[0]['gradient'], atol=2e-8, rtol=0)
        assert values[0]['geometry'] == []
        assert len(values[1]['geometry']) == 1, values[1]
        assert values[1]['geometry'][0]['geometry_passes'] == 1
        assert values[1]['geometry'][0]['consumers'] > 1
        assert any(row['enabled'] and row['cache_hit_blocks'] > 0 for row in values[1]['caches'])
        assert counts[0]['scf'] == 2 and all(row['scf'] == 0 for row in counts[1:])
        output.write_text(json.dumps(dict(results=values, builders=counts), indent=2))
        print('root response smoke PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('mp2', 'ccsd'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.method, args.output)
    except BaseException:
        import traceback
        from mpi4py import MPI
        traceback.print_exc()
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
