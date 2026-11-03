"""C20 SCF and one orbital-response pullback, with preconditioning enabled.

The scalar probe is 0.5*||W.T C_occ||_F**2 for fixed seeded AO weights W.
It is invariant to occupied-orbital rotations and has a nonzero SCF response.
Its coordinate derivative is a diagnostic, not a correlated molecular force.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import resource
import time

from benchmark_affinity import configure_cpu_workload, ensure_worker_stack, check_worker_affinity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xyz', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--max-memory-mb', type=float, required=True)
    parser.add_argument('--df-memory-mb', type=float, required=True)
    args = parser.parse_args()
    cpus = configure_cpu_workload(args.threads)
    os.environ['PYSCFAD_PROFILE_BACKWARD_PHASES'] = '1'
    os.environ['PYSCFAD_PRECOND_CACHE_MEMORY_MB'] = str(args.max_memory_mb * 1e6 / 1024**2)
    os.environ['PYSCFAD_SCF_OCCUPIED_CACHE'] = '0'
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pyscf import gto as pgto, scf as pscf
    import pyscfad
    from pyscfad import config, gto, scf
    from pyscfad.df import _df_jk_opt
    from pyscfad.scf import hf
    from pyscfad._src.scipy.sparse import linalg as solver_impl
    from pyscfad.lib import _cotangent_buffer
    from pyscfad.lib._threading import dense_blas_scope
    from pyscfad.tools.linear_solver import eager_gmres

    expected_source = Path(os.environ['LNO_BENCHMARK_SOURCE_ROOT'])
    if not Path(pyscfad.__file__).resolve().is_relative_to(expected_source):
        raise RuntimeError('SCF worker did not import its immutable snapshot')
    output = args.output
    output.mkdir(exist_ok=True)
    started = time.perf_counter()
    timings = {}
    statistics = dict(preconditioner_setups=[], gmres_solves=[])

    def save(name, value):
        path = output/name
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        tmp.replace(path)

    def report(message):
        print(f'[BENCHMARK {time.perf_counter()-started:.1f}s] {message}', flush=True)

    config.update('pyscfad_scf_implicit_diff', True)
    config.update('pyscfad_scf_first_order_custom', False)
    config.update('pyscfad_moleintor_opt', False)
    config.update('pyscfad_scf_df_precond', True)

    # Instrument existing implementations without changing their equations,
    # tolerances, thread settings, or the number of requested reverse passes.
    original_preconditioner = _df_jk_opt.gen_coulomb_precond

    def traced_preconditioner(obj, occupied, virtual, gaps):
        begin = time.perf_counter()
        plan = _df_jk_opt._precond_memory_plan(
            obj, occupied.shape[0], obj.get_naoaux(), occupied.shape[1], virtual.shape[1])
        function = original_preconditioner(obj, occupied, virtual, gaps)
        if function is None:
            raise RuntimeError(f'Preconditioner could not fit requested memory: {plan}')
        closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        cached = closure['cached'].cell_contents is not None
        entry = dict(plan=plan, cached=cached, setup_seconds=time.perf_counter()-begin,
                     application_count=0, application_seconds=0.)
        statistics['preconditioner_setups'].append(entry)
        report(f'Preconditioner ready: cached={cached}, setup={entry["setup_seconds"]:.3f}s')
        save('response_progress.json', statistics)

        def apply(vector):
            begin = time.perf_counter()
            value = function(vector)
            entry['application_count'] += 1
            entry['application_seconds'] += time.perf_counter()-begin
            return value
        return apply

    _df_jk_opt.gen_coulomb_precond = traced_preconditioner
    original_gmres = solver_impl.scipy_gmres

    def traced_gmres(operator, rhs, **kwargs):
        entry = dict(iterations=0, residual_history=[], rhs_norm=float(np.linalg.norm(rhs)),
                     rtol=kwargs.get('rtol'), atol=kwargs.get('atol'),
                     restart=kwargs.get('restart'), maxiter=kwargs.get('maxiter'),
                     preconditioner_present=kwargs.get('M') is not None)
        statistics['gmres_solves'].append(entry)

        def iteration(residual):
            entry['iterations'] += 1
            entry['residual_history'].append(float(residual))
            report(f'GMRES iteration {entry["iterations"]}: preconditioned residual {residual:.6e}')
            save('response_progress.json', statistics)

        kwargs.update(callback=iteration, callback_type='pr_norm')
        begin = time.perf_counter()
        solution, info = original_gmres(operator, rhs, **kwargs)
        entry.update(solve_seconds=time.perf_counter()-begin, info=int(info))
        begin = time.perf_counter()
        residual = np.asarray(operator @ solution) - rhs
        entry.update(true_residual_norm=float(np.linalg.norm(residual)),
                     residual_check_seconds=time.perf_counter()-begin)
        tolerance = max(float(kwargs['atol']), float(kwargs['rtol'])*entry['rhs_norm'])
        entry['residual_tolerance'] = tolerance
        entry['converged'] = info == 0 and entry['true_residual_norm'] <= tolerance*(1+1e-5)
        save('response_progress.json', statistics)
        if not entry['converged']:
            raise RuntimeError(f'Orbital response failed residual check: {entry}')
        return solution, info

    solver_impl.scipy_gmres = traced_gmres
    mol = gto.Mole(atom=str(args.xyz), basis='cc-pvtz', unit='Angstrom',
                   max_memory=args.max_memory_mb, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    if symbols.count('C') != 20 or symbols.count('H') != 42:
        raise ValueError('This isolated benchmark requires C20H42')
    nocc = mol.nelectron//2
    refmol = pgto.M(atom=str(args.xyz), basis='cc-pvtz', unit='Angstrom',
                    max_memory=args.max_memory_mb, verbose=0)
    dfobj = pscf.RHF(refmol).density_fit().with_df
    dfobj._compatible_format = True
    cderi = output/'scf_cderi.h5'
    dfobj._cderi_to_save = str(cderi)
    dfobj.max_memory = args.df_memory_mb
    report('BEGIN DF construction')
    begin = time.perf_counter()
    dfobj.build()
    timings['df_build_seconds'] = time.perf_counter()-begin
    report(f'END DF construction: {timings["df_build_seconds"]:.3f}s')

    forward = {}

    def scf_outputs(m):
        mf = scf.RHF(m).density_fit()
        mf.with_df.incore = False
        mf.with_df.max_memory = args.df_memory_mb
        mf.with_df.precond_total_memory = args.max_memory_mb
        mf.with_df.precond_workspace_memory = args.df_memory_mb
        mf.with_df.attach_outcore_cderi(str(cderi))
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-9, 1e-6, None
        with eager_gmres():
            mf.kernel()
        if not mf.converged:
            raise RuntimeError('C20 SCF did not converge')
        forward['converged'] = bool(mf.converged)
        return mf.e_tot, mf.mo_coeff[:, :nocc]

    report('BEGIN SCF forward and VJP construction')
    begin = time.perf_counter()
    outputs, pullback = jax.vjp(scf_outputs, mol)
    jax.block_until_ready(outputs)
    timings['scf_forward_vjp_seconds'] = time.perf_counter()-begin
    coeff = np.asarray(outputs[1])
    weights = np.random.default_rng(71).normal(size=(mol.nao, 8))/np.sqrt(mol.nao)
    projections = weights.T @ coeff
    coefficient_bar = weights @ projections
    probe = float(.5*np.sum(projections**2))
    np.savez(output/'orbital_probe.npz', weights=weights, occupied_coefficients=coeff,
             occupied_coefficient_cotangent=coefficient_bar)
    settings = dict(basis='cc-pvtz', geometry=str(args.xyz), nao=mol.nao,
        naux=dfobj.get_naoaux(), nocc=nocc, scf_converged=forward['converged'],
        hf_energy_eh=float(outputs[0]), probe_value=probe, probe_seed=71,
        probe='0.5 * ||W.T @ C_occ||_F^2; W fixed seeded AO weights / sqrt(nao)',
        gradient_meaning='Coordinate derivative of the orbital probe, not correlated forces',
        energy_cotangent=0., reverse_passes=1, scf_df_precond=True,
        max_memory_mb=args.max_memory_mb, df_memory_mb=args.df_memory_mb,
        scf_energy_tolerance=1e-9, scf_gradient_tolerance=1e-6,
        threads=args.threads, cpu_affinity=cpus,
        snapshot_sha256=os.environ['LNO_BENCHMARK_SNAPSHOT_SHA256'])
    save('settings.json', settings)
    save('timings.json', timings)
    report(f'END SCF forward: {timings["scf_forward_vjp_seconds"]:.3f}s; EHF={float(outputs[0]):.12f}')
    report('BEGIN orbital response reverse: preconditioner ON only')
    begin = time.perf_counter()
    with _cotangent_buffer.reverse_scope(), dense_blas_scope():
        gradient, = pullback((jnp.zeros_like(outputs[0]), jnp.asarray(coefficient_bar)))
        jax.block_until_ready(gradient)
    timings['orbital_reverse_seconds'] = time.perf_counter()-begin
    coordinate_gradient = np.asarray(gradient.coords)
    if not np.isfinite(coordinate_gradient).all():
        raise RuntimeError('Nonfinite orbital-probe coordinate derivative')
    if not statistics['preconditioner_setups'] or not statistics['gmres_solves']:
        raise RuntimeError('The requested orbital response/preconditioner was not exercised')
    if not all(s['cached'] for s in statistics['preconditioner_setups']):
        raise RuntimeError('C20 preconditioner unexpectedly streamed; inspect its memory plan')
    np.save(output/'orbital_probe_coordinate_derivative.npy', coordinate_gradient)
    timings['total_seconds'] = time.perf_counter()-started
    result = dict(settings=settings, timings=timings, response=statistics,
        coordinate_derivative=coordinate_gradient.tolist(),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        affinity=check_worker_affinity(cpus),
        completed_at_local=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat())
    save('result.json', result)
    report(f'END orbital response: {timings["orbital_reverse_seconds"]:.3f}s')


if __name__ == '__main__':
    ensure_worker_stack()
    main()
