"""Prepare frozen benchmark copies with identical phase-timing hooks.

This reads and rewrites source text only; it imports no scientific packages.
Production source is never instrumented by this helper.
"""
import ast
from pathlib import Path
import shutil


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def prepare(source, destination):
    shutil.copytree(source, destination)
    root = Path(destination) / 'pyscfad/df'
    path = root / '_df_jk_factorized.py'
    s = path.read_text()
    s = replace_once(s, '_SOLVE_MEMORY_MB = 64.0',
                     '_SOLVE_MEMORY_MB = 64.0\n_reverse_timing_hook = None')
    s = replace_once(s, '    started = time.perf_counter()',
                     '    started = time.perf_counter()\n    timing_hook = _reverse_timing_hook')
    s = replace_once(s, '    for packed in dfobj.loop(rows):',
                     '    initial_read_seconds = initial_unpack_seconds = initial_dense_seconds = 0.0\n'
                     '    next_started = time.perf_counter()\n'
                     '    for packed in dfobj.loop(rows):\n'
                     '        initial_read_seconds += time.perf_counter() - next_started\n'
                     '        unpack_started = time.perf_counter()')
    s = replace_once(s, '        B = lib.unpack_tril(packed)',
                     '        B = lib.unpack_tril(packed)\n'
                     '        initial_unpack_seconds += time.perf_counter() - unpack_started\n'
                     '        dense_started = time.perf_counter()')
    s = replace_once(s, '        if with_k:\n            del r\n    if end != naux:',
                     '        if with_k:\n            del r\n'
                     '        initial_dense_seconds += time.perf_counter() - dense_started\n'
                     '        next_started = time.perf_counter()\n    if end != naux:')
    s = replace_once(s, '    auxmol = dfobj.auxmol',
                     '    df_started = time.perf_counter()\n    auxmol = dfobj.auxmol')
    s = replace_once(s, '    del low, q, g',
                     '    del low, q, g\n    metric_transform_ended = time.perf_counter()')
    s = replace_once(s, '    molbar, auxbar = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(',
                     '    density_setup_ended = time.perf_counter()\n'
                     '    molbar, auxbar = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader(')
    s = replace_once(s, '    def metric_cholesky(aux):',
                     '    coordinate_ended = time.perf_counter()\n\n    def metric_cholesky(aux):')
    end = '    auxbar = _cderi_vjp._tree_add(auxbar, metric_pullback(np.asarray(lowbar))[0])'
    s = replace_once(s, end, end + '''
    if timing_hook is not None:
        jax.block_until_ready((molbar, auxbar))
        df_ended = time.perf_counter()
        timing_hook(dict(
            kind='factorized_df', df_seconds=df_ended-df_started,
            factor_response_prefix_seconds=df_started-started,
            metric_transform_seconds=metric_transform_ended-df_started,
            density_setup_seconds=density_setup_ended-metric_transform_ended,
            coordinate_seconds=coordinate_ended-density_setup_ended,
            metric_reverse_seconds=df_ended-coordinate_ended,
            initial_read_seconds=initial_read_seconds,
            initial_unpack_seconds=initial_unpack_seconds,
            initial_dense_seconds=initial_dense_seconds,
            rank=int(rank), factor_response=bool(factor_response),
            with_j=bool(with_j), with_k=bool(with_k)))''')
    ast.parse(s)
    path.write_text(s)

    path = root / '_cderi_vjp.py'
    s = path.read_text()
    s = replace_once(s, '_INT3C_MO_VJP_BLOCK_MB = 512.0',
                     '_INT3C_MO_VJP_BLOCK_MB = 512.0\n_reverse_timing_hook = None')
    node = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)
                and n.name == '_int3c_mo_deriv_coords_vjp_from_z_reader')
    lines = s.splitlines(True)
    region = ''.join(lines[node.lineno-1:node.end_lineno])
    doc_end = node.body[0].end_lineno - node.lineno + 1
    region_lines = region.splitlines(True)
    region_lines[doc_end:doc_end] = [
        '    timing_hook = _reverse_timing_hook\n',
        '    helper_started = time.perf_counter()\n']
    changed = ''.join(region_lines)
    changed = replace_once(changed, '    profile = _profile_enabled()',
                           '    profile = _profile_enabled() or timing_hook is not None')
    old = '    return _coords_tree_like(mol, mol_coords), _coords_tree_like(auxmol, aux_coords)'
    changed = replace_once(changed, old, '''    result = _coords_tree_like(mol, mol_coords), _coords_tree_like(auxmol, aux_coords)
    if timing_hook is not None:
        timing_hook(dict(kind='coordinate_df', seconds=time.perf_counter()-helper_started,
                         density_seconds=density_seconds, native_seconds=native_seconds,
                         fallback_seconds=fallback_seconds, native_blocks=int(native_blocks),
                         fallback_blocks=int(fallback_blocks),
                         density_reader=read_density_aux_block is not None))
    return result''')
    s = replace_once(s, region, changed)
    ast.parse(s)
    path.write_text(s)

    path = root / '_df_jk_opt.py'
    s = path.read_text()
    s = replace_once(s, '_FAST_EXCHANGE_DM_DATA = OrderedDict()',
                     '_FAST_EXCHANGE_DM_DATA = OrderedDict()\n_reverse_timing_hook = None')
    old = "                             f'{time.perf_counter() - t_bwd:.2f} s')\n            return None, dm_bar"
    new = """                             f'{time.perf_counter() - t_bwd:.2f} s')
                if _reverse_timing_hook is not None:
                    _reverse_timing_hook(dict(kind='implicit_matvec',
                                             seconds=time.perf_counter()-t_bwd))
            return None, dm_bar"""
    s = replace_once(s, old, new)
    ast.parse(s)
    path.write_text(s)


if __name__ == '__main__':
    w = Path(__file__).resolve().parent
    prepare(w / 'baseline', w / 'jobs/paired/baseline_code')
    prepare(w / 'candidate', w / 'jobs/paired/candidate_code')
