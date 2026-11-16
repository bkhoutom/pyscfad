"""Share synthetic MP2 fixture grids with the CC API without duplicating them."""


def cc_options(baseline, matrix):
    baseline = {} if baseline is None else baseline.copy()
    matrix = {} if matrix is None else matrix.copy()
    roots = baseline.pop('laplace_roots', None)
    weights = baseline.pop('laplace_weights', None)
    matrix.pop('laplace_roots', None)
    matrix.pop('laplace_weights', None)
    return dict(baseline_controls=baseline, matrix_controls=matrix,
                mp2_grid=None if roots is None and weights is None else (roots, weights))
