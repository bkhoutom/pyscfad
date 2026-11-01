"""Run reverse-path scientific and fallback regressions on a compute node."""
import os
from pathlib import Path
import pyscfad
import pytest

assert Path(pyscfad.__file__).resolve().is_relative_to(Path('code').resolve())
print('Package under test:', pyscfad.__file__, flush=True)
raise SystemExit(pytest.main([
    '-q',
    'code/pyscfad/df/test/test_reverse_packed_density.py',
    'code/pyscfad/df/test/test_reverse_dense_density.py',
    'code/pyscfad/df/test/test_reverse_generic_geometry.py',
    'code/pyscfad/df/test/test_reverse_precond_cache.py',
    'code/pyscfad/df/test/test_scf_precond.py',
    'code/pyscfad/df/test/test_reverse_native_slices.py',
    'code/pyscfad/df/test/test_cderi_vjp.py',
    'code/pyscfad/df/test/test_df_jk_factors.py',
    'code/pyscfad/df/test/test_df_jk_implicit.py',
    'code/pyscfad/df/test/test_df_jk_activity.py',
    'code/pyscfad/df/test/test_df_jk_lowrank.py',
    'code/pyscfad/df/test/test_df_incore.py',
    'code/pyscfad/mp/test/test_reverse_saved_lov.py',
    'code/pyscfad/mp/test/test_dfmp2_outcore.py',
    'code/pyscfad/mp/test/test_dfmp2_native.py',
    f'--junitxml=results.{os.environ["SLURM_JOB_ID"]}.xml',
]))
