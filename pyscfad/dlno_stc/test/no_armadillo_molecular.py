"""Fresh-process deterministic DLNO-CCSD(T) force and restart acceptance."""
import sys,importlib.abc,json
from pathlib import Path
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name=='pyscfad.dlno_stc._stc_mp2':
            raise ImportError('Deterministic route imported legacy Armadillo extension')
sys.meta_path.insert(0,RejectLegacy())
import numpy as np
from pyscfad import config
from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test._cc_options import cc_options
config.update('pyscfad_dfccsd_native',True)
config.update('pyscfad_dfccsd_cpp',True)
config.update('pyscfad_scf_implicit_diff',True)
config.update('pyscfad_scf_first_order_custom',False)
mol=accept.molecule();mol.max_memory=8000
mf=accept.build_mf(mol);domains=accept.selection(mf)
controls=dict(accept.controls(),threads=2)
options=dict(**cc_options(controls,controls),ccsd_t=True,triples_controls=dict(mode='deterministic'))
static=ccsd.build_static_selections(mf,domains,thresh_occ=1e-4,thresh_vir=3e-5,
                                    **cc_options(controls,controls))
checkpoint=Path(sys.argv[1])/'checkpoint'
e,grad=ccsd.value_and_grad(mol,accept.build_mf,static,checkpoint_dir=checkpoint,**options)
er,gr=ccsd.value_and_grad(mol,accept.build_mf,static,checkpoint_dir=checkpoint,resume=True,**options)
np.testing.assert_allclose(er,e,atol=1e-11,rtol=0)
np.testing.assert_allclose(gr.coords,grad.coords,atol=1e-9,rtol=1e-8)
maps=Path('/proc/self/maps').read_text()
assert 'libarmadillo' not in maps and 'libmkl_' not in maps and 'libcc_vjp' not in maps
assert 'pyscfad.dlno_stc._stc_mp2' not in sys.modules
Path(sys.argv[1]).joinpath('no_armadillo.json').write_text(json.dumps(dict(energy=float(e),gradient=np.asarray(grad.coords).tolist(),
    restart_energy_error=abs(float(er-e)),restart_gradient_error=float(np.max(np.abs(gr.coords-grad.coords))),
    legacy_extension_loaded=False,armadillo_loaded=False,mkl_loaded=False,libcc_vjp_loaded=False),indent=2))
print('Fresh-process molecular force and checkpoint restart passed',flush=True)
