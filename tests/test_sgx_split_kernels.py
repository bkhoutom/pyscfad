"""Compare split FP32 kernels directly with the original expression graphs."""
import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from pyscfad.sgx import _native


KERNELS = ('VD_F_D', 'VD_F_F', 'VD_G_D', 'VD_G_F', 'VD_G_G', 'VRR_G_F', 'VRR_G_G')
ROOT = Path(__file__).resolve().parents[1] / 'pyscfadlib/pyscfadlib/sgx'


@pytest.fixture(scope='module')
def direct_kernels(tmp_path_factory):
    library = _native.library()
    compiler = shutil.which('g++')
    if library is None or compiler is None:
        pytest.skip('Built SGX library and a C++ compiler required')
    directory = tmp_path_factory.mktemp('sgx-split-reference')
    references, wrappers = [], []
    for name in KERNELS:
        # Reconstruct the original FP32 source independently of the splitter.
        source = (ROOT / 'generated' / (name + '.C')).read_text().replace('double', 'float')
        source = re.sub(r'(?<![\w.])(\d+\.\d*|\.\d+)([eE][+-]?\d+)?(?![\w.])',
                        lambda match: match[0] + 'f', source)
        syntax = re.sub(r'//[^\n]*', '', source)
        declaration = re.search(r'void ' + name + r'\((.*?)\)\s*\{', syntax, re.DOTALL)[1]
        declaration = re.sub(r'//[^\n]*', '', declaration)
        declaration = re.sub(r'\bT\b', 'float', declaration)
        declaration = ', '.join(arg.strip() for arg in declaration.split(','))
        arguments = ', '.join(re.search(r'(\w+)$', arg)[1]
                              for arg in declaration.split(','))
        function = name + ('<float>' if name.startswith('VRR') else '')
        wrapper = (f'extern "C" void check_{name}({declaration}) {{\n'
                   f'  pyscfad_sgx::fp32::{function}({arguments});\n}}\n')
        reference = directory / (name + '.cpp')
        source = source.replace('namespace pyscfad_sgx {',
                                'namespace pyscfad_sgx { namespace reference {')
        reference.write_text(source + '\n}\n' + wrapper.replace('::fp32::', '::reference::'))
        references.append(str(reference))
        header = name + '.h' if name.startswith('VRR') else 'derivatives.h'
        wrappers.append(f'#include "{header}"\n' + wrapper)
    wrapper_path = directory / 'wrappers.cpp'
    wrapper_path.write_text('\n'.join(wrappers))
    common = [compiler, '-std=c++17', '-O0', '-shared', '-fPIC',
              '-I' + str(ROOT / 'generated/fp32')]
    reference_path = directory / 'reference.so'
    actual_path = directory / 'actual.so'
    subprocess.run(common + references + ['-o', str(reference_path)], check=True,
                   capture_output=True, text=True)
    native_path = Path(library._name).resolve()
    subprocess.run(common + [str(wrapper_path), str(native_path),
                   '-Wl,-rpath,' + str(native_path.parent), '-o', str(actual_path)],
                   check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(reference_path)), ctypes.CDLL(str(actual_path))


@pytest.mark.parametrize('name', KERNELS)
@pytest.mark.parametrize('npb,nprim,ng', [(1, 1, 1), (2, 6, 7), (2, 6, 128)])
def test_split_expression_graphs(direct_kernels, name, npb, nprim, ng):
    rng = np.random.default_rng(811)
    la, lb = ('SPDFG'.index(letter) for letter in name.split('_')[1:])
    derivative = name.startswith('VD')
    nm = la + lb + int(derivative)
    p = rng.uniform(.8, 2., nprim).astype('f')
    # Positive, smoothly decaying Boys-like seeds and signed coefficients.
    boys = (rng.uniform(.05, .2, (nprim, 1, ng))
            / (2 * np.arange(nm + 1)[None, :, None] + 1)).astype('f')
    boys[1::2] *= -.7
    grids = [rng.uniform(-.2, .2, (nprim, ng)).astype('f') for _ in range(3)]
    centers = [rng.uniform(-.2, .2, nprim).astype('f') for _ in range(3)]
    exponents = [rng.uniform(.4, 1.5, nprim // npb).astype('f'),
                 rng.uniform(.4, 1.5, npb).astype('f')]
    arrays = [boys, p, *grids, *centers]
    shape = ((7 if derivative else 1) * (la + 1) * (la + 2) // 2
             * (lb + 1) * (lb + 2) // 2, ng)
    ptr = ctypes.c_void_p
    argtypes = [ptr] * 9 + [ctypes.c_float] * 3
    if derivative:
        argtypes += [ptr] * 2 + [ctypes.c_int] * 3
    else:
        argtypes += [ctypes.c_int] * 2
    results = []
    for library in direct_kernels:
        out = np.full(shape, np.nan, dtype='f')
        fn = getattr(library, 'check_' + name)
        fn.argtypes, fn.restype = argtypes, None
        args = [ptr(a.ctypes.data) for a in [out, *arrays]] + [.1, -.2, .15]
        if derivative:
            args += [ptr(a.ctypes.data) for a in exponents] + [npb]
        fn(*args, nprim, ng)
        assert np.isfinite(out).all()
        results.append(out)
    np.testing.assert_allclose(results[1], results[0], atol=2e-6, rtol=3e-5)
