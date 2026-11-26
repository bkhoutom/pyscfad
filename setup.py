"""Build deterministic kernels against OpenMP and SciPy's BLAS API."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import pybind11

        cmake = shutil.which('cmake')
        if cmake is None:
            raise RuntimeError('Building pyscfad requires CMake and an OpenMP-capable C++ compiler')
        source = Path(__file__).resolve().parent / 'pyscfad/cc/cpp'
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        temporary = Path(self.build_temp).resolve() / ext.name
        temporary.mkdir(parents=True, exist_ok=True)
        config = 'Debug' if self.debug else 'Release'
        subprocess.run([
            cmake, '-S', str(source), '-B', str(temporary),
            *shlex.split(os.environ.get('CMAKE_ARGS', '')),
            f'-DCMAKE_BUILD_TYPE={config}',
            f'-DPython_EXECUTABLE={sys.executable}',
            f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output}',
        ], check=True)
        command = [cmake, '--build', str(temporary), '--config', config,
                   '--target', '_ccsd_kernels']
        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            command += ['--parallel', str(self.parallel or 2)]
        subprocess.run(command, check=True)


setup(ext_modules=[Extension('pyscfad.cc._ccsd_kernels', sources=[])],
      cmdclass={'build_ext': CMakeBuild})
