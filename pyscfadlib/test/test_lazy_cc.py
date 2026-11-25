"""Legacy triples remain available without eagerly loading their runtime."""
import os
import subprocess
import sys
from types import SimpleNamespace
from pyscfadlib import lib


def test_lazy_cc_loads_once(monkeypatch):
    calls = []
    def load(name):
        calls.append(name)
        return SimpleNamespace(first=1, second=2)
    monkeypatch.setattr(lib, 'load_library', load)
    proxy = lib._LazyCDLL('legacy')
    assert not calls
    assert proxy.first == 1
    assert proxy.second == 2
    assert calls == ['legacy']


def test_dlno_import_does_not_load_legacy_cc():
    if sys.platform != 'linux':
        return
    code = '''
from pathlib import Path
import pyscfadlib
from pyscfad.dlno_stc import ccsd
assert 'libcc_vjp.so' not in Path('/proc/self/maps').read_text()
assert pyscfadlib.libcc_vjp.lnoccsdt_contract is not None
assert 'libcc_vjp.so' in Path('/proc/self/maps').read_text()
'''
    subprocess.run([sys.executable, '-c', code], env=os.environ.copy(), check=True)
