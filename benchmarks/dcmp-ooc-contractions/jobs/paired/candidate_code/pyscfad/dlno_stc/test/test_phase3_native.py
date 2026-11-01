"""Final native surface contains only selected and complete contractions."""
from pyscfad.dlno_stc.test.test_backend import native


def test_weighted_binding_removed():
    module = native()
    assert hasattr(module, 'solve_selected')
    assert hasattr(module, 'solve_full')
    assert not hasattr(module, 'solve')
