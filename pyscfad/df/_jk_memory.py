"""Explicit bounded J/K workspace independent of unrelated resident state."""

import math

from pyscf import lib


def workspace_memory(dfobj):
    """Available decimal MB for the existing block planners.

    Ordinary callers retain PySCF's process-ceiling interpretation of
    ``max_memory``. An explicit workspace is advisory scratch, not an RSS cap;
    the native planners still reserve only 22--30% for their primary panels.
    """
    workspace = getattr(dfobj, 'jk_workspace_memory', None)
    if workspace is None:
        return dfobj.max_memory - lib.current_memory()[0]
    if not math.isfinite(workspace) or workspace <= 0:
        raise ValueError('jk_workspace_memory must be finite and positive (decimal MB)')
    return float(workspace)


def forward_df(dfobj):
    """Give PySCF's unchanged J/K planner an explicit scratch allowance."""
    if getattr(dfobj, 'jk_workspace_memory', None) is None:
        return dfobj
    workspace = workspace_memory(dfobj)
    # StreamObject.copy preserves the attached factor source. copy.copy would
    # invoke DF's pickle hooks, which deliberately reset factors and auxmol.
    view = dfobj.copy()
    view.max_memory = lib.current_memory()[0] + workspace
    return view
