# Copyright 2023-2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mpi4py import MPI
import numpy
from pyscfad.ops import stop_trace
from pyscfad.lno import lno_base, _profile
from pyscfad.lno.tools import autofrag, map_lo_to_frag

def partition_jobs(frag_lolist, frag_wghtlist):
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    nfrag = len(frag_lolist)
    indices = [i for i in range(nfrag) if i % nproc == rank]
    lolist = [frag_lolist[i] for i in indices]
    wghtlist = [frag_wghtlist[i] for i in indices]
    return lolist, wghtlist, indices


def _partition_optional_list(values, indices):
    if values is None:
        return None
    return [values[i] for i in indices]


class LNO(lno_base.LNO):
    def kernel(self,
               frag_lolist=None,
               frag_wghtlist=None,
               frag_atmlist=None,
               lo_type=None,
               no_type=None,
               frag_nonvlist=None,
               orbloc=None,
               lo_init_guess=None,
               lo_symmetry=False,
               lo_options=None,
               job_partition_list=None):
        if lo_type is None:
            lo_type = self.lo_type
        if no_type is None:
            no_type = self.no_type
        if orbloc is None:
            orbloc = self.get_lo(lo_type=lo_type, init_guess=lo_init_guess,
                                 symmetry=lo_symmetry,  options=lo_options)

        # LO assignment to fragments
        if frag_lolist is None:
            if frag_atmlist is None:
                #log.info('Grouping LOs by single-atom fragments')
                frag_atmlist = stop_trace(autofrag)(self.mol)
            else:
                #log.info('Grouping LOs by user input atom-based fragments')
                pass
            frag_lolist = stop_trace(map_lo_to_frag)(self.mol, orbloc, frag_atmlist,
                                                          verbose=self.verbose)
        elif frag_lolist == '1o':
            #log.info('Using single-LO fragment')
            frag_lolist = numpy.arange(orbloc.shape[1]).reshape(-1,1)
        else:
            #log.info('Using user input LO-fragment assignment')
            pass

        if job_partition_list is not None:
            tmp = []
            for i in job_partition_list:
                tmp.append(frag_lolist[i])
            frag_lolist = tmp

        nfrag = len(frag_lolist)
        if frag_wghtlist is None:
            frag_wghtlist = numpy.ones(nfrag)
        else:
            assert len(frag_wghtlist) == len(frag_lolist)

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        rank = comm.Get_rank()
        root_verbose = comm.bcast(int(self.verbose) if rank == 0 else None, root=0)
        root_collect_profile = False
        if rank == 0:
            root_collect_profile = bool(getattr(self, 'profile_fragments', False))
        collect_profile = comm.bcast(root_collect_profile, root=0)
        print_profile = collect_profile and root_verbose >= 2 and nproc > 1

        frag_lolist_local, frag_wghtlist_local, indices = partition_jobs(
            frag_lolist, frag_wghtlist)
        frag_nonvlist_local = _partition_optional_list(frag_nonvlist, indices)

        profile_start = len(_profile.get_fragment_profile())
        orig_verbose = self.verbose
        orig_profile_fragments = bool(getattr(self, 'profile_fragments', False))
        orig_profile_print = bool(getattr(self, 'profile_print', True))
        orig_profile_mpi_indices = getattr(self, 'profile_mpi_indices', None)
        orig_profile_mpi_nfrag = getattr(self, 'profile_mpi_nfrag', None)
        orig_profile_mpi_print = bool(getattr(self, 'profile_mpi_print', False))
        profile_indices = tuple(indices)
        if print_profile:
            self.verbose = 0
            self.profile_fragments = True
            self.profile_print = False
            self.profile_mpi_indices = profile_indices
            self.profile_mpi_nfrag = nfrag
            self.profile_mpi_print = True

        try:
            frag_res_local = lno_base.kernel(
                self,
                orbloc,
                frag_lolist_local,
                frag_nonvlist=frag_nonvlist_local,
            )
        finally:
            self.verbose = orig_verbose
            self.profile_fragments = orig_profile_fragments
            self.profile_print = orig_profile_print
            self.profile_mpi_indices = orig_profile_mpi_indices
            self.profile_mpi_nfrag = orig_profile_mpi_nfrag
            self.profile_mpi_print = orig_profile_mpi_print

        if print_profile:
            local_rows = _profile.get_fragment_profile()[profile_start:]
            local_rows = [
                _profile.remap_fragment_profile_row(
                    _profile.sanitize_fragment_profile_row_for_mpi(row),
                    profile_indices,
                    nfrag,
                )
                for row in local_rows
            ]
            gathered = comm.gather(local_rows, root=0)
            if rank == 0:
                rows = [row for rank_rows in gathered for row in rank_rows]
                rows.sort(key=_profile._profile_sort_key)
                _profile.print_fragment_profile_rows(rows)
        self._post_proc(frag_res_local, frag_wghtlist_local)
