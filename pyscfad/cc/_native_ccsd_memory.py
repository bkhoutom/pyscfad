# Copyright 2026 The PySCFAD Authors (Apache-2.0).
"""Live-buffer bounds for the integrated real RCCSD residual.

Inputs and solver history are already in process RSS. Each phase below counts
native owned storage, including resident buffers; only populated resident bytes
are credited against that total. Phase peaks are alternatives, not additive.
These are layout-independent bounds, not predictions of allocator/runtime RSS.
"""


def _tails(v, block):
    return {block, v % block} - {0}


def _private(m, n, k, threads):
    """Match parallel_gemm's reduction partition and private output storage."""
    if (m*n*k >= 200000 and threads > 1 and k//m > 3 and k//n > 3
            and m*n*threads <= k*max(m, n)):
        return 8*m*n*threads
    return 0


def residual_memory(o, v, block, ladder, *, resident=False, existing=None,
                    vvvv_copy_bytes=0, damping=False, short_lifetimes=True):
    """Bound one update, using the actual native ladder panel/thread plan.

    Only the reused voov block layout is cached, through the fvv contraction.
    Ordinary E() results/packing own storage outside the arena; C()/with_result()
    work uses that shared arena. The compatibility flag retains the older bound
    for an extension built before selective caching and lazy tau lifetimes.
    """
    o, v, block = int(o), int(v), int(block)
    if o < 1 or v < 1 or not 1 <= block <= v:
        raise ValueError('invalid CCSD memory dimensions or virtual block')
    existing = existing or {}
    threads = int(ladder['threads'])
    d, a = 8*o*o*v*v, 8*(o*v+o*o*v*v)
    occupied, quartic = 8*o**3*v, 8*o**4
    singles, oo, vv = 8*o*v, 8*o*o, 8*v*v
    fock = singles+oo+vv
    indices = 8*(v*v+o*(o+1)//2)
    integral = 8*(o*v**3+o**3*v) if resident else 0
    base = a+indices+integral+(2*d if resident else 0)+vvvv_copy_bytes
    credit = int(existing.get('populated_bytes', 0))
    capacity = int(existing.get('scratch_capacity_bytes', 0))
    phases, arenas = {}, {}

    def phase(name, live, scratch):
        nonlocal capacity
        # The allocation is reused, and keeps its largest capacity until the
        # update/solve ends. Growth drops the old allocation before replacing it.
        capacity = max(capacity, int(scratch))
        arenas[name] = capacity
        phases[name] = base+live+capacity

    def private(shapes):
        return max((_private(*shape, threads) for shape in shapes), default=0)

    phase('vvvv', 0, ladder['scratch_bytes'])
    phase('fock', fock+2*singles+
          max(oo+private([(o,o,v)]), vv+private([(v,v,o)])), 0)

    tile = min(32, block)
    unpacked = 8*tile*o*v*v if not resident else 0
    tau = 0 if resident else 2*d
    ovvv_private = private([shape for p in _tails(v,tile) for shape in
        [(1,v*v,o*p), (1,p*v,o*v), (o*o,v*o,p*v), (o,v,p*o*v)]])
    ovvv_scratch = 8*(max(o*o*v*v,tile*o*v*v,tile*o*o*v)+o*v+tile*v+24)
    # woo/tmp_acc remain alive while unpacking the second full w tensor.
    phase('ovvv', fock+tau+d+8*o*o*(v*(v+1)//2)+occupied+
          max(d,unpacked+ovvv_private), ovvv_scratch)

    db, ob, sb = 8*o*o*block*v, 8*o**3*block, 8*o*block
    live_groups=[]
    for p in _tails(v,block):
        dp, op, sp = 8*o*o*p*v, 8*o**3*p, 8*o*p
        live_groups += [
            (2 if short_lifetimes else 1)*dp+private([(v,v,p*o*o)]),  # theta_fock + cached voov
            dp+private([(o,o,p*v*o)]),             # cached voov already released
            op+dp+singles+private([(o,o*o*p,v)]),   # ordinary E -> tmp
            quartic+op+sp+private([(o,o**3,p)]),    # ordinary E -> woooo
            dp+singles+private([(1,o*p,o*v)]),      # singles E packing
            private([(1,o*o,o*p),(o*o,o*o,p*v)]),  # direct foo/woooo
        ]
        for q in _tails(v,block):
            # In the full-block path half-tau and response-tau have disjoint
            # lifetimes. Partial blocks retain both for reuse across p blocks.
            one_tau = d if short_lifetimes and block == v else 0
            live_groups += [one_tau+private([(p*o,o*v,o*q)]),
                            one_tau+dp+private([(p*o,q*o,o*v)])]  # VOov + response
    ordinary = max(live_groups)
    ovov_scratch = max(3*db,db+ob+singles)+192
    # Two w tensors persist. With partial blocks, the two reusable tau tensors
    # can overlap too; full-block tau storage is counted at its own consumers.
    working = 2*d+(2*d if block < v else 0) if short_lifetimes else 8*d
    phase('ovov', fock+tau+working+quartic+sb+ordinary, ovov_scratch)
    phase('oooo', fock+tau+quartic+
          private([(o,v,o*o*p) for p in _tails(v,block)]), db+ob+192)
    phase('finish', fock+quartic+oo+vv+
          private([(o,o,v),(v,v,o),(o,v,o),(o,v,v)]),
          d+max(singles,oo,vv)+192)
    # Norm/energy reductions and a damped update happen after native locals
    # have gone away. Resident buffers/arena still live through solver work.
    retained = integral+(2*d if resident else 0)+indices+(capacity if resident else 0)
    post = retained+(2 if damping else 1)*a+8*o*o*v
    phases['solver_updates'] = post
    return dict(phase_bytes=phases, arena_bytes=arenas,
                peak_bytes=max(phases.values()), populated_credit_bytes=credit,
                incremental_bytes=max(0,max(phases.values())-credit),
                retained_bytes=retained, integral_bytes=integral,
                amplitude_bytes=2*d, index_bytes=indices)


def input_bytes(o, v, naux):
    """Resident host ERIs in the current DF interface (distinct input layouts)."""
    p=v*(v+1)//2
    return 8*((o+v)**2+(o+v)+o**4+o**3*v+3*o*o*v*v+
              o*v*p+p*p+naux*(o*v+2*p))
