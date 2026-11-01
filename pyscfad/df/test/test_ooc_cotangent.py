"""Deferred OOC geometry must preserve complete, independently active VJPs."""
import importlib
import importlib.util

import jax
import numpy as np
import pytest

from pyscfad import config_update, df, gto
from pyscfad.df import _df_jk_factorized as factorized
from pyscfad.lib import _cotangent_buffer


def coordinator():
    name = 'pyscfad.df._ooc_cotangent'
    assert importlib.util.find_spec(name) is not None, 'OOC reverse coordinator missing'
    return importlib.import_module(name)


@pytest.fixture
def source(tmp_path):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path/'cderi.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    return obj


def zeros(tree):
    return jax.tree_util.tree_map(np.zeros_like, tree)


def add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x+y, a, b)


def close(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b),strict=True):
        np.testing.assert_allclose(x, y, atol=3e-10, rtol=3e-10)


def test_equal_coordinate_views_with_different_strides_are_distinct():
    oc=coordinator()
    coords=np.eye(3)
    assert np.array_equal(coords,coords.T)
    assert oc._array_identity(coords)!=oc._array_identity(coords.T)


def test_scope_uses_compact_lowrank_seed_without_changing_geometry(source,monkeypatch):
    oc=coordinator()
    rng=np.random.default_rng(99)
    s=rng.normal(size=(source.mol.nao,2))
    q=rng.normal(size=(source.mol.nao,1))
    seeds=(s@s.T,-q@q.T)
    with config_update('pyscfad_moleintor_opt',True):
        expected,_=factorized._backward(True,True,False,(source,s),seeds)
        monkeypatch.setenv('PYSCFAD_DF_OOC_COMPACT','1')
        with oc.geometry_scope(source,enabled=True) as scope:
            explicit,_=factorized._backward(True,True,False,(source,s),seeds)
            actual=scope.finish(explicit)
        close(actual,expected)
        assert scope.stats.get('compact_contributions',0)==1


@pytest.mark.parametrize('force_disk',[False,True])
def test_mo_and_scf_seeds_share_one_final_geometry_pass(source,force_disk):
    from pyscfad.df import _cderi_vjp
    oc=coordinator()
    rng=np.random.default_rng(100)
    mo=rng.normal(size=(source.mol.nao,source.mol.nao))
    orbs=(0,2,2,5)
    out=_cderi_vjp._stream_nr_e2_from_cderi_source(source._cderi_to_save,mo,orbs,1000,aosym='s2')
    seed=rng.normal(size=out.shape)
    s=mo[:,:2]
    jkseeds=(np.eye(source.mol.nao),-.3*np.eye(source.mol.nao))
    with config_update('pyscfad_moleintor_opt',True):
        m,a=_cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar(source.mol,source.auxmol,
                source._cderi_to_save,mo,seed,orbs,1000,saved_output=out)
        reference,_=factorized._backward(True,True,False,(source,s),jkseeds)
        reference.mol=add(reference.mol,m)
        reference.auxmol=add(reference.auxmol,a)
        for scale in (1.,-.5,0.):
            with oc.geometry_scope(source,enabled=True,force_disk=force_disk) as scope:
                assert oc.defer_mo_geometry(source._fast_exchange_cache_token,
                        source._cderi_to_save,source.mol,source.auxmol,mo,orbs,
                        out,scale*seed,geometry_activity=(True,True))
                explicit,_=factorized._backward(True,True,False,(source,s),tuple(scale*x for x in jkseeds))
                actual=scope.finish(explicit)
            close(actual,jax.tree_util.tree_map(lambda x:scale*x,reference))
            assert scope.stats['mo_consumers']==1
            assert scope.stats['geometry_passes']==1


def test_mo_raw_seed_cleanup_and_source_ownership(source,tmp_path,monkeypatch):
    from pyscf import lib
    oc=coordinator()
    monkeypatch.setattr(lib.param,'TMPDIR',str(tmp_path))
    mo=np.eye(source.mol.nao)
    orbs=(0,2,2,5)
    raw=np.ones((source.get_naoaux(),6))
    before=set(tmp_path.iterdir())
    with pytest.raises(ValueError,match='sentinel'):
        with oc.geometry_scope(source,enabled=True,force_disk=True) as scope:
            assert not oc.defer_mo_geometry(None,source._cderi_to_save,
                    source.mol,source.auxmol,mo,orbs,raw,raw)
            assert oc.defer_mo_geometry(source._fast_exchange_cache_token,
                    source._cderi_to_save,source.mol,source.auxmol,mo,orbs,raw,raw)
            assert scope.stats['mo_raw_disk_seeds']==1
            assert set(tmp_path.iterdir())!=before
            raise ValueError('sentinel')
    assert set(tmp_path.iterdir())==before


@pytest.mark.parametrize('different_factors', [False, True])
@pytest.mark.parametrize('force_disk', [False, True])
def test_fused_scope_preserves_geometry_and_immediate_factors(source, different_factors, force_disk):
    oc = coordinator()
    rng = np.random.default_rng(805)
    s = rng.normal(size=(source.mol.nao, 2))
    ss = [s, s.copy(), rng.normal(size=s.shape) if different_factors else s]
    seeds = [(rng.normal(size=(source.mol.nao,)*2),rng.normal(size=(source.mol.nao,)*2)) for _ in ss]
    with config_update('pyscfad_moleintor_opt', True):
        expected = [factorized._backward(True,True,True,(source,x),y) for x,y in zip(ss,seeds)]
        for scale in (1., -.5, 0., 2.):
            with oc.geometry_scope(source, enabled=True, force_disk=force_disk) as scope:
                bars = [factorized._backward(True,True,True,(source,x),tuple(scale*z for z in y)) for x,y in zip(ss,seeds)]
                for (_, got), (_, want) in zip(bars,expected):
                    np.testing.assert_allclose(got,scale*want,atol=3e-11)
                explicit = zeros(source)
                for bar,_ in bars:
                    explicit=add(explicit,bar)
                actual = scope.finish(explicit)
            reference=zeros(source)
            for bar,_ in expected:
                reference=add(reference,jax.tree_util.tree_map(lambda z:scale*z,bar))
            close(actual,reference)
            assert scope.stats['consumers']==3
            assert scope.stats['seed_groups']==(2 if different_factors else 1)
            assert scope.stats['geometry_passes']==1


def test_scope_cleanup_and_nested_root_suspension(source):
    oc=coordinator()
    s=np.ones((source.mol.nao,1))*.1
    seeds=(np.eye(source.mol.nao),np.eye(source.mol.nao)*-.5)
    with config_update('pyscfad_moleintor_opt',True):
        with pytest.raises(RuntimeError,match='unconsumed'):
            with oc.geometry_scope(source,enabled=True):
                factorized._backward(True,True,False,(source,s),seeds)
        with pytest.raises(ValueError,match='sentinel'):
            with oc.geometry_scope(source,enabled=True):
                factorized._backward(True,True,False,(source,s),seeds)
                raise ValueError('sentinel')
        with oc.geometry_scope(source,enabled=True) as scope:
            with pytest.raises(RuntimeError,match='nested'):
                with oc.geometry_scope(source,enabled=True):
                    pass
            with _cotangent_buffer.suspend():
                explicit,_=factorized._backward(True,True,False,(source,s),seeds)
            assert scope.stats['consumers']==0
            close(scope.finish(explicit),explicit)


def test_unrelated_source_same_file_is_not_captured(source):
    oc=coordinator()
    other=df.DF(source.mol,auxbasis='weigend',incore=False)
    other.auxmol=source.auxmol
    other.attach_outcore_cderi(source._cderi_to_save)
    s=np.ones((source.mol.nao,1))*.1
    seeds=(np.eye(source.mol.nao),np.eye(source.mol.nao))
    with config_update('pyscfad_moleintor_opt',True):
        expected,_=factorized._backward(True,True,False,(other,s),seeds)
        with oc.geometry_scope(source,enabled=True) as scope:
            actual,_=factorized._backward(True,True,False,(other,s),seeds)
            assert scope.stats['consumers']==0
            scope.finish(zeros(source))
        close(actual,expected)


def test_independent_geometry_activity_is_preserved(source):
    oc=coordinator()
    s=np.ones((source.mol.nao,1))*.1
    seeds=(np.eye(source.mol.nao),np.eye(source.mol.nao)*-.5)
    with config_update('pyscfad_moleintor_opt',True):
        expected,_=factorized._backward(True,True,False,(source,s),seeds)
        for mask in ((True,False),(False,True),(False,False)):
            with oc.geometry_scope(source,enabled=True) as scope:
                explicit,_=factorized._backward(True,True,False,(source,s),seeds,geometry_activity=mask)
                actual=scope.finish(zeros(source) if explicit is None else explicit)
            reference=zeros(source)
            if mask[0]:reference.mol=expected.mol
            if mask[1]:reference.auxmol=expected.auxmol
            close(actual,reference)


def test_complete_scf_pullback_has_one_geometry_pass(tmp_path):
    from pyscfad import scf
    oc=coordinator()
    mol=gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                 basis='6-31g',verbose=0,max_memory=1000)
    mol.build(trace_exp=False,trace_ctr_coeff=False)
    obj=df.DF(mol,auxbasis='weigend',incore=False)
    obj._cderi_to_save=str(tmp_path/'scf-cderi.h5')
    with config_update('pyscfad_moleintor_opt',True), config_update('pyscfad_scf_implicit_diff',True):
        obj.build()
        def outputs(inputs):
            m,d=inputs
            mf=scf.RHF(m).density_fit(with_df=d)
            mf.conv_tol=1e-12
            mf.conv_tol_grad=1e-9
            mf.kernel()
            return mf.e_tot,mf.mo_energy,mf.mo_coeff
        value,pullback=jax.vjp(outputs,(mol,obj))
        seed=tuple(np.ones_like(x)*.01 for x in value)
        reference,=pullback(seed)
        for scale in (1.,-.5,0.):
            with oc.geometry_scope(obj,enabled=True,force_disk=True) as scope:
                bars,=pullback(tuple(scale*x for x in seed))
                actual=(bars[0],scope.finish(bars[1]))
            close(actual,jax.tree_util.tree_map(lambda x:scale*x,reference))
            assert scope.stats['consumers']==3,scope.stats
            assert scope.stats['geometry_passes']==1,scope.stats


@pytest.mark.parametrize('force_disk',[False,True])
def test_complete_dfmp2_reverse_chain_preserves_gradient(tmp_path,monkeypatch,force_disk):
    from pyscfad import scf
    from pyscfad.mp import dfmp2
    from pyscfad.df import _cderi_vjp
    oc=coordinator()
    monkeypatch.setenv('PYSCFAD_DF_OOC_CAPTURE_FACTORS','1')
    monkeypatch.setenv('PYSCFAD_DF_OOC_COMPACT','1')
    monkeypatch.setenv('PYSCFAD_SCF_OCCUPIED_CACHE','1')
    mol=gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                 basis='6-31g',verbose=0,max_memory=8000)
    mol.build(trace_exp=False,trace_ctr_coeff=False)
    path=str(tmp_path/'full-chain.h5')
    def df_state(m):
        d=df.DF(m,auxbasis='weigend',incore=False)
        d.attach_outcore_cderi(path)
        return m,d
    def scf_state(inputs):
        m,d=inputs
        mf=scf.RHF(m).density_fit(with_df=d)
        mf.conv_tol=1e-12
        mf.conv_tol_grad=1e-9
        mf.kernel()
        assert mf.converged
        return mf.mol,mf.with_df,mf.e_tot,mf.mo_coeff,mf.mo_energy
    def energy(state):
        m,d,e,c,eps=state
        mf=scf.RHF(m).density_fit(with_df=d)
        mf.e_tot,mf.mo_coeff,mf.mo_energy=e,c,eps
        mf.mo_occ=mf.get_occ()
        mf.converged=True
        mp=dfmp2.MP2(mf)
        mp.kernel(with_t2=False)
        return mp.e_tot
    with (config_update('pyscfad_moleintor_opt',True),
          config_update('pyscfad_scf_implicit_diff',True),
          config_update('pyscfad_scf_df_precond',True)):
        builder=df.DF(mol,auxbasis='weigend',incore=False)
        builder._cderi_to_save=path
        builder.build()
        inputs,df_pull=jax.vjp(df_state,mol)
        state,scf_pull=jax.vjp(scf_state,inputs)
        value,mp_pull=jax.vjp(energy,state)
        state_bar,=mp_pull(np.ones_like(value))
        bars,=scf_pull(state_bar)
        reference,=df_pull(bars)
        original=_cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader
        calls=[]
        def counted(*a,**kw):
            calls.append(1)
            return original(*a,**kw)
        monkeypatch.setattr(_cderi_vjp,'_int3c_mo_deriv_coords_vjp_from_z_reader',counted)
        for scale in (1.,-.5,0.):
            calls.clear()
            with oc.geometry_scope(inputs[1],enabled=True,force_disk=force_disk) as scope:
                state_bar,=mp_pull(np.ones_like(value)*scale)
                bars,=scf_pull(state_bar)
                actual,=df_pull((bars[0],scope.finish(bars[1])))
            close(actual,jax.tree_util.tree_map(lambda x:scale*x,reference))
            assert scope.stats['mo_consumers']==1,scope.stats
            assert scope.stats['consumers']==3,scope.stats
            assert scope.stats['geometry_passes']==len(calls)==1,scope.stats
