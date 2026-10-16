import importlib
import numpy as np
import pytest

def test_exact_neighbors_independent_queries_and_ties():
    r=importlib.import_module('orion.retrieval')
    ids,ds=r.exact_knn(np.array([[0.,0.],[2.,0.],[0.,2.]]),np.array([[1.,0.],[0.,1.]]),2)
    np.testing.assert_array_equal(ids,[[0,1],[0,2]])
    np.testing.assert_array_equal(ds,np.ones((2,2)))

def test_approximate_returns_actual_distances_and_high_recall_on_small_data():
    r=importlib.import_module('orion.retrieval'); rng=np.random.default_rng(8)
    x=rng.normal(size=(80,6)); q=rng.normal(size=(9,6))
    ids,ds=r.approximate_knn(x,q,4)
    expected=np.sum((q[:,None,:]-x[ids])**2,axis=2)
    np.testing.assert_allclose(ds,expected,atol=1e-12)
    exact,_=r.exact_knn(x,q,4)
    assert np.mean([len(set(a)&set(b))/4 for a,b in zip(ids,exact)])>=.8

def test_validation_and_empty_queries():
    r=importlib.import_module('orion.retrieval')
    assert r.exact_knn(np.ones((3,2)),np.empty((0,2)),2)[0].shape==(0,2)
    for k in (0,4):
        with pytest.raises(ValueError): r.exact_knn(np.ones((3,2)),np.ones((1,2)),k)
