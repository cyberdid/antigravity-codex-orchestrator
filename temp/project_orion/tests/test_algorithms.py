import numpy as np
from orion.baselines import ClusteringBaselines as B
from orion.risk_graph import RiskGraphClusterer, ClusteringConfig

def test_mutual_nearest_neighbors_excludes_one_direction_bridge():
    x=np.array([[0.],[.1],[.3]])
    p=B.reciprocal_nearest_neighbor_graph(x,k=1,threshold=-1)
    assert p[0]==p[1] and p[2]!=p[1]

def test_threshold_and_unknown_k_ahc():
    x=np.array([[0.,0.],[.01,0.],[4.,0.],[4.01,0.],[8.,0.],[8.01,0.]])
    p=B.agglomerative_clustering_baseline(x,distance_threshold=.1)
    assert len(np.unique(p))==3
    q=B.thresholded_graph_connected_components(np.array([[1.,0.],[.99,.01],[0.,1.]]),threshold=.9)
    assert q[0]==q[1] and q[0]!=q[2]

def test_risk_graph_canonical_clusters_and_real_abstention():
    x=np.array([[1.,0.,0.],[.999,.01,0.],[0.,1.,0.],[.01,.999,0.],[0.,0.,1.]])
    p=RiskGraphClusterer(max_neighbors=2,calibration_factor=1).cluster(x)
    np.testing.assert_array_equal(p,[0,0,1,1,-1])

def test_component_cohesion_blocks_chaining():
    # endpoints .62 apart while adjacent pairs are .9: complete-link blocks bridge
    theta=np.array([0,.45,.9]); x=np.c_[np.cos(theta),np.sin(theta)]
    cfg=ClusteringConfig(max_neighbors=2,reciprocal_threshold=.8,abstention_threshold=.8,cohesion_threshold=.8,use_local_calibration=False,use_adaptive_neighborhood=False)
    p=RiskGraphClusterer(cfg).cluster(x)
    assert not (p[0]==p[1]==p[2])
    p=RiskGraphClusterer(ClusteringConfig(**{**cfg.__dict__,'use_cohesion_constraint':False})).cluster(x)
    np.testing.assert_array_equal(p,[0,0,0])

def test_abstention_toggle_retains_singleton_partition():
    x=np.eye(3)
    np.testing.assert_array_equal(RiskGraphClusterer(use_abstention=False).cluster(x),[0,1,2])
