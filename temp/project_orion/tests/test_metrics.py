import json
import numpy as np
import pytest
from orion.metrics import calculate_metrics

@pytest.mark.parametrize('prediction,precision,recall,bp,br', [
    ([0,0,1,1],1,1,1,1), ([0,1,2,3],None,0,1,.5),
    ([0,0,0,0],1/3,1,.5,1), ([0,0,-1,-1],1,.5,1,1),
    ([-1,-1,-1,-1],None,0,None,None),
])
def test_literal_partitions(prediction, precision, recall, bp, br):
    m=calculate_metrics([0,0,1,1],prediction)
    for key,expected in [('pairwise_precision',precision),('pairwise_recall',recall),('b_cubed_precision',bp),('b_cubed_recall',br)]:
        assert m[key] == pytest.approx(expected) if expected is not None else m[key] is None
    assert m['coverage']==sum(p!=-1 for p in prediction)/4
    json.dumps(m,allow_nan=False)

def test_partial_abstention_global_denominators():
    m=calculate_metrics([0,0,1,1],[7,7,-1,-1])
    assert (m['true_positive_pairs'],m['false_positive_pairs'],m['false_negative_pairs'])==(1,0,1)
    assert m['b_cubed_recall_global']==.5
    assert m['accepted_observations']==2
    assert m['pairwise_f1']==pytest.approx(2/3)

def test_fmr_is_distinct_from_fdp_and_point_contamination():
    m=calculate_metrics([0,0,1,1],[0,0,0,1])
    assert m['false_match_rate']==.5 # 2 FP / 4 impostor pairs
    assert m['false_discovery_proportion']==pytest.approx(2/3)
    assert m['point_contamination']==.25

def test_empty_and_outlier_semantics():
    m=calculate_metrics([],[])
    assert m['coverage'] is None and m['adjusted_rand_index'] is None
    assert calculate_metrics([-1,-1,0,0],[0,0,1,1])['false_positive_pairs']==1

def test_label_and_row_permutation_invariance():
    a=calculate_metrics([0,0,1,1,2],[3,3,3,-1,4])
    b=calculate_metrics([8,9,8,7,9],[5,-1,5,2,5]) # rows 0,3,1,4,2
    assert a==b

@pytest.mark.parametrize('truth,pred',[([0],[0,1]),([[0]],[[0]]),([0],[float('nan')]),([0],[-2])])
def test_invalid_labels_rejected(truth,pred):
    with pytest.raises(ValueError): calculate_metrics(truth,pred)

def test_randomized_pairs_match_independent_enumeration():
    rng=np.random.default_rng(17)
    for size in range(1,30):
        truth=rng.integers(-1,4,size=size); pred=rng.integers(-1,5,size=size)
        tp=fp=fn=tn=0
        for i in range(size):
            for j in range(i):
                same_true=truth[i]!=-1 and truth[i]==truth[j]
                same_pred=pred[i]!=-1 and pred[i]==pred[j]
                tp+=same_true and same_pred; fp+=not same_true and same_pred
                fn+=same_true and not same_pred; tn+=not same_true and not same_pred
        m=calculate_metrics(truth,pred)
        assert [m[key] for key in ['true_positive_pairs','false_positive_pairs','false_negative_pairs','true_negative_pairs']]==[tp,fp,fn,tn]

def test_label_integer_overflow_is_rejected():
    with pytest.raises(ValueError): calculate_metrics([0],[1e30])
