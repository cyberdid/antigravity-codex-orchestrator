import numpy as np
from orion.data import SyntheticEmbeddingGenerator
from orion.config import Difficulty

def test_local_rng_and_repeated_instance_generation():
    np.random.seed(812); expected=np.random.random(4); np.random.seed(812)
    g=SyntheticEmbeddingGenerator(seed=4,num_entities=3,observations_per_entity=10,embedding_dim=8)
    a=g.generate(); b=g.generate()
    np.testing.assert_array_equal(a.embeddings,b.embeddings)
    np.testing.assert_array_equal(np.random.random(4),expected)
    np.testing.assert_allclose(np.linalg.norm(a.embeddings,axis=1),1)

def test_outliers_labeled_and_population_budget_preserved():
    d=SyntheticEmbeddingGenerator.generate(seed=4,num_entities=3,observations_per_entity=20,dimension=8,outliers_ratio=.2,unequal_populations=True)
    assert len(d.labels)==60
    assert np.sum(d.labels==-1)==12
    np.testing.assert_array_equal(d.outlier_mask,d.labels==-1)

def test_domain_shift_is_explicit_and_controlled():
    a=SyntheticEmbeddingGenerator.generate(seed=2,dimension=8,domain_shift=0)
    b=SyntheticEmbeddingGenerator.generate(seed=2,dimension=8,domain_shift=.4)
    np.testing.assert_array_equal(a.labels,b.labels)
    np.testing.assert_array_equal(a.domains,b.domains)
    assert set(a.domains)=={0,1}
    np.testing.assert_array_equal(a.embeddings[a.domains==0],b.embeddings[b.domains==0])
    assert not np.allclose(a.embeddings[a.domains==1],b.embeddings[b.domains==1])

def test_difficulty_increases_within_entity_dispersion():
    means=[]
    for diff in Difficulty:
        d=SyntheticEmbeddingGenerator.generate(seed=7,dimension=32,outliers_ratio=0,difficulty=diff)
        x=d.embeddings[d.labels==0]; means.append(np.mean(x@x.T))
    assert all(a>b for a,b in zip(means,means[1:]))
