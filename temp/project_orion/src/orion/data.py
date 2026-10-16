"""Locally seeded synthetic unit vectors with known identities/domain/outliers."""
from dataclasses import dataclass
import numpy as np
from .config import SyntheticConfig

@dataclass
class DatasetResult:
    embeddings: np.ndarray
    labels: np.ndarray
    outlier_mask: np.ndarray
    domains: np.ndarray
    def __iter__(self): return iter((self.embeddings,self.labels))
    def __getitem__(self,idx): return (self.embeddings,self.labels)[idx]
    def __len__(self): return 2

class hybridmethod:
    def __init__(self,func): self.func=func
    def __get__(self,instance,owner):
        return lambda *args,**kwargs:self.func(owner if instance is None else instance,*args,**kwargs)

def normalize(x):
    x=np.asarray(x,dtype=float)
    return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-15)

class SyntheticEmbeddingGenerator(SyntheticConfig):
    def _generate_data(self):
        c=self
        if min(c.num_entities,c.observations_per_entity,c.embedding_dim)<1:
            raise ValueError('positive entity, observation and dimension counts required')
        if not 0<=c.outliers_ratio<=1 or not 0<=c.inter_entity_similarity<=1:
            raise ValueError('ratios must be in [0,1]')
        if min(c.variance,c.noise_level,c.domain_shift)<0:
            raise ValueError('noise and domain shift must be nonnegative')
        rng=np.random.default_rng(c.seed)
        means=normalize(rng.normal(size=(c.num_entities,c.embedding_dim)))
        common=normalize(rng.normal(size=(1,c.embedding_dim)))
        means=normalize(np.sqrt(1-c.inter_entity_similarity)*means+np.sqrt(c.inter_entity_similarity)*common)
        n=c.num_entities*c.observations_per_entity
        counts=np.full(c.num_entities,c.observations_per_entity)
        if c.unequal_populations:
            counts=1+rng.multinomial(n-c.num_entities,rng.dirichlet(np.ones(c.num_entities)))
        labels=np.repeat(np.arange(c.num_entities),counts)
        params=c.get_adjusted_params()
        noise=rng.normal(size=(n,c.embedding_dim))
        residual=rng.normal(size=(n,c.embedding_dim))
        x=means[labels]+(params['variance']*noise+params['noise_level']*residual)/np.sqrt(c.embedding_dim)
        domains=rng.integers(0,2,size=n)
        shift=normalize(rng.normal(size=(1,c.embedding_dim)))[0]
        x[domains==1]+=c.domain_shift*shift
        x=normalize(x)
        outlier_mask=np.zeros(n,dtype=bool)
        outlier_mask[rng.choice(n,int(n*c.outliers_ratio),replace=False)]=True
        labels[outlier_mask]=-1
        x[outlier_mask]=normalize(rng.normal(size=(int(outlier_mask.sum()),c.embedding_dim)))
        return DatasetResult(x,labels,outlier_mask,domains)

    @hybridmethod
    def generate(self_or_cls,**kwargs):
        if isinstance(self_or_cls,type):
            if 'dimension' in kwargs: kwargs['embedding_dim']=kwargs.pop('dimension')
            return self_or_cls(**kwargs)._generate_data()
        if kwargs: raise TypeError('instance generate accepts no overrides')
        return self_or_cls._generate_data()
