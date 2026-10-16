"""Experimental conservative mutual-neighbor merging heuristic.

Canonical components use union-find. Candidate edges are sparse. Local score
thresholds are heuristics, not calibrated probabilities. Complete-link cohesion
checks every cross-component pair before merging. Unresolved singletons abstain.
"""
from dataclasses import dataclass
import numpy as np
from .baselines import observations,knn_without_self
from .data import normalize

@dataclass
class ClusteringConfig:
    max_neighbors: int=20
    expansion_factor: float=1.2
    reciprocal_threshold: float=.8
    calibration_factor: float=1.1
    abstention_threshold: float=.7
    cohesion_threshold: float=.6
    use_adaptive_neighborhood: bool=True
    use_reciprocal_consistency: bool=True
    use_local_calibration: bool=True
    use_cohesion_constraint: bool=True
    use_abstention: bool=True

class RiskGraphClusterer:
    def __init__(self,config=None,**kwargs):
        self.config=config if config is not None else ClusteringConfig(**kwargs)
        if kwargs and config is not None: raise ValueError('pass config or keyword parameters')
        if self.config.max_neighbors<1 or self.config.expansion_factor<=0 or self.config.calibration_factor<=0:
            raise ValueError('positive neighborhood and calibration parameters required')

    def cluster(self,data_or_indices,similarity_matrix=None,level=0.,**kwargs):
        c=self.config
        # level is retained only for API compatibility: ground-truth outlier
        # fraction must never change decision thresholds on held-out data.
        if similarity_matrix is None:
            x=normalize(observations(data_or_indices)); n=len(x)
            ids,ds=knn_without_self(x,c.max_neighbors)
            scores=1-ds/2
            external_ids=None
        else:
            external_ids=list(data_or_indices); s=np.asarray(similarity_matrix,dtype=float)
            if s.shape!=(len(external_ids),len(external_ids)) or not np.allclose(s,s.T):
                raise ValueError('similarity matrix must be symmetric and match node count')
            n=len(s); k=min(c.max_neighbors,max(0,n-1))
            ids=np.empty((n,k),dtype=int); scores=np.empty((n,k))
            for i in range(n):
                candidates=np.array([j for j in range(n) if j!=i])
                order=np.lexsort((candidates,-s[i,candidates]))[:k]
                ids[i]=candidates[order]; scores[i]=s[i,ids[i]]
        neighbors=[]
        local=np.full(n,-1.)
        for row,sim in zip(ids,scores):
            if len(sim):
                # Expand within a distance multiple of the best neighbor, up
                # to max_neighbors. No hidden full pairwise similarity matrix.
                keep=(1-sim)<=max(1e-12,(1-sim[0])*c.expansion_factor) if c.use_adaptive_neighborhood else np.ones(len(sim),bool)
                neighbors.append(set(row[keep]))
            else: neighbors.append(set())
        for i,sim in enumerate(scores):
            if len(sim): local[i]=float(np.mean(sim)-np.std(sim)/c.calibration_factor)
        edges=[]
        for i,(row,sim) in enumerate(zip(ids,scores)):
            for j,score in zip(row,sim):
                if j not in neighbors[i]: continue
                if c.use_reciprocal_consistency and i not in neighbors[j]: continue
                if score<c.reciprocal_threshold or score<c.abstention_threshold: continue
                if c.use_local_calibration and score<max(local[i],local[j]): continue
                edges.append((-float(score),min(i,int(j)),max(i,int(j))))
        edges=sorted(set(edges))
        parent=list(range(n)); members={i:[i] for i in range(n)}
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]; i=parent[i]
            return i
        def cohesive(a,b):
            if not c.use_cohesion_constraint: return True
            if similarity_matrix is not None:
                return all(s[i,j]>=c.cohesion_threshold for i in members[a] for j in members[b])
            # A row at a time prevents allocation of a large cluster square.
            return all(np.min(x[members[b]]@x[i])>=c.cohesion_threshold for i in members[a])
        for _,i,j in edges:
            a,b=root(i),root(j)
            if a==b or not cohesive(a,b): continue
            a,b=min(a,b),max(a,b)
            parent[b]=a; members[a].extend(members.pop(b))
        labels=np.full(n,-1,dtype=int); next_id=0
        for a,rows in sorted(members.items()):
            if c.use_abstention and len(rows)==1: continue
            labels[rows]=next_id; next_id+=1
        if external_ids is not None: return dict(zip(external_ids,map(int,labels)))
        return labels
