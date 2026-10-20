"""Bounded-memory squared Euclidean retrieval, sorted by distance then row ID.

exact_knn scans blocks. approximate_knn builds independent random projection
sorted indexes, retrieves a bounded window per index and reranks the union in
original space. It is an actual approximate algorithm; recall is data dependent.
For unit vectors squared L2 = 2 * cosine distance. Queries need not be indexed
vectors and no self-neighbor is implicitly removed.
"""
import numpy as np
from scipy.spatial.distance import cdist


def _inputs(vectors,queries,k):
    x,q=np.asarray(vectors,dtype=float),np.asarray(queries,dtype=float)
    if x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1] or not x.shape[1]:
        raise ValueError('vectors and queries need matching nonempty feature axes')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)):
        raise ValueError('vectors must be finite')
    if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 1<=k<=len(x):
        raise ValueError('k must be between 1 and index size')
    return x,q


def exact_knn(vectors,queries,k,*,query_batch_size=32,index_batch_size=8192):
    x,q=_inputs(vectors,queries,k)
    if min(query_batch_size,index_batch_size)<1: raise ValueError('positive batches required')
    ids=np.empty((len(q),k),dtype=np.int64); distances=np.empty((len(q),k))
    for start in range(0,len(q),query_batch_size):
        batch=q[start:start+query_batch_size]
        best_d=np.empty((len(batch),0)); best_i=np.empty((len(batch),0),dtype=np.int64)
        for offset in range(0,len(x),index_batch_size):
            block=x[offset:offset+index_batch_size]
            ds=cdist(batch,block,metric='sqeuclidean')
            candidate_d=np.concatenate((best_d,ds),axis=1)
            candidate_i=np.concatenate((best_i,np.broadcast_to(np.arange(offset,offset+len(block)),ds.shape)),axis=1)
            # Stable row-ID tie breaking across every index block.
            order=np.lexsort((candidate_i,candidate_d),axis=1)[:,:k]
            best_d=np.take_along_axis(candidate_d,order,axis=1)
            best_i=np.take_along_axis(candidate_i,order,axis=1)
        ids[start:start+len(batch)]=best_i; distances[start:start+len(batch)]=best_d
    return ids,distances


class ProjectionWindowIndex:
    """Deterministic projection-window ANN index, no quadratic allocations."""
    def __init__(self,vectors,*,num_projections=8,window=256,seed=0):
        self.vectors=np.asarray(vectors,dtype=float)
        _inputs(self.vectors,self.vectors[:0],1)
        if num_projections<1 or window<1: raise ValueError('positive index parameters required')
        self.window=int(window)
        directions=np.random.default_rng(seed).normal(size=(self.vectors.shape[1],num_projections))
        directions/=np.linalg.norm(directions,axis=0)
        self.directions=directions
        projected=self.vectors@directions
        self.orders=np.argsort(projected,axis=0,kind='stable')
        self.values=np.take_along_axis(projected,self.orders,axis=0)

    def query(self,queries,k):
        x,q=_inputs(self.vectors,queries,k)
        ids=np.empty((len(q),k),dtype=np.int64); distances=np.empty((len(q),k))
        for row,query in enumerate(q):
            candidates=[]
            for axis,value in enumerate(query@self.directions):
                middle=np.searchsorted(self.values[:,axis],value)
                half=max(self.window,k)//2
                start=max(0,min(len(x)-max(self.window,k),middle-half))
                candidates.append(self.orders[start:min(len(x),start+max(self.window,k)),axis])
            cand=np.unique(np.concatenate(candidates))
            ds=np.sum((x[cand]-query)**2,axis=1)
            order=np.lexsort((cand,ds))[:k]
            ids[row]=cand[order]; distances[row]=ds[order]
        return ids,distances


def approximate_knn(vectors,queries,k):
    _inputs(vectors,queries,k)
    return ProjectionWindowIndex(vectors).query(queries,k)
