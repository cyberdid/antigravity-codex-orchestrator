"""Unknown-K graph and threshold-AHC baselines; no truth labels are read."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from .data import normalize
from .retrieval import exact_knn


def observations(data):
    return np.asarray(data.embeddings if hasattr(data,'embeddings') else data,dtype=float)


def knn_without_self(x,k):
    n=len(x)
    if n<2: return np.empty((n,0),dtype=int),np.empty((n,0))
    k=min(k,n-1)
    ids,ds=exact_knn(x,x,k+1)
    out_i=np.empty((n,k),dtype=int); out_d=np.empty((n,k))
    for row in range(n):
        keep=ids[row]!=row
        out_i[row]=ids[row][keep][:k]; out_d[row]=ds[row][keep][:k]
    return out_i,out_d


def components(n,edges):
    if not n: return np.empty(0,dtype=int)
    if edges:
        a,b=zip(*edges)
        graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(n,n)).tocsr()
    else: graph=coo_matrix((n,n)).tocsr()
    return connected_components(graph,directed=False)[1]


class ClusteringBaselines:
    @staticmethod
    def thresholded_graph_connected_components(data,threshold=.8,radius=None):
        """Cosine threshold graph; optional radius uses raw Euclidean distances.

        Blocked pair enumeration has quadratic compute but bounded working
        memory. Intended for small clustering cohorts, not scalability claims.
        """
        from scipy.spatial.distance import cdist
        x=observations(data); n=len(x); edges=[]
        if radius is None: x=normalize(x)
        for start in range(0,n,32):
            if radius is None: keep=(x[start:start+32]@x.T)>=threshold
            else: keep=cdist(x[start:start+32],x)<=radius
            for local,row in enumerate(range(start,min(start+32,n))):
                edges.extend((row,int(j)) for j in np.flatnonzero(keep[local]) if j>row)
        return components(n,edges)

    @staticmethod
    def agglomerative_clustering_baseline(data,n_clusters=None,distance_threshold=.7):
        """Unknown-K average-link Euclidean AHC; explicit K is oracle control."""
        x=observations(data)
        if len(x)<2: return np.zeros(len(x),dtype=int)
        return AgglomerativeClustering(n_clusters=n_clusters,
            distance_threshold=distance_threshold if n_clusters is None else None,
            linkage='average').fit_predict(x)

    @staticmethod
    def reciprocal_nearest_neighbor_graph(data,k=10,threshold=.8):
        """Euclidean kNN (excluding self), mutual membership AND cosine gate."""
        x=observations(data); unit=normalize(x)
        ids,_=knn_without_self(x,k)
        neighborhoods=[set(row) for row in ids]
        edges=[]
        for i,neighbors in enumerate(ids):
            for j in neighbors:
                if i<j and i in neighborhoods[j] and unit[i]@unit[j]>=threshold:
                    edges.append((i,int(j)))
        return components(len(x),edges)


def threshold_graph_clustering(data,level=0.,**params):
    return ClusteringBaselines.thresholded_graph_connected_components(observations(data),**params)

def agglomerative_clustering(data,level=0.,**params):
    return ClusteringBaselines.agglomerative_clustering_baseline(observations(data),**params)

def rr_graph_clustering(data,level=0.,**params):
    return ClusteringBaselines.reciprocal_nearest_neighbor_graph(observations(data),**params)

def run_baselines(data):
    return {'threshold_graph':threshold_graph_clustering(data),
            'agglomerative':agglomerative_clustering(data),'rr_graph':rr_graph_clustering(data)}
