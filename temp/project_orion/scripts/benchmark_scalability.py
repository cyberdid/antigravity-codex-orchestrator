"""Real retrieval benchmark with independent query vectors and saved neighbors.

Local process RSS is sampled, not a claim of an exact allocator peak. Exact
references scan the entire index for the reported query subset. No clustering
ARI is inferred from retrieval. Large N is retrieval-only.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from orion.data import normalize
from orion.retrieval import exact_knn,ProjectionWindowIndex
from orion.runner import file_sha,write_json

class RSSSampler:
    def __init__(self): self.stop=threading.Event(); self.peak=psutil.Process().memory_info().rss
    def _sample(self):
        while not self.stop.wait(.005): self.peak=max(self.peak,psutil.Process().memory_info().rss)
    def __enter__(self): self.thread=threading.Thread(target=self._sample,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.peak=max(self.peak,psutil.Process().memory_info().rss); self.stop.set(); self.thread.join()


def run_benchmark(size,use_approx=True,*,output_dir,dimension=192,query_count=64,k=10,seed=712,window=256):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    if size<k or min(query_count,dimension)<1: raise ValueError('invalid benchmark sizes')
    name=f'n{size}-d{dimension}-seed{seed}'
    if (output/f'{name}-neighbors.npz').exists(): raise FileExistsError(name)
    rng=np.random.default_rng(seed)
    with RSSSampler() as rss:
        start=time.perf_counter()
        # Avoid a dense index-by-index matrix and retain exact inputs on disk.
        vectors=normalize(rng.normal(size=(size,dimension)))
        queries=normalize(rng.normal(size=(query_count,dimension)))
        generation_sec=time.perf_counter()-start
        np.save(output/f'{name}-vectors.npy',vectors); np.save(output/f'{name}-queries.npy',queries)
        start=time.perf_counter(); exact_ids,exact_distances=exact_knn(vectors,queries,k); exact_sec=time.perf_counter()-start
        build_sec=query_sec=None
        if use_approx:
            start=time.perf_counter(); index=ProjectionWindowIndex(vectors,window=window); build_sec=time.perf_counter()-start
            start=time.perf_counter(); approximate_ids,approximate_distances=index.query(queries,k); query_sec=time.perf_counter()-start
            recall=float(np.mean([len(set(a)&set(b))/k for a,b in zip(exact_ids,approximate_ids)]))
        else:
            approximate_ids=exact_ids.copy(); approximate_distances=exact_distances.copy(); recall=1.
        neighbors=output/f'{name}-neighbors.npz'
        np.savez_compressed(neighbors,exact_ids=exact_ids,exact_distances=exact_distances,
                            approximate_ids=approximate_ids,approximate_distances=approximate_distances)
    return {'size':size,'dimension':dimension,'query_count':query_count,'k':k,'seed':seed,
            'algorithm':'projection_window_candidates_exact_rerank' if use_approx else 'blocked_exact_scan',
            'num_projections':8 if use_approx else None,'window':window if use_approx else None,
            'distance':'squared_euclidean','query_protocol':'independent unit-normalized Gaussian queries; exact reference against full index',
            'recall_at_k':recall,'generation_sec':generation_sec,'exact_query_sec':exact_sec,
            'approximate_build_sec':build_sec,'approximate_query_sec':query_sec,
            'process_sampled_peak_rss_mb':rss.peak/(1024**2),
            'neighbors_file':neighbors.name,'neighbors_sha256':file_sha(neighbors),
            'vectors_file':f'{name}-vectors.npy','queries_file':f'{name}-queries.npy',
            'vectors_sha256':file_sha(output/f'{name}-vectors.npy'),'queries_sha256':file_sha(output/f'{name}-queries.npy')}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--sizes',nargs='+',type=int,default=[1000,10000,100000,200000])
    p.add_argument('--queries',type=int,default=64); args=p.parse_args()
    rows=[run_benchmark(n,output_dir=args.output,query_count=args.queries) for n in args.sizes]
    write_json(Path(args.output)/'results.json',{'benchmarks':rows,'skipped':[]})

if __name__=='__main__': main()
