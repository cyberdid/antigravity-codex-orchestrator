"""Create a fresh, auditable experiment; never overwrite historical artifacts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from orion.runner import run_protocol,write_json,finalize_manifest,file_sha
from benchmark_scalability import run_benchmark


def profile_config(profile):
    if profile=='smoke':
        return {'profile':'smoke','development_seeds':[1],'evaluation_seeds':[55],
                'generator':{'num_entities':4,'observations_per_entity':12,'embedding_dim':32,
                             'variance':.35,'noise_level':.1,'inter_entity_similarity':.4,'outliers_ratio':.1},
                'scenarios':[{'domain_shift':0.},{'domain_shift':.35}],
                'threshold_grid':[.4,.65,.85],
                'sensitivity_grid':{'merge_threshold':[.4,.8],'domain_shift':[0.,.35]}}
    return {'profile':'full','development_seeds':[1,42,1024],'evaluation_seeds':[55,99,2048],
            'generator':{'num_entities':10,'observations_per_entity':30,'embedding_dim':192,
                         'variance':.5,'noise_level':.15,'inter_entity_similarity':.55,
                         'difficulty':'hard','unequal_populations':True},
            'scenarios':[{'outliers_ratio':rate,'domain_shift':shift}
                         for rate in [.01,.05,.1,.2,.5] for shift in [0.,.35]],
            'threshold_grid':[.25,.4,.6,.8], 'maximum_development_fdp':.01,
            'sensitivity_grid':{'cluster_population':[5,10,20], 'noise':[.01,.15,.3],
                                'dimensionality':[64,128,192], 'overlap':[.2,.55,.8],
                                'domain_shift':[0.,.25,.5], 'outlier_rate':[.01,.1,.3],
                                'neighborhood_size':[5,20,50], 'merge_threshold':[.25,.5,.8]}}


def execute(output,profile='smoke'):
    output=Path(output)
    config=profile_config(profile)
    payload=run_protocol(config,output)
    sizes=[1000] if profile=='smoke' else [1000,10000,100000,200000]
    rows=[]
    for n in sizes:
        print(f'retrieval n={n}',flush=True)
        rows.append(run_benchmark(n,output_dir=output/'scalability',query_count=16 if profile=='smoke' else 64,
                                  dimension=192,k=10,window=256))
    write_json(output/'scalability/results.json',{'benchmarks':rows,'skipped':[],
        'scope':'retrieval only; fixed query subset scans full index for exact reference; no large-N clustering claim'})
    claims=[{'id':'evaluation-count','kind':'artifact_summary','value':len(payload['results']),
             'source':'evaluation/results.json','source_sha256':file_sha(output/'evaluation/results.json')},
            {'id':'ablation-count','kind':'artifact_summary','value':len(payload['ablations']),
             'source':'ablation/results.json','source_sha256':file_sha(output/'ablation/results.json')}]
    for row in rows:
        claims.append({'id':f'recall-{row["size"]}','kind':'retrieval_recall_at_k','value':row['recall_at_k'],
                       'size':row['size'],'k':row['k'],'query_count':row['query_count'],
                       'source':'scalability/'+row['neighbors_file'],'source_sha256':row['neighbors_sha256']})
    write_json(output/'claims.json',{'schema_version':2,'status':'candidate_measurements_require_independent_verification',
                                    'claims':claims,'limitations':['synthetic data only','no novelty established',
                                    'ANN recall is measured for finite independent Gaussian query subset',
                                    'sampled process RSS can miss short peaks; trusted controller observes resources independently',
                                    'full rerun is confirmatory only for this frozen synthetic protocol; sensitivity is descriptive']})
    finalize_manifest(output)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--profile',choices=['smoke','full'],default='smoke')
    args=p.parse_args(); execute(args.output,args.profile)
    print(f'Artifacts saved to {args.output}',flush=True)

if __name__=='__main__': main()
