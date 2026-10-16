"""Frozen development selection and disjoint, paired synthetic evaluation."""
from dataclasses import asdict,dataclass,field,replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from .data import SyntheticEmbeddingGenerator
from .metrics import calculate_metrics
from .baselines import threshold_graph_clustering,agglomerative_clustering,rr_graph_clustering
from .risk_graph import RiskGraphClusterer,ClusteringConfig

METHODS=('threshold_graph','agglomerative','rr_graph','risk_aware')


def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def file_sha(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dataset_sha(data):
    h=sha256()
    for x in (data.embeddings,data.labels,data.domains,data.outlier_mask):
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def parameters(method,threshold):
    if method=='threshold_graph': return {'threshold':threshold}
    if method=='agglomerative': return {'distance_threshold':float(np.sqrt(2*(1-threshold)))}
    if method=='rr_graph': return {'k':20,'threshold':threshold}
    return asdict(ClusteringConfig(reciprocal_threshold=threshold,abstention_threshold=threshold,cohesion_threshold=max(0.,threshold-.1)))


def predict(method,data,params):
    funcs={'threshold_graph':threshold_graph_clustering,'agglomerative':agglomerative_clustering,'rr_graph':rr_graph_clustering}
    if method=='risk_aware': return RiskGraphClusterer(**params).cluster(data)
    return funcs[method](data,**params)


def _record(data,method,params,output,filename,**metadata):
    start=time.perf_counter(); pred=predict(method,data,params); runtime=time.perf_counter()-start
    path=Path(output)/'predictions'/f'{filename}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,embeddings=data.embeddings,labels_true=data.labels,labels_pred=pred,domains=data.domains,outlier_mask=data.outlier_mask)
    strata={}
    for name,mask in [('domain_0',data.domains==0),('domain_1',data.domains==1),('outliers',data.outlier_mask),('known',~data.outlier_mask)]:
        strata[name]=calculate_metrics(data.labels[mask],pred[mask])
    return {'method':method,'parameters':params,'metrics':calculate_metrics(data.labels,pred),
            'stratified_metrics':strata,'runtime_sec':runtime,'status':'success',
            'dataset_sha256':dataset_sha(data),'predictions_file':str(path.relative_to(output)),
            'predictions_sha256':file_sha(path),**metadata}


def _selection_rank(rows,maximum_fdp):
    """Max mean global pair F1 subject to mean FDP <= budget; no-pair is ineligible.

    If no candidate meets budget, choose smallest mean FDP, then highest F1.
    All candidates and eligibility are retained for scrutiny.
    """
    mean_f1=float(np.mean([r['metrics']['pairwise_f1'] or 0 for r in rows]))
    fdps=[r['metrics']['false_discovery_proportion'] for r in rows]
    valid=all(x is not None for x in fdps)
    mean_fdp=float(np.mean(fdps)) if valid else None
    eligible=valid and mean_fdp<=maximum_fdp
    return (int(eligible),mean_f1 if eligible else -(mean_fdp if valid else 2),mean_f1),{'mean_pairwise_f1':mean_f1,'mean_fdp':mean_fdp,'eligible':eligible}


def run_protocol(config,output):
    output=Path(output)
    dev=list(config.get('development_seeds',[1,42,1024])); evaluation=list(config.get('evaluation_seeds',[55,99,2048]))
    if not dev or not evaluation or set(dev)&set(evaluation):
        raise ValueError('nonempty disjoint development and evaluation seeds required')
    if len(set(dev))!=len(dev) or len(set(evaluation))!=len(evaluation): raise ValueError('seeds must be unique')
    output.mkdir(parents=True,exist_ok=True)
    # Do not overwrite either historical artifacts or an interrupted fresh run.
    if any(output.iterdir()): raise FileExistsError(f'output directory is not empty: {output}')
    write_json(output/'config.json',config)
    base=config.get('generator',{})
    scenarios=config.get('scenarios',[{}])
    grid=config.get('threshold_grid',[.4,.6,.75,.9])
    if not grid or not scenarios: raise ValueError('nonempty grid and scenarios required')
    dev_rows=[]; selection={}; selection_scores={}
    for method in METHODS:
        candidates=[]
        for candidate,t in enumerate(grid):
            params=parameters(method,float(t)); rows=[]
            for seed in dev:
                for scenario_idx,scenario in enumerate(scenarios):
                    data=SyntheticEmbeddingGenerator.generate(**{**base,**scenario,'seed':seed})
                    row=_record(data,method,params,output,f'dev-{method}-{candidate}-{seed}-{scenario_idx}',
                                split='development',seed=seed,scenario=scenario,scenario_id=scenario_idx,candidate=candidate)
                    rows.append(row); dev_rows.append(row)
            rank,summary=_selection_rank(rows,config.get('maximum_development_fdp',.01))
            candidates.append((rank,-candidate,params,summary))
        winner=max(candidates,key=lambda x:(x[0],x[1]))
        selection[method]=winner[2]
        selection_scores[method]=[{'parameters':x[2],**x[3]} for x in candidates]
    frozen={'schema_version':2,'development_seeds':dev,'evaluation_seeds':evaluation,
            'methods':selection,'scores':selection_scores,
            'selection_rule':'maximum mean global pair F1 with mean FDP <= budget; if none eligible minimum mean FDP then F1; no accepted pairs ineligible',
            'maximum_development_fdp':config.get('maximum_development_fdp',.01),
            'config_sha256':file_sha(output/'config.json')}
    write_json(output/'development/results.json',{'results':dev_rows})
    write_json(output/'selected_config.json',frozen)
    frozen_sha=file_sha(output/'selected_config.json')
    results=[]; ablations=[]; sensitivity=[]
    toggles={'full_system':None,'without_reciprocal_consistency':'use_reciprocal_consistency',
             'without_local_calibration':'use_local_calibration','without_cohesion_constraint':'use_cohesion_constraint',
             'without_abstention':'use_abstention','without_adaptive_neighborhood':'use_adaptive_neighborhood'}
    for seed in evaluation:
        for scenario_idx,scenario in enumerate(scenarios):
            data=SyntheticEmbeddingGenerator.generate(**{**base,**scenario,'seed':seed})
            for method in METHODS:
                results.append(_record(data,method,selection[method],output,f'eval-{method}-{seed}-{scenario_idx}',
                                       split='evaluation',seed=seed,scenario=scenario,scenario_id=scenario_idx,selected_config_sha256=frozen_sha))
            for variant,toggle in toggles.items():
                params={**selection['risk_aware']}
                if toggle: params[toggle]=False
                ablations.append(_record(data,'risk_aware',params,output,f'ablation-{variant}-{seed}-{scenario_idx}',
                                         split='evaluation',variant=variant,seed=seed,scenario=scenario,scenario_id=scenario_idx,selected_config_sha256=frozen_sha))
    for parameter,values in config.get('sensitivity_grid',{'merge_threshold':[.4,.6,.8]}).items():
        gen_fields={'cluster_population':'num_entities','noise':'noise_level','dimensionality':'embedding_dim',
                    'overlap':'inter_entity_similarity','domain_shift':'domain_shift','outlier_rate':'outliers_ratio'}
        for value_idx,value in enumerate(values):
            for seed in evaluation:
                gen={**base,'seed':seed}; params={**selection['risk_aware']}
                if parameter in gen_fields: gen[gen_fields[parameter]]=value
                elif parameter=='neighborhood_size': params['max_neighbors']=int(value)
                elif parameter=='merge_threshold':
                    params['abstention_threshold']=float(value); params['reciprocal_threshold']=float(value)
                else: raise ValueError(f'unknown sensitivity parameter: {parameter}')
                data=SyntheticEmbeddingGenerator.generate(**gen)
                sensitivity.append(_record(data,'risk_aware',params,output,f'sensitivity-{parameter}-{value_idx}-{seed}',
                                           split='evaluation_sensitivity',seed=seed,parameter=parameter,value=value,
                                           generator_parameters=gen,selected_config_sha256=frozen_sha))
    if file_sha(output/'selected_config.json')!=frozen_sha: raise RuntimeError('frozen selection changed during evaluation')
    payload={'schema_version':2,'scientific_scope':'synthetic unit vectors only; no novelty or real speaker validity established',
             'selected_config_sha256':frozen_sha,'results':results,'ablations':ablations,'sensitivity':sensitivity}
    write_json(output/'evaluation/results.json',{'results':results})
    write_json(output/'ablation/results.json',{'results':ablations})
    write_json(output/'sensitivity/results.json',{'results':sensitivity,'pairing':'same seed for every factor value; algorithm-only factors reuse identical data'})
    write_json(output/'results.json',payload)
    finalize_manifest(output)
    return payload


def finalize_manifest(output):
    output=Path(output)
    try: revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): revision=None
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','psutil')}
    try:
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--',str(Path(__file__).resolve().parents[2])],cwd=Path(__file__).parent,text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError): dirty=None
    source_root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(source_root)):file_sha(p) for p in sorted((source_root/'orion').glob('*.py'))}
    project_root=source_root.parent
    for folder in ('scripts','configs'):
        for p in sorted((project_root/folder).glob('*')):
            if p.is_file(): sources[str(p.relative_to(project_root))]=file_sha(p)
    for filename in ('pyproject.toml','README.md'):
        if (project_root/filename).is_file(): sources[filename]=file_sha(project_root/filename)
    manifest={'schema_version':2,'python':sys.version,'platform':platform.platform(),'dependencies':versions,
              'code_revision':revision,'dirty_worktree':dirty,'source_sha256':sources,'measurement_note':'local timings and process RSS are descriptive; independent controller evidence required for trusted acceptance',
              'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    write_json(output/'manifest.json',manifest)
    return manifest


@dataclass
class ExperimentConfig:
    name:str='orion2'
    seed:int=42
    methods:list=field(default_factory=lambda:list(METHODS))
    mode:str='evaluation'
    output_path:str='outputs/orion2'

class ExperimentRunner:
    def __init__(self,config): self.config=config; self.results=[]
    def run(self):
        self.results=run_protocol({'development_seeds':[self.config.seed],'evaluation_seeds':[self.config.seed+1]},self.config.output_path)
        return self.results
    def save_results(self): return self.results

class EvaluationRunner:
    """Compatibility adapter; redoes development instead of trusting old fake selection."""
    def __init__(self,dev_config_path,eval_config_path):
        self.dev_config=json.loads(Path(dev_config_path).read_text())
        self.eval_config=json.loads(Path(eval_config_path).read_text()); self.results=[]
    def run_evaluation(self,output_path='outputs/orion2'):
        dev=self.dev_config.get('development',self.dev_config); ev=self.eval_config.get('evaluation',self.eval_config)
        config={**dev,**ev,'development_seeds':dev.get('seeds',[1,42,1024]),'evaluation_seeds':ev.get('seeds',[55,99,2048])}
        self.results=run_protocol(config,output_path)
        return self.results
