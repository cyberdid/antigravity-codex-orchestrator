"""Trusted fixtures contain literal partitions and inputs, never candidate imports."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from antigravity_hybrid_orchestrator.integrity.oracle import reference_metrics


def _write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,allow_nan=False))


def _digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_digest(arrays):
    h=hashlib.sha256()
    for name in ('embeddings','labels_true','domains','outlier_mask'):
        a=arrays[name]; h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _seal(root):
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {'schema_version':2,'source_sha256':{'orion/metrics.py':'a'*64},'code_revision':'0'*40,'dirty_worktree':False}
    manifest['files']={str(p.relative_to(root)):_digest(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    _write(root/'manifest.json',manifest)


def _parameters(method,t):
    if method=='threshold_graph': return {'threshold':t}
    if method=='agglomerative': return {'distance_threshold':float(np.sqrt(2*(1-t)))}
    if method=='rr_graph': return {'k':20,'threshold':t}
    return {'max_neighbors':20,'expansion_factor':1.2,'reciprocal_threshold':t,'calibration_factor':1.1,
            'abstention_threshold':t,'cohesion_threshold':max(0.,t-.1),
            'use_adaptive_neighborhood':True,'use_reciprocal_consistency':True,
            'use_local_calibration':True,'use_cohesion_constraint':True,'use_abstention':True}


@pytest.fixture
def study(tmp_path):
    root=tmp_path/'submitted'; root.mkdir()
    config={'profile':'smoke','development_seeds':[1],'evaluation_seeds':[55],
        'generator':{'num_entities':4,'observations_per_entity':12,'embedding_dim':32,'variance':.35,'noise_level':.1,'inter_entity_similarity':.4,'outliers_ratio':.1},
        'scenarios':[{'domain_shift':0.},{'domain_shift':.35}], 'threshold_grid':[.4,.65,.85],
        'sensitivity_grid':{'merge_threshold':[.4,.8],'domain_shift':[0.,.35]}}
    _write(root/'config.json',config)
    methods=['threshold_graph','agglomerative','rr_graph','risk_aware']
    selected={'schema_version':2,'development_seeds':[1],'evaluation_seeds':[55],
              'methods':{m:_parameters(m,.4) for m in methods},'maximum_development_fdp':.01,
              'config_sha256':_digest(root/'config.json'),'scores':{}}
    for m in methods:
        selected['scores'][m]=[{'parameters':_parameters(m,t),'mean_pairwise_f1':1.,'mean_fdp':0.,'eligible':True} for t in [.4,.65,.85]]
    _write(root/'selected_config.json',selected); frozen=_digest(root/'selected_config.json')
    def record(seed,shift,method,params,tag,**metadata):
        rng=np.random.default_rng(seed); x=rng.normal(size=(48,32)); x[:,0]+=shift*(np.arange(48)%2)
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        truth=np.repeat(np.arange(4),12); truth[-4:]=-1
        arrays={'embeddings':x,'labels_true':truth,'labels_pred':truth.copy(),'domains':np.arange(48)%2,'outlier_mask':truth==-1}
        rel=f'predictions/{tag}.npz'; path=root/rel; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**arrays)
        return {'method':method,'parameters':params,'metrics':reference_metrics(truth.tolist(),truth.tolist()),
                'status':'success','runtime_sec':.001,'seed':seed,'dataset_sha256':_dataset_digest(arrays),
                'predictions_file':rel,'predictions_sha256':_digest(path),**metadata}
    dev=[]; ev=[]; ab=[]; sens=[]
    for method in methods:
        for candidate,t in enumerate([.4,.65,.85]):
            for i,shift in enumerate([0.,.35]):
                dev.append(record(1,shift,method,_parameters(method,t),f'dev-{method}-{candidate}-{i}',candidate=candidate,split='development',scenario_id=i,scenario={'domain_shift':shift}))
    for i,shift in enumerate([0.,.35]):
        for method in methods:
            ev.append(record(55,shift,method,_parameters(method,.4),f'eval-{method}-{i}',split='evaluation',scenario_id=i,scenario={'domain_shift':shift},selected_config_sha256=frozen))
        variants={'full_system':None,'without_reciprocal_consistency':'use_reciprocal_consistency','without_local_calibration':'use_local_calibration',
                  'without_cohesion_constraint':'use_cohesion_constraint','without_abstention':'use_abstention','without_adaptive_neighborhood':'use_adaptive_neighborhood'}
        for variant,toggle in variants.items():
            params=_parameters('risk_aware',.4)
            if toggle: params[toggle]=False
            ab.append(record(55,shift,'risk_aware',params,f'ablation-{variant}-{i}',variant=variant,split='evaluation',scenario_id=i,scenario={'domain_shift':shift},selected_config_sha256=frozen))
    for parameter,values in config['sensitivity_grid'].items():
        for value in values:
            gen={**config['generator'],'seed':55}; params=_parameters('risk_aware',.4)
            if parameter=='domain_shift': gen['domain_shift']=value
            else: params.update(abstention_threshold=value,reciprocal_threshold=value)
            sens.append(record(55,gen.get('domain_shift',0.),'risk_aware',params,f'sens-{parameter}-{value}',
                               parameter=parameter,value=value,split='evaluation_sensitivity',generator_parameters=gen,selected_config_sha256=frozen))
    for name,rows in [('development',dev),('evaluation',ev),('ablation',ab),('sensitivity',sens)]: _write(root/f'{name}/results.json',{'results':rows})
    _write(root/'results.json',{'schema_version':2,'selected_config_sha256':frozen,'results':ev,'ablations':ab,'sensitivity':sens})
    rng=np.random.default_rng(712); x=rng.normal(size=(1000,192)); x/=np.linalg.norm(x,axis=1,keepdims=True)
    q=rng.normal(size=(16,192)); q/=np.linalg.norm(q,axis=1,keepdims=True)
    dist=((q[:,None,:]-x[None,:,:])**2).sum(axis=2); ids=np.argsort(dist,axis=1,kind='stable')[:,:10]; ds=np.take_along_axis(dist,ids,axis=1)
    folder=root/'scalability'; folder.mkdir()
    np.save(folder/'vectors.npy',x); np.save(folder/'queries.npy',q)
    np.savez_compressed(folder/'neighbors.npz',exact_ids=ids,exact_distances=ds,approximate_ids=ids,approximate_distances=ds)
    row={'size':1000,'dimension':192,'query_count':16,'k':10,'seed':712,'algorithm':'projection_window_candidates_exact_rerank',
         'num_projections':8,'window':256,'distance':'squared_euclidean','recall_at_k':1.,'generation_sec':.01,'exact_query_sec':.01,
         'approximate_build_sec':.01,'approximate_query_sec':.01,'process_sampled_peak_rss_mb':200.,
         **{f'{name}_file':f'{name}.npy' for name in ('vectors','queries')},'neighbors_file':'neighbors.npz',
         'vectors_sha256':_digest(folder/'vectors.npy'),'queries_sha256':_digest(folder/'queries.npy'),'neighbors_sha256':_digest(folder/'neighbors.npz')}
    _write(folder/'results.json',{'benchmarks':[row],'skipped':[]})
    claims=[{'id':'evaluation-count','kind':'artifact_summary','value':8,'source':'evaluation/results.json','source_sha256':_digest(root/'evaluation/results.json')},
            {'id':'ablation-count','kind':'artifact_summary','value':12,'source':'ablation/results.json','source_sha256':_digest(root/'ablation/results.json')},
            {'id':'recall-1000','kind':'retrieval_recall_at_k','value':1.,'size':1000,'k':10,'query_count':16,
             'source':'scalability/neighbors.npz','source_sha256':_digest(folder/'neighbors.npz')}]
    _write(root/'claims.json',{'schema_version':2,'claims':claims}); _seal(root)
    return root


def test_independent_fixture_passes_and_links_claims(study):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    result=validate_study_directory(study,'smoke')
    assert result['failures']==[]
    assert result['measurements']['evaluation_count']==8
    assert result['measurements']['development_count']==24
    assert result['measurements']['recall_at_1000']==1.
    assert {x['claim_id'] for x in result['claim_links']}=={'evaluation-count','ablation-count','recall-1000'}

@pytest.mark.parametrize('mutation,expected',[('metrics','metric'),('claims','claim'),('selection','selection'),('pairing','dataset'),('neighbors','neighbor'),('seeds','config'),('digest','digest')])
def test_self_consistent_tampering_rejected(study,mutation,expected):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    if mutation=='metrics':
        p=study/'evaluation/results.json'; doc=json.loads(p.read_text()); doc['results'][0]['metrics']['b_cubed_precision']=.123; _write(p,doc)
    elif mutation=='claims':
        p=study/'claims.json'; doc=json.loads(p.read_text()); doc['claims'][0]['value']=999; _write(p,doc)
    elif mutation=='selection':
        p=study/'selected_config.json'; doc=json.loads(p.read_text()); doc['methods']['agglomerative']['n_clusters']=4; _write(p,doc)
    elif mutation=='pairing':
        p=study/'ablation/results.json'; doc=json.loads(p.read_text()); doc['results'][1]['dataset_sha256']='0'*64; _write(p,doc)
    elif mutation=='neighbors':
        p=study/'scalability/neighbors.npz'; arrays=dict(np.load(p,allow_pickle=False)); arrays['exact_ids'][0]=np.arange(10); np.savez_compressed(p,**arrays)
        r=study/'scalability/results.json'; doc=json.loads(r.read_text()); doc['benchmarks'][0]['neighbors_sha256']=_digest(p); _write(r,doc)
    elif mutation=='seeds':
        p=study/'config.json'; doc=json.loads(p.read_text()); doc['evaluation_seeds']=[1]; _write(p,doc)
    else: (study/'predictions/dev-threshold_graph-0-0.npz').write_bytes(b'changed')
    _seal(study)
    result=validate_study_directory(study,'smoke')
    assert result['failures']
    assert any(expected in f.lower() for f in result['failures']),result['failures']


def test_fake_historical_structure_is_rejected(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    _write(tmp_path/'results.json',{'metrics':{'ari':.85,'recall':.98}})
    assert validate_study_directory(tmp_path,'smoke')['failures']


def test_symlink_and_unsafe_numpy_are_rejected(study,tmp_path):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    p=study/'predictions/dev-threshold_graph-0-0.npz'; p.unlink(); p.symlink_to(tmp_path/'secret.npz')
    assert validate_study_directory(study,'smoke')['failures']


def test_replica_comparison_ignores_timing_but_detects_prediction_change(study,tmp_path):
    import shutil
    from antigravity_hybrid_orchestrator.integrity.study import compare_studies
    replica=tmp_path/'replica'; shutil.copytree(study,replica)
    p=replica/'evaluation/results.json'; doc=json.loads(p.read_text()); doc['results'][0]['runtime_sec']=100.; _write(p,doc)
    assert compare_studies(study,replica)==[]
    arr=replica/'predictions/eval-threshold_graph-0.npz'; arrays=dict(np.load(arr,allow_pickle=False)); arrays['labels_pred'][0]=123; np.savez_compressed(arr,**arrays)
    assert any('prediction' in f.lower() for f in compare_studies(study,replica))

def test_pickle_object_arrays_are_rejected_without_loading(study):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    p=study/'predictions/dev-threshold_graph-0-0.npz'
    arrays=dict(np.load(p,allow_pickle=False)); arrays['labels_pred']=np.array([{'unsafe':'object'}]*48,dtype=object)
    np.savez_compressed(p,**arrays)
    doc=json.loads((study/'development/results.json').read_text()); doc['results'][0]['predictions_sha256']=_digest(p)
    _write(study/'development/results.json',doc); _seal(study)
    result=validate_study_directory(study,'smoke')
    assert any('object arrays' in f.lower() for f in result['failures'])


def test_changed_prediction_is_rejected_after_resealing_metrics_and_claims(study):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    p=study/'predictions/ablation-full_system-0.npz'; arrays=dict(np.load(p,allow_pickle=False)); arrays['labels_pred']=np.zeros(48,dtype=int)
    np.savez_compressed(p,**arrays)
    doc=json.loads((study/'ablation/results.json').read_text()); row=doc['results'][0]
    row['predictions_sha256']=_digest(p); row['metrics']=reference_metrics(arrays['labels_true'].tolist(),arrays['labels_pred'].tolist())
    _write(study/'ablation/results.json',doc)
    composite=json.loads((study/'results.json').read_text()); composite['ablations']=doc['results']; _write(study/'results.json',composite)
    claims=json.loads((study/'claims.json').read_text()); claims['claims'][1]['source_sha256']=_digest(study/'ablation/results.json'); _write(study/'claims.json',claims)
    _seal(study)
    result=validate_study_directory(study,'smoke')
    assert any('full-system ablation prediction' in f for f in result['failures'])


def test_replica_comparison_detects_silent_config_change(study,tmp_path):
    import shutil
    from antigravity_hybrid_orchestrator.integrity.study import compare_studies
    replica=tmp_path/'replica'; shutil.copytree(study,replica)
    doc=json.loads((replica/'selected_config.json').read_text()); doc['methods']['risk_aware']['max_neighbors']=19; _write(replica/'selected_config.json',doc)
    assert any('selection' in f for f in compare_studies(study,replica))


def test_unknown_profile_rejected(study):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    assert validate_study_directory(study,'candidate-chosen')['failures']

@pytest.mark.parametrize('artifact,field',[('selected_config.json','methods'),('claims.json','claims')])
def test_malformed_nested_schema_returns_rejection(study,artifact,field):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    path=study/artifact; doc=json.loads(path.read_text()); doc[field]=[17]; _write(path,doc); _seal(study)
    assert validate_study_directory(study,'smoke')['failures']

def test_claim_failure_marks_a_check_rejected(study):
    from antigravity_hybrid_orchestrator.integrity.study import validate_study_directory
    p=study/'claims.json'; doc=json.loads(p.read_text()); doc['claims'][0]['value']=999; _write(p,doc); _seal(study)
    result=validate_study_directory(study,'smoke')
    assert any(c['status']=='REJECTED' for c in result['checks'])
    assert result['claim_links']==[]
