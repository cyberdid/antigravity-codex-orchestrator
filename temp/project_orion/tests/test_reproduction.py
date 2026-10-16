import importlib
import json
from pathlib import Path
import numpy as np
import pytest

def test_package_metadata_parses():
    import tomllib
    cfg=tomllib.loads((Path(__file__).resolve().parents[1]/'pyproject.toml').read_text())
    assert 'scikit-learn' in ' '.join(cfg['project']['dependencies'])

def test_protocol_saves_predictions_freezes_selection_and_pairs_ablations(tmp_path):
    runner=importlib.import_module('orion.runner')
    config={'development_seeds':[1], 'evaluation_seeds':[55], 'generator':{'num_entities':3,'observations_per_entity':5,'embedding_dim':8,'outliers_ratio':.1}, 'threshold_grid':[.6,.8], 'sensitivity_grid':{'merge_threshold':[.6,.8]}}
    payload=runner.run_protocol(config,tmp_path)
    selected=json.loads((tmp_path/'selected_config.json').read_text())
    assert selected['development_seeds']==[1] and selected['evaluation_seeds']==[55]
    assert set(selected['methods'])=={'threshold_graph','agglomerative','rr_graph','risk_aware'}
    assert payload['results'] and payload['ablations'] and payload['sensitivity']
    paired=set()
    for record in payload['results']+payload['ablations']+payload['sensitivity']:
        data=np.load(tmp_path/record['predictions_file'])
        from orion.metrics import calculate_metrics
        assert record['metrics']==calculate_metrics(data['labels_true'],data['labels_pred'])
        if record.get('variant'): paired.add(record['dataset_sha256'])
    assert len(paired)==1
    assert (tmp_path/'manifest.json').is_file()
    with pytest.raises(FileExistsError): runner.run_protocol(config,tmp_path)

def test_protocol_rejects_overlapping_seeds(tmp_path):
    runner=importlib.import_module('orion.runner')
    with pytest.raises(ValueError,match='disjoint'):
        runner.run_protocol({'development_seeds':[1],'evaluation_seeds':[1]},tmp_path)

def test_scalability_benchmark_returns_measured_neighbors(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    benchmark=importlib.import_module('benchmark_scalability')
    record=benchmark.run_benchmark(100,use_approx=True,output_dir=tmp_path,dimension=8,query_count=7,k=3)
    data=np.load(tmp_path/record['neighbors_file'])
    assert data['approximate_ids'].shape==(7,3)
    expected=sum(len(set(a)&set(b)) for a,b in zip(data['approximate_ids'],data['exact_ids']))/(7*3)
    assert record['recall_at_k']==expected
    assert record['query_count']==7 and record['process_sampled_peak_rss_mb']>0
    assert 'ari' not in record

def test_manifest_binds_actual_source_files(tmp_path):
    from orion.runner import finalize_manifest,file_sha
    manifest=finalize_manifest(tmp_path)
    actual=Path(importlib.import_module('orion.metrics').__file__)
    assert manifest['source_sha256']['orion/metrics.py']==file_sha(actual)
    assert 'dirty_worktree' in manifest
