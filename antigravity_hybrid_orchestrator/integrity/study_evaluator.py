"""Require independent numerical validation and isolated study replication."""

from pathlib import Path
from uuid import uuid4

from .evaluator import EvaluationDecision
from .evidence import contained_file, source_paths


def evaluate_study(evaluator):
    from .study import validate_study_directory, compare_studies
    submitted = (evaluator.workspace / evaluator.study_directory).resolve()
    if not submitted.is_relative_to(evaluator.workspace):
        raise ValueError('Submitted study must be inside candidate workspace')
    core = evaluator.evaluate('orion-core-v1')
    if not core.accepted:
        return core
    checks = [{'id':'core-contract', 'status':'ACCEPTED', 'evidence_id':core.evidence_id}]
    validation = validate_study_directory(submitted, evaluator.study_profile)
    failures = list(validation['failures'])
    execution = {'isolated':True, 'not_started_reason':'Submitted study failed independent validation'}
    hashes = evaluator.store.get(core.evidence_id)['source_hashes']
    artifacts, replica_hashes = {}, {}
    replica = None
    status = 'REJECTED' if failures else 'ACCEPTED'
    if not failures:
        # Bind the submitted bytes before starting the independent reproduction.
        submitted_files = [path.relative_to(evaluator.workspace).as_posix()
                           for path in submitted.rglob('*') if path.is_file()]
        artifacts = evaluator.store.hash_files(submitted_files)
        retained = evaluator.store.directory / 'replicas' / uuid4().hex
        retained.mkdir(parents=True,mode=0o700)
        try:
            result, execution, replica_sources, error = evaluator._invoke(
                {'kind':'study', 'profile':evaluator.study_profile}, retained_output=retained,
                timeout_s=900 if evaluator.study_profile == 'full' else 120)
            if error:
                failures.append('Independent replication: ' + error)
                status = 'SKIPPED_RESOURCE_LIMIT' if execution.get('timed_out') or execution.get('oom_killed') else 'REJECTED'
            else:
                replica = retained/'study'
                independent = validate_study_directory(replica,evaluator.study_profile)
                failures.extend(independent['failures'])
                failures.extend(compare_studies(submitted,replica))
                if replica_sources != hashes:
                    failures.append('Candidate source changed between oracle and replication')
                replica_hashes = evaluator.store.hash_replica_files(
                    [path.relative_to(evaluator.store.directory).as_posix()
                     for path in retained.rglob('*') if path.is_file()])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            status = 'INCONCLUSIVE'
            failures.append(str(exc))
        try:
            if evaluator.store.hash_files(artifacts) != artifacts or evaluator.store.hash_files(hashes) != hashes or source_paths(evaluator.workspace) != sorted(hashes):
                failures.append('Submitted source/artifacts changed during independent verification')
        except (OSError,ValueError):
            failures.append('Submitted source/artifacts disappeared during verification')
    if failures and status == 'ACCEPTED':
        status = 'REJECTED'
    checks.extend(validation.get('checks', []))
    checks.append({'id':'independent-study-replication', 'status':status, 'failures':failures[:50]})
    record = evaluator.store.append({
        'contract_id':'orion-study-v1', 'contract_version':1, 'profile':evaluator.study_profile,
        'status':status, 'source_hashes':hashes, 'artifacts':artifacts,
        'replica_hashes':replica_hashes, 'checks':checks, 'execution':execution,
        'measurements':validation.get('measurements', {}),
        'claim_links':validation.get('claim_links', []) if status == 'ACCEPTED' else [],
        'core_evidence_id':core.evidence_id,
        'replica_directory':str(replica) if replica else None,
    })
    return EvaluationDecision(status,evidence_id=record['evidence_id'],checks=checks,feedback='; '.join(failures[:30]))
