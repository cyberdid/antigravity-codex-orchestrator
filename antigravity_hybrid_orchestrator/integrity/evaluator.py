"""Scientific promotion controlled outside the candidate's execution boundary."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import secrets
import selectors
import shutil
import subprocess
import tempfile
import time

from antigravity_hybrid_orchestrator.integrity.evidence import EvidenceStore, source_paths
from antigravity_hybrid_orchestrator.integrity.oracle import (
    check_metric_results, check_neighbors, metric_cases, squared_distance,
)
from antigravity_hybrid_orchestrator.runtime.sandbox import DockerSandbox


CONTRACTS = ('metrics-v1', 'retrieval-v1', 'orion-core-v1', 'orion-study-v1')


@dataclass
class EvaluationDecision:
    status: str
    verification_level: str = 'scientific'
    evidence_id: str = ''
    checks: list = field(default_factory=list)
    feedback: str = ''

    @property
    def accepted(self):
        return self.status == 'ACCEPTED'

    def to_dict(self):
        return asdict(self)


# This adapter contains only the public calling convention and challenge inputs.
# Expected outputs, comparison logic and trusted measurements are never mounted.
ADAPTER = '''import json, sys
import numpy as np
sys.path.insert(0, '/workspace/candidate/src')
request = json.load(open('/workspace/request.json'))
if request['kind'] == 'study':
    import subprocess
    subprocess.run([sys.executable, '/workspace/candidate/scripts/run_protocol.py',
                    '--profile', request['profile'], '--output', '/output/study'], check=True)
    output = {'profile':request['profile'], 'study_directory':'study'}
elif request['kind'] == 'metrics':
    from orion.metrics import calculate_metrics
    output = [calculate_metrics(np.asarray(c['truth'],dtype=int), np.asarray(c['prediction'],dtype=int)) for c in request['cases']]
else:
    from orion.retrieval import exact_knn, approximate_knn
    vectors = np.asarray(request['vectors'], dtype=float)
    queries = np.asarray(request['queries'], dtype=float)
    output = {}
    for name, function in [('exact',exact_knn),('approximate',approximate_knn)]:
        indices, distances = function(vectors, queries, request['k'])
        output[name] = {'indices':np.asarray(indices).tolist(), 'distances':np.asarray(distances).tolist()}
with open('/output/result.json','w') as stream:
    json.dump(output, stream, allow_nan=False)
'''


class ResearchIntegrityEvaluator:
    def __init__(self, workspace, evidence_directory, *, image='antigravity-orion:local',
                 timeout_s=120., execution_control=None, study_profile='full', study_directory='orion2-run'):
        self.workspace = Path(workspace).resolve(strict=True)
        self.store = EvidenceStore(evidence_directory, self.workspace)
        self.image = image
        self.timeout_s = timeout_s
        self.execution_control = execution_control
        self.study_profile = study_profile
        self.study_directory = study_directory

    def _copy_source(self, destination):
        import hashlib
        hashes = {}
        total = 0
        for relative in source_paths(self.workspace):
            source = self.workspace / relative
            total += source.stat().st_size
            if total > 128 * 1024 * 1024 or len(hashes) >= 10000:
                raise ValueError('Candidate source snapshot exceeds resource policy')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        return hashes

    def _invoke(self, request, *, retained_output=None, timeout_s=None):
        from contextlib import nullcontext
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        execution = {'isolated': True, 'started_at_utc': started_at,
                     'exit_code': None, 'timed_out': False,
                     'sampled_cgroup_memory_peak_bytes': None,
                     'resource_measurement': 'Linux cgroup memory.peak sampled by host; null if unavailable'}
        output_context = nullcontext(str(retained_output)) if retained_output is not None else tempfile.TemporaryDirectory(prefix='rie-output-')
        with tempfile.TemporaryDirectory(prefix='rie-input-') as input_dir, output_context as output_dir:
            stage = Path(input_dir)
            source_hashes = self._copy_source(stage / 'candidate')
            (stage / 'request.json').write_text(json.dumps(request, allow_nan=False))
            (stage / 'invoke.py').write_text(ADAPTER)
            sandbox = DockerSandbox(stage, self.image, output_directory=output_dir,
                                    max_file_size_bytes=512 * 1024 * 1024 if request['kind'] == 'study' else 16 * 1024 * 1024)
            cmd = sandbox.build_command('python3 /workspace/invoke.py', read_only=True)
            # Retain the terminal container until inspection, then remove in finally.
            cmd.remove('--rm')
            proc = None
            output = bytearray()
            resource_path = None
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        env=sandbox.client_environment(), start_new_session=True)
                last_inspect = 0
                last_stats = 0
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while selector.get_map() or proc.poll() is None:
                        now = time.monotonic()
                        stop = self.execution_control.check() if self.execution_control else None
                        if now - start >= (timeout_s if timeout_s is not None else self.timeout_s) or (stop and stop.stopped):
                            execution['timed_out'] = True
                            execution['stop_reason'] = stop.reason if stop and stop.stopped else 'evaluation_timeout'
                            sandbox.cleanup()
                            proc.terminate()
                            break
                        if len(output) > 2 * 1024 * 1024:
                            raise ValueError('Candidate process output exceeds policy')
                        if resource_path is None and now - last_inspect > .25:
                            last_inspect = now
                            info = subprocess.run(['docker', 'inspect', '-f', '{{.State.Pid}}', sandbox.container_name],
                                                  capture_output=True, text=True, timeout=2,
                                                  env=sandbox.client_environment())
                            try:
                                pid = int(info.stdout.strip())
                                cgroup = Path(f'/proc/{pid}/cgroup').read_text().strip().split('0::', 1)[1].splitlines()[0]
                                resource_path = Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'memory.peak'
                            except (ValueError, IndexError, OSError):
                                pass
                        if resource_path is not None:
                            try:
                                peak = int(resource_path.read_text())
                                execution['sampled_cgroup_memory_peak_bytes'] = max(execution['sampled_cgroup_memory_peak_bytes'] or 0, peak)
                            except (ValueError, OSError):
                                pass
                        if request['kind'] == 'study' and now - last_stats > 2:
                            last_stats = now
                            try:
                                stats = subprocess.run(['docker','stats','--no-stream','--format','{{.MemUsage}}',sandbox.container_name],
                                                       capture_output=True,text=True,timeout=2,env=sandbox.client_environment())
                                import re
                                match = re.match(r'([\d.]+)([a-zA-Z]+)', stats.stdout.strip())
                                units = {'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3,'kB':1000,'MB':1000**2,'GB':1000**3}
                                if match and match[2] in units:
                                    memory = int(float(match[1]) * units[match[2]])
                                    execution['sampled_container_working_set_peak_bytes'] = max(execution.get('sampled_container_working_set_peak_bytes',0),memory)
                            except subprocess.TimeoutExpired:
                                pass
                        for key, _ in selector.select(.05):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if chunk:
                                output.extend(chunk)
                            else:
                                selector.unregister(key.fileobj)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                execution['exit_code'] = proc.returncode
                execution['duration_monotonic_s'] = time.monotonic() - start
                info = subprocess.run(['docker','inspect','-f','{{.State.OOMKilled}}',sandbox.container_name],
                                      capture_output=True,text=True,timeout=2,env=sandbox.client_environment())
                execution['oom_killed'] = info.stdout.strip() == 'true'
                if execution['timed_out'] or execution['oom_killed']:
                    return None, execution, source_hashes, 'Resource limit reached'
                if proc.returncode:
                    return None, execution, source_hashes, output.decode(errors='replace')[-2000:]
                path = Path(output_dir) / 'result.json'
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
                    return None, execution, source_hashes, 'Missing or oversized candidate result'
                result = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
                return result, execution, source_hashes, ''
            finally:
                try:
                    sandbox.cleanup()
                finally:
                    if proc is not None:
                        if proc.poll() is None:
                            proc.kill()
                        proc.wait()
                        if proc.stdout:
                            proc.stdout.close()

    def evaluate(self, contract_id):
        if contract_id not in CONTRACTS:
            raise ValueError(f'Unknown controller contract: {contract_id}')
        if contract_id == 'orion-study-v1':
            from antigravity_hybrid_orchestrator.integrity.study_evaluator import evaluate_study
            return evaluate_study(self)
        kinds = ['metrics', 'retrieval'] if contract_id == 'orion-core-v1' else [contract_id.split('-')[0]]
        checks, executions, measurements, hashes, challenges = [], [], {}, {}, []
        status = 'ACCEPTED'
        for kind in kinds:
            if kind == 'metrics':
                cases = metric_cases(secrets.randbits(64))
                request = {'kind':kind, 'cases':cases}
            else:
                rng = random.Random(secrets.randbits(64))
                vectors = [[rng.uniform(-1, 1) for _ in range(9)] for _ in range(83)]
                queries = [[rng.uniform(-1, 1) for _ in range(9)] for _ in range(11)]
                # Exact-point lookup is a minimal query-dependence contract,
                # independent of the measured quality of approximate retrieval.
                queries += [list(vector) for vector in vectors[-5:]]
                request = {'kind':kind, 'vectors':vectors, 'queries':queries, 'k':5}
            try:
                output, execution, source_hashes, error = self._invoke(request)
                challenges.append({'request':request, 'response':output, 'error':error})
                executions.append(execution)
                if hashes and hashes != source_hashes:
                    raise ValueError('Candidate changed during multi-contract evaluation')
                hashes = source_hashes
                failures = []
                check_status = 'ACCEPTED'
                if error:
                    failures = [error]
                    if execution.get('timed_out') or execution.get('oom_killed'):
                        check_status = 'SKIPPED_RESOURCE_LIMIT'
                    elif execution.get('exit_code') == 125:
                        check_status = 'INCONCLUSIVE'
                elif kind == 'metrics':
                    failures = check_metric_results(cases, output)
                    measurements['metric_cases_passed'] = len(cases) if not failures else 0
                else:
                    if not isinstance(output, dict):
                        failures = ['Invalid retrieval envelope']
                    else:
                        for mode in ('exact', 'approximate'):
                            result = output.get(mode, {})
                            if not isinstance(result, dict):
                                failures.append('Invalid nested retrieval envelope')
                            else:
                                failures += check_neighbors(vectors, queries, result.get('indices'), result.get('distances'), 5, exact=mode == 'exact')
                        if not failures:
                            if any(output['approximate']['distances'][i][0] > 1e-12 for i in range(11, 16)):
                                failures.append('Approximate index fails exact-point query-dependence challenge')
                            measurements['recall_at_k'] = sum(len(set(a) & set(b)) / 5 for a, b in zip(output['exact']['indices'], output['approximate']['indices'])) / len(queries)
                if failures and check_status == 'ACCEPTED':
                    check_status = 'REJECTED'
                checks.append({'id': kind + '-oracle-v1', 'status':check_status, 'failures':failures[:30]})
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
                checks.append({'id':kind + '-oracle-v1', 'status':'INCONCLUSIVE', 'failures':[str(exc)]})
        statuses = {check['status'] for check in checks}
        status = next((value for value in ('REJECTED','INCONCLUSIVE','SKIPPED_RESOURCE_LIMIT') if value in statuses), 'ACCEPTED')
        # Bind to the source actually evaluated, never to a later version.
        try:
            stable = not hashes or (self.store.hash_files(hashes) == hashes and source_paths(self.workspace) == sorted(hashes))
        except (OSError, ValueError):
            stable = False
        if not stable:
            status = 'REJECTED'
            checks.append({'id':'source-stability', 'status':'REJECTED', 'failures':['Candidate changed after evaluation']})
        record = self.store.append({'contract_id':contract_id, 'contract_version':1,
                                    'status':status, 'checks':checks, 'measurements':measurements,
                                    'execution': executions[0] if len(executions) == 1 else {'isolated':True,'children':executions},
                                    'source_hashes':hashes, 'challenges':challenges})
        feedback = '; '.join(f'{check["id"]}: ' + '; '.join(check.get('failures', [])) for check in checks if check['status'] != 'ACCEPTED')
        return EvaluationDecision(status=status, evidence_id=record['evidence_id'], checks=checks, feedback=feedback)

    def evaluate_subtask(self, subtask, contract_id):
        paths = set(subtask.outputs + subtask.affected_files)
        reports = [path for path in paths if Path(path).name in ('claims.json', 'research_report.md')]
        if reports:
            if contract_id == 'orion-study-v1':
                return self.evaluate(contract_id).to_dict()
            from antigravity_hybrid_orchestrator.integrity.evidence import (
                contained_file, validate_claims, validate_narrative,
            )
            try:
                claims_paths = {str(Path(path).parent / 'claims.json') for path in reports}
                failures = []
                claims_by_dir = {}
                for path in claims_paths:
                    payload = json.loads(contained_file(self.workspace, path).read_text())
                    claims = payload['claims']
                    failures.extend(validate_claims(claims, self.store))
                    claims_by_dir[str(Path(path).parent)] = claims
                for path in reports:
                    if Path(path).name == 'research_report.md':
                        narrative = contained_file(self.workspace, path).read_text()
                        failures.extend(validate_narrative(narrative, claims_by_dir.get(str(Path(path).parent), [])))
                status = 'REJECTED' if failures else 'ACCEPTED'
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failures, status = [str(exc)], 'REJECTED'
            record = self.store.append({'status':status, 'contract_id':'claim-links-v1',
                                       'checks':failures, 'artifacts':self.store.hash_files(reports)})
            return EvaluationDecision(status, evidence_id=record['evidence_id'], feedback='; '.join(failures)).to_dict()
        if contract_id in ('metrics-v1', 'retrieval-v1'):
            return self.evaluate(contract_id).to_dict()
        if any(path.endswith('/metrics.py') for path in paths):
            return self.evaluate('metrics-v1').to_dict()
        if any(path.endswith('/retrieval.py') or path.endswith('/benchmark_scalability.py') for path in paths):
            return self.evaluate('retrieval-v1').to_dict()
        return EvaluationDecision('ACCEPTED', verification_level='behavioral',
                                  feedback='Shell verification only; final scientific gate remains mandatory').to_dict()
