"""Owner-only, append-only-by-API controller evidence with content bindings.

The external store plus container mount policy protects against candidate writes.
Hash chaining detects modification; it is not a signature or protection from the
host account administrator. Evidence never accepts candidate-authored attestations.
"""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

SOURCE_IGNORES = {'.git', '.venv', 'venv', '__pycache__', '.antigravity_runs',
                  '.pytest_cache', 'node_modules', 'artifacts', 'reports', 'outputs', 'orion2-run'}


def source_paths(workspace):
    paths = []
    for base, directories, files in os.walk(workspace, followlinks=False):
        directories[:] = [d for d in directories if d not in SOURCE_IGNORES]
        for directory in directories:
            if (Path(base) / directory).is_symlink():
                raise ValueError('Candidate source symlinks are not allowed')
        for name in files:
            source = Path(base) / name
            if source.is_symlink():
                raise ValueError('Candidate source symlinks are not allowed')
            if source.suffix in {'.py', '.pyi', '.json', '.toml'}:
                paths.append(source.relative_to(workspace).as_posix())
    return sorted(paths)


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contained_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Evidence path must be workspace relative')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError('Evidence file is missing or escapes workspace')
    return resolved


class EvidenceStore:
    def __init__(self, directory, workspace):
        self.directory = Path(directory).resolve()
        self.workspace = Path(workspace).resolve()
        if self.directory.is_relative_to(self.workspace):
            raise ValueError('Trusted evidence must be outside candidate workspace')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)

    def hash_files(self, paths):
        return self._hash_files(self.workspace, paths)

    def hash_replica_files(self, paths):
        if any(not Path(path).parts or Path(path).parts[0] != 'replicas' for path in paths):
            raise ValueError('Replica bindings must be inside controller replicas directory')
        return self._hash_files(self.directory, paths)

    @staticmethod
    def _hash_files(root, paths):
        result = {}
        for relative in sorted(set(paths)):
            target = contained_file(root, relative)
            digest = hashlib.sha256()
            with target.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            result[str(relative)] = digest.hexdigest()
        return result

    def append(self, payload):
        index_path = self.directory / 'index.jsonl'
        fd = os.open(index_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a+', encoding='utf-8') as index:
            fcntl.flock(index, fcntl.LOCK_EX)
            index.seek(0)
            previous = None
            for line in index:
                if line.strip():
                    previous = json.loads(line)['record_digest']
            record = dict(payload, schema_version=1, evidence_id=uuid4().hex,
                          previous_digest=previous)
            record['record_digest'] = digest_object(record)
            path = self.directory / (record['evidence_id'] + '.json')
            entry_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(entry_fd, 'w', encoding='utf-8') as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            index.write(json.dumps({'evidence_id': record['evidence_id'], 'record_digest': record['record_digest']}) + '\n')
            index.flush()
            os.fsync(index.fileno())
            return record

    def get(self, evidence_id):
        if not re.fullmatch(r'[0-9a-f]{32}', evidence_id):
            raise ValueError('Invalid evidence id')
        return json.loads((self.directory / (evidence_id + '.json')).read_text())

    def validate(self, evidence_id):
        try:
            record = self.get(evidence_id)
            recorded_digest = record.pop('record_digest')
            if digest_object(record) != recorded_digest:
                return False
            previous = None
            seen = False
            with (self.directory / 'index.jsonl').open() as stream:
                for line in stream:
                    entry = json.loads(line)
                    linked = self.get(entry['evidence_id'])
                    digest = linked.pop('record_digest')
                    if digest_object(linked) != digest or entry['record_digest'] != digest or linked.get('previous_digest') != previous:
                        return False
                    previous = digest
                    if entry['evidence_id'] == evidence_id:
                        seen = True
            if not seen:
                return False
            for field in ('artifacts', 'source_hashes'):
                bindings = record.get(field, {})
                if self.hash_files(bindings) != bindings:
                    return False
                if field == 'source_hashes' and bindings and source_paths(self.workspace) != sorted(bindings):
                    return False
            replicas = record.get('replica_hashes', {})
            if self.hash_replica_files(replicas) != replicas:
                return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False


def validate_claims(claims, store):
    failures = []
    if not isinstance(claims, list) or not claims:
        return ['No machine-readable claims supplied']
    operators = {'>=': lambda a, b: a >= b, '<=': lambda a, b: a <= b,
                 '==': lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)}
    for claim in claims:
        try:
            if not isinstance(claim, dict) or not claim.get('claim_id'):
                raise ValueError('Missing claim id')
            value = claim['value']
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid claim threshold')
            ids = claim['evidence_ids']
            if not isinstance(ids, list) or not ids:
                raise ValueError('Missing evidence')
            for evidence_id in ids:
                if not store.validate(evidence_id):
                    raise ValueError('Missing, altered or stale evidence')
                record = store.get(evidence_id)
                if record['status'] != 'ACCEPTED':
                    raise ValueError('Evidence not accepted')
                measured = record['measurements'][claim['metric_id']]
                if not isinstance(measured, (int, float)) or isinstance(measured, bool) or not math.isfinite(measured) or not operators[claim['operator']](measured, value):
                    raise ValueError('Claim disagrees with trusted measurement')
        except (KeyError, TypeError, ValueError, OSError) as exc:
            failures.append(f'{claim.get("claim_id", "unknown") if isinstance(claim, dict) else "unknown"}: {exc}')
    return failures


def validate_narrative(text, claims):
    if not isinstance(text, str) or not text.strip():
        return ['Empty narrative report']
    if not isinstance(claims, list) or not claims:
        return ['No machine-readable claims supplied for narrative report']
    failures = []
    scrubbed = text
    allowed_values = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get('claim_id') or claim.get('id') or '')
        evidence_ids = [str(eid) for eid in claim.get('evidence_ids', []) if eid]
        if not claim_id or claim_id not in text:
            failures.append(f'Narrative report omits claim citation: {claim_id or "unknown"}')
        if evidence_ids and not any(eid in text for eid in evidence_ids):
            failures.append(f'Narrative report omits evidence citation for {claim_id or "unknown"}')
        if claim_id:
            scrubbed = scrubbed.replace(claim_id, ' ')
        for eid in evidence_ids:
            scrubbed = scrubbed.replace(eid, ' ')
        value = claim.get('value')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            allowed_values.append(float(value))
    for match in re.finditer(r'(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(%?)(?!\w)', scrubbed):
        raw = float(match.group(1))
        candidates = (raw / 100.0, raw) if match.group(2) == '%' else (raw,)
        if not any(math.isclose(candidate, allowed, rel_tol=1e-6, abs_tol=1e-6)
                   for candidate in candidates for allowed in allowed_values):
            failures.append(f'Unchecked numeric narrative claim: {match.group(0)}')
    return failures

