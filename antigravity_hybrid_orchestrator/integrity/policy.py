"""Persist operator execution policy outside the writable candidate workspace."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def safe_run_path(run_log):
    """Keep stable lexical identity; reject candidate-controlled link redirection."""
    path = Path(os.path.abspath(run_log))
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Run log path may not contain symlinks')
    return path


class RunPolicyStore:
    @staticmethod
    def _path(run_log):
        run_log = safe_run_path(run_log)
        base = Path(os.environ.get('ANTIGRAVITY_STATE_HOME', Path.home()/'.local/state/antigravity')).resolve()
        directory = base / 'policies'
        workspace = run_log.parent
        if directory.is_relative_to(workspace):
            raise ValueError('Controller policies must be outside candidate run directory')
        key = hashlib.sha256(str(run_log).encode()).hexdigest()
        return directory / (key + '.json')

    @classmethod
    def save(cls, run_log, policy):
        path = cls._path(run_log)
        workspace = Path(policy['workspace']).resolve()
        if path.is_relative_to(workspace):
            raise ValueError('Controller policy must be outside candidate workspace')
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.chmod(path.parent,0o700)
        fd, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(policy,stream,allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, run_log):
        path = cls._path(run_log)
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        if not isinstance(result,dict):
            raise ValueError('Invalid trusted run policy')
        return result
