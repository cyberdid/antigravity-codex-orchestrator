"""Paper I task set: frozen manifest and task invariants through the benchmark's own setup."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper_tasks"))

import benchmark_gemma_orion_phase2b as bench  # noqa: E402
import registry  # noqa: E402

PAPER = {tid: (setup, hold) for tid, setup, hold in bench.TASKS if tid.startswith("P")}
BY_ID = {task.id: task for task in registry.ALL}


def test_manifest_matches_task_content():
    import json

    manifest = json.loads(registry.MANIFEST.read_text())
    assert set(manifest) == set(BY_ID) and len(manifest) == 25
    for task in registry.ALL:
        assert manifest[task.id]["sha256"] == registry.digest(task)


@pytest.mark.parametrize("task_id", sorted(PAPER))
def test_reference_passes_visible_and_holdout_buggy_fails(tmp_path, task_id):
    setup, hold = PAPER[task_id]
    buggy = tmp_path / "buggy"
    fixed = tmp_path / "fixed"
    buggy.mkdir()
    fixed.mkdir()
    setup(buggy)
    setup(fixed)
    for name, text in BY_ID[task_id].reference.items():
        (fixed / name).write_text(text)
    run = lambda ws: subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=ws, capture_output=True).returncode == 0
    assert run(fixed) and hold(fixed)
    assert not run(buggy) and not hold(buggy)
