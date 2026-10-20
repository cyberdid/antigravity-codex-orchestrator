"""Grade the cloud-written contract tests stored in a benchmark JSON against reference code.

A correct module passes on the reference implementation; a useful one fails on the
original buggy code. Usage: python scripts/analysis/grade_contract_tests.py <benchmark.json>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_gemma_orion_phase2b as bench  # noqa: E402
from references import REFERENCES  # noqa: E402


def run(prefix, source, reference):
    task = next(t for t in bench.TASKS if t[0].startswith(prefix + "_"))
    ws = Path(tempfile.mkdtemp())
    task[1](ws)
    if reference:
        for name, text in REFERENCES[prefix].items():
            (ws / name).write_text(text)
        assert task[2](ws), f"reference for {prefix} fails its own holdout"
    (ws / "test_spec_contract.py").write_text(source)
    try:
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_spec_contract"], cwd=ws,
                              capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, ["TIMEOUT"]
    lines = proc.stderr.splitlines()
    return sum(line.endswith("... ok") for line in lines), [l for l in lines if l.endswith(("... FAIL", "... ERROR"))]


def main(path):
    data = json.load(open(path))
    sources = {}
    for v in data["policies"].values():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                src = t["metadata"].get("spec_tests_source")
                if src:
                    sources.setdefault(t["task_id"].split("_")[0], src)
    wrong_total = 0
    for prefix in sorted(sources, key=lambda p: int(p[1:])):
        if prefix not in REFERENCES:
            print(f"{prefix}: no reference available")
            continue
        ok_r, bad_r = run(prefix, sources[prefix], True)
        ok_o, bad_o = run(prefix, sources[prefix], False)
        wrong_total += len(bad_r)
        print(f"{prefix}: reference {ok_r} ok / {len(bad_r)} failing; original {ok_o} ok / {len(bad_o)} failing")
        for line in bad_r:
            print("    WRONG:", line)
    print(f"TOTAL wrong tests on reference code: {wrong_total}")


if __name__ == "__main__":
    main(sys.argv[1])
