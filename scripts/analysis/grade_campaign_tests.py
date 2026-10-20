"""Grade the contract-test modules of a Paper I campaign against reference and buggy code.

A test that fails on the reference fix is either over-constrained (asserts behaviour the
docstring does not state) or exposes a reference that departs from its docstring; each such
case is adjudicated by hand in the results.
Usage: python scripts/analysis/grade_campaign_tests.py benchmarks/campaign_a.json [--out FILE]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper_tasks"))
import registry  # noqa: E402


def run_module(files, source):
    with tempfile.TemporaryDirectory(prefix="grade-") as tmp:
        for name, text in files.items():
            (Path(tmp) / name).write_text(text)
        (Path(tmp) / "test_spec_contract.py").write_text(source)
        try:
            proc = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_spec_contract"], cwd=tmp,
                                  capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return 0, ["TIMEOUT"]
    lines = proc.stderr.splitlines()
    return sum(l.endswith("... ok") for l in lines), [l.split(" (")[0] for l in lines if l.endswith(("... FAIL", "... ERROR"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--out")
    args = parser.parse_args()
    tasks = {t.id: t for t in registry.ALL}
    try:
        from natural import load_tasks
        tasks.update({t.id: t for t in load_tasks()})
    except (FileNotFoundError, KeyError):
        pass
    data = json.loads(Path(args.results).read_text())
    sources = {}
    for pol, v in data["policies"].items():
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                if t["metadata"].get("spec_tests_source"):
                    sources[(pol, t["task_id"])] = t["metadata"]["spec_tests_source"]
    report, wrong = [], 0
    for (pol, tid), src in sorted(sources.items(), key=lambda kv: (kv[0][0], int(kv[0][1].split("_")[0][1:]))):
        task = tasks[tid]
        ok_ref, bad_ref = run_module({**task.files, **task.reference}, src)
        ok_bug, bad_bug = run_module(task.files, src)
        wrong += len(bad_ref)
        report.append({"policy": pol, "task": tid, "reference_ok": ok_ref, "reference_failing": bad_ref,
                       "buggy_ok": ok_bug, "buggy_failing": len(bad_bug)})
        print(f"{pol[-14:]:14s} {tid:24s} ref ok {ok_ref:2d} fail {len(bad_ref)} | buggy fail {len(bad_bug)}"
              + (f"  {bad_ref}" if bad_ref else ""))
    print(f"modules {len(report)}; tests failing on the reference: {wrong}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
