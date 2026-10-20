"""Quality of the cloud-written contract tests, measured without the worker model.

1. Over-constraint: every distinct implementation that passed the task's own visible
   tests and hidden holdout in a run *without* contract tests is an independent valid
   implementation; contract tests that fail on it are over-constrained (or the holdout
   is weaker than the docstring).
2. Mutation coverage: single-point AST mutants of a reference implementation; for the
   mutants the hidden holdout kills (non-equivalent w.r.t. the hidden spec), what share
   do the visible tests and the contract tests kill?

Usage: python scripts/analysis/oracle_quality.py [--out benchmarks/oracle_quality.json]
"""
import argparse
import ast
import copy
import hashlib
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

H17_REFERENCE = {"kvpairs.py": '''import re


def parse_kv(text: str) -> dict:
    if not text.strip():
        return {}
    result = {}
    for pair in text.split(';'):
        if '=' not in pair:
            raise ValueError(pair)
        key, value = pair.split('=', 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r'[A-Za-z0-9_]+', key) or key in result:
            raise ValueError(pair)
        result[key] = value
    return result
'''}
DATASETS = ["spec_contract_experiment.json", "spec_contract_control.json", "spec_contract_v2.json",
            "spec_contract_confirm_fresh.json", "spec_contract_confirm_all.json"]
TASKS = {t[0].split("_")[0]: t for t in bench.TASKS}


def fresh_workspace(prefix):
    task = TASKS[prefix]
    ws = Path(tempfile.mkdtemp(prefix=f"oq_{prefix}_"))
    subtask = task[1](ws)
    return ws, subtask


def code_files(ws):
    return {p.name: p.read_text() for p in sorted(ws.glob("*.py")) if not p.name.startswith("test_")}


def visible_modules(subtask):
    return [Path(f).stem for f in subtask.input_files if Path(f).name.startswith("test_")]


def run_unittest(ws, modules, timeout=20):
    try:
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", *modules], cwd=ws, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, ["TIMEOUT"]
    failing = [l.split(" ... ")[0] for l in proc.stderr.splitlines() if l.endswith(("... FAIL", "... ERROR"))]
    return proc.returncode == 0, failing


def holdout(prefix, ws):
    try:
        return bool(TASKS[prefix][2](ws))
    except subprocess.TimeoutExpired:
        return False


def collect():
    contracts, valid, shaped = {}, {}, {}
    for name in DATASETS:
        data = json.loads((ROOT / "benchmarks" / name).read_text())
        for pol, v in data["policies"].items():
            for sr in v["seed_runs"]:
                for t in sr["tasks"]:
                    prefix, meta = t["task_id"].split("_")[0], t["metadata"]
                    if meta.get("spec_tests_source"):
                        src = meta["spec_tests_source"]
                        contracts.setdefault(prefix, {})[hashlib.sha256(src.encode()).hexdigest()[:10]] = (name, src)
                    if t["pipeline_verified"] and meta.get("workspace_diff"):
                        bucket = valid if "SPECTESTS" not in pol else shaped
                        bucket.setdefault(prefix, []).append((name, pol, sr["seed"], meta["initial_workspace_sha256"], meta["workspace_diff"]))
    return contracts, valid, shaped


def reconstruct(prefix, initial_sha, diff, check_sha=True):
    ws, subtask = fresh_workspace(prefix)
    # SPECTESTS runs snapshot the workspace after installing the contract tests, so their SHA differs.
    if check_sha and bench._hash_snapshot(bench._snapshot_files(ws)) != initial_sha:
        return None, None, "task changed since the run"
    proc = subprocess.run(["patch", "-p1", "--batch", "--silent"], cwd=ws, input=diff, capture_output=True, text=True)
    if proc.returncode:
        return None, None, "patch failed"
    return ws, subtask, ""


# ------------------------------------------------------------------ mutation

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
        ast.And: ast.Or, ast.Or: ast.And}


def mutation_sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sites += [(node, "cmp", i) for i, op in enumerate(node.ops) if type(op) in SWAP]
        elif isinstance(node, (ast.BinOp, ast.BoolOp)) and type(node.op) in SWAP:
            sites.append((node, "op", None))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites += [(node, "inc", None), (node, "dec", None)]
        elif isinstance(node, ast.Raise):
            sites.append((node, "noraise", None))
        elif isinstance(node, ast.If):
            sites.append((node, "negate", None))
    return sites


def mutants(source):
    tree = ast.parse(source)
    count = len(mutation_sites(tree))
    for index in range(count):
        clone = copy.deepcopy(tree)
        node, kind, pos = mutation_sites(clone)[index]
        if kind == "cmp":
            node.ops[pos] = SWAP[type(node.ops[pos])]()
        elif kind == "op":
            node.op = SWAP[type(node.op)]()
        elif kind in ("inc", "dec"):
            node.value += 1 if kind == "inc" else -1
        elif kind == "noraise":
            node.__class__, node._fields = ast.Pass, ()
        elif kind == "negate":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        try:
            yield f"{kind}@{getattr(node, 'lineno', '?')}", ast.unparse(ast.fix_missing_locations(clone))
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "benchmarks/oracle_quality.json"))
    args = parser.parse_args()
    contracts, valid, shaped = collect()
    report = {"over_constraint": {}, "mutation": {}}
    references = dict(REFERENCES, H17=H17_REFERENCE)

    print("== Over-constraint on independent valid implementations ==")
    for prefix in sorted(TASKS, key=lambda p: int(p[1:])):
        seen, rows = set(), []
        for name, pol, seed, sha, diff in valid.get(prefix, []):
            ws, subtask, err = reconstruct(prefix, sha, diff)
            if err:
                continue
            key = hashlib.sha256(json.dumps(code_files(ws), sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            ok_v, _ = run_unittest(ws, visible_modules(subtask))
            if not (ok_v and holdout(prefix, ws)):
                continue  # reconstruction does not reproduce a verified state
            if prefix not in references:
                references[prefix] = code_files(ws)  # first verified implementation serves as reference
            for chash, (cname, src) in contracts.get(prefix, {}).items():
                (ws / "test_spec_contract.py").write_text(src)
                ok_c, failing = run_unittest(ws, ["test_spec_contract"])
                rows.append({"impl": key[:10], "from": f"{name}:{pol}:{seed}", "contract": f"{cname}:{chash}",
                             "passes": ok_c, "failing": failing})
        report["over_constraint"][prefix] = rows
        impls = len({r["impl"] for r in rows})
        bad = [r for r in rows if not r["passes"]]
        print(f"{prefix}: {impls} independent valid implementations x {len(contracts.get(prefix, {}))} contract modules; "
              f"{len(bad)} failing pairs" + (f" -> {sorted({f for r in bad for f in r['failing']})[:4]}" if bad else ""))

    # Tasks no baseline run solved: a holdout-passing SPECTESTS run serves as mutation reference only.
    for prefix, runs in shaped.items():
        for name, pol, seed, sha, diff in runs:
            if prefix in references:
                break
            ws, subtask, err = reconstruct(prefix, sha, diff, check_sha=False)
            if not err and run_unittest(ws, visible_modules(subtask))[0] and holdout(prefix, ws):
                references[prefix] = code_files(ws)
    print("\n== Mutation coverage (mutants of the reference) ==")
    for prefix in sorted(TASKS, key=lambda p: int(p[1:])):
        if prefix not in references:
            print(f"{prefix}: no reference")
            continue
        counts = {"total": 0, "holdout_killed": 0, "visible_of_holdout": 0, "contract_of_holdout": {},
                  "contract_only_not_holdout": {}, "mutants": []}
        for fname, source in references[prefix].items():
            for label, mutated in mutants(source):
                ws, subtask = fresh_workspace(prefix)
                for name, text in references[prefix].items():
                    (ws / name).write_text(text)
                (ws / fname).write_text(mutated)
                counts["total"] += 1
                killed_h = not holdout(prefix, ws)
                killed_v = not run_unittest(ws, visible_modules(subtask))[0]
                counts["holdout_killed"] += killed_h
                counts["visible_of_holdout"] += killed_h and killed_v
                changed = [b for a, b in zip(source.splitlines(), mutated.splitlines()) if a.strip() != b.strip()][:1]
                record = {"file": fname, "mutant": label, "changed": changed, "holdout": killed_h, "visible": killed_v, "contract": {}}
                counts["mutants"].append(record)
                for chash, (cname, src) in contracts.get(prefix, {}).items():
                    (ws / "test_spec_contract.py").write_text(src)
                    killed_c = not run_unittest(ws, ["test_spec_contract"])[0]
                    tag = f"{cname}:{chash}"
                    record["contract"][tag] = killed_c
                    counts["contract_of_holdout"][tag] = counts["contract_of_holdout"].get(tag, 0) + (killed_h and killed_c)
                    counts["contract_only_not_holdout"][tag] = counts["contract_only_not_holdout"].get(tag, 0) + (killed_c and not killed_h)
        report["mutation"][prefix] = counts
        h = counts["holdout_killed"] or 1
        cov = {k: round(v / h, 2) for k, v in counts["contract_of_holdout"].items()}
        print(f"{prefix}: {counts['total']} mutants, holdout kills {counts['holdout_killed']}; of those visible kill "
              f"{counts['visible_of_holdout'] / h:.2f}, contract kill {cov}; contract-only kills {counts['contract_only_not_holdout']}")
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
