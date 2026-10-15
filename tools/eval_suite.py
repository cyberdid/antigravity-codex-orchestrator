"""Baseline eval suite for the R&D engine.

Each goal runs in a fresh workspace through the real CLI (live mode by default).
Ground truth is checked by a separate script that the model never sees, so a run
counts as a pass only when the produced code actually behaves correctly.

Usage:
    .venv/bin/python tools/eval_suite.py [--architect haiku] [--mode live] [--goals E01,E02]
                                         [--out eval_results.json]

Metrics per goal: ground-truth pass, orchestrator verdict, subtasks completed and
failed, wall time, CLI-reported cloud cost, cloud and local token totals.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RUN_TIMEOUT_S = 3600  # per-run wall-clock cap; a run hitting it is recorded as failed

GOALS: Dict[str, Dict[str, str]] = {
    "E01-matrix": {
        "goal": ("Create matrix.py with functions multiply(a, b) for two 2-D lists of numbers "
                 "(matrix product) and transpose(a) returning the transposed 2-D list. "
                 "Add test_matrix.py with unittest tests for both functions."),
        "truth": (
            "import matrix as m\n"
            "assert m.multiply([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]\n"
            "assert m.transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]\n"
            "print('TRUTH OK')\n"
        ),
    },
    "E02-cli-flags": {
        "goal": ("Create cli.py using argparse with --count (int, default 3) and --prefix (string, "
                 "default 'item'). When run it prints count lines of the form '<prefix>-<i>' for i from 1. "
                 "Add test_cli.py with unittest tests that run the script via subprocess."),
        "truth": (
            "import subprocess, sys\n"
            "out = subprocess.run([sys.executable, 'cli.py', '--count', '2', '--prefix', 'x'],"
            " capture_output=True, text=True, check=True).stdout.split()\n"
            "assert out == ['x-1', 'x-2'], out\n"
            "out = subprocess.run([sys.executable, 'cli.py'], capture_output=True, text=True, check=True).stdout.split()\n"
            "assert out == ['item-1', 'item-2', 'item-3'], out\n"
            "print('TRUTH OK')\n"
        ),
    },
    "E03-lru-cache": {
        "goal": ("Create lru_cache.py with class LRUCache(capacity) providing get(key) (returns None on miss) "
                 "and put(key, value). Evict the least recently used entry when over capacity, and "
                 "treat get as a use. Add test_lru_cache.py with unittest tests for eviction order."),
        "truth": (
            "from lru_cache import LRUCache\n"
            "c = LRUCache(2)\n"
            "c.put('a', 1); c.put('b', 2)\n"
            "assert c.get('a') == 1\n"
            "c.put('c', 3)\n"
            "assert c.get('b') is None, 'b should be evicted'\n"
            "assert c.get('a') == 1 and c.get('c') == 3\n"
            "print('TRUTH OK')\n"
        ),
    },
    "E04-token-verify": {
        "goal": ("Create tokensign.py with sign(payload: str, secret: str) -> str returning "
                 "payload + '.' + hex HMAC-SHA256 of payload keyed by secret, and verify(token: str, secret: str) "
                 "-> bool that returns True only for an untampered token signed with that secret, using "
                 "hmac.compare_digest. Add test_token.py with unittest tests for valid, tampered and wrong-secret tokens."),
        "truth": (
            "import tokensign as t\n"
            "tok = t.sign('user=ada', 'k1')\n"
            "assert t.verify(tok, 'k1') is True\n"
            "assert t.verify(tok, 'k2') is False\n"
            "assert t.verify('user=root.' + tok.split('.', 1)[1], 'k1') is False\n"
            "print('TRUTH OK')\n"
        ),
    },
    "E05-sqlite-repo": {
        "goal": ("Create repo.py defining class Repo(path). It stores users in a SQLite table users(id INTEGER "
                 "PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL) created on first use. Methods: "
                 "add(name, email) -> int id; get(id) -> dict with keys id, name, email, or None; "
                 "list_all() -> list of such dicts ordered by id; delete(id) -> bool (True if a row was removed); "
                 "and transaction(), a context manager that commits on success and rolls back on any exception. "
                 "Add test_repo.py with unittest tests covering add/get, delete, and rollback."),
        "truth": (
            "import os, tempfile\n"
            "from repo import Repo\n"
            "d = tempfile.mkdtemp(); p = os.path.join(d, 'r.db')\n"
            "r = Repo(p)\n"
            "a = r.add('ada', 'ada@x.io'); b = r.add('bob', 'bob@x.io')\n"
            "assert r.get(a) == {'id': a, 'name': 'ada', 'email': 'ada@x.io'}, r.get(a)\n"
            "assert [u['name'] for u in r.list_all()] == ['ada', 'bob']\n"
            "assert r.delete(b) is True and r.delete(b) is False\n"
            "assert r.get(b) is None\n"
            "try:\n"
            "    with r.transaction():\n"
            "        r.add('eve', 'eve@x.io')\n"
            "        raise RuntimeError('boom')\n"
            "except RuntimeError:\n"
            "    pass\n"
            "assert [u['name'] for u in r.list_all()] == ['ada'], 'rollback failed'\n"
            "print('TRUTH OK')\n"
        ),
    },
    "E06-inventory-cli": {
        "goal": ("Create inventory.py, a command-line tool with a --db PATH option (default inventory.json). "
                 "Subcommands: 'add NAME QTY' stores the item (QTY is an int; adding an existing name adds to its "
                 "quantity), 'remove NAME' deletes the item and exits with code 1 if it does not exist, and "
                 "'list' prints one line per item as 'NAME QTY' sorted by name. The data is stored as JSON in the "
                 "db file. Add test_inventory.py with unittest tests that run the tool through subprocess with a "
                 "temporary --db path."),
        "truth": (
            "import subprocess, sys, tempfile, os, json\n"
            "d = tempfile.mkdtemp(); db = os.path.join(d, 'inv.json')\n"
            "def run(*a):\n"
            "    return subprocess.run([sys.executable, 'inventory.py', '--db', db, *a], capture_output=True, text=True)\n"
            "assert run('add', 'widget', '3').returncode == 0\n"
            "run('add', 'bolt', '5'); run('add', 'widget', '2')\n"
            "out = run('list').stdout.split('\\n')\n"
            "assert [l.split() for l in out if l.strip()] == [['bolt', '5'], ['widget', '5']], out\n"
            "assert run('remove', 'nope').returncode == 1\n"
            "assert run('remove', 'bolt').returncode == 0\n"
            "assert [l.split() for l in run('list').stdout.split('\\n') if l.strip()] == [['widget', '5']]\n"
            "assert json.load(open(db)) is not None\n"
            "print('TRUTH OK')\n"
        ),
    },
}


def run_goal(name: str, spec: Dict[str, str], architect: str, mode: str, root: Path) -> Dict[str, object]:
    workspace = Path(tempfile.mkdtemp(prefix=f"eval_{name}_"))
    started = time.time()
    cmd = [
        str(ROOT / "run.sh"), "--goal", spec["goal"], "--workspace", str(workspace),
        "--mode", mode, "--no-hud", "--architect", architect,
    ]
    env = os.environ.copy()
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=RUN_TIMEOUT_S, env=env)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        # A stuck run is a failed run, not a reason to abandon the whole suite.
        timed_out = True
        exit_code = -1
    wall = time.time() - started

    truth_path = workspace / "_ground_truth_check.py"
    truth_path.write_text(spec["truth"], encoding="utf-8")
    truth = subprocess.run([sys.executable, str(truth_path)], cwd=str(workspace),
                           capture_output=True, text=True, timeout=120)
    truth_path.unlink(missing_ok=True)

    record = _last_run_record(workspace)
    return {
        "goal": name,
        "architect": architect,
        "mode": mode,
        "ground_truth_pass": truth.returncode == 0,
        "ground_truth_output": (truth.stdout + truth.stderr).strip()[-300:],
        "orchestrator_exit": exit_code,
        "timed_out": timed_out,
        "orchestrator_success": record.get("success"),
        "completed": record.get("completed", []),
        "failed": record.get("failed", []),
        "wall_s": round(wall, 1),
        "cloud_cost_usd": record.get("cloud_cost_usd"),
        "cloud_tokens": record.get("cloud_tokens"),
        "local_tokens": record.get("local_tokens"),
        "workspace": str(workspace),
    }


def _last_run_record(workspace: Path) -> Dict[str, object]:
    """Reads the most recent run JSONL in the workspace: footer verdict + last telemetry snapshot."""
    files = sorted(glob.glob(str(workspace / ".antigravity_runs" / "run-*.jsonl")))
    if not files:
        return {}
    footer, telemetry = {}, {}
    with open(files[-1], encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "footer":
                footer = rec
            elif rec.get("kind") == "telemetry":
                telemetry = rec.get("summary", {}) or {}
    return {
        "success": footer.get("success"),
        "completed": footer.get("completed", []),
        "failed": footer.get("failed", []),
        "cloud_cost_usd": telemetry.get("cloud_cost_usd"),
        "cloud_tokens": telemetry.get("cloud_tokens"),
        "local_tokens": telemetry.get("local_tokens"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architect", default="haiku", choices=["haiku", "luna", "agy"])
    parser.add_argument("--mode", default="live", choices=["live", "mock"])
    parser.add_argument("--goals", default="", help="comma-separated goal ids (default: all)")
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--repeat", type=int, default=1, help="independent runs per goal (variance)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="load --out if it exists and skip (goal, repeat) pairs already recorded")
    args = parser.parse_args(argv)

    selected = [g.strip() for g in args.goals.split(",") if g.strip()] or list(GOALS)
    unknown = [g for g in selected if g not in GOALS]
    if unknown:
        parser.error(f"unknown goals: {unknown}; valid: {list(GOALS)}")

    results = []
    if args.skip_existing and Path(args.out).exists():
        results = json.loads(Path(args.out).read_text(encoding="utf-8"))
    done = {(r["goal"], r.get("repeat")) for r in results}
    for repeat in range(1, max(1, args.repeat) + 1):
        for name in selected:
            if (name, repeat) in done:
                print(f"== {name} run {repeat}/{args.repeat}: already recorded, skipping", flush=True)
                continue
            print(f"== {name} run {repeat}/{args.repeat} ({args.architect}, {args.mode})", flush=True)
            result = run_goal(name, GOALS[name], args.architect, args.mode, ROOT)
            result["repeat"] = repeat
            results.append(result)
            print(f"   truth={'PASS' if result['ground_truth_pass'] else 'FAIL'} "
                  f"orchestrator={'PASS' if result['orchestrator_success'] else 'FAIL'} "
                  f"completed={len(result['completed'])} failed={len(result['failed'])} "
                  f"wall={result['wall_s']}s cost={result['cloud_cost_usd']}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nper goal (ground truth pass / runs, mean wall, mean cost, fake successes):")
    for name in selected:
        rs = [r for r in results if r["goal"] == name]
        ok = sum(1 for r in rs if r["ground_truth_pass"])
        fake = sum(1 for r in rs if r["orchestrator_success"] and not r["ground_truth_pass"])
        wall = sum(r["wall_s"] for r in rs) / len(rs)
        costs = [r["cloud_cost_usd"] for r in rs if r["cloud_cost_usd"] is not None]
        cost = sum(costs) / len(costs) if costs else float("nan")
        print(f"  {name:20} {ok}/{len(rs)}  wall={wall:.0f}s  cost={cost:.4f}  fake={fake}")
    passed = sum(1 for r in results if r["ground_truth_pass"])
    print(f"\nground-truth pass rate: {passed}/{len(results)} -> {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
