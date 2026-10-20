"""Campaign B: repair tasks whose bugs were written by the local model, not by the author.

For each Campaign A specification the first file the agent may edit is reduced to a
skeleton (signatures and docstrings, bodies replaced by `...`); every other file stays
in its reference state. gemma4:12b writes the module once per generation seed
(temperature 0.8, no tests shown). The original module and function docstrings are restored
in the generated code (the bug must be in the bodies, not in the specification); a generation that
does not parse or misses a specified function is rejected. The first remaining generation that
violates at least one rule becomes the buggy file of task N<k>; specifications for which every seed is correct are
excluded and counted. Visible rules are chosen by the same seeded coin flip as Campaign A.

Generate and freeze:  python3 scripts/paper_tasks/natural.py   (needs Ollama; writes natural_tasks.json)
"""
from __future__ import annotations

import ast
import copy
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import framework as F  # noqa: E402
import registry  # noqa: E402

MODEL = "gemma4:12b"
GEN_SEEDS = (1, 2, 3, 4, 5)
OUT = HERE / "natural_tasks.json"
PROMPT = """You are implementing one module of a small Python package from its specification.

Module to write: `{target}` (signatures and docstrings below; the bodies are missing).

```python
{skeleton}
```
{others}
Write the complete contents of `{target}` so that every function does exactly what its docstring says.
Reply with a single ```python code block containing the whole file and nothing else."""


def skeleton(source: str) -> str:
    """Keep imports, constants, classes, signatures and docstrings; replace function bodies by `...`."""
    tree = ast.parse(source)

    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            doc = ast.get_docstring(node)
            node.body = ([ast.Expr(ast.Constant(doc))] if doc else []) + [ast.Expr(ast.Constant(Ellipsis))]
            return node

    return ast.unparse(Strip().visit(copy.deepcopy(tree)))


def _functions(tree) -> dict:
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            found.update({f"{node.name}.{item.name}": item for item in node.body
                          if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))})
    return found


def restore_spec(generated: str, original: str):
    """Put the original module and function docstrings back; None if a specified function is missing.

    The bug must live in function bodies, never in the specification the agent and the cloud read.
    """
    try:
        gen_tree, orig_tree = ast.parse(generated), ast.parse(original)
    except SyntaxError:
        return None
    gen_funcs, orig_funcs = _functions(gen_tree), _functions(orig_tree)
    if set(orig_funcs) - set(gen_funcs):
        return None
    for name, orig in orig_funcs.items():
        doc, node = ast.get_docstring(orig), gen_funcs[name]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
        if doc:
            node.body.insert(0, ast.Expr(ast.Constant(doc)))
    module_doc = ast.get_docstring(orig_tree)
    if ast.get_docstring(gen_tree) is not None:
        gen_tree.body = gen_tree.body[1:]
    if module_doc:
        gen_tree.body.insert(0, ast.Expr(ast.Constant(module_doc)))
    return ast.unparse(ast.fix_missing_locations(gen_tree)) + "\n"


def generate(prompt: str, seed: int) -> dict:
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0.8, "seed": seed, "num_predict": 4096, "num_ctx": 8192}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def extract(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def build_prompt(task: F.RuleTask, target: str) -> str:
    fixed = {**task.files, **task.reference}
    others = "".join(f"\nRead-only module `{name}` (already correct):\n```python\n{text}```\n"
                     for name, text in fixed.items() if name != target)
    return PROMPT.format(target=target, skeleton=skeleton(fixed[target]), others=others)


def natural_task(task: F.RuleTask, index: int, generated: str, target: str) -> F.RuleTask:
    fixed = {**task.files, **task.reference}
    return F.RuleTask(id=f"N{index}_{task.id.split('_', 1)[1]}", title=task.title, category=task.category,
                      files={**fixed, target: generated}, reference={target: fixed[target]},
                      affected=[target], rules=task.rules, imports=task.imports,
                      notes=f"natural bug from {MODEL}")


def main():
    records = []
    for index, task in enumerate(registry.ALL, start=1):
        target = task.affected[0]
        prompt = build_prompt(task, target)
        chosen = None
        attempts = []
        for seed in GEN_SEEDS:
            start = time.time()
            reply = generate(prompt, seed)
            code = restore_spec(extract(reply["response"]), {**task.files, **task.reference}[target])
            meta = {"seed": seed, "wall_s": round(time.time() - start, 1), "done_reason": reply.get("done_reason"),
                    "eval_count": reply.get("eval_count")}
            if code is None:  # unparsable or a specified function missing: not a usable task, next seed
                attempts.append(dict(meta, rejected="unparsable or incomplete"))
                print(f"{task.id} seed {seed}: rejected (unparsable or incomplete)", flush=True)
                continue
            fixed = {**task.files, **task.reference, target: code}
            failing = [r.name for r in task.rules if not F.rule_passes(fixed, task.imports, r)]
            attempts.append(dict(meta, failing_rules=failing))
            print(f"{task.id} seed {seed}: {len(failing)} failing rules", flush=True)
            if failing:
                chosen = (seed, code)
                break
        record = {"spec": task.id, "target": target, "attempts": attempts}
        if chosen:
            ntask = natural_task(task, index, chosen[1], target)
            ntask.choose_visible()
            record.update(task_id=ntask.id, generation_seed=chosen[0], buggy_source=chosen[1], visible_rules=ntask.visible)
        records.append(record)
    OUT.write_text(json.dumps({"model": MODEL, "generation_seeds": GEN_SEEDS, "records": records}, indent=1) + "\n")
    kept = [r for r in records if "task_id" in r]
    print(f"kept {len(kept)} natural tasks; {len(records) - len(kept)} specifications implemented correctly on every seed")


def load_tasks() -> list:
    """Frozen Campaign B tasks as RuleTask objects (visible rules from the frozen file)."""
    data = json.loads(OUT.read_text())
    by_id = {t.id: (i, t) for i, t in enumerate(registry.ALL, start=1)}
    tasks = []
    for record in data["records"]:
        if "task_id" not in record:
            continue
        index, spec = by_id[record["spec"]]
        ntask = natural_task(spec, index, record["buggy_source"], record["target"])
        ntask.visible = record["visible_rules"]
        tasks.append(ntask)
    return tasks


if __name__ == "__main__":
    main()
