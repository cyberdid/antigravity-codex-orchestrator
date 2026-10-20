"""Natural stratum: single-shot implementations by the local worker model from the public spec.

Usage: python scripts/orion3/generate_natural.py [--metrics 20] [--retrieval 10]
Writes benchmarks/orion3/corpus/N{M,R}{seed}_gemma/ with meta.json (raw response kept).
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = 'gemma4:12b'
PROMPT = """You are implementing part of a Python package from its public specification.

{spec}

Write the complete contents of the file `src/orion/{module}.py`, implementing {what} exactly as specified.
Reply with a single ```python code block containing the whole file and nothing else."""


def generate(prompt, seed):
    body = json.dumps({'model': MODEL, 'prompt': prompt, 'stream': False, 'think': False,
                       'options': {'temperature': 0.8, 'seed': seed, 'num_predict': 6144, 'num_ctx': 8192}}).encode()
    req = urllib.request.Request('http://localhost:11434/api/generate', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def extract(text):
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    return max(blocks, key=len) if blocks else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=int, default=20)
    parser.add_argument('--retrieval', type=int, default=10)
    parser.add_argument('--out', default=str(ROOT / 'benchmarks/orion3/corpus'))
    args = parser.parse_args()
    spec = (HERE / 'PUBLIC_SPEC.md').read_text()
    jobs = [('metrics', 'NM', '`calculate_metrics`', s) for s in range(1, args.metrics + 1)]
    jobs += [('retrieval', 'NR', '`exact_knn` and `approximate_knn`', s) for s in range(1, args.retrieval + 1)]
    for module, prefix, what, seed in jobs:
        ident = f'{prefix}{seed:02d}_gemma'
        base = Path(args.out) / ident
        if (base / 'meta.json').exists():
            continue
        start = time.time()
        resp = generate(PROMPT.format(spec=spec, module=module, what=what), seed)
        code = extract(resp['response'])
        pkg = base / 'workspace/src/orion'
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / '__init__.py').write_text('')
        (pkg / f'{module}.py').write_text(code)
        meta = {'id': ident, 'stratum': 'natural', 'kind': module, 'cls': 'natural', 'intended_label': None,
                'description': f'{MODEL} single shot, temperature 0.8, seed {seed}', 'model': MODEL, 'seed': seed,
                'eval_count': resp.get('eval_count'), 'done_reason': resp.get('done_reason'),
                'wall_s': round(time.time() - start, 1), 'raw_response': resp['response']}
        (base / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(ident, meta['done_reason'], meta['eval_count'], meta['wall_s'], 's', flush=True)


if __name__ == '__main__':
    main()
