"""Run ORION-3 Experiment I: every labelled artifact x arms A0-A3 x repetitions.

Usage: python scripts/orion3/run_experiment.py [--reps 3] [--out benchmarks/orion3/exp1_results.json] [--only ID...]
Results are saved after every evaluation; rerunning the command resumes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import arms  # noqa: E402


def digest_tree(path):
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default=str(ROOT / 'benchmarks/orion3/corpus'))
    parser.add_argument('--labels', default=str(ROOT / 'benchmarks/orion3/labels.json'))
    parser.add_argument('--out', default=str(ROOT / 'benchmarks/orion3/exp1_results.json'))
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    corpus, out = Path(args.corpus), Path(args.out)
    labels = json.loads(Path(args.labels).read_text())
    header = {
        'git_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        'git_dirty': bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        'corpus_sha256': digest_tree(corpus), 'labels_sha256': hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
        'reps': args.reps, 'arms': list(arms.ARMS), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    data = json.loads(out.read_text()) if out.exists() else {'header': header, 'evaluations': []}
    if data['header']['corpus_sha256'] != header['corpus_sha256'] or data['header']['labels_sha256'] != header['labels_sha256']:
        sys.exit('corpus or labels changed since this results file was started; refusing to mix')
    done = {(e['artifact'], e['arm'], e['rep']) for e in data['evaluations']}
    artifacts = sorted(a for a in labels if not args.only or a in args.only)
    for rep in range(1, args.reps + 1):
        for ident in artifacts:
            meta = json.loads((corpus / ident / 'meta.json').read_text())
            for arm in arms.ARMS:
                if (ident, arm, rep) in done:
                    continue
                result = arms.evaluate(arm, corpus / ident, meta)
                data['evaluations'].append(dict(result, artifact=ident, arm=arm, rep=rep, label=labels[ident]['label'],
                                                stratum=labels[ident]['stratum'], kind=meta['kind'], cls=meta.get('cls')))
                out.write_text(json.dumps(data, indent=1))
                print(f"rep {rep} {ident:32s} {arm}: {'ACCEPT' if result['accepted'] else 'reject'} "
                      f"[{result['status']}] {result['wall_s']}s  (truth: {labels[ident]['label']})", flush=True)
    data['header']['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    out.write_text(json.dumps(data, indent=1))
    print('DONE')


if __name__ == '__main__':
    main()
