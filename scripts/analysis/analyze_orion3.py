"""Analyse ORION-3 Experiment I results with the pre-registered endpoints.

Usage: python scripts/analysis/analyze_orion3.py [benchmarks/orion3/exp1_results.json]
"""
import json
import math
import sys
from collections import defaultdict

ARMS = ('A0', 'A1', 'A2', 'A3')


def wilson(k, n, z=1.96):
    if not n:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def exact_mcnemar(b, c):
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(path):
    data = json.load(open(path))
    print({k: data['header'].get(k) for k in ('git_commit', 'git_dirty', 'corpus_sha256', 'reps', 'started_utc', 'finished_utc')})
    ev = defaultdict(dict)  # (artifact, arm) -> {rep: accepted}
    info, wall = {}, defaultdict(list)
    for e in data['evaluations']:
        ev[e['artifact'], e['arm']][e['rep']] = e['accepted']
        info[e['artifact']] = e
        wall[e['arm']].append(e['wall_s'])
    labelled = {a: i for a, i in info.items() if i['label'] in ('valid', 'invalid')}
    ambiguous = sorted(a for a, i in info.items() if i['label'] not in ('valid', 'invalid'))
    invalid = sorted(a for a, i in labelled.items() if i['label'] == 'invalid')
    valid = sorted(a for a, i in labelled.items() if i['label'] == 'valid')
    print(f'\nartifacts: {len(info)} ({len(invalid)} invalid, {len(valid)} valid, {len(ambiguous)} ambiguous excluded: {ambiguous})')

    def first(a, arm):
        return ev[a, arm].get(1)

    def anyrep(a, arm):
        return any(ev[a, arm].values())

    print('\nArm | FSAR rep1 (95% CI) | FSAR any-of-reps | FRR rep1 (95% CI) | natural FSAR | designed FSAR | mean wall s')
    for arm in ARMS:
        fa = sum(bool(first(a, arm)) for a in invalid)
        fa_any = sum(anyrep(a, arm) for a in invalid)
        fr = sum(not first(a, arm) for a in valid)
        nat = [a for a in invalid if labelled[a]['stratum'] == 'natural']
        des = [a for a in invalid if labelled[a]['stratum'] == 'designed']
        lo, hi = wilson(fa, len(invalid))
        flo, fhi = wilson(fr, len(valid))
        print(f'{arm} | {fa}/{len(invalid)} = {fa / max(1, len(invalid)):.3f} ({lo:.2f}-{hi:.2f}) | {fa_any}/{len(invalid)} | '
              f'{fr}/{len(valid)} ({flo:.2f}-{fhi:.2f}) | {sum(bool(first(a, arm)) for a in nat)}/{len(nat)} | '
              f'{sum(bool(first(a, arm)) for a in des)}/{len(des)} | {sum(wall[arm]) / max(1, len(wall[arm])):.2f}')

    def compare(x, y, pool):
        # b: x accepts, y rejects (y better); c: y accepts, x rejects
        b = sum(bool(first(a, x)) and not first(a, y) for a in pool)
        c = sum(bool(first(a, y)) and not first(a, x) for a in pool)
        return b, c, exact_mcnemar(b, c)

    b, c, p = compare('A1', 'A3', invalid)
    print(f'\nPRIMARY  FSAR A3 vs A1 (invalid artifacts, rep 1): A1-only accepts {b}, A3-only accepts {c}, exact McNemar p = {p:.6f}')
    family = [(f'{x} vs {y}', *compare(x, y, invalid)) for x, y in (('A2', 'A3'), ('A1', 'A2'), ('A0', 'A1'))]
    order = sorted(range(len(family)), key=lambda i: family[i][3])
    holm, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(family) - rank) * family[i][3]))
        holm[i] = running
    print('Secondary FSAR comparisons (first arm accepts only / second arm accepts only, raw p, Holm p):')
    for i, (name, b, c, p) in enumerate(family):
        print(f'  {name}: {b} / {c}, p = {p:.4f}, Holm p = {holm[i]:.4f}')

    print('\nConsistency across repetitions (share of artifact x arm with identical decisions):')
    for arm in ARMS:
        keys = [a for a in labelled if len(ev[a, arm]) > 1]
        same = sum(len(set(ev[a, arm].values())) == 1 for a in keys)
        flips = sorted(a for a in keys if len(set(ev[a, arm].values())) > 1)
        print(f'  {arm}: {same}/{len(keys)}  flipping: {flips}')

    print('\nPer artifact (A=accepted on rep1, a=accepted on a later rep only, .=rejected on all reps):')
    print(f"{'artifact':34s} {'truth':8s} " + ' '.join(ARMS))
    for a in sorted(info):
        marks = []
        for arm in ARMS:
            marks.append('A ' if first(a, arm) else ('a ' if anyrep(a, arm) else '. '))
        print(f"{a:34s} {info[a]['label']:8s} " + ' '.join(marks) + f"  {info[a].get('cls') or ''}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'benchmarks/orion3/exp1_results.json')
