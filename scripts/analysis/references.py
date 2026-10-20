"""Reference (correct) implementations of the benchmark tasks, as full module files.

Used only for offline grading of generated contract tests; never shown to the worker
or to the cloud.
"""

REFERENCES = {
    "H1": {"fdr_control.py": '''from typing import Sequence
from fdr_types import FdrDecision

def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> FdrDecision:
    m = len(p_values)
    if m == 0:
        return FdrDecision(rejected_mask=[], num_rejected=0, critical_threshold=0.0)
    order = sorted(range(m), key=lambda i: p_values[i])
    k_max = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= rank / m * alpha:
            k_max = rank
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= k_max:
            rejected[idx] = True
    crit = k_max / m * alpha if k_max else 0.0
    return FdrDecision(rejected_mask=rejected, num_rejected=k_max, critical_threshold=round(crit, 10))
'''},
    "H2": {"pareto_judge.py": '''from typing import List, Sequence
from candidate_spec import HarnessCandidate

def dominates(a: HarnessCandidate, b: HarnessCandidate) -> bool:
    ka = (a.wall_clock_s, a.total_tokens, a.tool_errors)
    kb = (b.wall_clock_s, b.total_tokens, b.tool_errors)
    return all(x <= y for x, y in zip(ka, kb)) and any(x < y for x, y in zip(ka, kb))

def select_pareto_frontier(candidates: Sequence[HarnessCandidate]) -> List[str]:
    pool = [c for c in candidates if c.rie_accepted]
    frontier = [c for c in pool if not any(dominates(o, c) for o in pool if o.candidate_id != c.candidate_id)]
    frontier.sort(key=lambda c: (c.wall_clock_s, c.total_tokens, c.tool_errors, c.candidate_id))
    return [c.candidate_id for c in frontier]
'''},
    "H3": {"group_splitter.py": '''from typing import List, Sequence, Tuple
from sample_record import SampleRecord

def group_kfold_splits(records: Sequence[SampleRecord], n_splits: int = 3) -> List[Tuple[List[int], List[int]]]:
    if n_splits < 2:
        raise ValueError('n_splits must be >= 2')
    groups = list(dict.fromkeys(r.group_id for r in records))
    if len(groups) < n_splits:
        raise ValueError('not enough groups')
    fold_of = {g: i % n_splits for i, g in enumerate(groups)}
    folds = []
    for k in range(n_splits):
        test_ids = [r.sample_id for r in records if fold_of[r.group_id] == k]
        train_ids = [r.sample_id for r in records if fold_of[r.group_id] != k]
        folds.append((train_ids, test_ids))
    return folds
'''},
    "H4": {"weights.py": '''from typing import List, Sequence

def normalize(weights: Sequence[float]) -> List[float]:
    if not weights or any(w < 0 for w in weights):
        raise ValueError("bad weights")
    total = sum(weights)
    if total == 0:
        raise ValueError("zero total")
    return [w / total for w in weights]
''', "summary.py": '''from typing import Sequence
from weights import normalize

def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("length mismatch")
    w = normalize(weights)
    return sum(v * wi for v, wi in zip(values, w))
'''},
    "H5": {"intervals.py": '''from typing import List, Sequence, Tuple

def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if any(s > e for s, e in intervals):
        raise ValueError('start > end')
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result
'''},
    "H6": {"durations.py": '''import re

UNITS = {'h': 3600, 'm': 60, 's': 1}

def parse_duration(text: str) -> int:
    m = re.fullmatch(r'(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?', text.strip())
    if not text.strip() or not m:
        raise ValueError(text)
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s
'''},
    "H7": {"rolling.py": '''from typing import List, Sequence

def rolling_mean(values: Sequence[float], window: int) -> List[float]:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('window')
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]
'''},
    "H8": {"slugs.py": '''import re

def slugify(title: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '-', title).strip('-').lower()
    if not slug:
        raise ValueError(title)
    return slug
'''},
    "H9": {"semver.py": '''import re

def parse_semver(text: str):
    m = re.fullmatch(r'(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?', text)
    if not m:
        raise ValueError(text)
    return int(m[1]), int(m[2]), int(m[3]), m[4]
'''},
    "H10": {"ipv4.py": '''import re

def parse_ipv4(text: str):
    parts = text.split('.')
    if len(parts) != 4:
        raise ValueError(text)
    out = []
    for p in parts:
        if not re.fullmatch(r'0|[1-9][0-9]{0,2}', p) or int(p) > 255:
            raise ValueError(text)
        out.append(int(p))
    return tuple(out)
'''},
    "H11": {"flags.py": '''def parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in ('true', 'yes', 'on', '1'):
        return True
    if t in ('false', 'no', 'off', '0'):
        return False
    raise ValueError(text)
'''},
    "H12": {"colors.py": '''import re

def parse_hex_color(text: str) -> tuple:
    if not re.fullmatch(r'#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})', text):
        raise ValueError(text)
    h = text[1:]
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
'''},
    "H13": {"ranges.py": '''import re

def parse_range(text: str) -> list:
    m = re.fullmatch(r'([0-9]+)(?:-([0-9]+))?', text)
    if not m:
        raise ValueError(text)
    a = int(m[1])
    b = int(m[2]) if m[2] is not None else a
    if a > b:
        raise ValueError(text)
    return list(range(a, b + 1))
'''},
}
