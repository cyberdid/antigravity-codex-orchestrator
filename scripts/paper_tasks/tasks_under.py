"""Under-specified rule tasks P19-P25.

The docstrings are deliberately brief and leave behaviour unstated (invalid input,
edge conventions). Rules and holdout check only what the docstring states, so these
tasks measure whether contract tests over-constrain an incomplete specification.
"""
from framework import Rule, RuleTask

P19 = RuleTask(
    id="P19_dedupe", title="Fix dedupe", category="underspecified", affected=["seqs.py"],
    imports="from seqs import dedupe",
    files={"seqs.py": '''"""Sequence helpers."""


def dedupe(items: list) -> list:
    """Remove duplicates, keeping the first occurrence of each item in its original position."""
    return list(set(items))
'''},
    reference={"seqs.py": '''"""Sequence helpers."""


def dedupe(items: list) -> list:
    """Remove duplicates, keeping the first occurrence of each item in its original position."""
    seen, result = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
'''},
    rules=[
        Rule("order_kept", [("dedupe([3, 1, 3, 2, 1])", [3, 1, 2]), ("dedupe(['b', 'a', 'b'])", ["b", "a"])]),
        Rule("no_duplicates", [("dedupe([1, 2, 3])", [1, 2, 3])]),
        Rule("empty", [("dedupe([])", [])]),
    ])

P20 = RuleTask(
    id="P20_median", title="Fix median", category="underspecified", affected=["middle.py"],
    imports="from middle import median",
    files={"middle.py": '''"""Order statistics."""


def median(values: list) -> float:
    """Median of a non-empty list of numbers; the list passed in is left unchanged."""
    values.sort()
    return values[len(values) // 2]
'''},
    reference={"middle.py": '''"""Order statistics."""


def median(values: list) -> float:
    """Median of a non-empty list of numbers; the list passed in is left unchanged."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
'''},
    rules=[
        Rule("odd_length", [("median([3, 1, 2])", 2), ("median([7])", 7)]),
        Rule("even_length", [("median([4, 1, 3, 2])", 2.5), ("median([1, 3])", 2.0)]),
        Rule("input_unchanged", [("(lambda v: (median(v), v))([3, 1, 2])", (2, [3, 1, 2]))]),
    ])

P21 = RuleTask(
    id="P21_capitalize_words", title="Fix capitalize_words", category="underspecified", affected=["words.py"],
    imports="from words import capitalize_words",
    files={"words.py": '''"""Word casing."""


def capitalize_words(text: str) -> str:
    """Upper-case the first letter of every space-separated word; other characters are unchanged."""
    return text[:1].upper() + text[1:]
'''},
    reference={"words.py": '''"""Word casing."""


def capitalize_words(text: str) -> str:
    """Upper-case the first letter of every space-separated word; other characters are unchanged."""
    return " ".join(word[:1].upper() + word[1:] for word in text.split(" "))
'''},
    rules=[
        Rule("every_word", [("capitalize_words('hello world')", "Hello World"), ("capitalize_words('a b c')", "A B C")]),
        Rule("rest_unchanged", [("capitalize_words('mcDonald iPhone')", "McDonald IPhone")]),
        Rule("empty", [("capitalize_words('')", "")]),
    ])

P22 = RuleTask(
    id="P22_clamp", title="Fix clamp", category="underspecified", affected=["bounds.py"],
    imports="from bounds import clamp",
    files={"bounds.py": '''"""Bounds."""


def clamp(x: float, lo: float, hi: float) -> float:
    """Limit x to the closed range [lo, hi]."""
    return max(hi, min(lo, x))
'''},
    reference={"bounds.py": '''"""Bounds."""


def clamp(x: float, lo: float, hi: float) -> float:
    """Limit x to the closed range [lo, hi]."""
    return min(hi, max(lo, x))
'''},
    rules=[
        Rule("inside", [("clamp(5, 0, 10)", 5)]),
        Rule("below", [("clamp(-3, 0, 10)", 0)]),
        Rule("above", [("clamp(42, 0, 10)", 10)]),
        Rule("boundaries", [("clamp(0, 0, 10)", 0), ("clamp(10, 0, 10)", 10)]),
    ])

P23 = RuleTask(
    id="P23_flatten_once", title="Fix flatten_once", category="underspecified", affected=["nesting.py"],
    imports="from nesting import flatten_once",
    files={"nesting.py": '''"""Nested lists."""


def flatten_once(nested: list) -> list:
    """Flatten a list of lists by exactly one level."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_once(item))
        else:
            result.append(item)
    return result
'''},
    reference={"nesting.py": '''"""Nested lists."""


def flatten_once(nested: list) -> list:
    """Flatten a list of lists by exactly one level."""
    return [item for inner in nested for item in inner]
'''},
    rules=[
        Rule("one_level", [("flatten_once([[1, 2], [3]])", [1, 2, 3])]),
        Rule("keeps_deeper_lists", [("flatten_once([[1, [2, 3]], [[4]]])", [1, [2, 3], [4]])]),
        Rule("empty", [("flatten_once([])", []), ("flatten_once([[], []])", [])]),
    ])

P24 = RuleTask(
    id="P24_word_count", title="Fix word_count", category="underspecified", affected=["counting.py"],
    imports="from counting import word_count",
    files={"counting.py": '''"""Text statistics."""


def word_count(text: str) -> int:
    """Number of words in `text`, where words are separated by whitespace."""
    return len(text.split(" "))
'''},
    reference={"counting.py": '''"""Text statistics."""


def word_count(text: str) -> int:
    """Number of words in `text`, where words are separated by whitespace."""
    return len(text.split())
'''},
    rules=[
        Rule("basic", [("word_count('one two three')", 3)]),
        Rule("whitespace_runs", [("word_count('  one   two ')", 2), ("word_count('a\\tb\\nc')", 3)]),
        Rule("empty", [("word_count('')", 0), ("word_count('   ')", 0)]),
    ])

P25 = RuleTask(
    id="P25_safe_div", title="Fix safe_div", category="underspecified", affected=["arith.py"],
    imports="from arith import safe_div",
    files={"arith.py": '''"""Arithmetic helpers."""


def safe_div(a: float, b: float, default=None):
    """a / b, or `default` when b is zero."""
    try:
        return a // b
    except Exception:
        return 0
'''},
    reference={"arith.py": '''"""Arithmetic helpers."""


def safe_div(a: float, b: float, default=None):
    """a / b, or `default` when b is zero."""
    if b == 0:
        return default
    return a / b
'''},
    rules=[
        Rule("true_division", [("safe_div(7, 2)", 3.5), ("safe_div(6, 3)", 2.0)]),
        Rule("zero_default_none", [("safe_div(1, 0)", None)]),
        Rule("zero_custom_default", [("safe_div(1, 0, default=5)", 5), ("safe_div(1, 0.0, 'x')", "x")]),
    ])

TASKS = [P19, P20, P21, P22, P23, P24, P25]
