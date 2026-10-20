"""Multi-file rule tasks P9-P18: the rules or the defects span two modules."""
from framework import RAISES, Rule, RuleTask

P9 = RuleTask(
    id="P9_order_total", title="Fix order_total", category="multi-file", affected=["pricing.py"],
    imports="from models import Item\nfrom pricing import order_total",
    files={
        "models.py": '''"""Order data."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """One order line; prices are integer cents."""
    sku: str
    qty: int
    unit_price_cents: int
''',
        "pricing.py": '''"""Order pricing."""
from models import Item


def order_total(items: list, discount_percent: int = 0) -> int:
    """Total of `items` in cents after a percentage discount.

    Rules:
    1. The subtotal is the sum of qty * unit_price_cents; an empty order totals 0.
    2. The discount is discount_percent of the subtotal, rounded down to whole cents, and is
       subtracted from the subtotal.
    3. discount_percent is an int from 0 to 100; raise ValueError otherwise.
    4. Raise ValueError if any item has qty < 1 or unit_price_cents < 0, or if two items share a sku.
    """
    subtotal = sum(item.qty * item.unit_price_cents for item in items)
    return round(subtotal * (1 - discount_percent / 100))
''',
    },
    reference={"pricing.py": '''"""Order pricing."""
from models import Item


def order_total(items: list, discount_percent: int = 0) -> int:
    """Total of `items` in cents after a percentage discount.

    Rules:
    1. The subtotal is the sum of qty * unit_price_cents; an empty order totals 0.
    2. The discount is discount_percent of the subtotal, rounded down to whole cents, and is
       subtracted from the subtotal.
    3. discount_percent is an int from 0 to 100; raise ValueError otherwise.
    4. Raise ValueError if any item has qty < 1 or unit_price_cents < 0, or if two items share a sku.
    """
    if isinstance(discount_percent, bool) or not isinstance(discount_percent, int) or not 0 <= discount_percent <= 100:
        raise ValueError("discount_percent must be an int from 0 to 100")
    skus = [item.sku for item in items]
    if len(set(skus)) != len(skus):
        raise ValueError("duplicate sku")
    if any(item.qty < 1 or item.unit_price_cents < 0 for item in items):
        raise ValueError("invalid item")
    subtotal = sum(item.qty * item.unit_price_cents for item in items)
    return subtotal - subtotal * discount_percent // 100
'''},
    rules=[
        Rule("subtotal", [("order_total([Item('a', 2, 150), Item('b', 1, 99)])", 399), ("order_total([])", 0)]),
        Rule("discount_floored", [("order_total([Item('a', 1, 999)], 10)", 900), ("order_total([Item('a', 3, 333)], 15)", 850)]),
        Rule("full_discount", [("order_total([Item('a', 1, 500)], 100)", 0)]),
        Rule("discount_range", [("order_total([], 101)", RAISES), ("order_total([], -5)", RAISES), ("order_total([], 12.5)", RAISES)]),
        Rule("invalid_item", [("order_total([Item('a', 0, 100)])", RAISES), ("order_total([Item('a', 1, -1)])", RAISES)]),
        Rule("duplicate_sku", [("order_total([Item('a', 1, 1), Item('a', 2, 1)])", RAISES)]),
    ])

P10 = RuleTask(
    id="P10_total_distance", title="Fix total_distance", category="multi-file", affected=["conversions.py", "distance.py"],
    imports="from conversions import to_meters\nfrom distance import total_distance",
    files={
        "conversions.py": '''"""Length units."""

FACTORS = {"m": 1.0, "km": 1000.0, "ft": 0.3, "in": 0.025, "mi": 1609.0}


def to_meters(value: float, unit: str) -> float:
    """Convert `value` in `unit` to meters.

    Rules:
    1. Exact factors: m 1, km 1000, ft 0.3048, in 0.0254, mi 1609.344.
    2. Raise ValueError for an unknown unit or a negative value.
    """
    return value * FACTORS[unit]
''',
        "distance.py": '''"""Route length."""
from conversions import to_meters


def total_distance(segments: list) -> float:
    """Total length in meters of (value, unit) segments, rounded to 3 decimal places.

    Rules:
    1. Each segment is converted with conversions.to_meters; the sum is rounded once, at the end.
    2. An empty route has length 0.0.
    """
    return sum(round(to_meters(value, unit), 1) for value, unit in segments)
''',
    },
    reference={
        "conversions.py": '''"""Length units."""

FACTORS = {"m": 1.0, "km": 1000.0, "ft": 0.3048, "in": 0.0254, "mi": 1609.344}


def to_meters(value: float, unit: str) -> float:
    """Convert `value` in `unit` to meters.

    Rules:
    1. Exact factors: m 1, km 1000, ft 0.3048, in 0.0254, mi 1609.344.
    2. Raise ValueError for an unknown unit or a negative value.
    """
    if unit not in FACTORS:
        raise ValueError(f"unknown unit {unit!r}")
    if value < 0:
        raise ValueError("negative length")
    return value * FACTORS[unit]
''',
        "distance.py": '''"""Route length."""
from conversions import to_meters


def total_distance(segments: list) -> float:
    """Total length in meters of (value, unit) segments, rounded to 3 decimal places.

    Rules:
    1. Each segment is converted with conversions.to_meters; the sum is rounded once, at the end.
    2. An empty route has length 0.0.
    """
    return round(sum(to_meters(value, unit) for value, unit in segments), 3)
''',
    },
    rules=[
        Rule("exact_factors", [("to_meters(1, 'ft')", 0.3048), ("to_meters(1, 'in')", 0.0254), ("to_meters(1, 'mi')", 1609.344)]),
        Rule("metric", [("to_meters(2, 'km')", 2000.0), ("to_meters(5, 'm')", 5.0)]),
        Rule("unknown_unit", [("to_meters(1, 'yd')", RAISES)]),
        Rule("negative", [("to_meters(-1, 'm')", RAISES)]),
        Rule("sum_then_round", [("total_distance([(1, 'in'), (1, 'in'), (1, 'in')])", 0.076),
                                ("total_distance([(1, 'km'), (500, 'm')])", 1500.0)]),
        Rule("empty_route", [("total_distance([])", 0.0)]),
        Rule("segment_errors", [("total_distance([(1, 'yd')])", RAISES)]),
    ])

P11 = RuleTask(
    id="P11_can_book", title="Fix can_book", category="multi-file", affected=["timeslots.py", "booking.py"],
    imports="from timeslots import parse_slot\nfrom booking import can_book",
    files={
        "timeslots.py": '''"""Time slots."""


def parse_slot(text: str) -> tuple:
    """Parse "HH:MM-HH:MM" into (start, end) minutes after midnight.

    Rules:
    1. HH is 00-23 and MM is 00-59, each exactly two ASCII digits; no spaces.
    2. The end must be strictly after the start.
    3. Raise ValueError for anything else.
    """
    start, end = text.split("-")
    h1, m1 = start.split(":")
    h2, m2 = end.split(":")
    return int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
''',
        "booking.py": '''"""Room booking."""
from timeslots import parse_slot


def can_book(existing: list, new: str) -> bool:
    """True if slot `new` overlaps none of the `existing` slots (all in parse_slot format).

    Rules:
    1. Slots that only touch (one ends when the other starts) do not overlap.
    2. Every slot, existing or new, is validated with parse_slot.
    """
    start, end = parse_slot(new)
    for slot in existing:
        s, e = parse_slot(slot)
        if start <= e and s <= end:
            return False
    return True
''',
    },
    reference={
        "timeslots.py": '''"""Time slots."""
import re


def parse_slot(text: str) -> tuple:
    """Parse "HH:MM-HH:MM" into (start, end) minutes after midnight.

    Rules:
    1. HH is 00-23 and MM is 00-59, each exactly two ASCII digits; no spaces.
    2. The end must be strictly after the start.
    3. Raise ValueError for anything else.
    """
    match = re.fullmatch(r"([0-9]{2}):([0-9]{2})-([0-9]{2}):([0-9]{2})", text)
    if not match:
        raise ValueError(f"invalid slot {text!r}")
    h1, m1, h2, m2 = (int(g) for g in match.groups())
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        raise ValueError(f"invalid time in {text!r}")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if end <= start:
        raise ValueError("end must be after start")
    return start, end
''',
        "booking.py": '''"""Room booking."""
from timeslots import parse_slot


def can_book(existing: list, new: str) -> bool:
    """True if slot `new` overlaps none of the `existing` slots (all in parse_slot format).

    Rules:
    1. Slots that only touch (one ends when the other starts) do not overlap.
    2. Every slot, existing or new, is validated with parse_slot.
    """
    start, end = parse_slot(new)
    slots = [parse_slot(slot) for slot in existing]
    return all(end <= s or e <= start for s, e in slots)
''',
    },
    rules=[
        Rule("parse_basic", [("parse_slot('09:00-10:30')", (540, 630))]),
        Rule("two_digit_fields", [("parse_slot('9:00-10:00')", RAISES), ("parse_slot('09:00 - 10:00')", RAISES)]),
        Rule("field_ranges", [("parse_slot('24:00-24:30')", RAISES), ("parse_slot('10:60-11:00')", RAISES)]),
        Rule("end_after_start", [("parse_slot('10:00-10:00')", RAISES), ("parse_slot('11:00-10:00')", RAISES)]),
        Rule("touching_allowed", [("can_book(['09:00-10:00'], '10:00-11:00')", True), ("can_book(['10:00-11:00'], '09:00-10:00')", True)]),
        Rule("overlap_rejected", [("can_book(['09:00-10:00'], '09:30-10:30')", False), ("can_book(['09:00-12:00'], '10:00-11:00')", False)]),
        Rule("all_slots_validated", [("can_book(['bad'], '10:00-11:00')", RAISES), ("can_book([], '10:00')", RAISES)]),
        Rule("empty_calendar", [("can_book([], '00:00-23:59')", True)]),
    ])

P12 = RuleTask(
    id="P12_trimmed_mean", title="Fix trimmed_mean", category="multi-file", affected=["statcore.py", "robust.py"],
    imports="from statcore import mean\nfrom robust import trimmed_mean",
    files={
        "statcore.py": '''"""Basic statistics."""


def mean(values: list) -> float:
    """Arithmetic mean as a float. Raise ValueError for an empty list."""
    return sum(values) // len(values)
''',
        "robust.py": '''"""Robust statistics."""
import math

from statcore import mean


def trimmed_mean(values: list, fraction: float) -> float:
    """Mean after removing floor(len(values) * fraction) values from each end of the sorted data.

    Rules:
    1. 0 <= fraction < 0.5; raise ValueError otherwise.
    2. Raise ValueError for an empty list.
    3. The input list is not modified.
    4. The mean itself is statcore.mean.
    """
    values.sort()
    k = math.ceil(len(values) * fraction)
    return mean(values[k:len(values) - k])
''',
    },
    reference={
        "statcore.py": '''"""Basic statistics."""


def mean(values: list) -> float:
    """Arithmetic mean as a float. Raise ValueError for an empty list."""
    if not values:
        raise ValueError("mean of an empty list")
    return sum(values) / len(values)
''',
        "robust.py": '''"""Robust statistics."""
import math

from statcore import mean


def trimmed_mean(values: list, fraction: float) -> float:
    """Mean after removing floor(len(values) * fraction) values from each end of the sorted data.

    Rules:
    1. 0 <= fraction < 0.5; raise ValueError otherwise.
    2. Raise ValueError for an empty list.
    3. The input list is not modified.
    4. The mean itself is statcore.mean.
    """
    if not 0 <= fraction < 0.5:
        raise ValueError("fraction must be in [0, 0.5)")
    if not values:
        raise ValueError("empty list")
    ordered = sorted(values)
    k = math.floor(len(ordered) * fraction)
    return mean(ordered[k:len(ordered) - k])
''',
    },
    rules=[
        Rule("mean_float", [("mean([1, 2])", 1.5), ("mean([4])", 4.0)]),
        Rule("mean_empty", [("mean([])", RAISES)]),
        Rule("trim_floor", [("trimmed_mean([1, 2, 3, 100], 0.25)", 2.5), ("trimmed_mean([1, 2, 3, 4, 100], 0.3)", 3.0)]),
        Rule("no_trim", [("trimmed_mean([3, 1, 2], 0)", 2.0)]),
        Rule("fraction_range", [("trimmed_mean([1, 2], 0.5)", RAISES), ("trimmed_mean([1, 2], -0.1)", RAISES)]),
        Rule("empty", [("trimmed_mean([], 0.1)", RAISES)]),
        Rule("input_unchanged", [("(lambda v: (trimmed_mean(v, 0.0), v))([3, 1, 2])", (2.0, [3, 1, 2]))]),
    ])

P13 = RuleTask(
    id="P13_load_settings", title="Fix load_settings", category="multi-file", affected=["loader.py"],
    imports="from settings import DEFAULTS\nfrom loader import load_settings",
    files={
        "settings.py": '''"""Application defaults."""

DEFAULTS = {"host": "localhost", "port": 8080, "debug": False, "workers": 4}
''',
        "loader.py": '''"""Settings loading."""
from settings import DEFAULTS


def load_settings(overrides: dict) -> dict:
    """DEFAULTS updated with `overrides`, returned as a new dict.

    Rules:
    1. DEFAULTS itself is never modified.
    2. Raise ValueError for a key not in DEFAULTS.
    3. Each value must have the same type as its default (bool is not accepted for an int
       setting); raise ValueError otherwise.
    4. "port" must be from 1 to 65535 and "workers" at least 1; raise ValueError otherwise.
    """
    DEFAULTS.update(overrides)
    return DEFAULTS
''',
    },
    reference={"loader.py": '''"""Settings loading."""
from settings import DEFAULTS


def load_settings(overrides: dict) -> dict:
    """DEFAULTS updated with `overrides`, returned as a new dict.

    Rules:
    1. DEFAULTS itself is never modified.
    2. Raise ValueError for a key not in DEFAULTS.
    3. Each value must have the same type as its default (bool is not accepted for an int
       setting); raise ValueError otherwise.
    4. "port" must be from 1 to 65535 and "workers" at least 1; raise ValueError otherwise.
    """
    result = dict(DEFAULTS)
    for key, value in overrides.items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown setting {key!r}")
        if type(value) is not type(DEFAULTS[key]):
            raise ValueError(f"wrong type for {key!r}")
        result[key] = value
    if not 1 <= result["port"] <= 65535 or result["workers"] < 1:
        raise ValueError("port or workers out of range")
    return result
'''},
    rules=[
        Rule("merge", [("load_settings({'port': 9000})['port']", 9000), ("load_settings({})['host']", "localhost")]),
        Rule("defaults_untouched", [("(load_settings({'port': 1234}), DEFAULTS['port'])[1]", 8080)]),
        Rule("new_dict", [("load_settings({}) is DEFAULTS", False)]),
        Rule("unknown_key", [("load_settings({'colour': 'red'})", RAISES)]),
        Rule("type_checked", [("load_settings({'port': '80'})", RAISES), ("load_settings({'debug': 1})", RAISES),
                              ("load_settings({'workers': True})", RAISES)]),
        Rule("ranges", [("load_settings({'port': 0})", RAISES), ("load_settings({'port': 65536})", RAISES),
                        ("load_settings({'workers': 0})", RAISES)]),
    ])

P14 = RuleTask(
    id="P14_matmul", title="Fix matmul", category="multi-file", affected=["shapes.py", "ops.py"],
    imports="from shapes import shape\nfrom ops import matmul",
    files={
        "shapes.py": '''"""Matrix shapes (lists of lists)."""


def shape(m: list) -> tuple:
    """(rows, cols) of a rectangular, non-empty matrix.

    Rules:
    1. Raise ValueError if the matrix or any row is empty, or if rows differ in length.
    """
    return len(m), len(m[0])
''',
        "ops.py": '''"""Matrix operations."""
from shapes import shape


def matmul(a: list, b: list) -> list:
    """Matrix product a x b as a new list of lists.

    Rules:
    1. Both matrices are validated with shapes.shape.
    2. Raise ValueError if the number of columns of a differs from the number of rows of b.
    """
    rows, inner = shape(a)
    _, cols = shape(b)
    return [[sum(a[i][k] * b[j][k] for k in range(inner)) for j in range(cols)] for i in range(rows)]
''',
    },
    reference={
        "shapes.py": '''"""Matrix shapes (lists of lists)."""


def shape(m: list) -> tuple:
    """(rows, cols) of a rectangular, non-empty matrix.

    Rules:
    1. Raise ValueError if the matrix or any row is empty, or if rows differ in length.
    """
    if not m or any(not row for row in m) or len({len(row) for row in m}) != 1:
        raise ValueError("matrix must be rectangular and non-empty")
    return len(m), len(m[0])
''',
        "ops.py": '''"""Matrix operations."""
from shapes import shape


def matmul(a: list, b: list) -> list:
    """Matrix product a x b as a new list of lists.

    Rules:
    1. Both matrices are validated with shapes.shape.
    2. Raise ValueError if the number of columns of a differs from the number of rows of b.
    """
    rows, inner = shape(a)
    inner_b, cols = shape(b)
    if inner != inner_b:
        raise ValueError("inner dimensions differ")
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]
''',
    },
    rules=[
        Rule("shape_basic", [("shape([[1, 2, 3], [4, 5, 6]])", (2, 3))]),
        Rule("shape_ragged", [("shape([[1, 2], [3]])", RAISES)]),
        Rule("shape_empty", [("shape([])", RAISES), ("shape([[]])", RAISES)]),
        Rule("product", [("matmul([[1, 2]], [[3], [4]])", [[11]]), ("matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])", [[19, 22], [43, 50]])]),
        Rule("identity", [("matmul([[1, 0], [0, 1]], [[2, 3], [4, 5]])", [[2, 3], [4, 5]])]),
        Rule("dimension_mismatch", [("matmul([[1, 2]], [[1, 2]])", RAISES)]),
        Rule("validates_operands", [("matmul([[1], [2, 3]], [[1]])", RAISES)]),
    ])

P15 = RuleTask(
    id="P15_next_task", title="Fix next_task", category="multi-file", affected=["scheduler.py"],
    imports="from jobs import Job\nfrom scheduler import next_task",
    files={
        "jobs.py": '''"""Job records."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    """A queued job; a higher priority runs first."""
    name: str
    priority: int
    created: int
''',
        "scheduler.py": '''"""Job selection."""
from jobs import Job


def next_task(jobs: list):
    """The job to run next, or None for an empty queue.

    Rules:
    1. The highest priority wins (priorities may be negative).
    2. Ties are broken by the earliest `created`, then by `name` in alphabetical order.
    """
    return min(jobs, key=lambda job: (job.priority, job.created))
''',
    },
    reference={"scheduler.py": '''"""Job selection."""
from jobs import Job


def next_task(jobs: list):
    """The job to run next, or None for an empty queue.

    Rules:
    1. The highest priority wins (priorities may be negative).
    2. Ties are broken by the earliest `created`, then by `name` in alphabetical order.
    """
    if not jobs:
        return None
    return min(jobs, key=lambda job: (-job.priority, job.created, job.name))
'''},
    rules=[
        Rule("highest_priority", [("next_task([Job('a', 1, 0), Job('b', 5, 9)]).name", "b")]),
        Rule("negative_priority", [("next_task([Job('a', -3, 0), Job('b', -1, 5)]).name", "b")]),
        Rule("tie_created", [("next_task([Job('a', 2, 7), Job('b', 2, 3)]).name", "b")]),
        Rule("tie_name", [("next_task([Job('zed', 2, 3), Job('amy', 2, 3)]).name", "amy")]),
        Rule("empty_queue", [("next_task([])", None)]),
    ])

P16 = RuleTask(
    id="P16_make_slug", title="Fix make_slug", category="multi-file", affected=["translit.py", "slug.py"],
    imports="from slug import make_slug",
    files={
        "translit.py": '''"""German transliteration."""

TABLE = {"ä": "ae", "ö": "oe", "ü": "ue"}


def transliterate(text: str) -> str:
    """Replace ä, ö, ü with ae, oe, ue and ß with ss (lower-case input)."""
    return "".join(TABLE.get(ch, ch) for ch in text)
''',
        "slug.py": '''"""URL slugs."""
import re

from translit import transliterate


def make_slug(title: str, max_len: int = 40) -> str:
    """URL slug for `title`.

    Rules:
    1. Lower-case first, then transliterate with translit.transliterate.
    2. Every run of characters other than ASCII letters and digits becomes one hyphen;
       no leading or trailing hyphens.
    3. Cut to at most max_len characters, then remove any trailing hyphen.
    4. Raise ValueError if the result is empty or max_len < 1.
    """
    slug = re.sub(r"[^a-z0-9]", "-", transliterate(title.lower()))
    return slug[:max_len]
''',
    },
    reference={
        "translit.py": '''"""German transliteration."""

TABLE = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def transliterate(text: str) -> str:
    """Replace ä, ö, ü with ae, oe, ue and ß with ss (lower-case input)."""
    return "".join(TABLE.get(ch, ch) for ch in text)
''',
        "slug.py": '''"""URL slugs."""
import re

from translit import transliterate


def make_slug(title: str, max_len: int = 40) -> str:
    """URL slug for `title`.

    Rules:
    1. Lower-case first, then transliterate with translit.transliterate.
    2. Every run of characters other than ASCII letters and digits becomes one hyphen;
       no leading or trailing hyphens.
    3. Cut to at most max_len characters, then remove any trailing hyphen.
    4. Raise ValueError if the result is empty or max_len < 1.
    """
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    slug = re.sub(r"[^a-z0-9]+", "-", transliterate(title.lower())).strip("-")
    slug = slug[:max_len].rstrip("-")
    if not slug:
        raise ValueError("empty slug")
    return slug
''',
    },
    rules=[
        Rule("basic", [("make_slug('Hello World')", "hello-world")]),
        Rule("umlauts", [("make_slug('Über Größe')", "ueber-groesse")]),
        Rule("collapse_runs", [("make_slug('a -- b!!c')", "a-b-c"), ("make_slug('  Lead and trail  ')", "lead-and-trail")]),
        Rule("cut_without_hyphen", [("make_slug('abc def', 4)", "abc"), ("make_slug('abcdef', 3)", "abc")]),
        Rule("empty_result", [("make_slug('!!!')", RAISES)]),
        Rule("invalid_max_len", [("make_slug('abc', 0)", RAISES)]),
    ])

P17 = RuleTask(
    id="P17_rate_limiter", title="Fix the rate limiter", category="multi-file", affected=["limiter.py"],
    imports="from limiter import simulate",
    files={
        "window.py": '''"""Sliding time window."""


class Window:
    """Timestamps (seconds) of accepted events; `length` is the window size in seconds."""

    def __init__(self, length: float) -> None:
        self.length = length
        self.events = []

    def count_since(self, now: float) -> int:
        """Accepted events with timestamp t such that now - length < t <= now."""
        return sum(1 for t in self.events if now - self.length < t <= now)

    def add(self, now: float) -> None:
        self.events.append(now)
''',
        "limiter.py": '''"""Rate limiting."""
from window import Window


def simulate(times: list, limit: int, length: float) -> list:
    """For each event time (non-decreasing), whether a limiter accepts it.

    Rules:
    1. An event at time t is accepted if fewer than `limit` events were accepted in the window
       (t - length, t] (see window.Window.count_since); rejected events do not count.
    2. Raise ValueError if limit < 1, length <= 0 or the times decrease.
    """
    window = Window(length)
    result = []
    for t in times:
        ok = window.count_since(t) <= limit
        window.add(t)
        result.append(ok)
    return result
''',
    },
    reference={"limiter.py": '''"""Rate limiting."""
from window import Window


def simulate(times: list, limit: int, length: float) -> list:
    """For each event time (non-decreasing), whether a limiter accepts it.

    Rules:
    1. An event at time t is accepted if fewer than `limit` events were accepted in the window
       (t - length, t] (see window.Window.count_since); rejected events do not count.
    2. Raise ValueError if limit < 1, length <= 0 or the times decrease.
    """
    if limit < 1 or length <= 0:
        raise ValueError("limit must be >= 1 and length > 0")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("times must not decrease")
    window = Window(length)
    result = []
    for t in times:
        ok = window.count_since(t) < limit
        if ok:
            window.add(t)
        result.append(ok)
    return result
'''},
    rules=[
        Rule("limit_enforced", [("simulate([0, 0, 0], 2, 1)", [True, True, False])]),
        Rule("window_slides", [("simulate([0, 1, 2], 1, 1)", [True, True, True]), ("simulate([0, 0.5, 1], 1, 1)", [True, False, True])]),
        Rule("rejected_not_counted", [("simulate([0, 0.5, 1.2], 1, 1)", [True, False, True]),
                                      ("simulate([0, 0.6, 1.1, 1.5], 1, 1)", [True, False, True, False])]),
        Rule("invalid_parameters", [("simulate([0], 0, 1)", RAISES), ("simulate([0], 1, 0)", RAISES)]),
        Rule("decreasing_times", [("simulate([2, 1], 5, 1)", RAISES)]),
    ])

P18 = RuleTask(
    id="P18_format_money", title="Fix format_money", category="multi-file", affected=["fmt.py"],
    imports="from fmt import format_money",
    files={
        "currency.py": '''"""Currency table: code -> (symbol, decimal places)."""

CURRENCIES = {"USD": ("$", 2), "EUR": ("€", 2), "JPY": ("¥", 0), "KWD": ("KD", 3)}
''',
        "fmt.py": '''"""Money formatting."""
from currency import CURRENCIES


def format_money(minor_units: int, code: str) -> str:
    """Format an amount given in minor units (e.g. cents) of currency `code`.

    Rules:
    1. Use the symbol and number of decimal places from currency.CURRENCIES.
    2. Group the whole part in thousands with ","; no decimal point when there are 0 decimals.
    3. Negative amounts start with "-" before the symbol, e.g. "-$1,234.50".
    4. Raise ValueError for an unknown currency code.
    """
    symbol, _ = CURRENCIES[code]
    return f"{symbol}{minor_units / 100:.2f}"
''',
    },
    reference={"fmt.py": '''"""Money formatting."""
from currency import CURRENCIES


def format_money(minor_units: int, code: str) -> str:
    """Format an amount given in minor units (e.g. cents) of currency `code`.

    Rules:
    1. Use the symbol and number of decimal places from currency.CURRENCIES.
    2. Group the whole part in thousands with ","; no decimal point when there are 0 decimals.
    3. Negative amounts start with "-" before the symbol, e.g. "-$1,234.50".
    4. Raise ValueError for an unknown currency code.
    """
    if code not in CURRENCIES:
        raise ValueError(f"unknown currency {code!r}")
    symbol, places = CURRENCIES[code]
    sign = "-" if minor_units < 0 else ""
    whole, fraction = divmod(abs(minor_units), 10 ** places)
    text = f"{whole:,}"
    if places:
        text += "." + str(fraction).zfill(places)
    return f"{sign}{symbol}{text}"
'''},
    rules=[
        Rule("two_decimals", [("format_money(123450, 'USD')", "$1,234.50"), ("format_money(5, 'EUR')", "€0.05")]),
        Rule("zero_decimals", [("format_money(1234, 'JPY')", "¥1,234")]),
        Rule("three_decimals", [("format_money(1234567, 'KWD')", "KD1,234.567")]),
        Rule("grouping", [("format_money(100000000, 'USD')", "$1,000,000.00")]),
        Rule("negative", [("format_money(-123450, 'USD')", "-$1,234.50"), ("format_money(-7, 'JPY')", "-¥7")]),
        Rule("unknown_code", [("format_money(1, 'XXX')", RAISES)]),
    ])

TASKS = [P9, P10, P11, P12, P13, P14, P15, P16, P17, P18]
