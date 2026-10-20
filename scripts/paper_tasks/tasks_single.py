"""Single-module rule tasks P1-P8 (Paper I confirmatory campaign)."""
from framework import RAISES, Rule, RuleTask

P1 = RuleTask(
    id="P1_split_amount", title="Fix split_amount", category="single", affected=["money.py"],
    imports="from money import split_amount",
    files={"money.py": '''"""Money helpers."""


def split_amount(total_cents: int, parts: int) -> list:
    """Split `total_cents` into `parts` integer shares.

    Rules:
    1. The shares sum to total_cents and differ from each other by at most 1 cent.
    2. Larger shares come first.
    3. Raise ValueError if parts < 1 or total_cents < 0.
    4. Both arguments must be int; bool is not accepted (raise ValueError).
    """
    share = total_cents // parts
    result = [share] * parts
    result[-1] += total_cents - share * parts
    return result
'''},
    reference={"money.py": '''"""Money helpers."""


def split_amount(total_cents: int, parts: int) -> list:
    """Split `total_cents` into `parts` integer shares.

    Rules:
    1. The shares sum to total_cents and differ from each other by at most 1 cent.
    2. Larger shares come first.
    3. Raise ValueError if parts < 1 or total_cents < 0.
    4. Both arguments must be int; bool is not accepted (raise ValueError).
    """
    for value in (total_cents, parts):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("integer arguments required")
    if parts < 1 or total_cents < 0:
        raise ValueError("invalid amount or parts")
    quotient, remainder = divmod(total_cents, parts)
    return [quotient + 1] * remainder + [quotient] * (parts - remainder)
'''},
    rules=[
        Rule("even_split", [("split_amount(10, 5)", [2, 2, 2, 2, 2]), ("split_amount(0, 2)", [0, 0])]),
        Rule("at_most_one_cent_apart", [("split_amount(5, 3)", [2, 2, 1]), ("split_amount(7, 4)", [2, 2, 2, 1])]),
        Rule("larger_first", [("split_amount(100, 3)", [34, 33, 33])]),
        Rule("invalid_parts", [("split_amount(5, 0)", RAISES), ("split_amount(5, -2)", RAISES)]),
        Rule("negative_total", [("split_amount(-1, 2)", RAISES)]),
        Rule("bool_rejected", [("split_amount(True, 1)", RAISES), ("split_amount(5, True)", RAISES)]),
        Rule("non_int_rejected", [("split_amount(5.0, 2)", RAISES)]),
    ])

P2 = RuleTask(
    id="P2_wrap_words", title="Fix wrap_words", category="single", affected=["textwrap2.py"],
    imports="from textwrap2 import wrap_words",
    files={"textwrap2.py": '''"""Greedy word wrapping."""


def wrap_words(text: str, width: int) -> list:
    """Wrap `text` into lines of at most `width` characters.

    Rules:
    1. Words are separated by any run of whitespace; leading and trailing whitespace is ignored.
    2. Words on a line are joined by single spaces, and lines are filled greedily.
    3. A word longer than `width` is split into pieces of exactly `width` characters (the last
       piece may be shorter); the pieces are then placed like separate words.
    4. Empty or all-whitespace text gives [].
    5. Raise ValueError if width < 1.
    """
    lines, current = [], ""
    for word in text.split(" "):
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word) if current else word
    if current:
        lines.append(current)
    return lines
'''},
    reference={"textwrap2.py": '''"""Greedy word wrapping."""


def wrap_words(text: str, width: int) -> list:
    """Wrap `text` into lines of at most `width` characters.

    Rules:
    1. Words are separated by any run of whitespace; leading and trailing whitespace is ignored.
    2. Words on a line are joined by single spaces, and lines are filled greedily.
    3. A word longer than `width` is split into pieces of exactly `width` characters (the last
       piece may be shorter); the pieces are then placed like separate words.
    4. Empty or all-whitespace text gives [].
    5. Raise ValueError if width < 1.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    pieces = []
    for word in text.split():
        pieces.extend(word[i:i + width] for i in range(0, len(word), width))
    lines, current = [], ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > width:
            lines.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        lines.append(current)
    return lines
'''},
    rules=[
        Rule("greedy_fill", [("wrap_words('a b c', 3)", ["a b", "c"]), ("wrap_words('hello world', 5)", ["hello", "world"])]),
        Rule("whitespace_runs", [("wrap_words('  a   b  ', 10)", ["a b"]), ("wrap_words('a\\tb\\nc', 5)", ["a b c"])]),
        Rule("long_word_split", [("wrap_words('abcdefgh', 3)", ["abc", "def", "gh"]),
                                 ("wrap_words('ab cdefg h', 3)", ["ab", "cde", "fg", "h"])]),
        Rule("empty_text", [("wrap_words('', 4)", []), ("wrap_words('   ', 4)", [])]),
        Rule("invalid_width", [("wrap_words('a', 0)", RAISES), ("wrap_words('a', -3)", RAISES)]),
    ])

P3 = RuleTask(
    id="P3_days_in_month", title="Fix days_in_month", category="single", affected=["calendar2.py"],
    imports="from calendar2 import days_in_month",
    files={"calendar2.py": '''"""Calendar helpers."""


def days_in_month(year: int, month: int) -> int:
    """Number of days in `month` of `year` in the proleptic Gregorian calendar.

    Rules:
    1. A year is a leap year if divisible by 4, except years divisible by 100 that are not
       divisible by 400.
    2. February has 29 days in leap years and 28 otherwise; April, June, September and November
       have 30; the other months have 31.
    3. Raise ValueError unless 1 <= month <= 12 and year >= 1.
    4. Both arguments must be int; bool is not accepted (raise ValueError).
    """
    if month == 2:
        return 29 if year % 4 == 0 else 28
    return 30 if month in (4, 6, 9, 11) else 31
'''},
    reference={"calendar2.py": '''"""Calendar helpers."""


def days_in_month(year: int, month: int) -> int:
    """Number of days in `month` of `year` in the proleptic Gregorian calendar.

    Rules:
    1. A year is a leap year if divisible by 4, except years divisible by 100 that are not
       divisible by 400.
    2. February has 29 days in leap years and 28 otherwise; April, June, September and November
       have 30; the other months have 31.
    3. Raise ValueError unless 1 <= month <= 12 and year >= 1.
    4. Both arguments must be int; bool is not accepted (raise ValueError).
    """
    for value in (year, month):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("integer arguments required")
    if not 1 <= month <= 12 or year < 1:
        raise ValueError("invalid year or month")
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31
'''},
    rules=[
        Rule("month_lengths", [("days_in_month(2023, 4)", 30), ("days_in_month(2023, 12)", 31), ("days_in_month(2023, 1)", 31)]),
        Rule("leap_by_four", [("days_in_month(2024, 2)", 29), ("days_in_month(2023, 2)", 28)]),
        Rule("century_rule", [("days_in_month(1900, 2)", 28), ("days_in_month(2000, 2)", 29), ("days_in_month(2100, 2)", 28)]),
        Rule("invalid_month", [("days_in_month(2023, 0)", RAISES), ("days_in_month(2023, 13)", RAISES)]),
        Rule("invalid_year", [("days_in_month(0, 5)", RAISES)]),
        Rule("bool_rejected", [("days_in_month(True, 2)", RAISES), ("days_in_month(2023, True)", RAISES)]),
    ])

P4 = RuleTask(
    id="P4_letter_grade", title="Fix letter_grade", category="single", affected=["grades.py"],
    imports="from grades import letter_grade",
    files={"grades.py": '''"""Grading."""


def letter_grade(score: float) -> str:
    """Letter grade for a percentage score.

    Rules:
    1. "A" for score >= 90, "B" for >= 80, "C" for >= 70, "D" for >= 60, otherwise "F";
       every threshold is inclusive.
    2. Raise ValueError unless 0 <= score <= 100; NaN is invalid.
    3. score must be an int or float; bool is not accepted (raise ValueError).
    """
    if score > 90:
        return "A"
    if score > 80:
        return "B"
    if score > 70:
        return "C"
    if score > 60:
        return "D"
    return "F"
'''},
    reference={"grades.py": '''"""Grading."""
import math


def letter_grade(score: float) -> str:
    """Letter grade for a percentage score.

    Rules:
    1. "A" for score >= 90, "B" for >= 80, "C" for >= 70, "D" for >= 60, otherwise "F";
       every threshold is inclusive.
    2. Raise ValueError unless 0 <= score <= 100; NaN is invalid.
    3. score must be an int or float; bool is not accepted (raise ValueError).
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("numeric score required")
    if math.isnan(score) or not 0 <= score <= 100:
        raise ValueError("score out of range")
    for threshold, grade in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= threshold:
            return grade
    return "F"
'''},
    rules=[
        Rule("bands", [("letter_grade(95)", "A"), ("letter_grade(85.5)", "B"), ("letter_grade(42)", "F")]),
        Rule("inclusive_thresholds", [("letter_grade(90)", "A"), ("letter_grade(80)", "B"), ("letter_grade(70)", "C"),
                                      ("letter_grade(60)", "D")]),
        Rule("range_ends", [("letter_grade(0)", "F"), ("letter_grade(100)", "A")]),
        Rule("out_of_range", [("letter_grade(100.1)", RAISES), ("letter_grade(-1)", RAISES)]),
        Rule("nan_rejected", [("letter_grade(float('nan'))", RAISES)]),
        Rule("bool_rejected", [("letter_grade(True)", RAISES)]),
    ])

P5 = RuleTask(
    id="P5_convert_temperature", title="Fix convert_temperature", category="single", affected=["temps.py"],
    imports="from temps import convert_temperature",
    files={"temps.py": '''"""Temperature conversion."""


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between "C", "F" and "K".

    Rules:
    1. Units are exactly "C", "F" or "K" (upper case); raise ValueError for anything else.
    2. The result is rounded to 2 decimal places.
    3. Raise ValueError if `value` is below absolute zero in `from_unit` (-273.15 C, -459.67 F, 0 K).
    4. Converting to the same unit returns the rounded value.
    """
    if from_unit == "C" and to_unit == "F":
        return value * 9 / 5 + 32
    if from_unit == "F" and to_unit == "C":
        return (value - 32) * 5 / 9
    return value
'''},
    reference={"temps.py": '''"""Temperature conversion."""

_TO_KELVIN = {"C": lambda v: v + 273.15, "F": lambda v: (v - 32) * 5 / 9 + 273.15, "K": lambda v: v}
_FROM_KELVIN = {"C": lambda k: k - 273.15, "F": lambda k: (k - 273.15) * 9 / 5 + 32, "K": lambda k: k}


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between "C", "F" and "K".

    Rules:
    1. Units are exactly "C", "F" or "K" (upper case); raise ValueError for anything else.
    2. The result is rounded to 2 decimal places.
    3. Raise ValueError if `value` is below absolute zero in `from_unit` (-273.15 C, -459.67 F, 0 K).
    4. Converting to the same unit returns the rounded value.
    """
    if from_unit not in _TO_KELVIN or to_unit not in _TO_KELVIN:
        raise ValueError("unknown unit")
    kelvin = _TO_KELVIN[from_unit](value)
    if kelvin < -1e-9:
        raise ValueError("below absolute zero")
    if from_unit == to_unit:
        return round(value, 2)
    return round(_FROM_KELVIN[to_unit](kelvin), 2)
'''},
    rules=[
        Rule("celsius_fahrenheit", [("convert_temperature(100, 'C', 'F')", 212.0), ("convert_temperature(32, 'F', 'C')", 0.0)]),
        Rule("kelvin", [("convert_temperature(0, 'K', 'C')", -273.15), ("convert_temperature(25, 'C', 'K')", 298.15)]),
        Rule("rounding", [("convert_temperature(98.6, 'F', 'C')", 37.0), ("convert_temperature(1, 'C', 'F')", 33.8)]),
        Rule("same_unit", [("convert_temperature(20.123, 'C', 'C')", 20.12)]),
        Rule("absolute_zero", [("convert_temperature(-500, 'C', 'F')", RAISES), ("convert_temperature(-1, 'K', 'C')", RAISES)]),
        Rule("unit_names", [("convert_temperature(10, 'c', 'F')", RAISES), ("convert_temperature(10, 'C', 'X')", RAISES)]),
    ])

P6 = RuleTask(
    id="P6_chunk_indices", title="Fix chunk_indices", category="single", affected=["chunks.py"],
    imports="from chunks import chunk_indices",
    files={"chunks.py": '''"""Batching helpers."""


def chunk_indices(n: int, size: int) -> list:
    """Half-open (start, end) index pairs that cover range(n) in chunks of `size`.

    Rules:
    1. Chunks are consecutive, start at 0 and end exactly at n; only the last chunk may be shorter.
    2. n == 0 gives [].
    3. Raise ValueError if size < 1 or n < 0.
    """
    return [(start, start + size) for start in range(0, n, size)]
'''},
    reference={"chunks.py": '''"""Batching helpers."""


def chunk_indices(n: int, size: int) -> list:
    """Half-open (start, end) index pairs that cover range(n) in chunks of `size`.

    Rules:
    1. Chunks are consecutive, start at 0 and end exactly at n; only the last chunk may be shorter.
    2. n == 0 gives [].
    3. Raise ValueError if size < 1 or n < 0.
    """
    if size < 1 or n < 0:
        raise ValueError("size must be >= 1 and n >= 0")
    return [(start, min(start + size, n)) for start in range(0, n, size)]
'''},
    rules=[
        Rule("exact_multiple", [("chunk_indices(6, 3)", [(0, 3), (3, 6)])]),
        Rule("short_last_chunk", [("chunk_indices(10, 3)", [(0, 3), (3, 6), (6, 9), (9, 10)]), ("chunk_indices(2, 5)", [(0, 2)])]),
        Rule("empty", [("chunk_indices(0, 5)", [])]),
        Rule("invalid_size", [("chunk_indices(5, 0)", RAISES)]),
        Rule("negative_n", [("chunk_indices(-1, 2)", RAISES)]),
    ])

P7 = RuleTask(
    id="P7_parse_csv_line", title="Fix parse_csv_line", category="single", affected=["csvline.py"],
    imports="from csvline import parse_csv_line",
    files={"csvline.py": '''"""Minimal CSV field splitting."""


def parse_csv_line(line: str) -> list:
    """Split one CSV line into fields.

    Rules:
    1. Fields are separated by commas; whitespace is kept as is; an empty line is one empty field.
    2. A field may be wrapped in double quotes; inside it commas are literal and "" stands for one ".
    3. Raise ValueError for an unterminated quoted field or for any character other than a comma
       right after a closing quote.
    """
    return line.split(",")
'''},
    reference={"csvline.py": '''"""Minimal CSV field splitting."""


def parse_csv_line(line: str) -> list:
    """Split one CSV line into fields.

    Rules:
    1. Fields are separated by commas; whitespace is kept as is; an empty line is one empty field.
    2. A field may be wrapped in double quotes; inside it commas are literal and "" stands for one ".
    3. Raise ValueError for an unterminated quoted field or for any character other than a comma
       right after a closing quote.
    """
    fields, i, n = [], 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            value = []
            while True:
                if i >= n:
                    raise ValueError("unterminated quoted field")
                if line[i] == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        value.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                value.append(line[i])
                i += 1
            if i < n and line[i] != ",":
                raise ValueError("unexpected character after closing quote")
            fields.append("".join(value))
        else:
            end = line.find(",", i)
            end = n if end == -1 else end
            fields.append(line[i:end])
            i = end
        if i >= n:
            return fields
        i += 1
'''},
    rules=[
        Rule("plain_fields", [("parse_csv_line('a,b,c')", ["a", "b", "c"]), ("parse_csv_line('a,,b')", ["a", "", "b"])]),
        Rule("whitespace_kept", [("parse_csv_line(' a , b')", [" a ", " b"])]),
        Rule("empty_line", [("parse_csv_line('')", [""])]),
        Rule("quoted_comma", [("parse_csv_line('\"a,b\",c')", ["a,b", "c"])]),
        Rule("escaped_quote", [("parse_csv_line('\"say \"\"hi\"\"\",x')", ['say "hi"', "x"])]),
        Rule("unterminated", [("parse_csv_line('\"abc')", RAISES)]),
        Rule("junk_after_quote", [("parse_csv_line('\"a\"b,c')", RAISES)]),
    ])

P8 = RuleTask(
    id="P8_compare_versions", title="Fix compare_versions", category="single", affected=["versions.py"],
    imports="from versions import compare_versions",
    files={"versions.py": '''"""Dotted version comparison."""


def compare_versions(a: str, b: str) -> int:
    """Compare dotted versions such as "1.10.2"; return -1, 0 or 1.

    Rules:
    1. Components are compared numerically, left to right ("1.10" > "1.9").
    2. Missing trailing components count as 0 ("1.0" == "1"); leading zeros are ignored ("1.01" == "1.1").
    3. Every component is a non-empty run of ASCII digits; raise ValueError otherwise
       (including an empty string).
    """
    return (a > b) - (a < b)
'''},
    reference={"versions.py": '''"""Dotted version comparison."""
import re


def _parts(text: str) -> list:
    parts = text.split(".")
    if any(not re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError(f"invalid version {text!r}")
    return [int(part) for part in parts]


def compare_versions(a: str, b: str) -> int:
    """Compare dotted versions such as "1.10.2"; return -1, 0 or 1.

    Rules:
    1. Components are compared numerically, left to right ("1.10" > "1.9").
    2. Missing trailing components count as 0 ("1.0" == "1"); leading zeros are ignored ("1.01" == "1.1").
    3. Every component is a non-empty run of ASCII digits; raise ValueError otherwise
       (including an empty string).
    """
    x, y = _parts(a), _parts(b)
    width = max(len(x), len(y))
    x, y = x + [0] * (width - len(x)), y + [0] * (width - len(y))
    return (x > y) - (x < y)
'''},
    rules=[
        Rule("numeric_order", [("compare_versions('1.2', '1.10')", -1), ("compare_versions('2', '1.9.9')", 1)]),
        Rule("missing_components", [("compare_versions('1.0', '1')", 0), ("compare_versions('1.2.0.0', '1.2')", 0)]),
        Rule("leading_zeros", [("compare_versions('1.01', '1.1')", 0)]),
        Rule("equal", [("compare_versions('3.4.5', '3.4.5')", 0)]),
        Rule("empty_component", [("compare_versions('1..2', '1')", RAISES), ("compare_versions('', '1')", RAISES)]),
        Rule("non_digit", [("compare_versions('1.a', '1')", RAISES), ("compare_versions('1', '1.-2')", RAISES)]),
    ])

TASKS = [P1, P2, P3, P4, P5, P6, P7, P8]
