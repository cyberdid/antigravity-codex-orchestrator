"""Generate the numeric table bodies of the paper from the benchmark files (both languages).

Tables are never edited by hand: every cell is recomputed from benchmarks/*.json.
Writes docs/paper/tex/tables/{en,uk}/T{2,3,4,5}.tex. Usage: python3 docs/paper/tex/gen_tables.py
"""
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = ROOT / "benchmarks"
SEEDS = (303, 404, 505)
BASE, TESTS, LIST = "P1_STAGE_ADAPTIVE", "P1_STAGE_ADAPTIVE_SPECTESTS", "P1_STAGE_ADAPTIVE_CHECKLIST"
ARM_NAMES = {"en": {BASE: "Baseline", TESTS: "Contract tests", LIST: "Checklist (control)"},
             "uk": {BASE: "Базова політика", TESTS: "Контрактні тести", LIST: "Чекліст (контроль)"}}


def num(x, lang, digits=0):
    """Thousands separator and decimal mark per language."""
    text = f"{x:,.{digits}f}"
    if lang == "uk":
        text = text.replace(",", "\\,").replace(".", "{,}")
    return text


def pval(p, lang):
    """Two significant decimals at p >= 0.01, four below; half-up rounding as in the text."""
    from decimal import ROUND_HALF_UP, Decimal
    quantum = Decimal("0.01") if p >= 0.1 else (Decimal("0.001") if p >= 0.01 else Decimal("0.0001"))
    text = str(Decimal(repr(p)).quantize(quantum, rounding=ROUND_HALF_UP))
    return text.replace(".", "{,}") if lang == "uk" else text


def signed(x, lang):
    text = f"{x:+.2f}"
    return text.replace(".", "{,}") if lang == "uk" else text


def sign_p(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


def cells(name, arms, tag=""):
    out = {}
    for pol, v in json.loads((BENCH / name).read_text())["policies"].items():
        if pol not in arms:
            continue
        for sr in v["seed_runs"]:
            for t in sr["tasks"]:
                out[(arms[pol], tag + t["task_id"].split("_")[0], sr["seed"])] = t
    return out


def silent(t):
    return bool(t["worker_reported_success"]) and not t["pipeline_verified"]


def task_diffs(c):
    tasks = sorted({t for (a, t, _) in c if a == "S"}, key=lambda x: (x[0], int("".join(ch for ch in x if ch.isdigit()))))
    return [(sum(c["S", t, s]["pipeline_verified"] for s in SEEDS) - sum(c["B", t, s]["pipeline_verified"] for s in SEEDS)) / 3
            for t in tasks]


def task_stats(diffs):
    w, l = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    nz, obs = [d for d in diffs if d], sum(diffs)
    hits = sum(abs(sum(d if m >> i & 1 else -d for i, d in enumerate(nz))) >= abs(obs) - 1e-12 for m in range(2 ** len(nz)))
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(20000))
    return w, l, len(diffs) - w - l, sign_p(w, l), hits / 2 ** len(nz), obs / len(diffs), boots[500], boots[19499]


def table2(lang):
    e_arms = {BASE: "B", TESTS: "S"}
    fresh = cells("spec_contract_confirm_fresh.json", e_arms)
    rest = cells("spec_contract_confirm_all.json", e_arms)
    labels = {"en": ["Fresh H14--H19 (R4)", "H1--H13 (R5)", "All 19 tasks"],
              "uk": ["Нові H14--H19 (R4)", "H1--H13 (R5)", "Усі 19 задач"]}[lang]
    rows = []
    for label, c in zip(labels, (fresh, rest, {**fresh, **rest})):
        w, l, e, sp, pp, mean, lo, hi = task_stats(task_diffs(c))
        rows.append(f"{label} & {w}/{l}/{e} & {pval(sp, lang)} & {pval(pp, lang)} & {signed(mean, lang)} ({signed(lo, lang)}, {signed(hi, lang)}) \\\\")
    return "\n".join(rows)


def campaign_rows(name, lang):
    data = json.loads((BENCH / name).read_text())["policies"]
    rows = []
    for pol in (BASE, TESTS, LIST):
        recs = [t for sr in data[pol]["seed_runs"] for t in sr["tasks"]]
        gen, tok = {}, {}
        for r in recs:
            m = r["metadata"]
            gen[r["task_id"]] = m.get("spec_tests_wall_s") or m.get("spec_checklist_wall_s") or 0
            tok[r["task_id"]] = m.get("spec_tests_tokens") or m.get("spec_checklist_tokens") or 0
        n = len(recs)
        g = "---" if not sum(gen.values()) else f"{num(sum(gen.values()), lang)} s"
        rows.append(f"{ARM_NAMES[lang][pol]} & {sum(r['pipeline_verified'] for r in recs)}/{n} & {sum(silent(r) for r in recs)}/{n} & "
                    f"{num(sum(r['wall_clock_s'] for r in recs), lang)} s & {g} & {num(sum(tok.values()), lang)} \\\\")
    return "\n".join(rows)


def table4(lang):
    a_arms = {BASE: "B", TESTS: "S"}
    pooled = {**cells("campaign_a.json", a_arms, "A"), **cells("campaign_b.json", a_arms, "N"),
              **cells("spec_contract_confirm_fresh.json", a_arms, "E"), **cells("spec_contract_confirm_all.json", a_arms, "E")}
    tasks = sorted({t for (_, t, _) in pooled})

    def total(arm, key):
        rows = [v for (a, _, _), v in pooled.items() if a == arm]
        return sum(key(v) for v in rows), len(rows)

    def task_level(bad):
        better = sum(sum(bad(pooled["S", t, s]) for s in SEEDS) < sum(bad(pooled["B", t, s]) for s in SEEDS) for t in tasks)
        worse = sum(sum(bad(pooled["S", t, s]) for s in SEEDS) > sum(bad(pooled["B", t, s]) for s in SEEDS) for t in tasks)
        return better, worse, sign_p(better, worse)

    out = []
    names = {"en": ["Silent false successes", "Verified completion"], "uk": ["Тихі хибні успіхи", "Верифіковане виконання"]}[lang]
    for name, key, bad in ((names[0], silent, silent), (names[1], lambda t: t["pipeline_verified"], lambda t: not t["pipeline_verified"])):
        (kb, n), (ks, _) = total("B", key), total("S", key)
        b, w, p = task_level(bad)
        share = lambda k: f" ({num(100 * k / n, lang)}\\%)" if key is not silent else ""
        out.append(f"{name} & {kb}/{n}{share(kb)} & \\textbf{{{ks}/{n}}}{share(ks)} & {b}/{w} ({pval(p, lang)}) \\\\")
    return "\n".join(out)


def main():
    for lang in ("en", "uk"):
        target = HERE / "tables" / lang
        target.mkdir(parents=True, exist_ok=True)
        (target / "T2.tex").write_text(table2(lang) + "\n")
        (target / "T3.tex").write_text(campaign_rows("campaign_a.json", lang) + "\n")
        (target / "T4.tex").write_text(table4(lang) + "\n")
        (target / "T5.tex").write_text(campaign_rows("campaign_b.json", lang) + "\n")
    print("tables written:", sorted(p.name for p in (HERE / "tables" / "en").iterdir()))


if __name__ == "__main__":
    main()
