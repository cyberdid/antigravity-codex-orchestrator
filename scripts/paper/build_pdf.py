"""Build docs/paper/manuscript.pdf (and the .tex source) from docs/paper/manuscript.md.

The Markdown stays the single source; this converter covers the subset the
manuscript uses (headings, paragraphs, bold/italic/code, links, lists, pipe
tables, figures with a following "**Figure N.**" caption, the draft banner).
Usage: python3 scripts/paper/build_pdf.py   (needs `tectonic` on PATH)
"""
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/paper"
BUILD = PAPER / "build"

SYMBOLS = {"∈": r"$\in$", "∧": r"$\wedge$", "∨": r"$\vee$", "≥": r"$\geq$", "≈": r"$\approx$",
           "→": r"$\rightarrow$", "×": r"$\times$", "²": r"$^{2}$", "…": r"\ldots{}"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.4cm]{geometry}
\usepackage{fontspec}
\usepackage{graphicx,tabularx,booktabs,float,xcolor,enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\setlist{itemsep=2pt,topsep=4pt}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\renewcommand{\tabularxcolumn}[1]{m{#1}}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcommand{\code}[1]{\texttt{#1}}
\graphicspath{{./}{../}}
\begin{document}
"""


def esc(text):
    return "".join(SPECIAL.get(c, c) for c in text)


def code(text):
    out = esc(text)
    for sep in (r"\_", "/", ".", "-"):
        out = out.replace(sep, sep + r"\allowbreak{}")
    return r"\code{" + out + "}"


def inline(text):
    """Convert inline Markdown to LaTeX; code spans and links are protected first."""
    parts = []

    def stash(value):
        parts.append(value)
        return f"\x00{len(parts) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: stash(code(m.group(1))), text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  lambda m: stash(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}{" + esc(m.group(1)) + "}"), text)
    text = re.sub(r"(?<![({])(https?://\S+?)(?=[\s)]|$)", lambda m: stash(r"\url{" + m.group(1).replace("%", r"\%").replace("#", r"\#") + "}"), text)
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", text)
    for sym, rep in SYMBOLS.items():
        text = text.replace(sym, rep)
    text = re.sub(r'"([^"]*)"', r"``\1''", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: parts[int(m.group(1))], text)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, align, body = cells[0], cells[1], cells[2:]
    spec = "".join("R" if a.endswith(":") and not a.startswith(":") else "L" for a in align)
    lines = [r"\begin{center}\small", r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\toprule",
             " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(inline(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return "\n".join(lines)


def convert(md):
    lines = md.splitlines()
    out, i, title = [], 0, ""
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i][1:].strip())
                i += 1
            out.append(r"\noindent\fcolorbox{orange}{orange!8}{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule}{\small "
                       + inline(" ".join(block)) + "}}\n")
            continue
        if line.startswith("# "):
            title = inline(line[2:])
            out.append(r"\begin{center}{\LARGE\bfseries " + title + r"\par}\end{center}")
            i += 1
            continue
        if line.startswith("## "):
            out.append(r"\section*{" + inline(line[3:]) + "}")
            i += 1
            continue
        if line.startswith("### "):
            out.append(r"\subsection*{" + inline(line[4:]) + "}")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table(rows))
            continue
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            caption = ""
            if i < len(lines) and lines[i].startswith("**Figure"):
                caption = inline(lines[i])
                i += 1
            path = (PAPER / m.group(1)).with_suffix(".pdf")
            # Natural size x 1.25 (figures are drawn at print size), capped at the text width (~6.1 in).
            import struct
            png = (PAPER / m.group(1)).read_bytes()
            inches = min(6.1, struct.unpack(">I", png[16:20])[0] / 300 * 1.25)
            out.append(r"\begin{figure}[tbp]\centering\includegraphics[width=" + f"{inches:.2f}in" + r",height=0.42\textheight,keepaspectratio]{"
                       + str(path.relative_to(PAPER)) + r"}\par\vspace{4pt}{\small " + caption + r"\par}\end{figure}")
            continue
        if re.match(r"\s*(- |\d+\. )", line):
            ordered = bool(re.match(r"\s*\d+\. ", line))
            env = "enumerate" if ordered else "itemize"
            items = []
            while i < len(lines) and re.match(r"\s*(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^\s*(- |\d+\. )", "", lines[i]))
                i += 1
            out.append(r"\begin{" + env + "}\n" + "\n".join(r"\item " + inline(t) for t in items) + "\n" + r"\end{" + env + "}")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"(#|\||!\[|> |\s*- |\s*\d+\. )", lines[i]):
            para.append(inline(lines[i].rstrip()) + (r"\newline" if lines[i].endswith("  ") else ""))
            i += 1
        out.append("\n".join(para) + "\n")
    return PREAMBLE + "\n".join(out) + "\n\\end{document}\n"


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    tex = BUILD / "manuscript.tex"
    tex.write_text(convert((PAPER / "manuscript.md").read_text()))
    subprocess.run(["tectonic", "--keep-logs", "--outdir", str(BUILD), str(tex)], check=True)
    shutil.copy(BUILD / "manuscript.pdf", PAPER / "manuscript.pdf")
    shutil.copy(tex, PAPER / "manuscript.tex")
    print("written", PAPER / "manuscript.pdf")


if __name__ == "__main__":
    main()
