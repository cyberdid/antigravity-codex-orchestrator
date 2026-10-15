"""Cyberpunk neon theme for the R&D terminal HUD.

Centralizes the palette, box styles, status glyphs and small text helpers so the
whole dashboard reads as one neon-on-black system.
"""

from __future__ import annotations

from rich import box
from rich.text import Text

# --- Neon palette -----------------------------------------------------------
CYAN = "#00f0ff"       # primary / structure
MAGENTA = "#ff2bd6"    # tools / actions
GREEN = "#39ff14"      # success / go
AMBER = "#ffb000"      # verifying / caution
PINK = "#ff3864"       # failure / stop
PURPLE = "#b026ff"     # blocked / cloud
BLUE = "#2de2e6"       # secondary accent
DIM = "#5c6f7e"        # muted labels
GHOST = "#8b9bb4"      # secondary text

# Neon frames
BOX = box.DOUBLE_EDGE
BOX_HEAVY = box.HEAVY

# --- Status glyphs + styles -------------------------------------------------
_STATUS = {
    "pending": ("▹", DIM),
    "running": ("▶", CYAN),
    "verifying": ("⟳", AMBER),
    "repairing": ("↺", MAGENTA),
    "completed": ("✔", GREEN),
    "complete": ("✔", GREEN),
    "success": ("✔", GREEN),
    "failed": ("✖", PINK),
    "error": ("✖", PINK),
    "blocked": ("⊘", PURPLE),
    "streaming": ("▪", CYAN),
    "thinking": ("◈", BLUE),
    "generating": ("▪", CYAN),
    "idle": ("·", DIM),
    "running_pipeline": ("▶", CYAN),
    "scanning": ("◈", BLUE),
    "planning": ("◈", PURPLE),
    "executing": ("▶", CYAN),
    "complete_pipeline": ("✔", GREEN),
    "running ": ("▶", CYAN),
}


def neon_badge(status: str) -> Text:
    """Returns a neon glyph+label badge for a status string."""
    key = (status or "").strip().lower()
    glyph, color = _STATUS.get(key, ("◆", GHOST))
    label = key.upper() if key else "—"
    return Text.assemble((f"{glyph} ", f"bold {color}"), (label, color))


def neon_rule(char: str = "═", color: str = CYAN, width: int = 78) -> Text:
    """A neon horizontal rule."""
    return Text(char * width, style=f"bold {color}")


def ok_fail(success: bool) -> Text:
    """A compact OK/FAIL neon marker."""
    return Text("◉ OK", style=f"bold {GREEN}") if success else Text("◉ FAIL", style=f"bold {PINK}")


# A compact neon wordmark for headers/summaries.
WORDMARK = "▛▀▖ ANTIGRAVITY ∙ R&D ENGINE ▗▄▟"
