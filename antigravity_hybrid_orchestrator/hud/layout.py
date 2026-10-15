"""Layout configuration for the Rich interactive terminal HUD."""

from rich.layout import Layout


def build_orchestrator_layout(terminal_width: int = 120) -> Layout:
    """Constructs the responsive hierarchical Rich Layout.

    Layout Tree:
    - Root:
      - header (size: 3)
      - body (ratio: 1)
        - left (ratio: 1)
          - cloud (ratio: 1)
          - telemetry (ratio: 1)
        - right (ratio: 2)
          - workforce (ratio: 2)
            - builder_a (ratio: 1)
            - builder_b (ratio: 1)
          - critic (ratio: 1)
      - footer (size: 8)

    If terminal width < 100, workforce panels are stacked vertically to avoid clipping.
    """
    root = Layout(name="root")

    header = Layout(name="header", size=3)
    body = Layout(name="body", ratio=1)
    footer = Layout(name="footer", size=8)

    root.split_column(header, body, footer)

    # Split body into left (Cloud & Telemetry) and right (Workforce Arena & Critic)
    left = Layout(name="left", ratio=1)
    right = Layout(name="right", ratio=2)
    body.split_row(left, right)

    # Left column: Cloud Architect on top, Telemetry Table on bottom
    left.split_column(
        Layout(name="cloud", ratio=1),
        Layout(name="telemetry", ratio=1),
    )

    # Right column: Workforce candidates on top, Blind Critic on bottom
    workforce = Layout(name="workforce", ratio=2)
    critic = Layout(name="critic", ratio=1)
    right.split_column(workforce, critic)

    # Within workforce: side-by-side if terminal is wide enough, stacked if narrow
    if terminal_width < 100:
        workforce.split_column(
            Layout(name="builder_a", ratio=1),
            Layout(name="builder_b", ratio=1),
        )
    else:
        workforce.split_row(
            Layout(name="builder_a", ratio=1),
            Layout(name="builder_b", ratio=1),
        )

    return root
