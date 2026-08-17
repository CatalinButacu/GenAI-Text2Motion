from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SRC = Path("src/text2motion")
FIG = Path("paper/figures")
DIAGRAMS = Path("paper/diagrams")

WIDTH, HEIGHT = 900, 570
INK = "#111417"
GREY = "#8b949e"
ACCENT = "#35617f"
SURFACE = "#ffffff"
SUNK = "#eef1f4"
ACCENT_SOFT = "#e3edf3"
FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]

LAYERS = ["motion", "tokenization", "generation", "evaluation", "streaming", "studio", "app"]

NODES = {
    "app": (365, 30, 170, 54, "composition root / CLI", ACCENT_SOFT, ACCENT),
    "evaluation": (90, 150, 170, 54, "FID / R-precision", SURFACE, INK),
    "streaming": (365, 150, 170, 54, "decoder / service", SURFACE, INK),
    "studio": (640, 150, 170, 54, "aitviewer", SURFACE, INK),
    "generation": (228, 280, 170, 54, "mamba / transformer", SURFACE, INK),
    "tokenization": (228, 390, 170, 54, "FSQ / RVQ", SURFACE, INK),
    "motion": (
        60,
        490,
        784,
        54,
        "263 representation / kinematics / dataset / preparation",
        SUNK,
        INK,
    ),
}

ROUTES = {
    ("evaluation", "generation"): [(175, 204), (250, 280)],
    ("evaluation", "motion"): [(140, 204), (140, 490)],
    ("streaming", "generation"): [(420, 204), (383, 280)],
    ("streaming", "tokenization"): [(450, 204), (450, 417), (402, 417)],
    ("streaming", "motion"): [(490, 204), (490, 490)],
    ("studio", "motion"): [(725, 204), (725, 490)],
    ("generation", "tokenization"): [(313, 334), (313, 386)],
    ("generation", "motion"): [(228, 307), (190, 307), (190, 490)],
    ("tokenization", "motion"): [(313, 444), (313, 490)],
    ("app", "evaluation"): [(405, 84), (205, 150)],
    ("app", "streaming"): [(450, 84), (450, 146)],
    ("app", "studio"): [(495, 84), (695, 150)],
    ("app", "generation"): [(380, 84), (330, 276)],
    ("app", "tokenization"): [(365, 57), (52, 57), (52, 413), (224, 413)],
    ("app", "motion"): [(535, 57), (866, 57), (866, 513), (844, 513)],
}


def package_edges() -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        owner = path.relative_to(SRC).parts[0]
        if owner not in LAYERS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            parts = node.module.split(".")
            if len(parts) > 1 and parts[0] == "text2motion" and parts[1] in LAYERS:
                if parts[1] != owner:
                    edges.add((owner, parts[1]))
    return edges


def verify(edges: set[tuple[str, str]]) -> None:
    rank = {name: index for index, name in enumerate(LAYERS)}
    upward = [(a, b) for a, b in edges if rank[b] > rank[a]]
    if upward:
        raise SystemExit(f"layering violated, cannot draw a layered figure: {sorted(upward)}")

    drawn = set(ROUTES)
    if drawn != edges:
        raise SystemExit(
            "figure is stale -- the drawn graph no longer matches the code.\n"
            f"  in code but undrawn: {sorted(edges - drawn)}\n"
            f"  drawn but not in code: {sorted(drawn - edges)}"
        )


def fan_in(edges: set[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _, dst in edges:
        counts[dst] += 1
    return counts


def write_svg(path: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="Segoe UI, DejaVu Sans, sans-serif">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
        "<defs>",
        f'<marker id="g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{GREY}"/></marker>',
        f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{ACCENT}"/></marker>',
        "</defs>",
    ]

    for (src, _dst), points in ROUTES.items():
        accent = src == "app"
        d = "M " + " L ".join(f"{x} {y}" for x, y in points)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{ACCENT if accent else GREY}" '
            f'stroke-width="{1.7 if accent else 1.3}" marker-end="url(#{"a" if accent else "g"})"/>'
        )

    for name, (x, y, w, h, subtitle, fill, stroke) in NODES.items():
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{1.5 if name in ("app", "motion") else 1.2}"/>'
        )
        cx = x + w / 2
        parts.append(
            f'<text x="{cx}" y="{y + 23}" text-anchor="middle" font-size="14" fill="{INK}" '
            f'font-family="Consolas, monospace">{name}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{y + 40}" text-anchor="middle" font-size="10.5" fill="{GREY}">'
            f"{subtitle}</text>"
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", path)


def write_png(path: Path) -> None:
    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    for (src, _dst), points in ROUTES.items():
        accent = src == "app"
        color = ACCENT if accent else GREY
        width = 1.7 if accent else 1.3
        for start, end in zip(points[:-2], points[1:-1], strict=False):
            ax.plot(*zip(start, end, strict=False), color=color, lw=width, solid_capstyle="round")
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=11,
                color=color,
                lw=width,
                shrinkA=0,
                shrinkB=0,
            )
        )

    for name, (x, y, w, h, subtitle, fill, stroke) in NODES.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=4",
                facecolor=fill,
                edgecolor=stroke,
                lw=1.5 if name in ("app", "motion") else 1.2,
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y + 21,
            name,
            ha="center",
            va="center",
            fontsize=10.5,
            family="monospace",
            color=INK,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + 38,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.6,
            color=GREY,
            zorder=4,
        )

    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def write_dot(path: Path, edges: set[tuple[str, str]]) -> None:
    lines = [
        "// Package dependency graph. GENERATED by scripts/figures/architecture_diagram.py",
        "// -- edges are derived from the source AST, so this file cannot drift from the code.",
        "digraph architecture {",
        "    rankdir=TB;",
        '    graph [fontname="Segoe UI", fontsize=12, labelloc="t", '
        'label="Package dependencies: an arrow points at the dependency"];',
        '    node  [shape=box, style="rounded,filled", fontname="Segoe UI", fontsize=10, '
        'fillcolor="#ffffff"];',
        '    edge  [fontname="Segoe UI", fontsize=9, color="#8b949e"];',
        "",
        f'    app [fillcolor="{ACCENT_SOFT}", color="{ACCENT}"];',
        f'    motion [fillcolor="{SUNK}"];',
        "",
    ]
    for src, dst in sorted(edges):
        style = f' [color="{ACCENT}", penwidth=1.5]' if src == "app" else ""
        lines.append(f"    {src} -> {dst}{style};")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path)


def main() -> None:
    edges = package_edges()
    verify(edges)

    FIG.mkdir(parents=True, exist_ok=True)
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    write_svg(FIG / "architecture.svg")
    write_png(FIG / "architecture.png")
    write_dot(DIAGRAMS / "architecture.dot", edges)

    counts = fan_in(edges)
    print(
        f"\n{len(NODES)} packages, {len(edges)} edges, 0 upward. "
        f"motion fan-in {counts['motion']}/{len(NODES) - 1}, "
        f"app fan-out {sum(1 for a, _ in edges if a == 'app')}/{len(NODES) - 1}."
    )


if __name__ == "__main__":
    main()
