#!/usr/bin/env python3
"""Build manuscript-ready C1 materials from locked B6/B7 results.

The script never reads prediction files or CAID labels.  It consumes only the
already locked aggregate/statistical outputs from B6 and B7.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
B6 = (
    ROOT
    / "artifacts/b6_locked_results_20260826/outputs/analysis/"
    "b6_paired_model_statistics/formal"
)
B7 = (
    ROOT
    / "artifacts/b7_locked_results_20260826/outputs/analysis/"
    "b7_error_complementarity/formal"
)

MODEL_ORDER = [
    "A10-locked",
    "LoRA-DR-Suite 650M",
    "PUNCH2-Light Paper-8",
    "PUNCH2-Light Released-13",
]
COMPARATOR_ORDER = [
    "LoRA-DR-Suite 650M",
    "PUNCH2-Light Released-13",
    "PUNCH2-Light Paper-8",
]
COLORS = {
    "A10-locked": "#0072B2",
    "LoRA-DR-Suite 650M": "#D55E00",
    "PUNCH2-Light Paper-8": "#8A8A8A",
    "PUNCH2-Light Released-13": "#009E73",
}
TRACK_ORDER = [
    ("CAID2", "disorder_nox"),
    ("CAID2", "disorder_pdb"),
    ("CAID3", "disorder_nox"),
    ("CAID3", "disorder_pdb"),
]
TRACK_LABELS = {
    ("CAID2", "disorder_nox"): "CAID2\nNOX",
    ("CAID2", "disorder_pdb"): "CAID2\nPDB",
    ("CAID3", "disorder_nox"): "CAID3\nNOX",
    ("CAID3", "disorder_pdb"): "CAID3\nPDB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/paper/c1_manuscript_materials",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value).replace("|", "\\|")

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{html.escape(title)}</title>",
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
            '.axis{stroke:#333;stroke-width:1.2}.grid{stroke:#D9D9D9;stroke-width:1}'
            '.small{font-size:13px}.label{font-size:15px}.title{font-size:22px;font-weight:700}'
            '.panel{font-size:18px;font-weight:700}</style>',
        ]

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        size: int = 14,
        anchor: str = "start",
        weight: str = "400",
        fill: str = "#222222",
        rotate: float | None = None,
    ) -> None:
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"'
            f'{transform}>{html.escape(str(value))}</text>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#333333",
        width: float = 1.2,
        dash: str | None = None,
    ) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1,
        radius: float = 0,
    ) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'rx="{radius:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str = "#FFFFFF",
        stroke_width: float = 1.5,
    ) -> None:
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def polygon(self, points: list[tuple[float, float]], *, fill: str) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(f'<polygon points="{encoded}" fill="{fill}"/>')

    def save(self, path: Path) -> None:
        write_text(path, "\n".join(self.parts + ["</svg>"]))


def interpolate_color(value: float, limit: float = 0.12) -> str:
    if not math.isfinite(value):
        return "#E6E6E6"
    value = max(-limit, min(limit, value)) / limit
    neutral = (247, 247, 247)
    target = (33, 102, 172) if value > 0 else (178, 24, 43)
    fraction = abs(value)
    rgb = tuple(round(neutral[i] + fraction * (target[i] - neutral[i])) for i in range(3))
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def draw_architecture(path: Path) -> None:
    svg = Svg(1500, 690, "Locked A10 cross-architecture ensemble")
    svg.text(750, 42, "Locked A10 cross-architecture ensemble", size=25, anchor="middle", weight="700")
    svg.text(750, 70, "Frozen ESM-2 representations; 30 label-free equal-logit members", size=15, anchor="middle", fill="#555")

    def box(x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> None:
        svg.rect(x, y, w, h, fill="#FFFFFF", stroke=color, stroke_width=2.5, radius=12)
        svg.text(x + w / 2, y + 33, title, size=18, anchor="middle", weight="700", fill=color)
        subtitle_lines = subtitle.split("\n")
        start_y = y + 63 - (len(subtitle_lines) - 1) * 9
        for line_index, line in enumerate(subtitle_lines):
            svg.text(
                x + w / 2,
                start_y + line_index * 20,
                line,
                size=14,
                anchor="middle",
                fill="#444",
            )

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#555") -> None:
        svg.line(x1, y1, x2, y2, stroke=color, width=2.2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 10
        points = [
            (x2, y2),
            (x2 - size * math.cos(angle - 0.55), y2 - size * math.sin(angle - 0.55)),
            (x2 - size * math.cos(angle + 0.55), y2 - size * math.sin(angle + 0.55)),
        ]
        svg.polygon(points, fill=color)

    box(55, 245, 210, 92, "Protein sequence", "one residue → one score", "#333333")
    box(345, 225, 260, 132, "ESM-2 650M", "frozen backbone\nshared extraction", "#6A3D9A")
    arrow(265, 291, 345, 291)

    box(700, 125, 300, 120, "A2 family", "final layer (33) + multiscale\ncontext kernels 3/7/15/31", COLORS["A10-locked"])
    box(700, 365, 300, 120, "A8 family", "learned mix of layers 30–33\n+ multiscale context", COLORS["PUNCH2-Light Released-13"])
    arrow(605, 270, 700, 185, COLORS["A10-locked"])
    arrow(605, 312, 700, 425, COLORS["PUNCH2-Light Released-13"])

    box(1080, 125, 240, 120, "15 A2 members", "3 seeds × 5 folds\nindependent heads", COLORS["A10-locked"])
    box(1080, 365, 240, 120, "15 A8 members", "3 seeds × 5 folds\nindependent heads", COLORS["PUNCH2-Light Released-13"])
    arrow(1000, 185, 1080, 185, COLORS["A10-locked"])
    arrow(1000, 425, 1080, 425, COLORS["PUNCH2-Light Released-13"])

    box(1090, 555, 300, 88, "Uniform logit mean", "30 logits × 1/30 → sigmoid", "#CC79A7")
    arrow(1200, 245, 1200, 555, "#CC79A7")
    arrow(1230, 485, 1230, 555, "#CC79A7")

    svg.rect(45, 560, 760, 82, fill="#FFF7E6", stroke="#E69F00", stroke_width=1.8, radius=10)
    svg.text(65, 591, "Locking rule", size=17, weight="700", fill="#A56500")
    svg.text(65, 619, "Architecture, checkpoints, and 1/30 weights fixed before CAID2/CAID3 label access.", size=15)
    svg.save(path)


def draw_external_performance(point: pd.DataFrame, path: Path) -> None:
    svg = Svg(1500, 700, "External CAID2/CAID3 performance")
    svg.text(750, 40, "Locked external-test performance", size=24, anchor="middle", weight="700")
    panels = [
        ("ROC-AUC", "roc_auc_full_precision", 0.30, 1.00),
        ("AUPRC", "aucpr_trapezoid_full_precision", 0.25, 1.00),
    ]
    for panel_index, (title, metric, ymin, ymax) in enumerate(panels):
        left = 90 + panel_index * 735
        top, width, height = 105, 620, 470
        svg.text(left, 82, chr(65 + panel_index), size=20, weight="700")
        svg.text(left + width / 2, 82, title, size=19, anchor="middle", weight="700")
        for tick in [ymin + (ymax - ymin) * i / 5 for i in range(6)]:
            y = top + height - (tick - ymin) / (ymax - ymin) * height
            svg.line(left, y, left + width, y, stroke="#DDDDDD", width=1)
            svg.text(left - 12, y + 5, f"{tick:.2f}", size=13, anchor="end", fill="#555")
        svg.line(left, top, left, top + height, stroke="#333", width=1.4)
        svg.line(left, top + height, left + width, top + height, stroke="#333", width=1.4)
        group_width = width / len(TRACK_ORDER)
        offsets = [-22.5, -7.5, 7.5, 22.5]
        for track_index, (dataset, track) in enumerate(TRACK_ORDER):
            x0 = left + group_width * (track_index + 0.5)
            for model_index, model in enumerate(MODEL_ORDER):
                row = point[(point.dataset == dataset) & (point.track == track) & (point.model == model)].iloc[0]
                value = float(row[metric])
                y = top + height - (value - ymin) / (ymax - ymin) * height
                x = x0 + offsets[model_index]
                svg.line(x, top + height, x, y, stroke=COLORS[model], width=2)
                svg.circle(x, y, 6.5, fill=COLORS[model])
            for line_index, label in enumerate(TRACK_LABELS[(dataset, track)].split("\n")):
                svg.text(x0, top + height + 26 + line_index * 17, label, size=14, anchor="middle", weight="600")

    legend_y = 652
    legend_x = 255
    for model in MODEL_ORDER:
        svg.circle(legend_x, legend_y, 6.5, fill=COLORS[model])
        svg.text(legend_x + 13, legend_y + 5, model, size=14)
        legend_x += 295
    svg.save(path)


def parse_ci(value: str) -> tuple[float, float]:
    lower, upper = ast.literal_eval(value)
    return float(lower), float(upper)


def draw_forest(pairwise: pd.DataFrame, path: Path) -> None:
    data = pairwise[pairwise.metric == "roc_auc_full_precision"].copy()
    data["track_order"] = data.apply(lambda row: TRACK_ORDER.index((row.dataset, row.track)), axis=1)
    data["comparator_order"] = data.second_model.map(COMPARATOR_ORDER.index)
    data = data.sort_values(["track_order", "comparator_order"])
    rows = list(data.itertuples(index=False))
    svg = Svg(1650, 885, "Paired protein-bootstrap ROC-AUC differences")
    svg.text(825, 38, "Paired whole-protein bootstrap: A10 minus comparator", size=24, anchor="middle", weight="700")
    plot_left, plot_right, top, row_h = 650, 1370, 105, 56
    xmin, xmax = -0.11, 0.06

    def xmap(value: float) -> float:
        return plot_left + (value - xmin) / (xmax - xmin) * (plot_right - plot_left)

    for tick in [-0.10, -0.05, 0.00, 0.05]:
        x = xmap(tick)
        svg.line(x, top - 25, x, top + row_h * len(rows), stroke="#D0D0D0", width=1, dash="4 4" if tick else None)
        svg.text(x, top + row_h * len(rows) + 27, f"{tick:+.2f}", size=14, anchor="middle")
    svg.line(xmap(0), top - 25, xmap(0), top + row_h * len(rows), stroke="#222", width=2)
    svg.text((plot_left + plot_right) / 2, 850, "Full-precision ROC-AUC difference", size=16, anchor="middle", weight="600")
    svg.text(100, 78, "External track", size=15, weight="700")
    svg.text(300, 78, "Comparator", size=15, weight="700")
    svg.text(1410, 78, "Estimate [95% CI]", size=15, weight="700")

    last_track = None
    for index, row in enumerate(rows):
        y = top + index * row_h
        track_key = (row.dataset, row.track)
        if track_key != last_track:
            if index:
                svg.line(80, y - row_h / 2, 1570, y - row_h / 2, stroke="#BDBDBD", width=1.5)
            svg.text(100, y + 5, TRACK_LABELS[track_key].replace("\n", " "), size=15, weight="700")
            last_track = track_key
        color = COLORS[row.second_model]
        label = row.second_model.replace("PUNCH2-Light ", "PUNCH2-Light ")
        svg.text(300, y + 5, label, size=14)
        lower, upper = parse_ci(row.percentile_95_ci)
        svg.line(xmap(lower), y, xmap(upper), y, stroke=color, width=3)
        svg.line(xmap(lower), y - 6, xmap(lower), y + 6, stroke=color, width=2)
        svg.line(xmap(upper), y - 6, xmap(upper), y + 6, stroke=color, width=2)
        marker_radius = 8 if float(row.bootstrap_holm_p) < 0.05 else 6
        svg.circle(xmap(float(row.point_delta)), y, marker_radius, fill=color)
        suffix = " *" if float(row.bootstrap_holm_p) < 0.05 else ""
        svg.text(1410, y + 5, f"{row.point_delta:+.3f} [{lower:+.3f}, {upper:+.3f}]{suffix}", size=14)
    svg.text(100, 850, "* Holm-adjusted protein-bootstrap p < 0.05; Paper-8 belongs to a separate supplementary family.", size=13, fill="#555")
    svg.save(path)


def delta_frame(subgroup: pd.DataFrame, comparator: str) -> pd.DataFrame:
    selected = subgroup[subgroup.stratification == "known_label_disorder_fraction"]
    a10 = selected[selected.model == "A10-locked"]
    other = selected[selected.model == comparator]
    merged = a10.merge(
        other,
        on=["dataset", "track", "stratification", "stratum"],
        suffixes=("_a10", "_other"),
    )
    merged["delta"] = merged.roc_auc_full_precision_a10 - merged.roc_auc_full_precision_other
    return merged


def draw_heatmap(subgroup: pd.DataFrame, path: Path) -> None:
    comparators = ["LoRA-DR-Suite 650M", "PUNCH2-Light Released-13"]
    strata = ["0-0.10", "(0.10,0.30]", "(0.30,0.60]", "(0.60,1.00]"]
    svg = Svg(1500, 690, "Subgroup ROC-AUC differences by disorder fraction")
    svg.text(750, 40, "Descriptive subgroup ROC-AUC differences (A10 minus comparator)", size=23, anchor="middle", weight="700")
    for panel_index, comparator in enumerate(comparators):
        data = delta_frame(subgroup, comparator)
        left = 160 + panel_index * 700
        top, cell_w, cell_h = 140, 130, 92
        svg.text(left + 2 * cell_w, 88, comparator, size=18, anchor="middle", weight="700", fill=COLORS[comparator])
        for column, stratum in enumerate(strata):
            svg.text(left + (column + 0.5) * cell_w, 122, stratum, size=14, anchor="middle")
        for row_index, (dataset, track) in enumerate(TRACK_ORDER):
            if panel_index == 0:
                svg.text(left - 15, top + (row_index + 0.55) * cell_h, TRACK_LABELS[(dataset, track)].replace("\n", " "), size=15, anchor="end", weight="600")
            for column, stratum in enumerate(strata):
                match = data[(data.dataset == dataset) & (data.track == track) & (data.stratum == stratum)]
                value = float(match.delta.iloc[0]) if len(match) else float("nan")
                x, y = left + column * cell_w, top + row_index * cell_h
                fill = interpolate_color(value)
                svg.rect(x, y, cell_w - 4, cell_h - 4, fill=fill, stroke="#FFFFFF", stroke_width=2)
                text_fill = "#FFFFFF" if math.isfinite(value) and abs(value) > 0.075 else "#222222"
                label = "NA" if not math.isfinite(value) else f"{value:+.3f}"
                svg.text(x + (cell_w - 4) / 2, y + cell_h / 2 + 6, label, size=16, anchor="middle", weight="700", fill=text_fill)
    legend_left, legend_top, legend_w = 525, 570, 450
    steps = 60
    for index in range(steps):
        value = -0.12 + 0.24 * index / (steps - 1)
        svg.rect(legend_left + legend_w * index / steps, legend_top, legend_w / steps + 1, 20, fill=interpolate_color(value))
    svg.text(legend_left, legend_top + 43, "−0.12 (comparator higher)", size=13, anchor="start")
    svg.text(legend_left + legend_w, legend_top + 43, "+0.12 (A10 higher)", size=13, anchor="end")
    svg.text(750, 655, "Descriptive only; no new subgroup hypothesis tests were performed.", size=14, anchor="middle", fill="#555")
    svg.save(path)


def draw_segment_profiles(segment: pd.DataFrame, path: Path) -> None:
    models = ["A10-locked", "LoRA-DR-Suite 650M", "PUNCH2-Light Released-13"]
    bins = ["1-15", "16-30", "31-100", "101+"]
    states = ["disordered", "ordered"]
    svg = Svg(1740, 900, "Segment-level threshold error profiles")
    svg.text(870, 38, "Descriptive segment-level error profiles at the fixed 0.5 threshold", size=23, anchor="middle", weight="700")
    panel_w, panel_h = 350, 300
    left0, top0 = 150, 105
    col_gap, row_gap = 45, 95
    for state_index, state in enumerate(states):
        for track_index, (dataset, track) in enumerate(TRACK_ORDER):
            left = left0 + track_index * (panel_w + col_gap)
            top = top0 + state_index * (panel_h + row_gap)
            if state_index == 0:
                svg.text(left + panel_w / 2, top - 26, TRACK_LABELS[(dataset, track)].replace("\n", " "), size=17, anchor="middle", weight="700")
            if track_index == 0:
                svg.text(left - 90, top + panel_h / 2, state.capitalize(), size=17, anchor="middle", weight="700", rotate=-90)
            for tick in [0, 0.25, 0.5, 0.75, 1.0]:
                y = top + panel_h - tick * panel_h
                svg.line(left, y, left + panel_w, y, stroke="#E0E0E0", width=1)
                if track_index == 0:
                    svg.text(left - 10, y + 5, f"{tick:.2f}", size=12, anchor="end", fill="#555")
            svg.line(left, top, left, top + panel_h, stroke="#333", width=1.3)
            svg.line(left, top + panel_h, left + panel_w, top + panel_h, stroke="#333", width=1.3)
            x_positions = [left + panel_w * (index + 0.5) / len(bins) for index in range(len(bins))]
            for model in models:
                points: list[tuple[float, float]] = []
                for bin_index, bin_name in enumerate(bins):
                    match = segment[
                        (segment.dataset == dataset)
                        & (segment.track == track)
                        & (segment.true_state == state)
                        & (segment.segment_length_bin == bin_name)
                        & (segment.model == model)
                    ]
                    if len(match):
                        value = float(match.error_rate_at_0_5.iloc[0])
                        points.append((x_positions[bin_index], top + panel_h - value * panel_h))
                for first, second in zip(points, points[1:]):
                    svg.line(first[0], first[1], second[0], second[1], stroke=COLORS[model], width=2.4)
                for x, y in points:
                    svg.circle(x, y, 5.5, fill=COLORS[model])
            for x, label in zip(x_positions, bins):
                svg.text(x, top + panel_h + 24, label, size=12, anchor="middle")
    legend_x, legend_y = 470, 865
    for model in models:
        svg.line(legend_x, legend_y, legend_x + 30, legend_y, stroke=COLORS[model], width=3)
        svg.circle(legend_x + 15, legend_y, 5.5, fill=COLORS[model])
        svg.text(legend_x + 40, legend_y + 5, model, size=14)
        legend_x += 320
    svg.save(path)


def build_narratives(point: pd.DataFrame, pairwise: pd.DataFrame, subgroup: pd.DataFrame, segment: pd.DataFrame, out: Path) -> None:
    def metric(dataset: str, track: str, model: str, column: str) -> float:
        row = point[(point.dataset == dataset) & (point.track == track) & (point.model == model)]
        return float(row.iloc[0][column])

    def roc_pair(dataset: str, track: str, comparator: str) -> pd.Series:
        row = pairwise[
            (pairwise.dataset == dataset)
            & (pairwise.track == track)
            & (pairwise.second_model == comparator)
            & (pairwise.metric == "roc_auc_full_precision")
        ]
        return row.iloc[0]

    def ci_text(row: pd.Series) -> str:
        lower, upper = parse_ci(row.percentile_95_ci)
        return f"[{lower:+.3f}, {upper:+.3f}]"

    results = f"""# C1 Results draft

## Locked model and external evaluation

A10 was finalized before access to CAID2 or CAID3 evaluation labels. The model is a 30-member ensemble comprising two frozen-ESM-2-650M architectures, three random seeds, and five cross-validation folds. Fifteen A2 members use the final ESM-2 residue representation with a multiscale context head, whereas fifteen A8 members learn a scalar mixture of layers 30–33 before the same contextual prediction stage. Member logits are combined with fixed equal weights (1/30), followed by a sigmoid to produce one classic intrinsic-disorder probability per residue (Fig. 1).

## External performance

Across the two PDB-derived tracks, A10 achieved ROC-AUC values of {metric('CAID2','disorder_pdb','A10-locked','roc_auc_full_precision'):.3f} on CAID2 and {metric('CAID3','disorder_pdb','A10-locked','roc_auc_full_precision'):.3f} on CAID3, with corresponding AUPRC values of {metric('CAID2','disorder_pdb','A10-locked','aucpr_trapezoid_full_precision'):.3f} and {metric('CAID3','disorder_pdb','A10-locked','aucpr_trapezoid_full_precision'):.3f}. Performance on the NOX tracks was lower and dataset-dependent: ROC-AUC was {metric('CAID2','disorder_nox','A10-locked','roc_auc_full_precision'):.3f} on CAID2 and {metric('CAID3','disorder_nox','A10-locked','roc_auc_full_precision'):.3f} on CAID3, while AUPRC was {metric('CAID2','disorder_nox','A10-locked','aucpr_trapezoid_full_precision'):.3f} and {metric('CAID3','disorder_nox','A10-locked','aucpr_trapezoid_full_precision'):.3f}, respectively (Table 1; Fig. 2).

The contrast with the reproduced baselines was track-specific. On CAID2-PDB, A10 exceeded LoRA-DR-Suite 650M by {roc_pair('CAID2','disorder_pdb','LoRA-DR-Suite 650M').point_delta:+.3f} ROC-AUC (95% protein-bootstrap CI {ci_text(roc_pair('CAID2','disorder_pdb','LoRA-DR-Suite 650M'))}; Holm-adjusted bootstrap p={roc_pair('CAID2','disorder_pdb','LoRA-DR-Suite 650M').bootstrap_holm_p:.3f}). On CAID3-PDB, the corresponding difference was {roc_pair('CAID3','disorder_pdb','LoRA-DR-Suite 650M').point_delta:+.3f} (95% CI {ci_text(roc_pair('CAID3','disorder_pdb','LoRA-DR-Suite 650M'))}; adjusted p={roc_pair('CAID3','disorder_pdb','LoRA-DR-Suite 650M').bootstrap_holm_p:.3f}). In contrast, LoRA-DR-Suite was higher on CAID2-NOX by {abs(roc_pair('CAID2','disorder_nox','LoRA-DR-Suite 650M').point_delta):.3f}, whereas the CAID3-NOX difference was smaller and its protein-bootstrap interval crossed zero. A10 and PUNCH2-Light Released-13 had overlapping protein-bootstrap intervals on all four tracks, supporting comparable rather than universally superior performance (Fig. 3; Table 2).

## Error structure and complementarity

Post-lock descriptive analyses localized the principal performance differences. A10 showed its lowest relative performance in proteins with 0–10% known disordered residues, including ROC-AUC differences of approximately −0.090 versus LoRA-DR-Suite on CAID2-NOX and −0.058 on CAID3-NOX. The direction reversed in several higher-disorder strata, including a +0.105 difference versus LoRA-DR-Suite in the CAID2-NOX 30–60% stratum and +0.085 in the CAID3-NOX >60% stratum (Fig. 4). These subgroup results were descriptive and were not subjected to additional hypothesis tests.

For true disordered segments of at least 101 residues, A10's fixed-threshold error rate ranged from roughly 0.047 to 0.072 across the four external tracks, compared with approximately 0.226–0.311 for LoRA-DR-Suite and 0.120–0.133 for PUNCH2-Light Released-13. Conversely, A10 produced more false positives in long ordered regions on NOX, consistent with its positive calibration bias on CAID2-NOX (mean predicted score {metric('CAID2','disorder_nox','A10-locked','mean_score'):.3f} vs prevalence {metric('CAID2','disorder_nox','A10-locked','prevalence'):.3f}). Thus, the ensemble's main advantage was sensitivity to extended disorder, while its principal weakness was overprediction in low-disorder and extended ordered contexts (Fig. 5).

Thresholded predictions also remained complementary. Relative to LoRA-DR-Suite, A10 alone was correct for 8.8% and 7.8% of residues on the CAID2-PDB and CAID3-PDB tracks, respectively; the comparator alone was correct for 2.6% and 4.3%. On the NOX tracks this pattern reversed, with LoRA-DR-Suite-only correctness exceeding A10-only correctness. These observations explain why cross-track aggregate claims obscure meaningful biological-regime differences.
"""
    write_text(out / "C1_RESULTS_DRAFT.md", results)

    discussion = """# C1 Discussion draft

## Main finding

The locked A10 ensemble is not uniformly superior across every definition of disorder. Its value is more specific and scientifically interpretable: a leakage-audited cross-architecture ESM-2 ensemble that is particularly effective for PDB-derived disorder and extended disordered regions, while remaining competitive with the reproduced PUNCH2-Light release across four independent track/dataset combinations.

## Why the architecture may help

Two sources of representation diversity are combined. A2 emphasizes the final ESM-2 layer and explicitly aggregates local contexts at four scales. A8 exposes the predictor to a learned mixture of the last four protein-language-model layers. Averaging across architectures, folds, and seeds at the logit level reduces dependence on a single representation, split, or initialization. The strong results for long disordered segments are consistent with—although they do not by themselves prove—the value of combining multiscale local evidence with complementary layer mixtures.

## Track-specific behavior

The discrepancy between PDB and NOX tracks is central rather than incidental. PDB-derived labels favor A10, whereas NOX reveals a high-score bias in proteins dominated by ordered residues. This suggests that the current ensemble separates extended disorder well but is less well calibrated when disorder prevalence is low or when long ordered segments contain locally disorder-like sequence features. Because thresholds and weights were locked before CAID label access, these findings diagnose the model rather than define a post-hoc correction.

## Comparison with reproduced baselines

LoRA-DR-Suite and PUNCH2-Light provide mechanistically distinct comparators. LoRA adaptation of the ESM-2 backbone is stronger on NOX but weaker on both PDB tracks. PUNCH2-Light, which uses ProtT5 and one-hot features, is close to A10 overall and highly score-correlated, yet the models retain residue-level disagreement. This combination of comparable aggregate performance and non-identical errors motivates future label-free stacking or distillation, but such a model must be named and locked as a successor rather than reported as a modified A10.

## Future model development

The next model should target the identified failure regime using only training-set cross-validation: calibration regularization, stronger negative/ordered-region sampling, boundary-aware losses, or a lightweight gate that reduces overprediction in low-disorder proteins. A distilled single-head model could reduce the cost of the 30-member ensemble. Any such change requires a new preregistered model identifier and an untouched external evaluation protocol.
"""
    write_text(out / "C1_DISCUSSION_DRAFT.md", discussion)

    limitations = """# C1 Limitations

1. **Benchmark scope.** The locked external evidence is restricted to CAID2 and CAID3 classic-disorder tracks; conclusions should not be generalized to every disorder definition, binding disorder, or soft disorder.
2. **Track heterogeneity.** NOX and PDB labels represent different operational definitions. A single pooled performance claim would hide the observed regime dependence.
3. **Model cost.** A10 uses 30 prediction heads, although ESM-2 representation extraction is shared. Runtime and storage remain greater than for a single-head predictor.
4. **Calibration.** Fixed-threshold and Brier results reveal overprediction on low-prevalence NOX data. No external-label recalibration was permitted after locking.
5. **Comparator scope.** Confirmatory paired statistics cover the reproduced LoRA-DR-Suite and PUNCH2-Light variants. Broader official-server rankings are contextual and are not equivalent to locally reproduced, fully aligned predictions.
6. **Descriptive subgroup analysis.** B7 strata, segment profiles, boundaries, and complementarity are exploratory descriptions. They were not used for model selection and should not be presented as confirmatory hypothesis tests.
7. **Causal interpretation.** The observed segment-length strengths are consistent with the multiscale architecture, but the external evaluations alone do not establish which component causes the improvement.
"""
    write_text(out / "C1_LIMITATIONS.md", limitations)

    legends = """# C1 figure legends

**Figure 1. Locked A10 architecture and evaluation protocol.** Frozen ESM-2-650M residue representations feed two model families. A2 uses the final representation layer with a multiscale context head; A8 learns a mixture of layers 30–33 before contextual prediction. Three seeds and five folds per family yield 30 logits, combined by a fixed uniform logit mean. All model choices were fixed before CAID2/CAID3 label access.

**Figure 2. External-test performance on CAID2 and CAID3.** Full-precision ROC-AUC and AUPRC for A10-locked and three reproduced baseline variants. All models have complete residue coverage in the aligned B6/B7 comparison.

**Figure 3. Paired protein-bootstrap differences in ROC-AUC.** Points show A10 minus comparator; horizontal lines show percentile 95% confidence intervals from 2,000 paired whole-protein bootstrap replicates. Asterisks denote Holm-adjusted bootstrap p<0.05. Paper-8 comparisons belong to a separate supplementary correction family.

**Figure 4. Descriptive subgroup ROC-AUC differences by known-label disorder fraction.** Cells report A10 minus comparator within predeclared protein-level disorder-fraction strata. Blue favors A10 and red favors the comparator. No new subgroup hypothesis tests were performed.

**Figure 5. Segment-level fixed-threshold error profiles.** Error rates at a fixed probability threshold of 0.5 are shown separately for true disordered and ordered segments and by segment-length bin. The analysis is descriptive and was performed after model locking.
"""
    write_text(out / "C1_FIGURE_LEGENDS.md", legends)


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    inputs = {
        "b6_point_metrics": B6 / "b6_point_metrics.csv",
        "b6_pairwise_statistics": B6 / "b6_pairwise_statistics.csv",
        "b6_report": B6 / "b6_report.json",
        "b7_point_metrics": B7 / "b7_point_metrics.csv",
        "b7_subgroup_metrics": B7 / "b7_subgroup_metrics.csv",
        "b7_boundary_metrics": B7 / "b7_boundary_metrics.csv",
        "b7_segment_metrics": B7 / "b7_segment_metrics.csv",
        "b7_pairwise_complementarity": B7 / "b7_pairwise_complementarity.csv",
        "b7_cross_dataset_robustness": B7 / "b7_cross_dataset_robustness.csv",
        "b7_report": B7 / "b7_report.json",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing locked inputs: {missing}")

    b6_point = pd.read_csv(inputs["b6_point_metrics"])
    pairwise = pd.read_csv(inputs["b6_pairwise_statistics"])
    point = pd.read_csv(inputs["b7_point_metrics"])
    subgroup = pd.read_csv(inputs["b7_subgroup_metrics"])
    boundary = pd.read_csv(inputs["b7_boundary_metrics"])
    segment = pd.read_csv(inputs["b7_segment_metrics"])
    complementarity = pd.read_csv(inputs["b7_pairwise_complementarity"])
    robustness = pd.read_csv(inputs["b7_cross_dataset_robustness"])

    verification_metrics = [
        "roc_auc_full_precision",
        "aucpr_trapezoid_full_precision",
        "aps_full_precision",
        "f1_at_0_5_full_precision",
        "mcc_at_0_5_full_precision",
        "fmax_full_precision",
    ]
    merged = b6_point.merge(point, on=["dataset", "track", "model"], suffixes=("_b6", "_b7"))
    maximum_difference = max(
        float((merged[f"{metric}_b6"] - merged[f"{metric}_b7"]).abs().max())
        for metric in verification_metrics
    )
    if maximum_difference > 1e-12:
        raise ValueError(f"B6/B7 point metrics disagree: {maximum_difference}")

    table1 = point[
        [
            "dataset", "track", "model", "proteins", "residues", "prevalence",
            "roc_auc_full_precision", "aucpr_trapezoid_full_precision", "aps_full_precision",
            "f1_at_0_5_full_precision", "mcc_at_0_5_full_precision", "fmax_full_precision",
            "brier_score",
        ]
    ].copy()
    table1["track"] = table1.track.map({"disorder_nox": "NOX", "disorder_pdb": "PDB"})
    table1["model_order"] = table1.model.map(MODEL_ORDER.index)
    table1["track_order"] = table1.apply(lambda row: TRACK_ORDER.index((row.dataset, "disorder_" + row.track.lower())), axis=1)
    table1 = table1.sort_values(["track_order", "model_order"]).drop(columns=["track_order", "model_order"])
    table1.columns = [
        "Dataset", "Track", "Model", "Proteins", "Known residues", "Prevalence",
        "ROC-AUC", "AUPRC", "APS", "F1@0.5", "MCC@0.5", "Fmax", "Brier",
    ]
    table1.to_csv(out / "table_1_external_performance.csv", index=False)
    write_text(out / "table_1_external_performance.md", markdown_table(table1, 3))

    table2 = pairwise[pairwise.metric == "roc_auc_full_precision"][
        [
            "dataset", "track", "second_model", "comparison_family", "common_proteins",
            "point_delta", "percentile_95_ci", "bootstrap_holm_p", "delong_holm_p",
        ]
    ].copy()
    table2.columns = [
        "Dataset", "Track", "Comparator", "Correction family", "Common proteins",
        "A10 ROC delta", "Protein-bootstrap 95% CI", "Bootstrap Holm p", "DeLong Holm p",
    ]
    table2.to_csv(out / "table_2_paired_roc_statistics.csv", index=False)
    write_text(out / "table_2_paired_roc_statistics.md", markdown_table(table2, 4))

    pairwise.to_csv(out / "table_s1_all_paired_statistics.csv", index=False)
    subgroup.to_csv(out / "table_s2_all_subgroup_metrics.csv", index=False)
    segment.to_csv(out / "table_s3_all_segment_metrics.csv", index=False)
    boundary.to_csv(out / "table_s4_all_boundary_metrics.csv", index=False)
    complementarity.to_csv(out / "table_s5_all_complementarity_metrics.csv", index=False)
    robustness.to_csv(out / "table_s6_cross_dataset_robustness.csv", index=False)

    draw_architecture(out / "figure_1_locked_a10_architecture.svg")
    draw_external_performance(point, out / "figure_2_external_performance.svg")
    draw_forest(pairwise, out / "figure_3_paired_roc_forest.svg")
    draw_heatmap(subgroup, out / "figure_4_disorder_fraction_heatmap.svg")
    draw_segment_profiles(segment, out / "figure_5_segment_error_profiles.svg")
    build_narratives(point, pairwise, subgroup, segment, out)

    cn_readme = """# C1 论文材料说明

- `table_1`：四个外部测试轨道的主性能表。
- `table_2`：A10 相对复现基线的蛋白级配对 ROC-AUC 统计。
- `table_s1`–`table_s6`：完整补充统计、分层、片段、边界、互补性和跨数据集稳健性表。
- `figure_1`：A10 锁定结构与外部评测流程。
- `figure_2`：CAID2/CAID3 外部性能。
- `figure_3`：2,000 次整蛋白配对 bootstrap 森林图。
- `figure_4`、`figure_5`：B7 描述性误差分析，不作新增显著性主张。
- `C1_RESULTS_DRAFT.md`、`C1_DISCUSSION_DRAFT.md`、`C1_LIMITATIONS.md`：英文论文初稿。

论文的稳妥主张是：A10 在 PDB 轨道和长无序片段上有明显优势，与 PUNCH2-Light Released-13 整体相当；它不是所有无序定义上的普遍 SOTA。CAID2-NOX、低无序比例蛋白和长有序片段是下一代模型需要解决的主要弱点。
"""
    write_text(out / "README_CN.md", cn_readme)

    generated = sorted(
        path for path in out.iterdir()
        if path.is_file() and path.name not in {"c1_build_report.json", "c1_result_lock.sha256"}
    )
    lock_lines = [f"{sha256(path)}  {path.name}" for path in generated]
    write_text(out / "c1_result_lock.sha256", "\n".join(lock_lines))
    report = {
        "schema_version": 1,
        "experiment": "c1_manuscript_materials",
        "status": "pass",
        "locked_aggregate_inputs_only": True,
        "caid_labels_or_predictions_accessed": False,
        "b6_b7_point_metrics_max_abs_delta": maximum_difference,
        "confirmatory_source": "B6 paired whole-protein bootstrap with Holm correction",
        "descriptive_source": "B7 post-lock error and complementarity analysis",
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "generated_sha256": {path.name: sha256(path) for path in generated},
        "outputs": len(generated),
    }
    write_text(out / "c1_build_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
