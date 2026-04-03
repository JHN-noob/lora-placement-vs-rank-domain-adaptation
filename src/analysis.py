from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


AUXILIARY_RUN_PREFIXES = ("smoke_", "mini_")
DOMAIN_NAME_ALIASES = {
    "Keboola/Developer-Documentation-QA": "keboola_docs",
    "rojagtap/tech-qa": "techqa",
    "keboola_docs": "keboola_docs",
    "techqa": "techqa",
}
CANONICAL_BASE_RUNS = {
    "keboola_docs": "keboola_base",
    "techqa": "techqa_base",
}
PLACEMENT_ORDER = ["lower", "upper", "all"]
RANK_ORDER = [4, 8, 16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _is_auxiliary_run(run_name: str) -> bool:
    return any(run_name.startswith(prefix) for prefix in AUXILIARY_RUN_PREFIXES)


def _normalize_domain_name(domain_name: str | None, dataset_name: str | None, run_name: str) -> str:
    if domain_name in DOMAIN_NAME_ALIASES:
        return DOMAIN_NAME_ALIASES[domain_name]
    if dataset_name in DOMAIN_NAME_ALIASES:
        return DOMAIN_NAME_ALIASES[dataset_name]
    if run_name.startswith("techqa_"):
        return "techqa"
    if run_name.startswith("keboola_"):
        return "keboola_docs"
    if dataset_name and "keboola" in dataset_name.lower():
        return "keboola_docs"
    if dataset_name and "tech-qa" in dataset_name.lower():
        return "techqa"
    return (domain_name or dataset_name or "unknown").strip() or "unknown"


def _weighted_general_accuracy(boolq_acc: float, boolq_samples: int, piqa_acc: float, piqa_samples: int) -> float:
    total = boolq_samples + piqa_samples
    if total <= 0:
        return 0.0
    return ((boolq_acc * boolq_samples) + (piqa_acc * piqa_samples)) / total


def _extract_context_ratio(metrics: dict[str, Any]) -> float:
    summary = metrics.get("data", {}).get("domain_context_summary", {})
    if isinstance(summary, dict) and "resolved_ratio" in summary:
        return _safe_float(summary.get("resolved_ratio"))
    return 0.0


def _load_metrics_rows(outputs_root: str | Path, include_aux_runs: bool = False) -> list[dict[str, Any]]:
    runs_dir = Path(outputs_root) / "runs"
    if not runs_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment = metrics.get("experiment", {})
        run_name = str(experiment.get("run_name") or run_dir.name)
        if not include_aux_runs and _is_auxiliary_run(run_name):
            continue

        data_block = metrics.get("data", {})
        general = metrics.get("general", {})
        boolq = general.get("boolq", {})
        piqa = general.get("piqa", {})
        wikitext = general.get("wikitext", {})
        training = metrics.get("training", {})

        domain_dataset = data_block.get("domain_dataset", "")
        domain_name = _normalize_domain_name(
            data_block.get("domain_name"),
            domain_dataset,
            run_name,
        )

        boolq_acc = _safe_float(boolq.get("accuracy"))
        boolq_samples = _safe_int(boolq.get("samples"))
        piqa_acc = _safe_float(piqa.get("accuracy"))
        piqa_samples = _safe_int(piqa.get("samples"))

        rows.append(
            {
                "run_name": run_name,
                "domain_name": domain_name,
                "domain_dataset": domain_dataset,
                "run_type": str(experiment.get("run_type", "")),
                "placement": str(experiment.get("placement", "")),
                "rank": _safe_int(experiment.get("rank")),
                "seed": _safe_int(experiment.get("seed")),
                "domain_f1": _safe_float(metrics.get("domain", {}).get("f1")),
                "boolq_acc": boolq_acc,
                "boolq_samples": boolq_samples,
                "piqa_acc": piqa_acc,
                "piqa_samples": piqa_samples,
                "general_acc_weighted": _weighted_general_accuracy(
                    boolq_acc=boolq_acc,
                    boolq_samples=boolq_samples,
                    piqa_acc=piqa_acc,
                    piqa_samples=piqa_samples,
                ),
                "wikitext_ppl": _safe_float(wikitext.get("perplexity")),
                "trainable_parameters": _safe_int(training.get("trainable_parameters")),
                "runtime_seconds": _safe_float(training.get("runtime_seconds")),
                "domain_input_mode": str(data_block.get("domain_input_mode", "")),
                "domain_context_ratio": _extract_context_ratio(metrics),
            }
        )
    return rows


def _choose_base_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bases_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["run_type"] == "base":
            bases_by_domain[row["domain_name"]].append(row)

    selected: dict[str, dict[str, Any]] = {}
    for domain_name, candidates in bases_by_domain.items():
        canonical_name = CANONICAL_BASE_RUNS.get(domain_name, f"{domain_name}_base")
        selected[domain_name] = sorted(
            candidates,
            key=lambda row: (
                row["run_name"] != canonical_name,
                not row["run_name"].endswith("_base"),
                row["run_name"],
            ),
        )[0]
    return selected


def _deduplicate_rows(rows: list[dict[str, Any]], base_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row["run_type"] != "base":
            filtered.append(row)
            continue
        if base_rows.get(row["domain_name"], {}).get("run_name") == row["run_name"]:
            filtered.append(row)
    return filtered


def _apply_base_comparisons(rows: list[dict[str, Any]], base_rows: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        base_row = base_rows.get(row["domain_name"])
        if base_row is None:
            row["base_run_name"] = ""
            row["adaptation_gain"] = None
            row["forgetting_boolq"] = None
            row["forgetting_piqa"] = None
            row["forgetting_acc_weighted"] = None
            row["forgetting_ppl"] = None
            continue

        row["base_run_name"] = base_row["run_name"]
        row["adaptation_gain"] = row["domain_f1"] - base_row["domain_f1"]
        row["forgetting_boolq"] = base_row["boolq_acc"] - row["boolq_acc"]
        row["forgetting_piqa"] = base_row["piqa_acc"] - row["piqa_acc"]
        row["forgetting_acc_weighted"] = base_row["general_acc_weighted"] - row["general_acc_weighted"]
        row["forgetting_ppl"] = row["wikitext_ppl"] - base_row["wikitext_ppl"]


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum - minimum < 1e-12:
        return [0.5 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _apply_composite_scores(rows: list[dict[str, Any]]) -> None:
    rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["run_type"] == "adapter":
            rows_by_domain[row["domain_name"]].append(row)

    for domain_name, domain_rows in rows_by_domain.items():
        gains = [float(row["adaptation_gain"]) for row in domain_rows]
        retention_acc = [-float(row["forgetting_acc_weighted"]) for row in domain_rows]
        retention_ppl = [-float(row["forgetting_ppl"]) for row in domain_rows]

        gain_norm = _normalize_series(gains)
        acc_norm = _normalize_series(retention_acc)
        ppl_norm = _normalize_series(retention_ppl)

        for row, gain_value, acc_value, ppl_value in zip(domain_rows, gain_norm, acc_norm, ppl_norm):
            row["composite_score"] = (
                0.6 * gain_value
                + 0.3 * acc_value
                + 0.1 * ppl_value
            )

        ranked = sorted(domain_rows, key=lambda row: (-row["composite_score"], row["run_name"]))
        for index, row in enumerate(ranked, start=1):
            row["composite_rank"] = index

    for row in rows:
        if row["run_type"] == "base":
            row["composite_score"] = None
            row["composite_rank"] = None


def _serialize_rows(rows: Iterable[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field) for field in fieldnames}
            writer.writerow(payload)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return mean(values_list)


def _build_runs_summary(rows: list[dict[str, Any]], summary_dir: Path) -> Path:
    fieldnames = [
        "run_name",
        "base_run_name",
        "domain_name",
        "domain_dataset",
        "run_type",
        "placement",
        "rank",
        "seed",
        "domain_f1",
        "boolq_acc",
        "boolq_samples",
        "piqa_acc",
        "piqa_samples",
        "general_acc_weighted",
        "wikitext_ppl",
        "adaptation_gain",
        "forgetting_boolq",
        "forgetting_piqa",
        "forgetting_acc_weighted",
        "forgetting_ppl",
        "composite_score",
        "composite_rank",
        "trainable_parameters",
        "runtime_seconds",
        "domain_input_mode",
        "domain_context_ratio",
    ]
    rows_for_csv = []
    for row in rows:
        payload = dict(row)
        for key in (
            "domain_f1",
            "boolq_acc",
            "piqa_acc",
            "general_acc_weighted",
            "wikitext_ppl",
            "adaptation_gain",
            "forgetting_boolq",
            "forgetting_piqa",
            "forgetting_acc_weighted",
            "forgetting_ppl",
            "composite_score",
            "domain_context_ratio",
            "runtime_seconds",
        ):
            payload[key] = _round(payload.get(key))
        rows_for_csv.append(payload)

    path = summary_dir / "runs_summary.csv"
    _serialize_rows(rows_for_csv, fieldnames, path)
    return path


def _aggregate_by(
    rows: list[dict[str, Any]],
    keys: list[str],
    metrics: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    aggregated: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        payload = {key: value for key, value in zip(keys, group_key)}
        payload["run_count"] = len(group_rows)
        for out_key, src_key in metrics.items():
            payload[out_key] = _round(_mean(float(row[src_key]) for row in group_rows))
        aggregated.append(payload)

    return sorted(aggregated, key=lambda row: tuple(row[key] for key in keys))


def _build_group_summaries(adapter_rows: list[dict[str, Any]], summary_dir: Path) -> dict[str, Path]:
    placement_rows = _aggregate_by(
        adapter_rows,
        keys=["domain_name", "placement"],
        metrics={
            "mean_domain_f1": "domain_f1",
            "mean_adaptation_gain": "adaptation_gain",
            "mean_forgetting_acc_weighted": "forgetting_acc_weighted",
            "mean_composite_score": "composite_score",
        },
    )
    rank_domain_rows = _aggregate_by(
        adapter_rows,
        keys=["domain_name", "rank"],
        metrics={
            "mean_domain_f1": "domain_f1",
            "mean_adaptation_gain": "adaptation_gain",
            "mean_composite_score": "composite_score",
        },
    )
    rank_forgetting_rows = _aggregate_by(
        adapter_rows,
        keys=["domain_name", "rank"],
        metrics={
            "mean_forgetting_acc_weighted": "forgetting_acc_weighted",
            "mean_forgetting_ppl": "forgetting_ppl",
        },
    )
    composite_rows = sorted(
        (
            {
                "domain_name": row["domain_name"],
                "run_name": row["run_name"],
                "placement": row["placement"],
                "rank": row["rank"],
                "domain_f1": _round(row["domain_f1"]),
                "adaptation_gain": _round(row["adaptation_gain"]),
                "forgetting_acc_weighted": _round(row["forgetting_acc_weighted"]),
                "forgetting_ppl": _round(row["forgetting_ppl"]),
                "general_acc_weighted": _round(row["general_acc_weighted"]),
                "composite_score": _round(row["composite_score"]),
                "composite_rank": row["composite_rank"],
            }
            for row in adapter_rows
        ),
        key=lambda row: (row["domain_name"], row["composite_rank"], row["run_name"]),
    )

    base_rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adapter_rows:
        base_rows_by_domain[row["domain_name"]].append(row)

    domain_best_rows: list[dict[str, Any]] = []
    for domain_name, domain_rows in sorted(base_rows_by_domain.items()):
        best_f1 = max(domain_rows, key=lambda row: (row["domain_f1"], row["run_name"]))
        best_composite = max(domain_rows, key=lambda row: (row["composite_score"], row["run_name"]))
        best_retention = max(domain_rows, key=lambda row: (row["general_acc_weighted"], row["run_name"]))
        domain_best_rows.append(
            {
                "domain_name": domain_name,
                "best_domain_run": best_f1["run_name"],
                "best_domain_f1": _round(best_f1["domain_f1"]),
                "best_composite_run": best_composite["run_name"],
                "best_composite_score": _round(best_composite["composite_score"]),
                "best_general_run": best_retention["run_name"],
                "best_general_acc_weighted": _round(best_retention["general_acc_weighted"]),
            }
        )

    outputs = {
        "placement_vs_domain.csv": (
            placement_rows,
            ["domain_name", "placement", "run_count", "mean_domain_f1", "mean_adaptation_gain", "mean_forgetting_acc_weighted", "mean_composite_score"],
        ),
        "rank_vs_domain.csv": (
            rank_domain_rows,
            ["domain_name", "rank", "run_count", "mean_domain_f1", "mean_adaptation_gain", "mean_composite_score"],
        ),
        "rank_vs_forgetting.csv": (
            rank_forgetting_rows,
            ["domain_name", "rank", "run_count", "mean_forgetting_acc_weighted", "mean_forgetting_ppl"],
        ),
        "composite_ranking.csv": (
            composite_rows,
            ["domain_name", "run_name", "placement", "rank", "domain_f1", "adaptation_gain", "forgetting_acc_weighted", "forgetting_ppl", "general_acc_weighted", "composite_score", "composite_rank"],
        ),
        "domain_best_runs.csv": (
            domain_best_rows,
            ["domain_name", "best_domain_run", "best_domain_f1", "best_composite_run", "best_composite_score", "best_general_run", "best_general_acc_weighted"],
        ),
    }

    paths: dict[str, Path] = {}
    for filename, (rows_to_write, fieldnames) in outputs.items():
        path = summary_dir / filename
        _serialize_rows(rows_to_write, fieldnames, path)
        paths[filename] = path
    return paths


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return None
    return plt


def _maybe_import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        return None
    return Image, ImageDraw, ImageFont


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _pil_font(size: int):
    pillow = _maybe_import_pillow()
    if pillow is None:
        return None
    _, _, ImageFont = pillow
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_pil_text(draw: Any, position: tuple[float, float], text: str, size: int = 12, fill: str = "#111827", anchor: str | None = None) -> None:
    font = _pil_font(size)
    kwargs: dict[str, Any] = {"fill": fill, "font": font}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text(position, text, **kwargs)


def _draw_rotated_text(image: Any, position: tuple[int, int], text: str, size: int = 12, fill: str = "#111827") -> None:
    pillow = _maybe_import_pillow()
    if pillow is None:
        return
    Image, ImageDraw, _ = pillow
    font = _pil_font(size)
    temp = Image.new("RGBA", (220, 40), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), text, fill=fill, font=font)
    rotated = temp.rotate(90, expand=1)
    image.alpha_composite(rotated, dest=position)


def _linear_position(value: float, minimum: float, maximum: float, size: float) -> float:
    if maximum - minimum < 1e-12:
        return size / 2
    return (value - minimum) / (maximum - minimum) * size


def _color_scale(value: float, minimum: float, maximum: float) -> str:
    if value != value:
        return "#e5e7eb"
    if maximum - minimum < 1e-12:
        ratio = 0.5
    else:
        ratio = (value - minimum) / (maximum - minimum)
    start = (239, 246, 255)
    end = (3, 105, 161)
    rgb = tuple(int(start[index] + (end[index] - start[index]) * ratio) for index in range(3))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _axis_ticks(minimum: float, maximum: float, count: int = 5) -> list[float]:
    if maximum - minimum < 1e-12:
        return [minimum]
    return [minimum + (maximum - minimum) * (index / (count - 1)) for index in range(count)]


def _plot_gain_vs_forgetting_png(adapter_rows: list[dict[str, Any]], output_path: Path) -> bool:
    pillow = _maybe_import_pillow()
    if pillow is None or not adapter_rows:
        return False
    Image, ImageDraw, _ = pillow
    width, height = 920, 620
    margin_left, margin_right = 90, 30
    margin_top, margin_bottom = 50, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)

    x_values = [float(row["forgetting_acc_weighted"]) for row in adapter_rows]
    y_values = [float(row["adaptation_gain"]) for row in adapter_rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.1, 0.01)
    y_pad = max((y_max - y_min) * 0.1, 0.01)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    domain_names = _ordered_domain_names(adapter_rows)
    palette = ["#2563eb", "#ea580c", "#059669", "#dc2626"]
    colors = {domain: palette[index % len(palette)] for index, domain in enumerate(domain_names)}

    _draw_pil_text(draw, (width / 2, 24), "Adaptation Gain vs Weighted Forgetting", size=20, anchor="ma")
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill="#111827", width=2)
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="#111827", width=2)

    for tick in _axis_ticks(x_min, x_max):
        x = margin_left + _linear_position(tick, x_min, x_max, plot_width)
        draw.line((x, height - margin_bottom, x, height - margin_bottom + 6), fill="#111827", width=1)
        _draw_pil_text(draw, (x, height - margin_bottom + 18), f"{tick:.3f}", size=11, anchor="ma")

    for tick in _axis_ticks(y_min, y_max):
        y = margin_top + (plot_height - _linear_position(tick, y_min, y_max, plot_height))
        draw.line((margin_left - 5, y, margin_left, y), fill="#111827", width=1)
        draw.line((margin_left, y, width - margin_right, y), fill="#e5e7eb", width=1)
        _draw_pil_text(draw, (margin_left - 10, y), f"{tick:.3f}", size=11, anchor="rm")

    if x_min <= 0 <= x_max:
        zero_x = margin_left + _linear_position(0.0, x_min, x_max, plot_width)
        for offset in range(margin_top, height - margin_bottom, 10):
            draw.line((zero_x, offset, zero_x, min(offset + 5, height - margin_bottom)), fill="#9ca3af", width=1)
    if y_min <= 0 <= y_max:
        zero_y = margin_top + (plot_height - _linear_position(0.0, y_min, y_max, plot_height))
        for offset in range(margin_left, width - margin_right, 10):
            draw.line((offset, zero_y, min(offset + 5, width - margin_right), zero_y), fill="#9ca3af", width=1)

    for row in adapter_rows:
        x = margin_left + _linear_position(float(row["forgetting_acc_weighted"]), x_min, x_max, plot_width)
        y = margin_top + (plot_height - _linear_position(float(row["adaptation_gain"]), y_min, y_max, plot_height))
        color = colors[row["domain_name"]]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=color)
        _draw_pil_text(draw, (x + 8, y - 10), str(row["run_name"]), size=10)

    legend_x = width - 185
    legend_y = margin_top + 12
    for index, domain_name in enumerate(domain_names):
        y = legend_y + index * 22
        draw.ellipse((legend_x, y - 5, legend_x + 10, y + 5), fill=colors[domain_name], outline=colors[domain_name])
        _draw_pil_text(draw, (legend_x + 16, y), domain_name, size=11, anchor="lm")

    _draw_pil_text(draw, (width / 2, height - 20), "Weighted forgetting", size=12, anchor="ma")
    _draw_rotated_text(image, (8, height // 2 - 110), "Adaptation gain", size=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return True


def _plot_rank_metric_by_domain_png(
    adapter_rows: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    title: str,
    ylabel: str,
) -> bool:
    pillow = _maybe_import_pillow()
    if pillow is None or not adapter_rows:
        return False
    Image, ImageDraw, _ = pillow
    domain_names = _ordered_domain_names(adapter_rows)
    panel_width, panel_height = 420, 320
    width = 60 + panel_width * len(domain_names)
    height = 420
    colors = {"lower": "#2563eb", "upper": "#dc2626", "all": "#059669"}

    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_pil_text(draw, (width / 2, 24), title, size=20, anchor="ma")

    for panel_index, domain_name in enumerate(domain_names):
        panel_x = 40 + panel_index * panel_width
        plot_x = panel_x + 50
        plot_y = 70
        plot_width = panel_width - 90
        plot_height = panel_height - 80
        domain_rows = [row for row in adapter_rows if row["domain_name"] == domain_name]
        y_values = [float(row[metric_key]) for row in domain_rows]
        y_min, y_max = min(y_values), max(y_values)
        y_pad = max((y_max - y_min) * 0.1, 0.01)
        y_min -= y_pad
        y_max += y_pad

        _draw_pil_text(draw, (panel_x + panel_width / 2, 50), domain_name, size=14, anchor="ma")
        draw.line((plot_x, plot_y + plot_height, plot_x + plot_width, plot_y + plot_height), fill="#111827", width=2)
        draw.line((plot_x, plot_y, plot_x, plot_y + plot_height), fill="#111827", width=2)

        for tick in _axis_ticks(y_min, y_max):
            y = plot_y + (plot_height - _linear_position(tick, y_min, y_max, plot_height))
            draw.line((plot_x - 5, y, plot_x, y), fill="#111827", width=1)
            draw.line((plot_x, y, plot_x + plot_width, y), fill="#e5e7eb", width=1)
            _draw_pil_text(draw, (plot_x - 8, y), f"{tick:.3f}", size=10, anchor="rm")

        rank_positions = {
            rank: plot_x + index * (plot_width / max(1, len(RANK_ORDER) - 1))
            for index, rank in enumerate(RANK_ORDER)
        }
        for rank, x in rank_positions.items():
            draw.line((x, plot_y + plot_height, x, plot_y + plot_height + 5), fill="#111827", width=1)
            _draw_pil_text(draw, (x, plot_y + plot_height + 18), str(rank), size=10, anchor="ma")

        for placement in PLACEMENT_ORDER:
            placement_rows = [row for row in domain_rows if row["placement"] == placement]
            if not placement_rows:
                continue
            points = []
            for row in sorted(placement_rows, key=lambda item: item["rank"]):
                x = rank_positions[row["rank"]]
                y = plot_y + (plot_height - _linear_position(float(row[metric_key]), y_min, y_max, plot_height))
                points.append((x, y))
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[placement], outline=colors[placement])
            if len(points) >= 2:
                draw.line(points, fill=colors[placement], width=3)

        legend_x = plot_x + plot_width - 80
        legend_y = plot_y + 16
        for index, placement in enumerate(PLACEMENT_ORDER):
            y = legend_y + index * 18
            draw.line((legend_x, y, legend_x + 18, y), fill=colors[placement], width=3)
            _draw_pil_text(draw, (legend_x + 24, y), placement, size=10, anchor="lm")

    _draw_pil_text(draw, (width / 2, height - 20), "LoRA rank", size=12, anchor="ma")
    _draw_rotated_text(image, (10, height // 2 - 110), ylabel, size=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return True


def _plot_heatmap_png(
    adapter_rows: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    title: str,
) -> bool:
    pillow = _maybe_import_pillow()
    if pillow is None or not adapter_rows:
        return False
    Image, ImageDraw, _ = pillow
    domain_names = _ordered_domain_names(adapter_rows)
    panel_width, panel_height = 360, 260
    width = 40 + panel_width * len(domain_names)
    height = 360
    cell_width, cell_height = 72, 48

    all_values = [float(row[metric_key]) for row in adapter_rows if row.get(metric_key) is not None]
    min_value = min(all_values) if all_values else 0.0
    max_value = max(all_values) if all_values else 1.0

    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_pil_text(draw, (width / 2, 24), title, size=20, anchor="ma")

    for panel_index, domain_name in enumerate(domain_names):
        origin_x = 40 + panel_index * panel_width + 60
        origin_y = 80
        _draw_pil_text(draw, (origin_x + 1.5 * cell_width, 52), domain_name, size=14, anchor="ma")

        for col_index, rank in enumerate(RANK_ORDER):
            x = origin_x + col_index * cell_width + cell_width / 2
            _draw_pil_text(draw, (x, origin_y - 14), f"r{rank}", size=11, anchor="ma")

        for row_index, placement in enumerate(PLACEMENT_ORDER):
            y = origin_y + row_index * cell_height + cell_height / 2
            _draw_pil_text(draw, (origin_x - 12, y), placement, size=11, anchor="rm")
            for col_index, rank in enumerate(RANK_ORDER):
                matched = [
                    float(row[metric_key])
                    for row in adapter_rows
                    if row["domain_name"] == domain_name and row["placement"] == placement and row["rank"] == rank
                ]
                value = _mean(matched) if matched else float("nan")
                x = origin_x + col_index * cell_width
                y_cell = origin_y + row_index * cell_height
                fill = _color_scale(value, min_value, max_value)
                draw.rectangle((x, y_cell, x + cell_width, y_cell + cell_height), fill=fill, outline="white", width=1)
                if value == value:
                    _draw_pil_text(draw, (x + cell_width / 2, y_cell + cell_height / 2), f"{value:.3f}", size=11, anchor="mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return True


def _ordered_domain_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row["domain_name"] for row in rows})


def _plot_gain_vs_forgetting(adapter_rows: list[dict[str, Any]], output_path: Path) -> bool:
    plt = _maybe_import_matplotlib()
    if not adapter_rows:
        return False
    if plt is None:
        return _plot_gain_vs_forgetting_png(adapter_rows, output_path)

    domain_names = _ordered_domain_names(adapter_rows)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    color_map = {domain: colors[index % len(colors)] for index, domain in enumerate(domain_names)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in adapter_rows:
        ax.scatter(
            row["forgetting_acc_weighted"],
            row["adaptation_gain"],
            color=color_map[row["domain_name"]],
            s=80,
            alpha=0.9,
        )
        ax.annotate(
            row["run_name"],
            (row["forgetting_acc_weighted"], row["adaptation_gain"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.axvline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Weighted forgetting")
    ax.set_ylabel("Adaptation gain")
    ax.set_title("Adaptation Gain vs Weighted Forgetting")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[name], label=name, markersize=8)
        for name in domain_names
    ]
    ax.legend(handles=handles, title="Domain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _plot_rank_metric_by_domain(
    adapter_rows: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    title: str,
    ylabel: str,
) -> bool:
    plt = _maybe_import_matplotlib()
    if not adapter_rows:
        return False
    if plt is None:
        return _plot_rank_metric_by_domain_png(
            adapter_rows=adapter_rows,
            output_path=output_path,
            metric_key=metric_key,
            title=title,
            ylabel=ylabel,
        )

    domain_names = _ordered_domain_names(adapter_rows)
    fig, axes = plt.subplots(1, len(domain_names), figsize=(6 * len(domain_names), 4.8), squeeze=False)

    for axis, domain_name in zip(axes[0], domain_names):
        domain_rows = [row for row in adapter_rows if row["domain_name"] == domain_name]
        for placement in PLACEMENT_ORDER:
            placement_rows = [row for row in domain_rows if row["placement"] == placement]
            if not placement_rows:
                continue
            rank_to_metric = {
                row["rank"]: row[metric_key]
                for row in placement_rows
            }
            x_values = [rank for rank in RANK_ORDER if rank in rank_to_metric]
            y_values = [rank_to_metric[rank] for rank in x_values]
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=placement)

        axis.set_title(domain_name)
        axis.set_xlabel("LoRA rank")
        axis.set_ylabel(ylabel)
        axis.set_xticks(RANK_ORDER)
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _plot_heatmap(
    adapter_rows: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    title: str,
) -> bool:
    plt = _maybe_import_matplotlib()
    if not adapter_rows:
        return False
    if plt is None:
        return _plot_heatmap_png(
            adapter_rows=adapter_rows,
            output_path=output_path,
            metric_key=metric_key,
            title=title,
        )

    domain_names = _ordered_domain_names(adapter_rows)
    fig, axes = plt.subplots(1, len(domain_names), figsize=(5.5 * len(domain_names), 4.8), squeeze=False)

    for axis, domain_name in zip(axes[0], domain_names):
        domain_rows = [row for row in adapter_rows if row["domain_name"] == domain_name]
        matrix: list[list[float]] = []
        for placement in PLACEMENT_ORDER:
            matrix_row: list[float] = []
            for rank in RANK_ORDER:
                matched = [
                    row[metric_key]
                    for row in domain_rows
                    if row["placement"] == placement and row["rank"] == rank
                ]
                matrix_row.append(_mean(float(value) for value in matched) if matched else float("nan"))
            matrix.append(matrix_row)

        image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu")
        axis.set_title(domain_name)
        axis.set_xticks(range(len(RANK_ORDER)))
        axis.set_xticklabels(RANK_ORDER)
        axis.set_yticks(range(len(PLACEMENT_ORDER)))
        axis.set_yticklabels(PLACEMENT_ORDER)
        axis.set_xlabel("LoRA rank")
        axis.set_ylabel("Placement")

        for row_index, matrix_row in enumerate(matrix):
            for col_index, value in enumerate(matrix_row):
                if value == value:
                    axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=9)

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _write_summary_markdown(
    rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    base_rows: dict[str, dict[str, Any]],
    summary_dir: Path,
    include_aux_runs: bool,
) -> Path:
    summary_path = summary_dir / "summary.md"

    lines = [
        "# 요약",
        "",
        f"- summary 기본 설정에서 smoke/mini run 포함 여부: {'포함' if include_aux_runs else '제외'}",
        f"- 집계 대상 run 수: {len(rows)}",
        f"- domain 수: {len(base_rows)}",
        "- composite score는 보조 지표이며, raw metric과 domain별 base 대비 변화량을 우선 해석합니다.",
        "",
        "## Domain Base",
        "",
    ]

    for domain_name in sorted(base_rows):
        base_row = base_rows[domain_name]
        lines.append(
            f"- {domain_name} / {base_row['run_name']}: "
            f"domain F1 {base_row['domain_f1']:.4f}, "
            f"weighted general accuracy {base_row['general_acc_weighted']:.4f}, "
            f"WikiText PPL {base_row['wikitext_ppl']:.4f}"
        )

    lines.extend(["", "## Domain별 핵심 관찰", ""])
    for domain_name in sorted(base_rows):
        domain_rows = [row for row in adapter_rows if row["domain_name"] == domain_name]
        if not domain_rows:
            continue
        best_f1 = max(domain_rows, key=lambda row: (row["domain_f1"], row["run_name"]))
        best_composite = max(domain_rows, key=lambda row: (row["composite_score"], row["run_name"]))
        best_retention = min(domain_rows, key=lambda row: (row["forgetting_acc_weighted"], row["run_name"]))
        lines.append(
            f"- {domain_name} 최고 domain F1: {best_f1['run_name']} "
            f"(F1 {best_f1['domain_f1']:.4f}, gain {best_f1['adaptation_gain']:.4f})"
        )
        lines.append(
            f"- {domain_name} 최고 composite: {best_composite['run_name']} "
            f"(score {best_composite['composite_score']:.4f}, "
            f"F1 {best_composite['domain_f1']:.4f}, forgetting {best_composite['forgetting_acc_weighted']:.4f})"
        )
        lines.append(
            f"- {domain_name} 최소 weighted forgetting: {best_retention['run_name']} "
            f"(forgetting {best_retention['forgetting_acc_weighted']:.4f}, "
            f"weighted general accuracy {best_retention['general_acc_weighted']:.4f})"
        )

    lines.extend(["", "## Adapter Run", ""])
    for domain_name in sorted(base_rows):
        domain_rows = sorted(
            (row for row in adapter_rows if row["domain_name"] == domain_name),
            key=lambda row: (-row["composite_score"], -row["domain_f1"], row["run_name"]),
        )
        for row in domain_rows:
            lines.append(
                f"- {domain_name} / {row['run_name']}: "
                f"placement {row['placement']}, rank {row['rank']}, "
                f"domain F1 {row['domain_f1']:.4f}, "
                f"weighted general accuracy {row['general_acc_weighted']:.4f}, "
                f"WikiText PPL {row['wikitext_ppl']:.4f}, "
                f"gain {row['adaptation_gain']:.4f}, "
                f"forgetting {row['forgetting_acc_weighted']:.4f}, "
                f"composite {row['composite_score']:.4f}"
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _build_results_overview_markdown(
    rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    base_rows: dict[str, dict[str, Any]],
    results_dir: Path,
) -> Path:
    path = results_dir / "RESULTS_OVERVIEW.md"
    lines = [
        "# 결과 요약",
        "",
        "## 주요 결과 표",
        "",
        "| Domain | Base | 최고 Domain F1 Run | Domain F1 | Gain | Weighted General Acc | Weighted Forgetting |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for domain_name in sorted(base_rows):
        base_row = base_rows[domain_name]
        best_row = max(
            (row for row in adapter_rows if row["domain_name"] == domain_name),
            key=lambda row: (row["domain_f1"], row["run_name"]),
        )
        lines.append(
            f"| {domain_name} | {base_row['run_name']} | {best_row['run_name']} | "
            f"{best_row['domain_f1']:.4f} | {best_row['adaptation_gain']:.4f} | "
            f"{best_row['general_acc_weighted']:.4f} | {best_row['forgetting_acc_weighted']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 해석 메모",
            "",
            "- 핵심 해석은 raw metric 기준으로 합니다.",
            "- composite score는 후보를 빠르게 정렬하기 위한 보조 지표입니다.",
            "- Keboola 파일럿에서는 `all_r16_seed42`가 최고였고, TechQA 확장에서는 `techqa_lower_r8_seed42`가 최고였습니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_results_bundle(
    outputs_root: str | Path = "outputs",
    results_root: str | Path = "results",
    include_aux_runs: bool = False,
) -> dict[str, Any]:
    outputs_root = Path(outputs_root)
    results_root = Path(results_root)
    summary = summarize_results(outputs_root=outputs_root, include_aux_runs=include_aux_runs)

    summary_dir = outputs_root / "summary"
    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_names = [
        "runs_summary.csv",
        "placement_vs_domain.csv",
        "rank_vs_domain.csv",
        "rank_vs_forgetting.csv",
        "composite_ranking.csv",
        "domain_best_runs.csv",
        "summary.md",
    ]
    figure_names = [
        "gain_vs_forgetting_scatter.png",
        "rank_vs_domain.png",
        "rank_vs_forgetting.png",
        "placement_rank_domain_f1_heatmap.png",
        "placement_rank_composite_heatmap.png",
    ]

    copied_tables: dict[str, str] = {}
    copied_figures: dict[str, str] = {}
    for filename in table_names:
        source = summary_dir / filename
        if source.exists():
            target = tables_dir / filename
            shutil.copy2(source, target)
            copied_tables[filename] = _relative_path(target, results_root.parent)
    for filename in figure_names:
        source = summary_dir / filename
        if source.exists():
            target = figures_dir / filename
            shutil.copy2(source, target)
            copied_figures[filename] = _relative_path(target, results_root.parent)

    rows = _load_metrics_rows(outputs_root, include_aux_runs=include_aux_runs)
    base_rows = _choose_base_rows(rows)
    rows = _deduplicate_rows(rows, base_rows)
    _apply_base_comparisons(rows, base_rows)
    _apply_composite_scores(rows)
    adapter_rows = [row for row in rows if row["run_type"] == "adapter"]
    overview_path = _build_results_overview_markdown(rows, adapter_rows, base_rows, results_root)

    return {
        "results_root": _relative_path(results_root, results_root.parent),
        "tables": copied_tables,
        "figures": copied_figures,
        "overview_path": _relative_path(overview_path, results_root.parent),
        "summary": summary,
    }


def summarize_results(
    outputs_root: str | Path = "outputs",
    include_aux_runs: bool = False,
) -> dict[str, Any]:
    outputs_root = Path(outputs_root)
    project_root = outputs_root.parent
    summary_dir = outputs_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metrics_rows(outputs_root, include_aux_runs=include_aux_runs)
    base_rows = _choose_base_rows(rows)
    rows = _deduplicate_rows(rows, base_rows)
    _apply_base_comparisons(rows, base_rows)
    _apply_composite_scores(rows)

    rows.sort(
        key=lambda row: (
            row["domain_name"],
            row["run_type"] != "base",
            row["placement"],
            row["rank"],
            row["run_name"],
        )
    )

    runs_summary_path = _build_runs_summary(rows, summary_dir)
    adapter_rows = [row for row in rows if row["run_type"] == "adapter"]
    grouped_paths = _build_group_summaries(adapter_rows, summary_dir)

    plot_paths: dict[str, str] = {}
    plot_jobs = [
        (
            "gain_vs_forgetting_scatter",
            lambda: _plot_gain_vs_forgetting(adapter_rows, summary_dir / "gain_vs_forgetting_scatter.png"),
        ),
        (
            "rank_vs_domain",
            lambda: _plot_rank_metric_by_domain(
                adapter_rows,
                summary_dir / "rank_vs_domain.png",
                metric_key="domain_f1",
                title="Domain F1 by Rank and Placement",
                ylabel="Domain F1",
            ),
        ),
        (
            "rank_vs_forgetting",
            lambda: _plot_rank_metric_by_domain(
                adapter_rows,
                summary_dir / "rank_vs_forgetting.png",
                metric_key="forgetting_acc_weighted",
                title="Weighted Forgetting by Rank and Placement",
                ylabel="Weighted forgetting",
            ),
        ),
        (
            "placement_rank_domain_f1_heatmap",
            lambda: _plot_heatmap(
                adapter_rows,
                summary_dir / "placement_rank_domain_f1_heatmap.png",
                metric_key="domain_f1",
                title="Placement x Rank x Domain F1",
            ),
        ),
        (
            "placement_rank_composite_heatmap",
            lambda: _plot_heatmap(
                adapter_rows,
                summary_dir / "placement_rank_composite_heatmap.png",
                metric_key="composite_score",
                title="Placement x Rank Composite Score",
            ),
        ),
    ]
    for stem, plot_fn in plot_jobs:
        stale_path = summary_dir / f"{stem}.png"
        if stale_path.exists():
            stale_path.unlink()
        if plot_fn():
            png_path = summary_dir / f"{stem}.png"
            if png_path.exists():
                plot_paths[png_path.name] = str(png_path)

    summary_md_path = _write_summary_markdown(
        rows=rows,
        adapter_rows=adapter_rows,
        base_rows=base_rows,
        summary_dir=summary_dir,
        include_aux_runs=include_aux_runs,
    )

    return {
        "runs_count": len(rows),
        "adapter_run_count": len(adapter_rows),
        "domain_count": len(base_rows),
        "base_runs": {domain: row["run_name"] for domain, row in base_rows.items()},
        "runs_summary_path": _relative_path(runs_summary_path, project_root),
        "summary_markdown_path": _relative_path(summary_md_path, project_root),
        "grouped_csv_paths": {name: _relative_path(path, project_root) for name, path in grouped_paths.items()},
        "plot_paths": {name: _relative_path(Path(path), project_root) for name, path in plot_paths.items()},
    }
