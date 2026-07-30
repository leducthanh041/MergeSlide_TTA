#!/usr/bin/env python3
"""Plot routing-confidence landscape before/after MergeSlide_TTA.

This renderer uses saved outputs from tools/plot/plot_prompt_embedding_space.py
and avoids rerunning TITAN/TTA.  Instead of plotting raw slide embeddings, it
projects the 6D task-routing probability vectors into 2D and draws a background
confidence field.  The plot directly visualizes whether TTA moves WSIs toward
the correct task-routing region.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import matplotlib.patheffects as path_effects
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


LEFT_ZOOM_BOX_COLOR = "#0057b8"
RIGHT_ZOOM_BOX_COLOR = "#ff2b2b"
SOURCE_BOX_LINEWIDTH = 3.40
SOURCE_BOX_DASH = (0, (2.0, 2.0))
CONNECTOR_LINEWIDTH = 2.80
CONNECTOR_DASH = (0, (10.0, 5.0))
ZOOM_BORDER_LINEWIDTH = 3.80
ZOOM_BORDER_DASH = (0, (4.0, 2.0))
SHOW_CONFIDENCE_BACKGROUND = True
BACKGROUND_ALPHA_MIN = 0.025
BACKGROUND_ALPHA_MAX = 0.20


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def record_key(row: dict) -> tuple[int, int]:
    return as_int(row, "task_id"), as_int(row, "slide_index")


def collect_task_info(records: list[dict]) -> tuple[list[int], dict[int, str]]:
    task_ids = sorted({as_int(r, "task_id") for r in records})
    names = {}
    for row in records:
        task_id = as_int(row, "task_id")
        names[task_id] = row.get("task_name", f"Task {task_id}")
    return task_ids, names


def palette_by_task() -> dict[int, str]:
    # Saturated colors, chosen to stay distinct when printed.
    return {
        0: "#0057b8",  # BRCA
        1: "#ff7f00",  # RCC
        2: "#009e3d",  # NSCLC
        3: "#d7191c",  # ESCA
        4: "#6a00a8",  # TGCT
        5: "#7f3b08",  # CESC
    }


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / max(float(temperature), 1e-8)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)


def routing_probs(raw: np.lib.npyio.NpzFile, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    if "baseline_probs" in raw.files and "tta_probs" in raw.files:
        return raw["baseline_probs"].astype(np.float32), raw["tta_probs"].astype(np.float32)
    baseline_scores = raw["baseline_vectors"].astype(np.float32) @ raw["source_prompts"].astype(np.float32).T
    tta_scores = raw["tta_vectors"].astype(np.float32) @ raw["adapted_prompts"].astype(np.float32).T
    return softmax_np(baseline_scores, temperature), softmax_np(tta_scores, temperature)


def routing_stats(records: list[dict]) -> dict[str, float]:
    correct = [
        as_int(row, "pred_task") == as_int(row, "task_id")
        for row in records
    ]
    margins = [float(row["true_vs_wrong_margin"]) for row in records]
    return {
        "routing_acc": 100.0 * float(np.mean(correct)) if correct else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
    }


def routing_sets(
    baseline_records: list[dict],
    tta_records: list[dict],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    base_wrong = {
        record_key(row) for row in baseline_records
        if as_int(row, "pred_task") != as_int(row, "task_id")
    }
    tta_wrong = {
        record_key(row) for row in tta_records
        if as_int(row, "pred_task") != as_int(row, "task_id")
    }
    corrected = base_wrong - tta_wrong
    return base_wrong, tta_wrong, corrected


def fit_routing_projection(
    baseline_probs: np.ndarray,
    tta_probs: np.ndarray,
    reducer: str,
    tsne_perplexity: float,
    seed: int,
) -> tuple[PCA | None, np.ndarray, np.ndarray]:
    all_probs = np.concatenate([baseline_probs, tta_probs], axis=0)
    reducer = reducer.lower()
    if reducer == "pca":
        model = PCA(n_components=2, random_state=seed)
        all_xy = model.fit_transform(all_probs)
    elif reducer == "tsne":
        max_perplexity = max(2.0, (all_probs.shape[0] - 1) / 3.0)
        model = None
        all_xy = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=min(float(tsne_perplexity), max_perplexity),
            early_exaggeration=18.0,
        ).fit_transform(all_probs)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")
    n_base = baseline_probs.shape[0]
    return model, all_xy[:n_base], all_xy[n_base:]


def probs_to_background(
    prob: np.ndarray,
    palette: dict[int, str],
    n_tasks: int,
    grid_shape: tuple[int, int],
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    pred = np.argmax(prob, axis=1)
    conf = np.max(prob, axis=1)
    conf_norm = (conf - (1.0 / max(n_tasks, 1))) / max(1.0 - (1.0 / max(n_tasks, 1)), 1e-8)
    conf_norm = np.clip(conf_norm, 0.0, 1.0)
    alpha = alpha_min + (alpha_max - alpha_min) * conf_norm

    colors = np.ones((prob.shape[0], 3), dtype=np.float32)
    for task_id in range(n_tasks):
        hex_color = palette.get(task_id, "#777777").lstrip("#")
        rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0
        mask = pred == task_id
        colors[mask] = (1.0 - alpha[mask, None]) * 1.0 + alpha[mask, None] * rgb
    return colors.reshape(grid_shape[0], grid_shape[1], 3)


def confidence_background_pca(
    pca: PCA,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    palette: dict[int, str],
    n_tasks: int,
    grid_size: int,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Approximate the local routing distribution by inverting the PCA plane.
    prob = pca.inverse_transform(grid)
    prob = np.clip(prob, 1e-8, None)
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)

    return probs_to_background(
        prob=prob,
        palette=palette,
        n_tasks=n_tasks,
        grid_shape=(grid_size, grid_size),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )


def confidence_background_knn(
    known_xy: np.ndarray,
    known_probs: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    palette: dict[int, str],
    n_tasks: int,
    grid_size: int,
    alpha_min: float,
    alpha_max: float,
    k_neighbors: int,
    chunk_size: int = 20000,
) -> np.ndarray:
    """Interpolate a confidence field in non-invertible spaces such as t-SNE."""
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    k = min(max(int(k_neighbors), 1), known_xy.shape[0])
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(known_xy)

    out_prob = np.empty((grid.shape[0], known_probs.shape[1]), dtype=np.float32)
    for start in range(0, grid.shape[0], chunk_size):
        end = min(start + chunk_size, grid.shape[0])
        dist, idx = nn.kneighbors(grid[start:end], return_distance=True)
        scale = np.maximum(np.median(dist[:, -1]), 1e-6)
        weights = np.exp(-(dist ** 2) / (2.0 * scale ** 2)).astype(np.float32)
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        out_prob[start:end] = np.sum(known_probs[idx] * weights[:, :, None], axis=1)

    out_prob = np.clip(out_prob, 1e-8, None)
    out_prob = out_prob / np.clip(out_prob.sum(axis=1, keepdims=True), 1e-12, None)
    return probs_to_background(
        prob=out_prob,
        palette=palette,
        n_tasks=n_tasks,
        grid_shape=(grid_size, grid_size),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )


def shared_limits(*arrays: np.ndarray, pad: float = 0.10) -> tuple[tuple[float, float], tuple[float, float]]:
    all_xy = np.concatenate(arrays, axis=0)
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    mins = mins - pad * span
    maxs = maxs + pad * span
    return (float(mins[0]), float(maxs[0])), (float(mins[1]), float(maxs[1]))


def contract_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    factor: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Zoom the view around the shared center without changing point coordinates."""
    factor = float(np.clip(factor, 0.35, 1.0))
    x_center = (xlim[0] + xlim[1]) / 2.0
    y_center = (ylim[0] + ylim[1]) / 2.0
    x_half = (xlim[1] - xlim[0]) * factor / 2.0
    y_half = (ylim[1] - ylim[0]) * factor / 2.0
    return (
        (float(x_center - x_half), float(x_center + x_half)),
        (float(y_center - y_half), float(y_center + y_half)),
    )


def draw_panel(
    ax,
    title: str,
    records: list[dict],
    xy: np.ndarray,
    background: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    task_ids: list[int],
    task_names: dict[int, str],
    palette: dict[int, str],
    corrected_keys: set[tuple[int, int]],
) -> None:
    if SHOW_CONFIDENCE_BACKGROUND:
        ax.imshow(
            background,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            zorder=0,
        )
    ax.set_facecolor("white")

    for task_id in task_ids:
        idxs = [i for i, row in enumerate(records) if as_int(row, "task_id") == task_id]
        if not idxs:
            continue
        pts = xy[idxs]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=78,
            c=palette.get(task_id, "#777777"),
            alpha=0.98,
            marker="o",
            edgecolors="white",
            linewidths=0.40,
            label=task_names[task_id],
            zorder=4,
        )

    keys = [record_key(row) for row in records]
    corrected_idx = [i for i, key in enumerate(keys) if key in corrected_keys]

    if corrected_idx:
        pts = xy[corrected_idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=170,
            marker="o",
            facecolors="none",
            edgecolors="#00c853",
            linewidths=2.55,
            alpha=1.0,
            zorder=8,
        )

    ax.set_title(
        title,
        fontsize=34.0,
        fontweight="normal" if title == "MergeSlide" else "bold",
        pad=11.0,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.set_box_aspect(1)


def draw_corrected_arrows(
    ax,
    baseline_records: list[dict],
    tta_records: list[dict],
    baseline_xy: np.ndarray,
    tta_xy: np.ndarray,
    corrected_keys: set[tuple[int, int]],
    max_arrows: int,
) -> None:
    if not corrected_keys:
        return
    base_index = {record_key(row): i for i, row in enumerate(baseline_records)}
    tta_index = {record_key(row): i for i, row in enumerate(tta_records)}
    keys = sorted(k for k in corrected_keys if k in base_index and k in tta_index)
    if max_arrows > 0 and len(keys) > max_arrows:
        # Deterministic subsample to keep the plot readable.
        step = max(len(keys) // max_arrows, 1)
        keys = keys[::step][:max_arrows]
    for key in keys:
        b = baseline_xy[base_index[key]]
        t = tta_xy[tta_index[key]]
        ax.annotate(
            "",
            xy=(t[0], t[1]),
            xytext=(b[0], b[1]),
            arrowprops=dict(
                arrowstyle="->",
                color="#00a651",
                lw=1.35,
                alpha=0.58,
                shrinkA=1.5,
                shrinkB=1.5,
            ),
            zorder=6,
        )


def select_zoom_corrected_keys(
    tta_records: list[dict],
    tta_xy: np.ndarray,
    corrected_keys: set[tuple[int, int]],
    max_points: int,
) -> list[tuple[int, int]]:
    """Select a dense single-task subset of corrected WSIs for the zoom inset."""
    key_to_idx = {record_key(row): i for i, row in enumerate(tta_records)}
    key_to_task = {record_key(row): as_int(row, "task_id") for row in tta_records}
    keys = sorted(k for k in corrected_keys if k in key_to_idx)
    if not keys:
        return []

    grouped: dict[int, list[tuple[int, int]]] = {}
    for key in keys:
        grouped.setdefault(key_to_task[key], []).append(key)
    # Show the task where MergeSlide_TTA corrected the most WSIs. This creates a
    # readable zoom rather than a very large box spanning multiple task regions.
    keys = sorted(grouped.values(), key=lambda group: (len(group), -key_to_task[group[0]]), reverse=True)[0]

    if not keys or max_points <= 0 or len(keys) <= max_points:
        return keys

    pts = np.asarray([tta_xy[key_to_idx[k]] for k in keys], dtype=np.float32)
    k = min(max(int(max_points), 1), pts.shape[0])
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(pts)
    dist, idx = nn.kneighbors(pts, return_distance=True)
    best = int(np.argmin(dist[:, -1]))
    selected = [keys[int(i)] for i in idx[best]]
    return sorted(selected)


def select_dense_corrected_task_region(
    tta_records: list[dict],
    tta_xy: np.ndarray,
    corrected_keys: set[tuple[int, int]],
    task_name: str,
    fallback_task_id: int,
    max_points: int,
) -> tuple[int | None, list[tuple[int, int]]]:
    """Select a dense task region anchored at a corrected WSI."""
    task_id = None
    for row in tta_records:
        if row.get("task_name", "").lower() == task_name.lower():
            task_id = as_int(row, "task_id")
            break
    if task_id is None:
        task_ids = {as_int(row, "task_id") for row in tta_records}
        task_id = fallback_task_id if fallback_task_id in task_ids else None
    if task_id is None:
        return None, []

    tta_index = {record_key(row): i for i, row in enumerate(tta_records)}
    task_keys = [
        key for key, idx in tta_index.items()
        if as_int(tta_records[idx], "task_id") == task_id
    ]
    if not task_keys:
        return task_id, []

    corrected_task_keys = [key for key in task_keys if key in corrected_keys]
    task_pts = np.asarray([tta_xy[tta_index[key]] for key in task_keys], dtype=np.float32)
    k = min(max(int(max_points), 2), len(task_keys))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(task_pts)

    anchor_keys = corrected_task_keys if corrected_task_keys else task_keys
    anchor_pts = np.asarray([tta_xy[tta_index[key]] for key in anchor_keys], dtype=np.float32)
    dist, neighbors = nn.kneighbors(anchor_pts, return_distance=True)
    best_anchor = int(np.argmin(dist[:, -1]))
    selected = [task_keys[int(i)] for i in neighbors[best_anchor]]
    return task_id, sorted(selected)


def square_limits_for_points(points: np.ndarray, min_side: float, pad_ratio: float) -> tuple[tuple[float, float], tuple[float, float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, min_side)
    side = float(max(span[0], span[1]) * (1.0 + pad_ratio))
    side = max(side, float(min_side))
    return (
        (float(center[0] - side / 2.0), float(center[0] + side / 2.0)),
        (float(center[1] - side / 2.0), float(center[1] + side / 2.0)),
    )


def clip_segment_to_box(
    start: np.ndarray,
    end: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a 2D segment to an axis-aligned box using Liang-Barsky."""
    delta = end - start
    p = (-delta[0], delta[0], -delta[1], delta[1])
    q = (
        start[0] - xlim[0],
        xlim[1] - start[0],
        start[1] - ylim[0],
        ylim[1] - start[1],
    )
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(float(pi)) < 1e-12:
            if qi < 0:
                return None
            continue
        u = float(qi / pi)
        if pi < 0:
            u0 = max(u0, u)
        else:
            u1 = min(u1, u)
        if u0 > u1:
            return None
    return start + u0 * delta, start + u1 * delta


def draw_zoom_connectors(
    fig,
    source_ax,
    zoom_ax,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: str,
) -> None:
    """Connect a source dashed box to its zoom crop."""
    source_points = ((xlim[0], ylim[0]), (xlim[1], ylim[0]))
    zoom_points = ((0.0, 1.0), (1.0, 1.0))

    for source_xy, zoom_xy in zip(source_points, zoom_points):
        connector = ConnectionPatch(
            xyA=source_xy,
            xyB=zoom_xy,
            coordsA="data",
            coordsB="axes fraction",
            axesA=source_ax,
            axesB=zoom_ax,
            color=color,
            linewidth=CONNECTOR_LINEWIDTH,
            linestyle=CONNECTOR_DASH,
            alpha=0.96,
            clip_on=False,
            zorder=30,
        )
        connector.set_path_effects([
            path_effects.Stroke(linewidth=CONNECTOR_LINEWIDTH + 2.00, foreground="white", alpha=0.86),
            path_effects.Normal(),
        ])
        fig.add_artist(connector)


def emphasize_dashed_border(artist, stroke_width: float = 4.00) -> None:
    artist.set_path_effects([
        path_effects.Stroke(linewidth=stroke_width, foreground="white", alpha=0.90),
        path_effects.Normal(),
    ])


def source_box_visual_aspect(
    global_xlim: tuple[float, float],
    global_ylim: tuple[float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> float:
    """Return height/width aspect of a source rectangle as seen in the main panel."""
    global_width = max(float(global_xlim[1] - global_xlim[0]), 1e-8)
    global_height = max(float(global_ylim[1] - global_ylim[0]), 1e-8)
    box_width = max(float(xlim[1] - xlim[0]), 1e-8)
    box_height = max(float(ylim[1] - ylim[0]), 1e-8)
    aspect = (box_height / global_height) / (box_width / global_width)
    return float(np.clip(aspect, 0.55, 1.45))


def draw_corrected_zoom_inset(
    fig,
    ax,
    background: np.ndarray,
    global_xlim: tuple[float, float],
    global_ylim: tuple[float, float],
    tta_records: list[dict],
    baseline_records: list[dict],
    tta_xy: np.ndarray,
    baseline_xy: np.ndarray,
    corrected_keys: set[tuple[int, int]],
    task_ids: list[int],
    palette: dict[int, str],
    max_points: int,
    min_side_frac: float,
    pad_ratio: float,
    inset_position: tuple[float, float, float, float],
    color: str,
) -> None:
    selected_keys = select_zoom_corrected_keys(tta_records, tta_xy, corrected_keys, max_points=max_points)
    if len(selected_keys) < 2:
        return

    tta_index = {record_key(row): i for i, row in enumerate(tta_records)}
    base_index = {record_key(row): i for i, row in enumerate(baseline_records)}
    selected_tta_idx = [tta_index[k] for k in selected_keys if k in tta_index]
    selected_base_idx = [base_index[k] for k in selected_keys if k in base_index]
    selected_pts = tta_xy[selected_tta_idx]
    selected_base_pts = baseline_xy[selected_base_idx] if selected_base_idx else selected_pts

    global_side = max(global_xlim[1] - global_xlim[0], global_ylim[1] - global_ylim[0])
    xlim, ylim = square_limits_for_points(
        np.concatenate([selected_pts, selected_base_pts], axis=0),
        min_side=max(global_side * min_side_frac, 1e-6),
        pad_ratio=pad_ratio,
    )
    red_box_scale = 1.00
    center_x = (xlim[0] + xlim[1]) / 2.0
    center_y = (ylim[0] + ylim[1]) / 2.0
    half_width = (xlim[1] - xlim[0]) * red_box_scale / 2.0
    half_height = (ylim[1] - ylim[0]) * red_box_scale / 2.0
    xlim = (center_x - half_width, center_x + half_width)
    ylim = (center_y - half_height, center_y + half_height)

    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        fill=False,
        linestyle=SOURCE_BOX_DASH,
        linewidth=SOURCE_BOX_LINEWIDTH,
        edgecolor=color,
        alpha=1.0,
        zorder=10,
    )
    emphasize_dashed_border(rect, stroke_width=SOURCE_BOX_LINEWIDTH + 1.80)
    ax.add_patch(rect)

    inset = fig.add_axes(inset_position)
    if SHOW_CONFIDENCE_BACKGROUND:
        inset.imshow(
            background,
            extent=(global_xlim[0], global_xlim[1], global_ylim[0], global_ylim[1]),
            origin="lower",
            interpolation="bicubic",
            aspect="auto",
            zorder=0,
        )

    for task_id in task_ids:
        idxs = [
            i for i, row in enumerate(tta_records)
            if as_int(row, "task_id") == task_id
            and xlim[0] <= tta_xy[i, 0] <= xlim[1]
            and ylim[0] <= tta_xy[i, 1] <= ylim[1]
        ]
        if not idxs:
            continue
        pts = tta_xy[idxs]
        inset.scatter(
            pts[:, 0],
            pts[:, 1],
            s=140,
            c=palette.get(task_id, "#777777"),
            alpha=0.96,
            marker="o",
            edgecolors="white",
            linewidths=0.55,
            zorder=4,
        )

    # Faint pre-TTA baseline positions for the corrected WSIs. These are shown
    # only inside the zoom inset to explain the correction movement without
    # cluttering the main panel.
    for task_id in task_ids:
        base_idxs = [
            base_index[key] for key in selected_keys
            if key in base_index and as_int(baseline_records[base_index[key]], "task_id") == task_id
        ]
        if not base_idxs:
            continue
        pts = baseline_xy[base_idxs]
        inset.scatter(
            pts[:, 0],
            pts[:, 1],
            s=120,
            c=palette.get(task_id, "#777777"),
            alpha=0.40,
            marker="o",
            edgecolors="none",
            zorder=5,
        )

    for key in selected_keys:
        if key not in tta_index or key not in base_index:
            continue
        b = baseline_xy[base_index[key]]
        t = tta_xy[tta_index[key]]
        inset.annotate(
            "",
            xy=(t[0], t[1]),
            xytext=(b[0], b[1]),
            arrowprops=dict(
                arrowstyle="->",
                color="#00a651",
                lw=1.75,
                alpha=0.88,
                shrinkA=0.7,
                shrinkB=0.7,
            ),
            zorder=5,
            clip_on=True,
        )

    ring_pts = tta_xy[selected_tta_idx]
    inset.scatter(
        ring_pts[:, 0],
        ring_pts[:, 1],
        s=260,
        marker="o",
        facecolors="none",
        edgecolors="#00c853",
        linewidths=3.05,
        alpha=1.0,
        zorder=8,
    )

    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.set_box_aspect(source_box_visual_aspect(global_xlim, global_ylim, xlim, ylim))
    inset.set_anchor("N")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_linewidth(ZOOM_BORDER_LINEWIDTH)
        spine.set_linestyle(ZOOM_BORDER_DASH)
        spine.set_edgecolor(color)
        emphasize_dashed_border(spine, stroke_width=ZOOM_BORDER_LINEWIDTH + 1.70)

    draw_zoom_connectors(fig, ax, inset, xlim, ylim, color=color)


def draw_dense_corrected_zoom_inset(
    fig,
    ax,
    background: np.ndarray,
    global_xlim: tuple[float, float],
    global_ylim: tuple[float, float],
    tta_records: list[dict],
    baseline_records: list[dict],
    tta_xy: np.ndarray,
    baseline_xy: np.ndarray,
    corrected_keys: set[tuple[int, int]],
    task_ids: list[int],
    palette: dict[int, str],
    task_name: str,
    fallback_task_id: int,
    max_points: int,
    min_side_frac: float,
    pad_ratio: float,
    inset_position: tuple[float, float, float, float],
    color: str,
) -> None:
    task_id, selected_keys = select_dense_corrected_task_region(
        tta_records=tta_records,
        tta_xy=tta_xy,
        corrected_keys=corrected_keys,
        task_name=task_name,
        fallback_task_id=fallback_task_id,
        max_points=max_points,
    )
    if task_id is None or len(selected_keys) < 2:
        return

    tta_index = {record_key(row): i for i, row in enumerate(tta_records)}
    base_index = {record_key(row): i for i, row in enumerate(baseline_records)}
    selected_tta_idx = [tta_index[k] for k in selected_keys if k in tta_index]
    selected_tta_pts = tta_xy[selected_tta_idx]

    movement_keys = [key for key in selected_keys if key in corrected_keys and key in tta_index and key in base_index]
    if movement_keys:
        # Keep the blue source box local to the corrected endpoint cluster.
        # The movement arrows can enter this crop from outside, but including
        # far-away baseline starts would make the dashed region too large.
        limit_points = np.asarray([tta_xy[tta_index[key]] for key in movement_keys], dtype=np.float32)
    else:
        limit_points = selected_tta_pts

    global_side = max(global_xlim[1] - global_xlim[0], global_ylim[1] - global_ylim[0])
    xlim, ylim = square_limits_for_points(
        limit_points,
        min_side=max(global_side * min_side_frac, 1e-6),
        pad_ratio=pad_ratio,
    )
    box_scale = 1.00 if movement_keys else 0.92
    x_shift = 0.004 * (global_xlim[1] - global_xlim[0])
    y_shift = 0.002 * (global_ylim[1] - global_ylim[0])
    center_x = (xlim[0] + xlim[1]) / 2.0 + x_shift
    center_y = (ylim[0] + ylim[1]) / 2.0 + y_shift
    half_width = (xlim[1] - xlim[0]) * box_scale / 2.0
    half_height = (ylim[1] - ylim[0]) * box_scale / 2.0
    xlim = (center_x - half_width, center_x + half_width)
    ylim = (center_y - half_height, center_y + half_height)

    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        fill=False,
        linestyle=SOURCE_BOX_DASH,
        linewidth=SOURCE_BOX_LINEWIDTH,
        edgecolor=color,
        alpha=1.0,
        zorder=10,
    )
    emphasize_dashed_border(rect, stroke_width=SOURCE_BOX_LINEWIDTH + 1.80)
    ax.add_patch(rect)

    inset = fig.add_axes(inset_position)
    if SHOW_CONFIDENCE_BACKGROUND:
        inset.imshow(
            background,
            extent=(global_xlim[0], global_xlim[1], global_ylim[0], global_ylim[1]),
            origin="lower",
            interpolation="bicubic",
            aspect="auto",
            zorder=0,
        )

    # Show the crowded green-gradient region itself. Other tasks stay faint,
    # while the target-task points remain visible as small context markers.
    for draw_task_id in task_ids:
        context_idx = [
            i for i, row in enumerate(tta_records)
            if as_int(row, "task_id") == draw_task_id
            and xlim[0] <= tta_xy[i, 0] <= xlim[1]
            and ylim[0] <= tta_xy[i, 1] <= ylim[1]
        ]
        if not context_idx:
            continue
        pts = tta_xy[context_idx]
        inset.scatter(
            pts[:, 0],
            pts[:, 1],
            s=140.0,
            c=palette.get(draw_task_id, "#777777"),
            alpha=0.96 if draw_task_id == task_id else 0.42,
            marker="o",
            edgecolors="white" if draw_task_id == task_id else "none",
            linewidths=0.55 if draw_task_id == task_id else 0.0,
            zorder=4 if draw_task_id == task_id else 2,
        )

    corrected_local_keys = [
        key for key in corrected_keys
        if key in tta_index and key in base_index
        and xlim[0] <= tta_xy[tta_index[key], 0] <= xlim[1]
        and ylim[0] <= tta_xy[tta_index[key], 1] <= ylim[1]
    ]
    if corrected_local_keys:
        for key in corrected_local_keys:
            baseline_point = baseline_xy[base_index[key]]
            adapted_point = tta_xy[tta_index[key]]
            clipped = clip_segment_to_box(baseline_point, adapted_point, xlim, ylim)
            if clipped is None:
                continue
            arrow_start, arrow_end = clipped
            if (
                xlim[0] <= baseline_point[0] <= xlim[1]
                and ylim[0] <= baseline_point[1] <= ylim[1]
            ):
                inset.scatter(
                    baseline_point[0],
                    baseline_point[1],
                    s=120,
                    c=palette.get(as_int(tta_records[tta_index[key]], "task_id"), "#777777"),
                    alpha=0.40,
                    edgecolors="none",
                    zorder=5,
                )
            inset.annotate(
                "",
                xy=(arrow_end[0], arrow_end[1]),
                xytext=(arrow_start[0], arrow_start[1]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#00a651",
                    lw=1.75,
                    alpha=0.90,
                    shrinkA=0.4,
                    shrinkB=0.8,
                ),
                zorder=6,
            )

        corrected_pts = np.asarray(
            [tta_xy[tta_index[key]] for key in corrected_local_keys],
            dtype=np.float32,
        )
        inset.scatter(
            corrected_pts[:, 0],
            corrected_pts[:, 1],
            s=260,
            marker="o",
            facecolors="none",
            edgecolors="#00c853",
            linewidths=3.05,
            zorder=8,
        )

    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.set_box_aspect(source_box_visual_aspect(global_xlim, global_ylim, xlim, ylim))
    inset.set_anchor("N")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_linewidth(ZOOM_BORDER_LINEWIDTH)
        spine.set_linestyle(ZOOM_BORDER_DASH)
        spine.set_edgecolor(color)
        emphasize_dashed_border(spine, stroke_width=ZOOM_BORDER_LINEWIDTH + 1.70)

    draw_zoom_connectors(fig, ax, inset, xlim, ylim, color=color)


def plot_landscape(
    input_dir: Path,
    output_dir: Path,
    tag: str,
    reducer: str,
    temperature: float,
    grid_size: int,
    tsne_perplexity: float,
    seed: int,
    knn_k: int,
    max_arrows: int,
    zoom_max_points: int,
    zoom_min_side_frac: float,
    zoom_pad_ratio: float,
    task_zoom_max_points: int,
    task_zoom_min_side_frac: float,
    task_zoom_pad_ratio: float,
    main_view_zoom: float,
    width: float,
    height: float,
    dpi: int,
) -> tuple[Path, Path]:
    baseline_records = read_csv(input_dir / f"{tag}_baseline_points.csv")
    tta_records = read_csv(input_dir / f"{tag}_tta_points.csv")
    raw = np.load(input_dir / f"{tag}_raw_embeddings.npz")

    baseline_probs, tta_probs = routing_probs(raw, temperature=temperature)
    projection_model, baseline_xy, tta_xy = fit_routing_projection(
        baseline_probs,
        tta_probs,
        reducer=reducer,
        tsne_perplexity=tsne_perplexity,
        seed=seed,
    )
    task_ids, task_names = collect_task_info(baseline_records + tta_records)
    palette = palette_by_task()
    base_wrong, tta_wrong, corrected = routing_sets(baseline_records, tta_records)

    xlim, ylim = shared_limits(baseline_xy, tta_xy, pad=0.02)
    xlim, ylim = contract_limits(xlim, ylim, factor=main_view_zoom)
    if reducer == "pca":
        if projection_model is None:
            raise RuntimeError("PCA projection model was not created.")
        background = confidence_background_pca(
            pca=projection_model,
            xlim=xlim,
            ylim=ylim,
            palette=palette,
            n_tasks=len(task_ids),
            grid_size=grid_size,
            alpha_min=BACKGROUND_ALPHA_MIN,
            alpha_max=BACKGROUND_ALPHA_MAX,
        )
    else:
        background = confidence_background_knn(
            known_xy=np.concatenate([baseline_xy, tta_xy], axis=0),
            known_probs=np.concatenate([baseline_probs, tta_probs], axis=0),
            xlim=xlim,
            ylim=ylim,
            palette=palette,
            n_tasks=len(task_ids),
            grid_size=grid_size,
            alpha_min=BACKGROUND_ALPHA_MIN,
            alpha_max=BACKGROUND_ALPHA_MAX,
            k_neighbors=knn_k,
        )

    fig, axes = plt.subplots(1, 2, figsize=(width, height), constrained_layout=False)
    axes = np.asarray(axes).reshape(-1)

    # 2x2 layout: method panels on the top row and zoom crops on the bottom
    # row.  The zoom panels intentionally reuse the same physical size and gap
    # as the main panels so the comparison reads as a balanced figure.
    main_left = 0.032
    main_width = 0.445
    main_height = main_width * (width / height)
    panel_gap = 0.026
    adapt_left = main_left + main_width + panel_gap
    zoom_bottom = 0.160
    row_gap = 0.014
    main_bottom = zoom_bottom + main_height + row_gap
    blue_zoom_position = (main_left, zoom_bottom, main_width, main_height)
    red_zoom_position = (adapt_left, zoom_bottom, main_width, main_height)

    axes[0].set_position([main_left, main_bottom, main_width, main_height])
    axes[1].set_position([adapt_left, main_bottom, main_width, main_height])

    draw_panel(
        axes[0],
        "MergeSlide",
        baseline_records,
        baseline_xy,
        background,
        xlim,
        ylim,
        task_ids,
        task_names,
        palette,
        corrected,
    )
    draw_panel(
        axes[1],
        "MergeSlide_TTA (ours)",
        tta_records,
        tta_xy,
        background,
        xlim,
        ylim,
        task_ids,
        task_names,
        palette,
        corrected,
    )
    draw_corrected_arrows(
        axes[1],
        baseline_records,
        tta_records,
        baseline_xy,
        tta_xy,
        corrected,
        max_arrows=max_arrows,
    )
    draw_corrected_zoom_inset(
        fig,
        axes[1],
        background,
        xlim,
        ylim,
        tta_records,
        baseline_records,
        tta_xy,
        baseline_xy,
        corrected,
        task_ids,
        palette,
        max_points=zoom_max_points,
        min_side_frac=zoom_min_side_frac,
        pad_ratio=zoom_pad_ratio,
        inset_position=red_zoom_position,
        color=RIGHT_ZOOM_BOX_COLOR,
    )
    draw_dense_corrected_zoom_inset(
        fig,
        axes[1],
        background,
        xlim,
        ylim,
        tta_records,
        baseline_records,
        tta_xy,
        baseline_xy,
        corrected,
        task_ids,
        palette,
        task_name="NSCLC",
        fallback_task_id=2,
        max_points=task_zoom_max_points,
        min_side_frac=task_zoom_min_side_frac,
        pad_ratio=task_zoom_pad_ratio,
        inset_position=blue_zoom_position,
        color=LEFT_ZOOM_BOX_COLOR,
    )

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=palette.get(task_id, "#777777"),
            markeredgecolor="white",
            markeredgewidth=0.70,
            markersize=16.5,
            label=task_names[task_id],
        )
        for task_id in task_ids
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#00c853",
            markeredgewidth=3.10,
            markersize=22.0,
            label="Corrected by MergeSlide_TTA",
        ),
        Line2D([0], [0], color="#00a651", lw=4.1, alpha=0.86, label="Correction movement"),
    ]
    fig.legend(
        task_handles + status_handles,
        [h.get_label() for h in task_handles + status_handles],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=22.0,
        handlelength=1.45,
        borderpad=0.0,
        columnspacing=1.08,
        labelspacing=0.20,
        bbox_to_anchor=(0.5, 0.128),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{tag}_{reducer}_routing_confidence_landscape.png"
    pdf = output_dir / f"{tag}_{reducer}_routing_confidence_landscape.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="logs/Ablations/prompt_embedding_space")
    parser.add_argument("--output_dir", default="logs/Ablations/routing_confidence_landscape")
    parser.add_argument("--tag", default="ind_forward_fold6_tsne")
    parser.add_argument("--reducer", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=520)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn_k", type=int, default=32)
    parser.add_argument("--max_arrows", type=int, default=80)
    parser.add_argument("--zoom_max_points", type=int, default=8)
    parser.add_argument("--zoom_min_side_frac", type=float, default=0.052)
    parser.add_argument("--zoom_pad_ratio", type=float, default=0.20)
    parser.add_argument("--task_zoom_max_points", type=int, default=24)
    parser.add_argument("--task_zoom_min_side_frac", type=float, default=0.038)
    parser.add_argument("--task_zoom_pad_ratio", type=float, default=0.20)
    parser.add_argument("--main_view_zoom", type=float, default=0.84)
    parser.add_argument("--width", type=float, default=11.6)
    parser.add_argument("--height", type=float, default=13.4)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    png, pdf = plot_landscape(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        tag=args.tag,
        reducer=args.reducer,
        temperature=args.temperature,
        grid_size=args.grid_size,
        tsne_perplexity=args.tsne_perplexity,
        seed=args.seed,
        knn_k=args.knn_k,
        max_arrows=args.max_arrows,
        zoom_max_points=args.zoom_max_points,
        zoom_min_side_frac=args.zoom_min_side_frac,
        zoom_pad_ratio=args.zoom_pad_ratio,
        task_zoom_max_points=args.task_zoom_max_points,
        task_zoom_min_side_frac=args.task_zoom_min_side_frac,
        task_zoom_pad_ratio=args.task_zoom_pad_ratio,
        main_view_zoom=args.main_view_zoom,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
    )
    print(f"[DONE] {png}")
    print(f"[DONE] {pdf}")


if __name__ == "__main__":
    main()
