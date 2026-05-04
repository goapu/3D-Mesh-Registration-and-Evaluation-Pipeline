
"""
DistanceEvaluation.py
=====================

Author: Dilip Goswami

Evaluate 3D registration quality by computing unsigned vertex-to-surface
distances between a registered intraoperative mesh and a preoperative mucosa mesh.

Main evaluation:
    - One-way surface distance:
        registered intraoperative vertices -> preoperative mucosa surface

    - Optional symmetric surface distance:
        registered intraoperative vertices -> preoperative mucosa surface
        plus
        preoperative mucosa vertices -> registered intraoperative surface

Outputs:
    - summary_metrics_surface.csv
    - histogram_surface.png
    - cdf_surface.png, unless disabled
    - registered_model_heatmap.obj with per-vertex colors for MeshLab
    - heatmap_legend.txt

Optional correspondence evaluation:
    If an Excel file with 1-based vertex-index correspondences is provided,
    the script also computes vertex-to-vertex distances for those pairs.

Notes:
    - Input meshes must already be in the same coordinate system.
    - Distances are reported in the same unit as the mesh coordinates, assumed mm.
    - For partial intraoperative reconstructions, one-way intra->pre distance is
      usually more appropriate than symmetric distance.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pandas as pd
import trimesh


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Mesh IO
# =============================================================================

def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """
    Load a mesh robustly from disk.

    If the file loads as a trimesh.Scene, all geometries are concatenated into
    a single Trimesh object.

    Parameters
    ----------
    path:
        Path to the input mesh file.

    Returns
    -------
    trimesh.Trimesh
        Loaded triangle mesh.

    Raises
    ------
    FileNotFoundError
        If the mesh path does not exist.
    TypeError
        If the loaded object cannot be converted to a Trimesh.
    ValueError
        If the mesh has no vertices or faces.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load_mesh(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Empty scene loaded from: {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"No vertices found in mesh: {path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(
            f"No faces found in mesh: {path}. "
            "Triangle faces are required for surface-distance evaluation."
        )

    return mesh


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Create and return the output directory.

    If an empty path is provided, the script directory is used.
    """
    output_dir = str(output_dir).strip()

    if not output_dir:
        output_path = Path(__file__).resolve().parent
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# =============================================================================
# Metrics
# =============================================================================

def finite_distances(distances: np.ndarray) -> np.ndarray:
    """
    Convert distances to float64 and remove NaN/Inf values.
    """
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    return d[np.isfinite(d)]


def compute_metrics(distances: np.ndarray, prefix: str = "") -> Dict[str, float | int]:
    """
    Compute registration error summary statistics.

    Parameters
    ----------
    distances:
        Array of distance errors.
    prefix:
        Optional prefix for metric names.

    Returns
    -------
    dict
        Dictionary of summary metrics.
    """
    d = finite_distances(distances)

    if d.size == 0:
        raise ValueError("No valid finite distances available for metric computation.")

    metrics: Dict[str, float | int] = {
        f"{prefix}n": int(d.size),
        f"{prefix}mean_mm": float(np.mean(d)),
        f"{prefix}median_mm": float(np.median(d)),
        f"{prefix}std_mm": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        f"{prefix}rmse_mm": float(np.sqrt(np.mean(d ** 2))),
        f"{prefix}min_mm": float(np.min(d)),
        f"{prefix}max_mm": float(np.max(d)),
        f"{prefix}p90_mm": float(np.percentile(d, 90)),
        f"{prefix}p95_mm": float(np.percentile(d, 95)),
        f"{prefix}p99_mm": float(np.percentile(d, 99)),
    }

    for threshold in [0.5, 1.0, 2.0, 3.0, 5.0]:
        metrics[f"{prefix}pct_<=_{threshold}mm"] = float(
            100.0 * np.mean(d <= threshold)
        )

    return metrics


def save_metrics_csv(metrics: Dict[str, float | int], output_path: Path) -> None:
    """
    Save one-row metrics dictionary as CSV.
    """
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    logger.info("Saved metrics: %s", output_path)


# =============================================================================
# Plots
# =============================================================================

def save_histogram(
    distances: np.ndarray,
    output_path: Path,
    p95: float,
    mean: float,
    median: float,
    title: str,
) -> None:
    """
    Save histogram of distance errors.
    """
    d = finite_distances(distances)

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=60)
    plt.axvline(mean, linestyle="--", label=f"Mean = {mean:.2f} mm")
    plt.axvline(median, linestyle="--", label=f"Median = {median:.2f} mm")
    plt.axvline(p95, linestyle="--", label=f"P95 = {p95:.2f} mm")
    plt.xlabel("Distance error (mm)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Saved histogram: %s", output_path)


def save_cdf(
    distances: np.ndarray,
    output_path: Path,
    p95: float,
    title: str,
) -> None:
    """
    Save cumulative distribution function plot of distance errors.
    """
    d = finite_distances(distances)
    d = np.sort(d)
    y = np.arange(1, len(d) + 1) / float(len(d))

    plt.figure(figsize=(7, 5))
    plt.plot(d, y)
    plt.axvline(p95, linestyle="--", label=f"P95 = {p95:.2f} mm")
    plt.xlabel("Distance error (mm)")
    plt.ylabel("Cumulative fraction")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Saved CDF: %s", output_path)


# =============================================================================
# Heatmap OBJ export
# =============================================================================

def errors_to_rgb(errors: np.ndarray, clamp_max: float) -> np.ndarray:
    """
    Convert distance errors to RGB colors using the jet colormap.

    Low error is blue and high error is red. Values are clamped at clamp_max,
    typically P95, to make the heatmap robust to outliers.

    Parameters
    ----------
    errors:
        Per-vertex errors.
    clamp_max:
        Maximum value used for color normalization.

    Returns
    -------
    np.ndarray
        RGB array with shape (N, 3), values in [0, 1].
    """
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    invalid_mask = ~np.isfinite(e)

    denom = max(float(clamp_max), 1e-12)
    normalized = np.clip(e / denom, 0.0, 1.0)

    cmap = plt.get_cmap("jet")
    rgba = cmap(normalized)
    rgb = rgba[:, :3].copy()

    rgb[invalid_mask] = np.array([0.6, 0.6, 0.6], dtype=np.float64)
    return rgb


def write_obj_with_vertex_colors(
    mesh: trimesh.Trimesh,
    rgb: np.ndarray,
    output_path: Path,
) -> None:
    """
    Write an OBJ file with per-vertex colors.

    MeshLab supports vertex colors in the following OBJ format:

        v x y z r g b
        f i j k

    Parameters
    ----------
    mesh:
        Input mesh.
    rgb:
        RGB values in [0, 1], one color per vertex.
    output_path:
        Output OBJ path.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if rgb.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"RGB length must match vertex count. "
            f"Got {rgb.shape[0]} colors for {vertices.shape[0]} vertices."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# OBJ with per-vertex colors: v x y z r g b\n")
        for (x, y, z), (r, g, b) in zip(vertices, rgb):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")

    logger.info("Saved heatmap OBJ: %s", output_path)


def save_heatmap_legend(output_path: Path, clamp: float) -> None:
    """
    Save a text legend explaining the heatmap.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Heatmap legend\n")
        f.write("==============\n")
        f.write("Mesh: registered_model_heatmap.obj\n")
        f.write("Quantity: per-vertex distance from registered intraoperative mesh ")
        f.write("to preoperative mucosa surface\n")
        f.write("Colormap: jet\n")
        f.write("Blue: low error\n")
        f.write("Red: high error\n")
        f.write(f"Color clamp: P95 = {clamp:.6f} mm\n")
        f.write("Distances above the clamp value are saturated to red.\n")

    logger.info("Saved heatmap legend: %s", output_path)


# =============================================================================
# Surface distance using Open3D RaycastingScene
# =============================================================================

def trimesh_to_open3d_tensor_mesh(mesh: trimesh.Trimesh) -> o3d.t.geometry.TriangleMesh:
    """
    Convert a trimesh.Trimesh to an Open3D tensor TriangleMesh.
    """
    vertices = o3c.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    faces = o3c.Tensor(np.asarray(mesh.faces, dtype=np.int32))
    return o3d.t.geometry.TriangleMesh(vertices, faces)


def surface_distances_vertex_to_mesh(
    source_vertices: np.ndarray,
    target_mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    Compute unsigned distances from source vertices to a target triangle mesh.

    Parameters
    ----------
    source_vertices:
        Source points with shape (N, 3).
    target_mesh:
        Target triangle mesh.

    Returns
    -------
    np.ndarray
        Unsigned vertex-to-surface distances with shape (N,).
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)

    if source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
        raise ValueError(
            f"source_vertices must have shape (N, 3), got {source_vertices.shape}"
        )

    target_tensor_mesh = trimesh_to_open3d_tensor_mesh(target_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(target_tensor_mesh)

    query_points = o3c.Tensor(source_vertices.astype(np.float32))
    distances = scene.compute_distance(query_points).numpy().astype(np.float64)

    return distances


# =============================================================================
# Optional vertex-index correspondence evaluation
# =============================================================================

def load_mapping_excel(path: str | Path) -> List[Tuple[int, int]]:
    """
    Load 1-based vertex-index correspondences from an Excel file.

    The first two columns are interpreted as:

        column 0: preoperative vertex index, 1-based
        column 1: intraoperative vertex index, 1-based

    Non-numeric/header rows are skipped.

    Parameters
    ----------
    path:
        Path to Excel file.

    Returns
    -------
    list[tuple[int, int]]
        List of 1-based vertex-index pairs.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mapping Excel file not found: {path}")

    dataframe = pd.read_excel(path, header=None, usecols=[0, 1])

    mapping: List[Tuple[int, int]] = []

    for _, row in dataframe.iterrows():
        pre_idx = row.iloc[0]
        intra_idx = row.iloc[1]

        if pd.isna(pre_idx) or pd.isna(intra_idx):
            continue

        try:
            mapping.append((int(float(pre_idx)), int(float(intra_idx))))
        except (TypeError, ValueError):
            continue

    if not mapping:
        raise ValueError(f"No valid vertex-index pairs found in: {path}")

    return mapping


def mapping_distances(
    pre_mesh: trimesh.Trimesh,
    intra_mesh: trimesh.Trimesh,
    mapping: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute vertex-to-vertex distances for 1-based index correspondences.

    Parameters
    ----------
    pre_mesh:
        Preoperative mesh.
    intra_mesh:
        Registered intraoperative mesh.
    mapping:
        List of 1-based vertex-index pairs.

    Returns
    -------
    distances:
        Vertex-to-vertex distances.
    pre_ids_zero_based:
        Preoperative vertex indices, 0-based.
    intra_ids_zero_based:
        Intraoperative vertex indices, 0-based.
    """
    pre_vertices = np.asarray(pre_mesh.vertices, dtype=np.float64)
    intra_vertices = np.asarray(intra_mesh.vertices, dtype=np.float64)

    pre_ids = np.array([pre_idx - 1 for pre_idx, _ in mapping], dtype=np.int64)
    intra_ids = np.array([intra_idx - 1 for _, intra_idx in mapping], dtype=np.int64)

    bad = np.where(
        (pre_ids < 0)
        | (pre_ids >= len(pre_vertices))
        | (intra_ids < 0)
        | (intra_ids >= len(intra_vertices))
    )[0]

    if bad.size > 0:
        examples = bad[:10].tolist()
        raise IndexError(
            "Out-of-range vertex indices in mapping file. "
            f"Example invalid filtered rows: {examples}. "
            f"Preop vertices: {len(pre_vertices)}, "
            f"intra vertices: {len(intra_vertices)}. "
            "Expected 1-based vertex indices."
        )

    pre_points = pre_vertices[pre_ids]
    intra_points = intra_vertices[intra_ids]

    distances = np.linalg.norm(pre_points - intra_points, axis=1)

    return distances, pre_ids, intra_ids


def save_mapping_outputs(
    mapping: List[Tuple[int, int]],
    distances: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save mapping distance table and summary metrics.
    """
    metrics = compute_metrics(distances)
    save_metrics_csv(metrics, output_dir / "summary_metrics_mapping.csv")

    df = pd.DataFrame(
        {
            "preop_index_1based": [pre_idx for pre_idx, _ in mapping],
            "intra_index_1based": [intra_idx for _, intra_idx in mapping],
            "distance_mm": distances,
        }
    )
    mapping_csv = output_dir / "mapping_distances.csv"
    df.to_csv(mapping_csv, index=False)
    logger.info("Saved mapping distances: %s", mapping_csv)

    save_histogram(
        distances=distances,
        output_path=output_dir / "histogram_mapping.png",
        p95=float(metrics["p95_mm"]),
        mean=float(metrics["mean_mm"]),
        median=float(metrics["median_mm"]),
        title="Mapping-Based Vertex-to-Vertex Distance Histogram",
    )

    logger.info("===== Mapping-distance summary =====")
    logger.info("N pairs: %d", metrics["n"])
    logger.info("Mean: %.4f mm", metrics["mean_mm"])
    logger.info("Median: %.4f mm", metrics["median_mm"])
    logger.info("RMSE: %.4f mm", metrics["rmse_mm"])
    logger.info("P95: %.4f mm", metrics["p95_mm"])
    logger.info("Max: %.4f mm", metrics["max_mm"])


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_surface_distance(
    pre_mesh: trimesh.Trimesh,
    intra_mesh: trimesh.Trimesh,
    output_dir: Path,
    surface_mode: str,
    save_cdf_plot: bool,
) -> None:
    """
    Compute and save surface-distance evaluation outputs.
    """
    logger.info("Computing intraoperative vertices -> preoperative surface distance.")

    intra_vertices = np.asarray(intra_mesh.vertices, dtype=np.float64)
    d_intra_to_pre = surface_distances_vertex_to_mesh(intra_vertices, pre_mesh)

    if surface_mode == "symmetric":
        logger.info("Computing preoperative vertices -> intraoperative surface distance.")
        pre_vertices = np.asarray(pre_mesh.vertices, dtype=np.float64)
        d_pre_to_intra = surface_distances_vertex_to_mesh(pre_vertices, intra_mesh)
        d_surface = np.concatenate([d_intra_to_pre, d_pre_to_intra], axis=0)
    elif surface_mode == "oneway":
        d_surface = d_intra_to_pre
    else:
        raise ValueError(f"Unsupported surface mode: {surface_mode}")

    metrics = compute_metrics(d_surface)
    save_metrics_csv(metrics, output_dir / "summary_metrics_surface.csv")

    save_histogram(
        distances=d_surface,
        output_path=output_dir / "histogram_surface.png",
        p95=float(metrics["p95_mm"]),
        mean=float(metrics["mean_mm"]),
        median=float(metrics["median_mm"]),
        title=f"Surface Distance Histogram ({surface_mode})",
    )

    if save_cdf_plot:
        save_cdf(
            distances=d_surface,
            output_path=output_dir / "cdf_surface.png",
            p95=float(metrics["p95_mm"]),
            title=f"Surface Distance CDF ({surface_mode})",
        )

    # Heatmap is always based on one-way intra -> pre distance.
    clamp = float(np.percentile(finite_distances(d_intra_to_pre), 95))
    rgb = errors_to_rgb(d_intra_to_pre, clamp_max=clamp)

    heatmap_path = output_dir / "registered_model_heatmap.obj"
    write_obj_with_vertex_colors(intra_mesh, rgb, heatmap_path)
    save_heatmap_legend(output_dir / "heatmap_legend.txt", clamp)

    logger.info("===== Surface-distance summary =====")
    logger.info("Surface mode: %s", surface_mode)
    logger.info("N samples: %d", metrics["n"])
    logger.info("Mean: %.4f mm", metrics["mean_mm"])
    logger.info("Median: %.4f mm", metrics["median_mm"])
    logger.info("RMSE: %.4f mm", metrics["rmse_mm"])
    logger.info("P95: %.4f mm", metrics["p95_mm"])
    logger.info("Max: %.4f mm", metrics["max_mm"])
    logger.info("%% <= 1 mm: %.1f", metrics["pct_<=_1.0mm"])
    logger.info("%% <= 2 mm: %.1f", metrics["pct_<=_2.0mm"])
    logger.info("%% <= 3 mm: %.1f", metrics["pct_<=_3.0mm"])
    logger.info("%% <= 5 mm: %.1f", metrics["pct_<=_5.0mm"])


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 3D registration quality using vertex-to-surface distance "
            "and optional vertex-index correspondence distances."
        )
    )

    parser.add_argument(
        "--preop_obj",
        required=True,
        help="Preoperative mucosa OBJ in original/preoperative coordinate space.",
    )
    parser.add_argument(
        "--intra_obj",
        required=True,
        help="Registered intraoperative OBJ already transformed into preoperative space.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where evaluation outputs will be saved.",
    )
    parser.add_argument(
        "--mapping_xlsx",
        default="",
        help=(
            "Optional Excel file containing 1-based vertex-index correspondences "
            "in the first two columns: preop_index, intra_index."
        ),
    )
    parser.add_argument(
        "--surface_mode",
        choices=["oneway", "symmetric"],
        default="oneway",
        help=(
            "Surface evaluation mode. "
            "'oneway' computes registered intra->pre distances. "
            "'symmetric' combines intra->pre and pre->intra distances."
        ),
    )
    parser.add_argument(
        "--no_cdf",
        action="store_true",
        help="Skip CDF plot generation.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    logger.info("Loading preoperative mesh: %s", args.preop_obj)
    pre_mesh = load_mesh(args.preop_obj)

    logger.info("Loading registered intraoperative mesh: %s", args.intra_obj)
    intra_mesh = load_mesh(args.intra_obj)

    evaluate_surface_distance(
        pre_mesh=pre_mesh,
        intra_mesh=intra_mesh,
        output_dir=output_dir,
        surface_mode=args.surface_mode,
        save_cdf_plot=not args.no_cdf,
    )

    if args.mapping_xlsx.strip():
        logger.info("Running optional vertex-index correspondence evaluation.")
        mapping = load_mapping_excel(args.mapping_xlsx)
        distances, _, _ = mapping_distances(pre_mesh, intra_mesh, mapping)
        save_mapping_outputs(mapping, distances, output_dir)

    logger.info("Evaluation complete. Outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()