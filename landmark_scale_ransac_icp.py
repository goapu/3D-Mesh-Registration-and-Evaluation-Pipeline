"""
3D Mesh Registration Pipeline (Rigid Landmark + RANSAC + ICP)
============================================================

Author: Dilip Goswami, MSc Student, TU Berlin
Date: 06.02.2026 (Updated)

Overview
--------
Rigid anatomical registration between preoperative (mm) and intraoperative (unknown scale) nasal meshes.

Pipeline:
1. Landmark Extraction (Geometric centroids of anatomical intersections)
2. Scale Normalization (Landmark spread based - Robust to partial scans)
3. Rigid Alignment (Kabsch algorithm)
4. RANSAC Refinement (Global coarse alignment)
5. ICP Refinement (Local fine alignment)

Outputs:
- Transformed meshes at each stage
- Inverse matrices to map Intraop back to Preop space
- Convergence plots and logs
"""

# =============================================================================
# Imports
# =============================================================================

import os
import re
import csv
import argparse
import logging
import multiprocessing
from typing import Optional, Dict, Tuple

import numpy as np
import trimesh
import difflib
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

KEYWORD_PAIRS = [
    ("anterior", "inferior_turbinate"),
    ("posterior", "inferior_turbinate"),
    ("middle", "inferior_turbinate"),
    ("posterior", "middle_turbinate"),
    ("middle", "middle_turbinate"),
]

EXCLUDE_FILES = {"mucosaleft.obj", "model.obj"}

# Mesh RANSAC params (Open3D)
MESH_RANSAC_MAX_CORR_DIST = 3.0
MESH_RANSAC_MAX_ITER = 50000
MESH_RANSAC_CONFIDENCE = 0.999
MESH_RANSAC_MIN_CORR = 50

# ICP params
ICP_THRESHOLD = 3.0
ICP_MAX_ITER = 300
ICP_TOL_FITNESS = 1e-6
ICP_TOL_RMSE = 1e-6

# =============================================================================
# Geometry Utilities
# =============================================================================

def load_obj_vertices(path: str) -> Optional[np.ndarray]:
    try:
        mesh = trimesh.load_mesh(path, process=False)
        return np.asarray(mesh.vertices)
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        return None


def compute_centroid(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or len(points) == 0:
        return None
    return points.mean(axis=0)


def find_intersection(v1, v2, tolerance=1e-3):
    if v1 is None or v2 is None:
        return None
    tree1, tree2 = KDTree(v1), KDTree(v2)
    matches = tree1.query_ball_tree(tree2, r=tolerance)
    pts = [tuple(v1[i]) for i, idx in enumerate(matches) if idx]
    return np.unique(np.array(pts), axis=0) if pts else None


def apply_transform_trimesh(mesh: trimesh.Trimesh, T: np.ndarray) -> None:
    """Apply 4x4 transform T in-place to a trimesh mesh."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    V_h = np.hstack([V, np.ones((len(V), 1), dtype=np.float64)])
    V2 = (T @ V_h.T).T[:, :3]
    mesh.vertices = V2


def mesh_to_o3d_pcd(mesh: trimesh.Trimesh, voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    return pcd


# =============================================================================
# Landmark Extraction
# =============================================================================

def categorize_files(paths):
    categorized = {kw: [] for pair in KEYWORD_PAIRS for kw in pair}
    choanal_arc = None
    for p in paths:
        name = os.path.basename(p).lower()
        if re.search(r"choanal.*arc", name):
            choanal_arc = p
        for k1, k2 in KEYWORD_PAIRS:
            if k1 in name:
                categorized[k1].append(p)
            if k2 in name:
                categorized[k2].append(p)
    return categorized, choanal_arc


def process_folder(folder):
    obj_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".obj") and f.lower() not in EXCLUDE_FILES
    ]

    categorized, choanal_arc = categorize_files(obj_files)

    vertices = {}
    with multiprocessing.Pool() as pool:
        for p, v in zip(obj_files, pool.map(load_obj_vertices, obj_files)):
            if v is not None:
                vertices[p] = v

    centroids = []

    if choanal_arc and choanal_arc in vertices:
        c = compute_centroid(vertices[choanal_arc])
        centroids.append(("choanal_arc", c))
        logger.info(f"Choanal arc centroid: {c}")

    used = set()
    for k1, k2 in KEYWORD_PAIRS:
        for f1 in categorized[k1]:
            for f2 in categorized[k2]:
                if f1 == f2 or (f1, f2) in used:
                    continue
                pts = find_intersection(vertices.get(f1), vertices.get(f2))
                c = compute_centroid(pts)
                if c is not None:
                    label = f"{os.path.basename(f1)}_{os.path.basename(f2)}"
                    centroids.append((label, c))
                    logger.info(f"Intersection centroid {label}: {c}")
                    used.add((f1, f2))
                if len(centroids) >= 5:
                    return centroids

    return centroids


# =============================================================================
# Registration Utilities
# =============================================================================

def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_centroids(src: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    src_pts, tgt_pts = [], []
    used_targets = set()

    for k in src:
        matches = difflib.get_close_matches(k, tgt.keys(), n=5, cutoff=0.6)
        chosen = None
        for m in matches:
            if m not in used_targets:
                chosen = m
                break
        if chosen is not None:
            src_pts.append(src[k])
            tgt_pts.append(tgt[chosen])
            used_targets.add(chosen)
            logger.info(f"Matched {k} -> {chosen}")

    if len(src_pts) < 3:
        raise ValueError("Insufficient matched centroids for rigid registration (need >= 3).")

    return np.asarray(src_pts, dtype=np.float64), np.asarray(tgt_pts, dtype=np.float64)


def estimate_rigid_kabsch(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """
    Rigid transform (rotation + translation) that maps src_pts -> tgt_pts.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    tgt = np.asarray(tgt_pts, dtype=np.float64)

    c_src = src.mean(axis=0)
    c_tgt = tgt.mean(axis=0)

    X = src - c_src
    Y = tgt - c_tgt

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c_tgt - R @ c_src

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    logger.info(f"Rigid landmark transform: det(R)={np.linalg.det(R):.6f}")
    return T


# =============================================================================
# Scale Normalization (UPDATED)
# =============================================================================

def compute_scale_from_landmarks(src_pts: np.ndarray, tgt_pts: np.ndarray) -> float:
    """
    Computes isotropic scale factor based on the spread of matched landmarks.
    Formula: Scale = (Spread of Intraop) / (Spread of Preop)
    
    This is robust to partial meshes because it depends only on the relative
    positions of the landmarks, not the bounding box of the mesh.
    """
    if len(src_pts) < 2:
        logger.warning("Not enough landmarks for scale estimation. Defaulting to 1.0")
        return 1.0

    # Center the points
    src_centered = src_pts - src_pts.mean(axis=0)
    tgt_centered = tgt_pts - tgt_pts.mean(axis=0)

    # Compute Root Mean Square distance from centroid
    src_spread = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    tgt_spread = np.sqrt(np.mean(np.sum(tgt_centered**2, axis=1)))

    if src_spread <= 1e-9:
        raise ValueError("Source landmarks have zero spread (all points identical?).")
    
    scale = tgt_spread / src_spread
    logger.info(f"Landmark Spread (Preop): {src_spread:.4f}")
    logger.info(f"Landmark Spread (Intraop): {tgt_spread:.4f}")
    logger.info(f"Computed Unit Scale Factor: {scale:.6f}")
    return scale


# =============================================================================
# Mesh-level RANSAC Refinement
# =============================================================================

def build_correspondences_nn(src_pcd, tgt_pcd, max_corr_dist):
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    src_pts = np.asarray(src_pcd.points)
    corr = []
    for i, p in enumerate(src_pts):
        k, idx, dist2 = tgt_tree.search_knn_vector_3d(p, 1)
        if k == 1 and float(np.sqrt(dist2[0])) <= max_corr_dist:
            corr.append([i, int(idx[0])])
    return o3d.utility.Vector2iVector(np.asarray(corr, dtype=np.int32))


def run_ransac_refinement(
    src_mesh: trimesh.Trimesh,
    tgt_mesh: trimesh.Trimesh,
    max_corr_dist: float = MESH_RANSAC_MAX_CORR_DIST,
    max_iter: int = MESH_RANSAC_MAX_ITER,
    confidence: float = MESH_RANSAC_CONFIDENCE,
) -> np.ndarray:
    tgt_diag = float(np.linalg.norm(tgt_mesh.bounding_box.extents))
    voxel = max(0.5, 0.01 * tgt_diag)

    src_pcd = mesh_to_o3d_pcd(src_mesh, voxel_size=voxel)
    tgt_pcd = mesh_to_o3d_pcd(tgt_mesh, voxel_size=voxel)

    corr = build_correspondences_nn(src_pcd, tgt_pcd, max_corr_dist=max_corr_dist)
    if np.asarray(corr).shape[0] < MESH_RANSAC_MIN_CORR:
        logger.warning("Too few correspondences. Skipping RANSAC.")
        return np.eye(4, dtype=np.float64)

    logger.info(f"Mesh RANSAC correspondences: {np.asarray(corr).shape[0]}")

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd, tgt_pcd, corr,
        max_corr_dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr_dist)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, confidence)
    )

    logger.info(f"RANSAC fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
    return np.asarray(result.transformation, dtype=np.float64)


# =============================================================================
# ICP with Iteration Logging
# =============================================================================

def run_icp_with_logging(src_mesh, tgt_mesh, init, output_dir):
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(np.asarray(src_mesh.vertices, dtype=np.float64))
    tgt_pcd.points = o3d.utility.Vector3dVector(np.asarray(tgt_mesh.vertices, dtype=np.float64))

    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)

    csv_path = os.path.join(output_dir, "icp_convergence.csv")
    T = init.copy()
    reg = None
    prev_f, prev_r = None, None
    it = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "fitness", "rmse"])
        for it in range(1, ICP_MAX_ITER + 1):
            reg = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, ICP_THRESHOLD, T, estimator, criteria
            )
            T = reg.transformation
            writer.writerow([it, reg.fitness, reg.inlier_rmse])

            if prev_f is not None:
                if abs(reg.fitness - prev_f) < ICP_TOL_FITNESS and abs(reg.inlier_rmse - prev_r) < ICP_TOL_RMSE:
                    break
            prev_f, prev_r = reg.fitness, reg.inlier_rmse

    logger.info(f"ICP iterations: {it}")
    return np.asarray(T, dtype=np.float64), reg, it


# =============================================================================
# Helpers
# =============================================================================

def find_obj(folder, keyword):
    for f in os.listdir(folder):
        if f.lower().endswith(".obj") and keyword in f.lower():
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No OBJ file with keyword '{keyword}' found in {folder}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preop", required=True)
    parser.add_argument("--intraop", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- 1. Landmarks ---
    logger.info("Extracting landmarks...")
    pre = process_folder(args.preop)
    intra = process_folder(args.intraop)

    src = {normalize_name(n): p for n, p in pre}
    tgt = {normalize_name(n): p for n, p in intra}

    src_pts, tgt_pts = match_centroids(src, tgt)
    
    # --- 2. Scale Calculation (Robust Landmark-Based) ---
    logger.info("Calculating scale...")
    scale = compute_scale_from_landmarks(src_pts, tgt_pts)
    np.save(os.path.join(args.output, "scale.npy"), np.float64(scale))

    # Load meshes
    mucosa: trimesh.Trimesh = trimesh.load_mesh(find_obj(args.preop, "mucosa"), process=False)
    model: trimesh.Trimesh = trimesh.load_mesh(find_obj(args.intraop, "model"), process=False)

    # Apply Scale
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = float(scale)
    
    # Scale both the mesh and the source landmarks to common unit space
    mucosa.apply_scale(scale)
    src_pts_scaled = src_pts * float(scale)

    # --- 3. Rigid Alignment (Kabsch) ---
    logger.info("Running rigid landmark alignment...")
    T_affine = estimate_rigid_kabsch(src_pts_scaled, tgt_pts)
    np.save(os.path.join(args.output, "affine.npy"), T_affine)

    apply_transform_trimesh(mucosa, T_affine)
    mucosa.export(os.path.join(args.output, "mucosa_registered_affine.obj"))

    # --- 4. RANSAC Refinement ---
    logger.info("Running RANSAC refinement...")
    T_ransac = run_ransac_refinement(mucosa, model)
    np.save(os.path.join(args.output, "ransac.npy"), T_ransac)

    apply_transform_trimesh(mucosa, T_ransac)
    mucosa.export(os.path.join(args.output, "mucosa_registered_ransac.obj"))

    # --- 5. ICP Refinement ---
    logger.info("Running ICP refinement...")
    T_icp, reg, iters = run_icp_with_logging(mucosa, model, np.eye(4), args.output)
    np.save(os.path.join(args.output, "icp.npy"), T_icp)

    apply_transform_trimesh(mucosa, T_icp)
    mucosa.export(os.path.join(args.output, "mucosa_registered_icp.obj"))

    # --- 6. Final Transforms & Inverses ---
    # Forward: original preop -> intraop model space
    T_total = T_icp @ T_ransac @ T_affine @ S

    # Inverse: intraop model -> original preop space (mm)
    T_affine_inv = np.linalg.inv(T_affine)
    T_ransac_inv = np.linalg.inv(T_ransac)
    T_icp_inv = np.linalg.inv(T_icp)
    
    S_inv = np.eye(4, dtype=np.float64)
    S_inv[0, 0] = S_inv[1, 1] = S_inv[2, 2] = 1.0 / float(scale)

    T_total_inv = np.linalg.inv(T_total)

    np.save(os.path.join(args.output, "inv_scale.npy"), S_inv)
    np.save(os.path.join(args.output, "inv_affine.npy"), T_affine_inv)
    np.save(os.path.join(args.output, "inv_ransac.npy"), T_ransac_inv)
    np.save(os.path.join(args.output, "inv_icp.npy"), T_icp_inv)
    np.save(os.path.join(args.output, "inv_total.npy"), T_total_inv)

    # Apply inverse to intraop model -> registered_model.obj
    registered_model = model.copy()
    apply_transform_trimesh(registered_model, T_total_inv)
    registered_model.export(os.path.join(args.output, "registered_model.obj"))

    np.savez(
        os.path.join(args.output, "transforms.npz"),
        scale=np.float64(scale),
        affine_matrix=T_affine.astype(np.float64),
        ransac_matrix=T_ransac.astype(np.float64),
        icp_matrix=T_icp.astype(np.float64),
        total_matrix=T_total.astype(np.float64),
        inv_scale_matrix=S_inv.astype(np.float64),
        inv_affine_matrix=T_affine_inv.astype(np.float64),
        inv_ransac_matrix=T_ransac_inv.astype(np.float64),
        inv_icp_matrix=T_icp_inv.astype(np.float64),
        inv_total_matrix=T_total_inv.astype(np.float64),
    )

    # Plot convergence
    csv_path = os.path.join(args.output, "icp_convergence.csv")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1: data = data.reshape(1, -1) # Handle single iteration case
    
    plt.figure()
    plt.plot(data[:, 0], data[:, 2])
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("ICP Convergence")
    plt.grid(True)
    plt.savefig(os.path.join(args.output, "icp_convergence.png"), dpi=300)
    plt.close()

    if reg is not None:
        logger.info(f"Final ICP fitness: {reg.fitness:.4f}, RMSE: {reg.inlier_rmse:.4f}")

    logger.info("Pipeline Complete. Output saved to " + args.output)

if __name__ == "__main__":
    main()