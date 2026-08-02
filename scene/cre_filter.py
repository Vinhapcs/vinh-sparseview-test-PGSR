"""
scene/cre_filter.py — Co-visible Redundancy Elimination (CRE)
==============================================================

Filters a dense BasicPointCloud by removing low-confidence points that are
redundant co-visible observations of the same surface seen by multiple
training cameras.

Algorithm (Unified Point Cloud Approach)
-----------------------------------------
1.  Convert points / colors / confs to CUDA tensors.
2.  Sort all N points **descending** by MASt3R confidence (stored in normals[:,0]).
3.  Iterate over every training camera:
    a.  Load the GT depth map (.npy) -> CUDA.
    b.  Project all currently-ACTIVE points into camera image-space.
    c.  Keep only those that fall inside [0,W-1]x[0,H-1] with Z > 0.01.
    d.  Sample GT depth at integer (u,v) to get Z_gt.
    e.  Depth test: |Z - Z_gt| < depth_tolerance * Z_gt  -> "on-surface".
    f.  Redundancy check: among on-surface points that share the same pixel
        (u, v), keep only the FIRST one (highest confidence, because we
        pre-sorted); mark the rest as keep_mask = False.
4.  Return a new BasicPointCloud filtered by the final keep_mask.

Camera convention (PGSR / COLMAP, row-vector):
    world_view_transform is W2C stored TRANSPOSED (col-major for CUDA/OpenGL).
    Row-vector convention:
        pts_cam_h = pts_world_h @ world_view_transform
    so   pts_cam_h[k, j] = sum_i pts_h[k, i] * W2C[j, i]   (== W2C @ pts[k])
    Z component (cam-space depth) = pts_cam_h[:, 2]

Intrinsics derived from FoVx / FoVy:
    fx = W / (2 * tan(FoVx / 2))
    fy = H / (2 * tan(FoVy / 2))
    cx = W / 2,  cy = H / 2

Pixel coordinates (integer, 0-indexed):
    u = round(fx * X/Z + cx)
    v = round(fy * Y/Z + cy)
"""

import os
import math

import numpy as np
import torch

from utils.graphics_utils import BasicPointCloud


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_redundant_points(
    pcd: BasicPointCloud,
    train_cameras,           # list[Camera] – fully initialised PGSR Camera objects
    dataset_path: str,
    depth_tolerance: float = 0.05,
) -> BasicPointCloud:
    """
    Co-visible Redundancy Elimination (CRE).

    Parameters
    ----------
    pcd : BasicPointCloud
        Dense point cloud.  MASt3R confidence is stored in ``pcd.normals[:, 0]``.
    train_cameras : list[Camera]
        Must expose: .world_view_transform (4x4 CUDA), .FoVx, .FoVy,
        .image_width, .image_height, .image_name.
    dataset_path : str
        Scene root.  GT depths are at ``<dataset_path>/depths/<image_name>.npy``.
    depth_tolerance : float
        Relative depth agreement threshold: |Z - Z_gt| < tol * Z_gt.

    Returns
    -------
    BasicPointCloud
        Filtered cloud; normals (including embedded confidence) are preserved.
    """
    N = pcd.points.shape[0]
    if N == 0:
        print("[CRE] Empty point cloud -- skipping.")
        return pcd

    depths_dir = os.path.join(dataset_path, "depths")
    if not os.path.isdir(depths_dir):
        print(f"[CRE] Depths directory not found at '{depths_dir}' -- skipping CRE.")
        return pcd

    print(f"[CRE] Starting Co-visible Redundancy Elimination on {N:,} points ...")

    # ------------------------------------------------------------------
    # 1. Move everything to CUDA
    # ------------------------------------------------------------------
    pts   = torch.from_numpy(pcd.points.astype(np.float32)).cuda()    # (N, 3)
    cols  = torch.from_numpy(pcd.colors.astype(np.float32)).cuda()    # (N, 3)
    norms = torch.from_numpy(pcd.normals.astype(np.float32)).cuda()   # (N, 3)
    confs = norms[:, 0].clone()                                        # (N,)

    # ------------------------------------------------------------------
    # 2. Sort descending by confidence (highest quality point first)
    # ------------------------------------------------------------------
    sort_idx = torch.argsort(confs, descending=True, stable=True)      # (N,)
    pts   = pts[sort_idx]
    cols  = cols[sort_idx]
    norms = norms[sort_idx]
    # confs not needed after sorting; free memory
    del confs

    # ------------------------------------------------------------------
    # 3. Keep mask – True  means "still alive"
    # ------------------------------------------------------------------
    keep_mask = torch.ones(N, dtype=torch.bool, device='cuda')

    # ------------------------------------------------------------------
    # 4. Pre-compute homogeneous coords (reused every camera)
    # ------------------------------------------------------------------
    ones  = torch.ones(N, 1, device='cuda')
    pts_h = torch.cat([pts, ones], dim=1)   # (N, 4)

    # ------------------------------------------------------------------
    # 5. Per-camera loop
    # ------------------------------------------------------------------
    cams_processed = 0

    for cam in train_cameras:
        depth_path = os.path.join(depths_dir, f"{cam.image_name}.npy")
        if not os.path.isfile(depth_path):
            continue

        # ---- Load GT depth map -------------------------------------------
        depth_np = np.load(depth_path).astype(np.float32)
        if depth_np.ndim == 3:        # (1, H, W) -> (H, W)
            depth_np = depth_np[0]
        depth_gt = torch.from_numpy(depth_np).cuda()   # (H, W)

        H = cam.image_height
        W = cam.image_width

        # Resize if needed (minor resolution mismatch between depth and image)
        dh, dw = depth_gt.shape
        if dh != H or dw != W:
            depth_gt = torch.nn.functional.interpolate(
                depth_gt[None, None], size=(H, W), mode='nearest'
            ).squeeze()

        # ---- Intrinsics --------------------------------------------------
        fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
        fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
        cx = W * 0.5
        cy = H * 0.5

        # ---- Active points -----------------------------------------------
        active_idx = torch.where(keep_mask)[0]    # (M,) – global indices
        if active_idx.numel() == 0:
            break

        # ---- Project to camera space (row-vector convention) -------------
        #   world_view_transform (4x4) = W2C^T in column-major storage
        #   pts_cam_h = pts_world_h @ world_view_transform  gives cam coords
        W2C = cam.world_view_transform            # (4, 4) on CUDA

        active_pts_h = pts_h[active_idx]          # (M, 4)
        pts_cam_h    = active_pts_h @ W2C         # (M, 4)

        Xc = pts_cam_h[:, 0]
        Yc = pts_cam_h[:, 1]
        Zc = pts_cam_h[:, 2]

        # ---- Validity: positive depth ------------------------------------
        valid_z = Zc > 0.01

        # ---- Pixel coordinates (float then rounded) ----------------------
        Z_safe = Zc.clamp(min=1e-6)
        u_f = fx * (Xc / Z_safe) + cx
        v_f = fy * (Yc / Z_safe) + cy

        u = torch.round(u_f).long()
        v = torch.round(v_f).long()

        # ---- Image boundary check ----------------------------------------
        in_bounds = (
            valid_z
            & (u >= 0) & (u < W)
            & (v >= 0) & (v < H)
        )

        vis_local = torch.where(in_bounds)[0]     # indices into active_idx
        if vis_local.numel() == 0:
            continue

        u_vis  = u[vis_local]
        v_vis  = v[vis_local]
        Z_vis  = Zc[vis_local]

        # ---- Sample GT depth at projected pixels -------------------------
        Z_gt_vis = depth_gt[v_vis, u_vis]         # (K,)

        # ---- Relative depth test -----------------------------------------
        diff       = (Z_vis - Z_gt_vis).abs()
        on_surface = (Z_gt_vis > 0.0) & (diff < depth_tolerance * Z_gt_vis)

        surf_local = vis_local[on_surface]        # local indices into active_idx
        if surf_local.numel() == 0:
            continue

        # ---- Redundancy check: keep first per (u, v) pixel ---------------
        # Encode pixel as a single int64 key for fast dedup
        u_surf = u[surf_local]
        v_surf = v[surf_local]
        pixel_keys = v_surf * W + u_surf          # (S,)

        # Stable argsort preserves confidence-descending order within ties.
        # The 'first' occurrence for each key is the highest-confidence point.
        sort_order  = torch.argsort(pixel_keys, stable=True)
        sorted_keys = pixel_keys[sort_order]

        is_dup = torch.zeros(surf_local.numel(), dtype=torch.bool, device='cuda')
        is_dup[1:] = (sorted_keys[1:] == sorted_keys[:-1])

        # Map duplicates back to global keep_mask indices
        dup_local  = sort_order[is_dup]                         # into surf_local
        dup_global = active_idx[surf_local[dup_local]]          # into keep_mask

        if dup_global.numel() > 0:
            keep_mask[dup_global] = False

        cams_processed += 1

    # ------------------------------------------------------------------
    # 6. Report and apply mask
    # ------------------------------------------------------------------
    removed = int((~keep_mask).sum())
    kept    = int(keep_mask.sum())
    print(
        f"[CRE] Processed {cams_processed}/{len(train_cameras)} cameras. "
        f"Removed {removed:,} redundant points "
        f"-> {kept:,} remain ({100.0 * kept / N:.1f}% of original)."
    )

    idx_keep  = torch.where(keep_mask)[0]
    pts_out   = pts[idx_keep].cpu().numpy()
    cols_out  = cols[idx_keep].cpu().numpy()
    norms_out = norms[idx_keep].cpu().numpy()

    # ------------------------------------------------------------------
    # 7. Free CUDA memory
    # ------------------------------------------------------------------
    del pts, cols, norms, pts_h, ones, keep_mask, sort_idx
    torch.cuda.empty_cache()

    return BasicPointCloud(
        points=pts_out,
        colors=cols_out,
        normals=norms_out,
    )
