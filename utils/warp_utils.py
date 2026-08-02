"""
utils/warp_utils.py — Pi-GS §3.6 Depth Warping Utilities

Implements cross-view depth warping to generate pseudo-view supervision:
  1. find_nearest_cameras    — find K nearest cameras by Euclidean distance
  2. circle_through_3_points — circumscribed circle of 3 points in 3D (true circle)
  3. make_pseudo_camera      — build a Camera object from new R, T
  4. interpolate_camera_on_circle — circle-interpolated camera position + SLERP rotation
  5. warp_image_rgb          — forward-warp RGB image with Z-buffer and confidence masking

Camera convention (PGSR / COLMAP, row-vector):
    pts_cam = pts_world @ R + T
    pts_world = (pts_cam - T) @ R.T
    camera_center (world) = -T @ R.T   →   T = -camera_center @ R
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Nearest-camera lookup
# ──────────────────────────────────────────────────────────────────────────────

def find_nearest_cameras(cameras, target_cam, num_neighbors=2):
    """
    Return the `num_neighbors` cameras closest to `target_cam` by the
    Euclidean distance between camera centers in world space.
    Excludes the target camera itself (matched by uid).

    Args:
        cameras:       iterable of Camera objects (PGSR Camera)
        target_cam:    the reference Camera
        num_neighbors: how many neighbors to return

    Returns:
        list of Camera objects, sorted by ascending distance, length <= num_neighbors
    """
    target_center = target_cam.camera_center.float()
    dists = []
    for cam in cameras:
        if cam.uid == target_cam.uid:
            continue
        dist = torch.norm(cam.camera_center.float() - target_center).item()
        dists.append((dist, cam))
    dists.sort(key=lambda x: x[0])
    return [c[1] for c in dists[:num_neighbors]]


# ──────────────────────────────────────────────────────────────────────────────
# 2. True circle through 3 points in 3D
# ──────────────────────────────────────────────────────────────────────────────

def circle_through_3_points(p1, p2, p3):
    """
    Compute the circumscribed circle (circumcircle) passing through exactly
    three points in 3D space.  The circle lies in the unique plane defined by
    the three points.

    Geometry:
        - The plane normal is  n = (p2-p1) × (p3-p1).
        - Orthonormal basis (u, v) is constructed in that plane.
        - The circumcenter is found by solving the perpendicular-bisector
          equations in the 2D local frame (p1 as origin).

    Args:
        p1, p2, p3:  [3] float tensors — three points on the circle (same device)

    Returns:
        center:  [3] 3D world position of the circle center
        radius:  scalar (torch.Tensor) radius of the circle
        u_axis:  [3] first orthonormal basis vector of the circle plane
        v_axis:  [3] second orthonormal basis vector of the circle plane
    """
    v1 = p2 - p1
    v2 = p3 - p1

    # ── Plane normal ─────────────────────────────────────────────────────────
    normal = torch.linalg.cross(v1, v2)
    n_norm = torch.norm(normal)

    # ── Collinear fallback ───────────────────────────────────────────────────
    if n_norm < 1e-7:
        # Degenerate: return the midpoint of p1–p2 as "center" of a half-arc
        u = v1 / (torch.norm(v1) + 1e-8)
        # Perpendicular to u: find an axis minimally aligned with u
        idx = int(torch.argmin(torch.abs(u)).item())
        perp = torch.zeros(3, dtype=p1.dtype, device=p1.device)
        perp[idx] = 1.0
        v = perp - torch.dot(perp, u) * u
        v = v / (torch.norm(v) + 1e-8)
        center = (p1 + p2) * 0.5
        radius = torch.norm(p2 - p1) * 0.5
        return center, radius, u, v

    normal = normal / n_norm

    # ── Orthonormal basis for the circle plane ───────────────────────────────
    u = v1 / torch.norm(v1)
    v = torch.linalg.cross(normal, u)
    v = v / torch.norm(v)

    # ── Project p2, p3 into local 2D frame (p1 = origin) ────────────────────
    bx = torch.dot(p2 - p1, u)
    by = torch.dot(p2 - p1, v)
    cx = torch.dot(p3 - p1, u)
    cy = torch.dot(p3 - p1, v)

    # ── Circumcenter formula (2D) ─────────────────────────────────────────────
    # Solves the system: |circumcenter - pi|^2 = R^2  for i=1,2,3
    D = 2.0 * (bx * cy - by * cx)

    if D.abs() < 1e-8:
        # Nearly collinear in-plane: fallback to midpoint
        center = (p1 + p2) * 0.5
        radius = torch.norm(p2 - p1) * 0.5
        return center, radius, u, v

    ux = (cy * (bx ** 2 + by ** 2) - by * (cx ** 2 + cy ** 2)) / D
    uy = (bx * (cx ** 2 + cy ** 2) - cx * (bx ** 2 + by ** 2)) / D

    center = p1 + ux * u + uy * v
    radius = torch.norm(center - p1)
    return center, radius, u, v


# ──────────────────────────────────────────────────────────────────────────────
# 3. Pseudo-camera factory
# ──────────────────────────────────────────────────────────────────────────────

def make_pseudo_camera(cam_template, R_new_np, T_new_np):
    """
    Build a Camera-like object with the same intrinsics / image size as
    `cam_template` but with new extrinsics given as numpy arrays.

    All PGSR transform matrices (world_view_transform, projection_matrix,
    full_proj_transform, camera_center) are recomputed from scratch.
    The pseudo-camera has no image (preload_img=False).

    Args:
        cam_template:  any PGSR Camera to use as template
        R_new_np:      numpy [3,3] rotation matrix (world→cam, PGSR convention)
        T_new_np:      numpy [3,] translation vector (PGSR convention)

    Returns:
        pseudo Camera object suitable for gaussian_renderer.render()
    """
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    pseudo = copy.copy(cam_template)
    pseudo.R = R_new_np
    pseudo.T = T_new_np

    wvt = torch.tensor(
        getWorld2View2(R_new_np, T_new_np, cam_template.trans, cam_template.scale),
        dtype=torch.float32,
    ).transpose(0, 1).cuda()

    proj = getProjectionMatrix(
        znear=cam_template.znear,
        zfar=cam_template.zfar,
        fovX=cam_template.FoVx,
        fovY=cam_template.FoVy,
    ).transpose(0, 1).cuda()

    pseudo.world_view_transform = wvt
    pseudo.projection_matrix    = proj
    pseudo.full_proj_transform  = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    pseudo.camera_center        = wvt.inverse()[3, :3]

    # No real image for pseudo-cameras
    pseudo.original_image = None
    pseudo.preload_img    = False
    return pseudo


# ──────────────────────────────────────────────────────────────────────────────
# 4. Circle-interpolated pseudo-camera (Pi-GS §3.6)
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_camera_on_circle(cam_src, cam_tgt, cam_ref, t):
    """
    Generate a pseudo-camera at interpolation parameter t ∈ (0, 1) between
    cam_src and cam_tgt, using cam_ref as the third point that uniquely
    defines the circumscribed circle — exactly as described in Pi-GS §3.6.

    Position: the new camera center lies on the circumscribed circle of
              [cam_src.camera_center, cam_tgt.camera_center, cam_ref.camera_center],
              interpolated along the shorter arc from cam_src to cam_tgt.

    Rotation: SLERP between cam_src.R and cam_tgt.R.

    Translation: derived from the new center and rotation via the PGSR identity
                 T = −camera_center @ R
                 (from camera_center = −T @ R.T).

    Args:
        cam_src:  source Camera  (t=0 → cam_src position)
        cam_tgt:  target Camera  (t=1 → cam_tgt position)
        cam_ref:  reference Camera  (defines the circle together with the other two)
        t:        float in (0, 1)

    Returns:
        pseudo Camera object (shallow-copy of cam_src with updated extrinsics)
    """
    import scipy.spatial.transform

    pa = cam_src.camera_center.float()   # [3] world positions
    pb = cam_tgt.camera_center.float()
    pc = cam_ref.camera_center.float()

    center, radius, u_ax, v_ax = circle_through_3_points(pa, pb, pc)

    # ── Angles of pa, pb on the circle ───────────────────────────────────────
    def angle_of(p):
        dp = p - center
        return torch.atan2(torch.dot(dp, v_ax), torch.dot(dp, u_ax))

    angle_a = angle_of(pa)
    angle_b = angle_of(pb)

    # Shortest arc: normalise difference to (−π, π]
    diff = ((angle_b - angle_a) + math.pi) % (2.0 * math.pi) - math.pi
    angle_t = angle_a + float(t) * diff

    # ── New camera center on the circumscribed circle ─────────────────────────
    new_center = center + radius * (torch.cos(angle_t) * u_ax + torch.sin(angle_t) * v_ax)

    # ── SLERP rotation ───────────────────────────────────────────────────────
    Ra = cam_src.R if isinstance(cam_src.R, np.ndarray) else cam_src.R.cpu().numpy()
    Rb = cam_tgt.R if isinstance(cam_tgt.R, np.ndarray) else cam_tgt.R.cpu().numpy()

    key_rots = scipy.spatial.transform.Rotation.from_matrix([Ra, Rb])
    R_interp = scipy.spatial.transform.Slerp([0.0, 1.0], key_rots)([float(t)])[0].as_matrix()
    R_interp_t = torch.tensor(R_interp, dtype=torch.float32, device=pa.device)

    # ── Translation from circle position + rotation ───────────────────────────
    # PGSR: camera_center = −T @ R.T  ⟹  T = −camera_center @ R
    T_new = -(new_center.unsqueeze(0) @ R_interp_t).squeeze(0)

    return make_pseudo_camera(cam_src, R_interp, T_new.cpu().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# 5. Forward RGB warping with Z-buffer and confidence masking
# ──────────────────────────────────────────────────────────────────────────────

def warp_image_rgb(src_cam, pseudo_cam, depth_map, rgb_image,
                   conf_map=None, conf_threshold=0.2):
    """
    Forward-warp `rgb_image` from `src_cam` into `pseudo_cam` using
    `depth_map` for 3D reprojection.  Applies a confidence mask and resolves
    depth-order conflicts with a Z-buffer (smallest depth wins).

    Camera convention (PGSR, row vectors):
        pts_cam   = pts_world @ R + T
        pts_world = (pts_cam − T) @ R.T

    Args:
        src_cam:        source Camera object (PGSR Camera)
        pseudo_cam:     target pseudo-Camera (from interpolate_camera_on_circle)
        depth_map:      [H, W] float tensor — rendered plane depth from src_cam
        rgb_image:      [3, H, W] float tensor — GT RGB at src_cam, values in [0,1]
        conf_map:       [1,H,W] | [H,W] | None — per-pixel confidence in [0,1]
                        (e.g. rendered_alpha).  None ≡ all pixels confident.
        conf_threshold: pixels with conf < threshold are excluded from warping

    Returns:
        warped_rgb:  [3, H, W] float tensor (0 where no valid projection hit)
        valid_mask:  [H, W] bool tensor    (True where at least one src pixel hit)
    """
    H, W   = depth_map.shape
    device = depth_map.device

    # ── Source intrinsics (assume same for pseudo-cam — Pi-GS §3.6) ─────────
    fx    = float(src_cam.image_width  / (2.0 * math.tan(src_cam.FoVx / 2.0)))
    fy    = float(src_cam.image_height / (2.0 * math.tan(src_cam.FoVy / 2.0)))
    cx_px = W / 2.0
    cy_px = H / 2.0

    # ── Confidence mask ──────────────────────────────────────────────────────
    if conf_map is not None:
        conf_mask = (conf_map.squeeze().to(device) >= conf_threshold).reshape(-1)
    else:
        conf_mask = torch.ones(H * W, dtype=torch.bool, device=device)

    # ── Pixel grid ───────────────────────────────────────────────────────────
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )  # both [H, W]

    # ── Backproject: image pixel → src camera space ──────────────────────────
    d    = depth_map                               # [H, W]
    X_c  = (xx - cx_px) * d / fx
    Y_c  = (yy - cy_px) * d / fy
    pts_c = torch.stack([X_c, Y_c, d], dim=-1).reshape(-1, 3)  # [N, 3]

    # ── Src camera → world: pts_w = (pts_c − T) @ R.T ───────────────────────
    src_R = torch.tensor(src_cam.R,  device=device, dtype=torch.float32)  # [3,3]
    src_T = torch.tensor(src_cam.T,  device=device, dtype=torch.float32)  # [3]
    pts_w = (pts_c - src_T.unsqueeze(0)) @ src_R.transpose(0, 1)          # [N, 3]

    # ── World → target camera: pts_tgt = pts_w @ R_tgt + T_tgt ─────────────
    tgt_R = torch.tensor(pseudo_cam.R, device=device, dtype=torch.float32)
    tgt_T = torch.tensor(pseudo_cam.T, device=device, dtype=torch.float32)
    pts_tgt = pts_w @ tgt_R + tgt_T.unsqueeze(0)                          # [N, 3]

    Xt, Yt, Zt = pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2]

    # ── Project to target image plane ────────────────────────────────────────
    u_t = Xt * fx / (Zt + 1e-8) + cx_px
    v_t = Yt * fy / (Zt + 1e-8) + cy_px

    # ── Validity filter ───────────────────────────────────────────────────────
    depth_flat = d.reshape(-1)
    valid = (
        (Zt > 0.01) &
        (u_t >= 0.0) & (u_t <= W - 1.0) &
        (v_t >= 0.0) & (v_t <= H - 1.0) &
        conf_mask &
        (depth_flat > 0.01)
    )  # [N] bool

    if valid.sum() == 0:
        return (
            torch.zeros(3, H, W, device=device),
            torch.zeros(H, W, dtype=torch.bool, device=device),
        )

    u_v      = u_t[valid].long()
    v_v      = v_t[valid].long()
    Z_v      = Zt[valid]
    src_idx  = torch.where(valid)[0]      # flat source indices  [M]
    tgt_idx  = v_v * W + u_v             # flat target indices  [M]

    # ── Z-buffer (smallest depth wins) ───────────────────────────────────────
    # Paint in descending depth order → the last write (smallest Z) survives.
    sort_idx = torch.argsort(Z_v, descending=True)

    out_depth     = torch.full((H * W,), float('inf'), device=device)
    surviving_src = torch.zeros(H * W,  dtype=torch.long,  device=device)

    out_depth.scatter_(0,     tgt_idx[sort_idx], Z_v[sort_idx])
    surviving_src.scatter_(0, tgt_idx[sort_idx], src_idx[sort_idx])

    valid_tgt     = out_depth < float('inf')           # [H*W] bool
    valid_tgt_idx = torch.where(valid_tgt)[0]          # [M'] target flat indices
    src_gather    = surviving_src[valid_tgt_idx]       # [M'] source flat indices

    # ── Gather source colors ──────────────────────────────────────────────────
    rgb_flat    = rgb_image.reshape(3, -1)             # [3, H*W]
    warped_flat = torch.zeros(3, H * W, device=device)
    warped_flat[:, valid_tgt_idx] = rgb_flat[:, src_gather]

    warped_rgb = warped_flat.reshape(3, H, W)
    valid_mask = valid_tgt.reshape(H, W)

    return warped_rgb, valid_mask
