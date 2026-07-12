import torch
import torch.nn.functional as F
import math

def find_nearest_cameras(cameras, target_cam_center, num_neighbors=2):
    dists = []
    for idx, cam in enumerate(cameras):
        dist = torch.norm(cam.camera_center - target_cam_center)
        dists.append((dist.item(), cam))
    dists.sort(key=lambda x: x[0])
    return [c[1] for c in dists[:num_neighbors]]

def slerp(q1, q2, t):
    cos_half_theta = torch.sum(q1 * q2, dim=-1)
    
    q2_corrected = torch.where(cos_half_theta.unsqueeze(-1) < 0, -q2, q2)
    cos_half_theta = torch.abs(cos_half_theta)
    
    half_theta = torch.acos(torch.clamp(cos_half_theta, -1.0, 1.0))
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta**2)
    
    ratio_a = torch.where(sin_half_theta < 0.001, 1 - t, torch.sin((1 - t) * half_theta) / sin_half_theta)
    ratio_b = torch.where(sin_half_theta < 0.001, t, torch.sin(t * half_theta) / sin_half_theta)
    
    return ratio_a.unsqueeze(-1) * q1 + ratio_b.unsqueeze(-1) * q2_corrected

def interpolate_cameras(cam1, cam2, t):
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    # Interpolate translation
    T1 = torch.tensor(cam1.T, dtype=torch.float32) if not isinstance(cam1.T, torch.Tensor) else cam1.T.float()
    T2 = torch.tensor(cam2.T, dtype=torch.float32) if not isinstance(cam2.T, torch.Tensor) else cam2.T.float()
    T_interp = (1 - t) * T1 + t * T2
    
    # Interpolate rotation
    R1 = torch.tensor(cam1.R, dtype=torch.float32) if not isinstance(cam1.R, torch.Tensor) else cam1.R.float()
    R2 = torch.tensor(cam2.R, dtype=torch.float32) if not isinstance(cam2.R, torch.Tensor) else cam2.R.float()
    q1 = matrix_to_quaternion(R1)
    q2 = matrix_to_quaternion(R2)
    q_interp = slerp(q1, q2, t)
    R_interp = quaternion_to_matrix(q_interp)
    return R_interp, T_interp

def warp_image(src_cam, tgt_R, tgt_T, depth_map, conf_map=None):
    # depth_map: [H, W]
    H, W = depth_map.shape
    device = depth_map.device
    
    # Intrinsic matrix
    fx = float(src_cam.image_width / (2 * math.tan(src_cam.FoVx / 2.)))
    fy = float(src_cam.image_height / (2 * math.tan(src_cam.FoVy / 2.)))
    cx, cy = W / 2.0, H / 2.0
    
    # Pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.float()
    y = y.float()
    
    # To camera coordinate
    X_c = (x - cx) * depth_map / fx
    Y_c = (y - cy) * depth_map / fy
    Z_c = depth_map
    pts_c = torch.stack([X_c, Y_c, Z_c], dim=-1).reshape(-1, 3) # [N, 3]
    
    # To world coordinate
    src_R = torch.tensor(src_cam.R, device=device, dtype=torch.float32)
    src_T = torch.tensor(src_cam.T, device=device, dtype=torch.float32)
    pts_w = (pts_c - src_T) @ src_R.transpose(0, 1) # [N, 3]
    
    # To target camera coordinate
    tgt_R = tgt_R.to(device, dtype=torch.float32)
    tgt_T = tgt_T.to(device, dtype=torch.float32)
    pts_tgt_c = (pts_w @ tgt_R) + tgt_T # [N, 3]
    
    # To target image coordinate
    X_tgt, Y_tgt, Z_tgt = pts_tgt_c[:, 0], pts_tgt_c[:, 1], pts_tgt_c[:, 2]
    u_tgt = (X_tgt * fx / Z_tgt) + cx
    v_tgt = (Y_tgt * fy / Z_tgt) + cy
    
    # Filter points behind camera and outside image
    valid = (Z_tgt > 0) & (u_tgt >= 0) & (u_tgt <= W - 1) & (v_tgt >= 0) & (v_tgt <= H - 1)
    
    u_tgt = u_tgt[valid].long()
    v_tgt = v_tgt[valid].long()
    Z_tgt = Z_tgt[valid]
    
    # Z-buffer logic
    out_depth = torch.full((H * W,), float('inf'), device=device)
    flat_idx = v_tgt * W + u_tgt
    
    # argsort logic: we want min depth, so we sort descending, meaning 
    # smaller depths are scattered last and overwrite larger ones
    sorted_idx = torch.argsort(Z_tgt, descending=True) 
    flat_idx_sorted = flat_idx[sorted_idx]
    Z_tgt_sorted = Z_tgt[sorted_idx]
    
    out_depth.scatter_(0, flat_idx_sorted, Z_tgt_sorted)
    out_depth = out_depth.view(H, W)
    out_depth[out_depth == float('inf')] = 0.0
    return out_depth
