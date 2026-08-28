# Vinh Sparse-View PGSR — Pipeline Documentation

> Dự án này là một biến thể của **PGSR (Planar-based Gaussian Splatting Reconstruction)** được mở rộng để hoạt động tốt hơn trong điều kiện **sparse-view** (ít ảnh đầu vào), tích hợp điểm đám mây từ **MASt3R** và nhiều cơ chế giám sát bổ sung.

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Module: `scene/`](#3-module-scene)
4. [Module: `gaussian_renderer/`](#4-module-gaussian_renderer)
5. [Module: `utils/`](#5-module-utils)
6. [Module: `arguments/`](#6-module-arguments)
7. [Training Pipeline (`train.py`)](#7-training-pipeline-trainpy)
8. [Loss Functions đang Active](#8-loss-functions-đang-active)
9. [Ablation Flags — Tính năng đã comment out](#9-ablation-flags--tính-năng-đã-comment-out)
10. [Hyperparameters quan trọng](#10-hyperparameters-quan-trọng)
11. [Data Format yêu cầu](#11-data-format-yêu-cầu)

---

## 1. Tổng quan kiến trúc

```
Input Scene (COLMAP sparse)
        │
        ▼
┌─────────────────────┐
│  MASt3R Point Cloud │  ← points3D.ply, confidence score lưu trong normals[:,0]
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Co-visible Redundancy Elimination (CRE)│  ← scene/cre_filter.py  [MỚI]
│  Loại bỏ điểm trùng lặp dựa trên depth │
└─────────────────────────────────────────┘
        │                    │
        │ (filtered pcd)     │ (unfiltered pcd → scene.unfiltered_pcd)
        ▼                    ▼ (dùng cho depth warping, tránh lỗ hổng)
┌──────────────────┐
│  GaussianModel   │  ← 3D Gaussians được khởi tạo từ pcd đã lọc
└──────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│                    Training Loop                        │
│                                                        │
│  Render (PGSR diff_plane_rasterization)                │
│    → image, plane_depth, rendered_normal, rendered_alpha│
│                                                        │
│  Losses (active):                                      │
│    ✅ L1 + SSIM                                        │
│    ✅ Flatten Loss (PGSR §3.2)                         │
│    ✅ Depth Prior Loss (Pearson + scale-shift align)   │
│    🚫 Depth Warp Loss (Pi-GS §3.6) ← đã comment out  │
│                                                        │
│  Densify & Prune (standard 3DGS)                      │
└────────────────────────────────────────────────────────┘
```

---

## 2. Cấu trúc thư mục

```
vinh-sparseview-test-PGSR/
│
├── train.py                    # Entry point — vòng lặp training chính
├── render.py                   # Script render sau khi training
├── metrics.py                  # Tính PSNR, SSIM, LPIPS
│
├── scene/                      # Quản lý scene và dữ liệu
│   ├── __init__.py             # Scene class — khởi tạo cameras, CRE, Gaussians
│   ├── cre_filter.py           # [MỚI] Co-visible Redundancy Elimination
│   ├── dataset_readers.py      # Đọc dữ liệu COLMAP / Blender
│   ├── gaussian_model.py       # GaussianModel — quản lý tham số 3D Gaussians
│   ├── cameras.py              # Camera class (intrinsics + extrinsics)
│   ├── colmap_loader.py        # Parser cho COLMAP binary/text format
│   └── app_model.py            # [ABLATION] Appearance/exposure model
│
├── gaussian_renderer/
│   └── __init__.py             # Hàm render() — gọi diff_plane_rasterization
│
├── utils/
│   ├── loss_utils.py           # Tất cả loss functions
│   ├── warp_utils.py           # [ABLATION] Depth warping utilities
│   ├── graphics_utils.py       # BasicPointCloud, projection math
│   ├── camera_utils.py         # Chuyển đổi CameraInfo → Camera object
│   ├── general_utils.py        # Activation functions, LR scheduler
│   ├── image_utils.py          # PSNR
│   └── sh_utils.py             # Spherical Harmonics
│
├── arguments/
│   └── __init__.py             # ModelParams, OptimizationParams, PipelineParams
│
└── submodules/                 # CUDA extensions (diff_plane_rasterization, simple_knn)
```

---

## 3. Module: `scene/`

### `scene/__init__.py` — `Scene` class

Điểm khởi tạo trung tâm của toàn bộ pipeline.

**Luồng khởi tạo:**

1. Đọc `scene_info` từ COLMAP (`sparse/0/`) hoặc Blender (`transforms_train.json`)
2. Build `train_cameras` và `test_cameras` từ `CameraInfo`
3. Tính `nearest_id` cho mỗi camera (dùng cho multi-view, đã tắt)
4. **[MỚI]** Chạy CRE filter → lưu `scene.unfiltered_pcd` (pcd gốc dày đặc)
5. Khởi tạo `GaussianModel` từ pcd đã lọc

**Thuộc tính quan trọng:**

| Thuộc tính | Mô tả |
|---|---|
| `scene.train_cameras[1.0]` | Danh sách Camera object cho training |
| `scene.test_cameras[1.0]` | Danh sách Camera object cho test |
| `scene.cameras_extent` | Bán kính chuẩn hóa nerf++ |
| `scene.unfiltered_pcd` | `BasicPointCloud` gốc trước CRE (dùng cho depth warp) |
| `scene.gaussians` | `GaussianModel` object |

---

### `scene/cre_filter.py` — Co-visible Redundancy Elimination [MỚI]

**Mục đích:** Loại bỏ các điểm trùng lặp trong point cloud — những điểm có confidence thấp hơn nhưng chiếu vào cùng pixel với một điểm confidence cao hơn trên cùng bề mặt.

**Thuật toán:**

```
1. Đưa points, colors, normals lên CUDA
2. Sắp xếp tất cả N điểm giảm dần theo confidence (normals[:,0])
3. keep_mask = [True] * N
4. Với mỗi camera trong train_cameras:
   a. Load depth map GT từ depths/<image_name>.npy
   b. Lấy các điểm active (keep_mask == True)
   c. Project vào image space: pts_cam = pts_h @ world_view_transform
   d. Lọc điểm trong ảnh (0 ≤ u < W, 0 ≤ v < H, Z > 0.01)
   e. Depth test: |Z - Z_gt| < 0.05 * Z_gt  → "on-surface"
   f. Redundancy check: trong các điểm on-surface cùng pixel (u,v),
      chỉ giữ điểm FIRST (confidence cao nhất); set keep_mask=False cho phần còn lại
5. Trả về BasicPointCloud được lọc
```

**Điều kiện chạy:** Chỉ chạy nếu thư mục `<dataset>/depths/` tồn tại. Nếu không có, bỏ qua (graceful fallback).

**Lưu ý quan trọng:**
- `scene.unfiltered_pcd` giữ lại pcd đầy đủ (trước CRE) để depth warping không bị lỗ hổng
- `torch.cuda.empty_cache()` được gọi sau khi xong

---

### `scene/gaussian_model.py` — `GaussianModel`

Quản lý toàn bộ tham số của 3D Gaussians:

| Tham số | Ý nghĩa |
|---|---|
| `_xyz` | Vị trí 3D (N, 3) |
| `_features_dc`, `_features_rest` | Spherical Harmonics coefficients |
| `_scaling` | Scale log-space (N, 3) |
| `_rotation` | Quaternion (N, 4) |
| `_opacity` | Opacity logit-space (N, 1) |

**Phương thức quan trọng:**

- `create_from_pcd(pcd, cameras_extent)` — Khởi tạo từ `BasicPointCloud`
- `densify_and_prune(...)` — Clone/split/prune Gaussians theo gradient
- `get_normal(viewpoint_cam)` — Tính normal từ chiều scale nhỏ nhất
- `get_smallest_scale()` — Dùng cho Flatten Loss

---

### `scene/cameras.py` — `Camera`

Camera object chứa đầy đủ intrinsics + extrinsics. Các thuộc tính hay dùng:

| Thuộc tính | Giá trị |
|---|---|
| `world_view_transform` | W2C^T (4×4 CUDA tensor) — dùng cho projection |
| `full_proj_transform` | Projection matrix đầy đủ |
| `camera_center` | Vị trí camera trong world space |
| `Fx, Fy, Cx, Cy` | Focal length và principal point (pixels) |
| `FoVx, FoVy` | Field of view (radians) |
| `image_name` | Tên file ảnh (không có extension) |
| `nearest_id` | List index của nearest cameras (multi-view) |

**Camera convention (PGSR / COLMAP, row-vector):**
```
pts_cam = pts_world @ R + T
pts_world = (pts_cam - T) @ R.T
camera_center = -T @ R.T
```

---

### `scene/dataset_readers.py`

Đọc dữ liệu scene và trả về `SceneInfo`:

```python
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud   # points, colors, normals (normals[:,0] = MASt3R conf)
    train_cameras: list            # list[CameraInfo]
    test_cameras: list
    nerf_normalization: dict       # {'translate': ..., 'radius': ...}
    ply_path: str
```

Hỗ trợ 2 format:
- **COLMAP**: đọc từ `sparse/0/{images.bin, cameras.bin, points3D.ply}`
- **Blender**: đọc từ `transforms_train.json`

Train/test split theo thứ tự ưu tiên: `split.json` > `train_split.npy` > `sparse/0/test.txt` > `test.txt` > `--eval` flag (mỗi 8 ảnh)

---

## 4. Module: `gaussian_renderer/`

### `gaussian_renderer/__init__.py` — `render()`

Wrapper cho CUDA kernel `diff_plane_rasterization` (custom PGSR rasterizer).

**Signature:**
```python
render(viewpoint_camera, pc, pipe, bg_color,
       return_plane=True, return_depth_normal=False) -> dict
```

**Output dict:**

| Key | Shape | Mô tả |
|---|---|---|
| `render` | `[3, H, W]` | RGB rendered image |
| `plane_depth` | `[1, H, W]` | PGSR plane depth = `\|n_cam · p_cam\|` (⚠ KHÔNG phải z-depth) |
| `rendered_normal` | `[3, H, W]` | Camera-space normals (alpha-blended) |
| `rendered_alpha` | `[1, H, W]` | Gaussian coverage ∈ [0,1] — dùng làm confidence mask |
| `rendered_distance` | `[1, H, W]` | Alpha-blended planar distance |
| `visibility_filter` | `[N]` bool | Các Gaussian visible trong frame này |
| `radii` | `[N]` | Screen-space radius |
| `out_observe` | `[N]` | Số lần quan sát |
| `viewspace_points` | `[N, 3]` | Screenspace gradients (dùng cho densification) |

> ⚠ **Quan trọng:** `plane_depth` ≠ z-depth. Để chuyển đổi, dùng `plane_depth_to_z_depth()` trong `utils/warp_utils.py`.

**Khi `return_plane=False`:** Render nhanh hơn, chỉ trả về `render`, `radii`, `out_observe`.

---

## 5. Module: `utils/`

### `utils/loss_utils.py`

| Hàm | Trạng thái | Mô tả |
|---|---|---|
| `l1_loss(pred, gt)` | ✅ Active | Mean absolute error |
| `ssim(img1, img2)` | ✅ Active | Structural similarity |
| `confidence_aware_pearson_loss(pred, gt, conf)` | ✅ Active | Pearson correlation có mask confidence |
| `align_depth_ls(pred, gt, mask)` | ✅ Active | Scale-shift alignment bằng least-squares trước khi tính Pearson |
| `get_img_grad_weight(img)` | 🚫 Ablation | Image gradient weight cho single-view normal loss |
| `lncc(ref, nea)` | 🚫 Ablation | Local NCC cho multi-view photometric consistency |

### `utils/warp_utils.py` [ABLATION — đã comment out]

Tập hợp các hàm cho Pi-GS §3.6 depth warping:

| Hàm | Mô tả |
|---|---|
| `plane_depth_to_z_depth(plane_depth, rendered_normal, fx, fy, cx, cy)` | Chuyển PGSR plane_depth → z-depth thật |
| `find_nearest_cameras(cameras, target_cam, n)` | Tìm N camera gần nhất theo Euclidean distance |
| `circle_through_3_points(p1, p2, p3)` | Tính circumscribed circle qua 3 điểm trong 3D |
| `interpolate_camera_on_circle(cam_src, cam_tgt, cam_ref, t)` | Sinh pseudo-camera trên cung tròn + SLERP rotation |
| `make_pseudo_camera(template, R, T)` | Tạo Camera object với extrinsics mới |
| `warp_image_rgb(src_cam, pseudo_cam, depth_map, rgb)` | Forward-warp RGB với Z-buffer và confidence mask |

### `utils/graphics_utils.py`

| Thành phần | Mô tả |
|---|---|
| `BasicPointCloud` | NamedTuple: `(points, colors, normals)` — dtype numpy |
| `getWorld2View2(R, T, translate, scale)` | Tạo W2C matrix (4×4 numpy) |
| `getProjectionMatrix(znear, zfar, fovX, fovY)` | Projection matrix cho CUDA renderer |
| `fov2focal(fov, pixels)` | Chuyển FoV → focal length |
| `normal_from_depth_image(depth, K, E)` | Tính normal từ depth image |

---

## 6. Module: `arguments/`

### `arguments/__init__.py`

Ba nhóm tham số chính:

#### `ModelParams` — Tham số scene

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `source_path` | `""` | Đường dẫn đến dataset |
| `model_path` | `""` | Đường dẫn output |
| `sh_degree` | `3` | Bậc Spherical Harmonics |
| `images` | `"images"` | Thư mục ảnh |
| `eval` | `False` | Chế độ eval (train/test split mỗi 8) |
| `multi_view_num` | `8` | Số camera lân cận tối đa |
| `multi_view_max_angle` | `30°` | Góc tối đa khi tìm nearest camera |
| `multi_view_max_dis` | `1.5` | Khoảng cách tối đa khi tìm nearest camera |

#### `OptimizationParams` — Tham số training

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `iterations` | `30,000` | Số iteration |
| `densify_from_iter` | `500` | Bắt đầu densify |
| `densify_until_iter` | `15,000` | Dừng densify |
| `densify_grad_threshold` | `0.0002` | Ngưỡng gradient cho clone/split |
| `densify_abs_grad_threshold` | `0.0008` | Ngưỡng gradient tuyệt đối |
| `opacity_reset_interval` | `3,000` | Reset opacity định kỳ |
| `opacity_cull_threshold` | `0.005` | Prune Gaussian có opacity < ngưỡng |
| `max_all_points` | `6,000,000` | Giới hạn tổng số Gaussians |
| **`lambda_flatten`** | **`1.0`** | **✅ Flatten loss weight (PGSR core)** |
| **`lambda_depth`** | **`0.05`** | **✅ Depth prior loss weight** |
| **`lambda_depth_from_iter`** | **`1,000`** | **✅ Delay trước khi bật depth loss** |
| `lambda_depth_warp` | `0.1` | 🚫 Depth warp loss weight (đã tắt) |

---

## 7. Training Pipeline (`train.py`)

### Luồng khởi tạo (1 lần)

```python
gaussians = GaussianModel(sh_degree=3)
scene = Scene(dataset, gaussians)
    # → chạy CRE, khởi tạo Gaussians từ pcd đã lọc

gaussians.training_setup(opt)

# Load depth priors từ <source_path>/depths/*.npy
depth_priors = {cam_name: tensor[1,H,W]}
```

### Vòng lặp chính (mỗi iteration)

```
1. Pick ngẫu nhiên 1 viewpoint_cam

2. RENDER:
   render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                       return_plane=True, return_depth_normal=False)

3. IMAGE LOSS:
   Ll1   = L1(rendered, gt)
   SSIM  = (1 - SSIM(rendered, gt))
   loss  = 0.8 * Ll1 + 0.2 * SSIM

4. FLATTEN LOSS (PGSR §3.2):
   loss += lambda_flatten * mean(smallest_scale[visible])

5. DEPTH PRIOR LOSS:
   if iteration > 1000 and cam có depth prior:
       gt_depth = depth_priors[cam_name]            # [1,H,W]
       rendered_depth = render_pkg['plane_depth']   # [1,H,W]
       valid_mask = rendered_alpha > 0.5            # confidence mask
       aligned = align_depth_ls(rendered_depth, gt_depth, mask)
       loss += 0.05 * pearson_loss(aligned, gt_depth, confidence=valid_mask)

6. [ABLATION: depth_warp] — ĐÃ TẮT

7. BACKWARD + OPTIMIZER STEP

8. DENSIFY & PRUNE (iteration < 15,000)

9. OPACITY RESET (mỗi 3,000 iter)
```

---

## 8. Loss Functions đang Active

| Loss | Weight | Điều kiện | Mô tả |
|---|---|---|---|
| **L1** | `0.8` | Mọi iter | Pixel-wise absolute error |
| **SSIM** | `0.2` | Mọi iter | Structural similarity |
| **Flatten** | `1.0` (`lambda_flatten`) | Mọi iter (khi có visible Gaussian) | Phạt scale nhỏ nhất → ép Gaussian thành đĩa phẳng (PGSR §3.2) |
| **Depth Prior** | `0.05` (`lambda_depth`) | `iter > 1000`, camera có file `.npy` | Pearson correlation giữa rendered depth và GT depth, có scale-shift alignment |

---

## 9. Ablation Flags — Tính năng đã comment out

Tất cả tính năng comment out đều được đánh dấu `[ABLATION: <tên>]` trong code để dễ tìm.

| Tag | Tính năng | Vị trí | Cách bật lại |
|---|---|---|---|
| `[ABLATION: depth_warp]` | Pi-GS pseudo-view supervision | `train.py` lines 31–33, 178–196, 272–284, 321–414 | Uncomment import + 3 khối code |
| `[ABLATION: normal_prior]` | Monocular normal supervision | `train.py` lines 148–162, 302–319 | Uncomment + set `lambda_normal > 0` |
| `[ABLATION: single_view]` | Image-grad weighted depth-normal consistency | `train.py` lines 452–464 | Uncomment + set `single_view_weight > 0` |
| `[ABLATION: multi_view]` | NCC + Homography multi-view photometric | `train.py` lines 466–616 | Uncomment + set weights > 0 |
| `[ABLATION: app_model]` | Appearance / exposure compensation | `train.py` (nhiều chỗ) | Uncomment + bật `AppModel` |
| `[ABLATION: scale_loss]` | Min-scale regularization riêng biệt | `train.py` lines 445–450 | Uncomment + set `scale_loss_weight > 0` |
| `[ABLATION: two_phase_backward]` | Tách backward image-loss vs aux-loss | `train.py` lines 621–640 | Uncomment để densify clean hơn |
| `[ABLATION: confidence_grad]` | InstantSplat confidence-aware gradient | `train.py` lines 642–647 | Uncomment khi cần |
| `[ABLATION: multi_view_trim]` | Prune Gaussian ít được quan sát | `train.py` lines 696–706 | Uncomment + bật `use_multi_view_trim` |

---

## 10. Hyperparameters quan trọng

### Thay đổi loss weights

Sửa trong `arguments/__init__.py` → class `OptimizationParams`:

```python
self.lambda_flatten = 1.0    # Flatten loss — PGSR core, không nên tắt
self.lambda_depth   = 0.05   # Depth prior — giảm nếu depth prior có noise
self.lambda_depth_from_iter = 1000  # Delay để Gaussians ổn định trước
```

### Thay đổi depth_tolerance trong CRE

Sửa trong `scene/__init__.py`:

```python
filtered_pcd = filter_redundant_points(
    pcd=scene_info.point_cloud,
    train_cameras=_train_cams_for_cre,
    dataset_path=self.source_path,
    depth_tolerance=0.05,   # ← thay đổi ở đây (0.05 = 5% relative error)
)
```

### Tắt CRE hoàn toàn

Xóa hoặc comment thư mục `depths/` trong dataset, hoặc đặt `depth_tolerance=0.0`.

---

## 11. Data Format yêu cầu

```
<dataset_path>/
├── sparse/0/
│   ├── cameras.bin          # COLMAP intrinsics
│   ├── images.bin           # COLMAP extrinsics
│   └── points3D.ply         # Dense point cloud từ MASt3R
│                            # normals[:,0] = MASt3R confidence score
├── images/
│   ├── frame_001.jpg
│   └── ...
├── depths/                  # [TÙY CHỌN] GT depth maps
│   ├── frame_001.npy        # shape: (H,W) hoặc (1,H,W), float32, đơn vị mét
│   └── ...
└── split.json               # [TÙY CHỌN] train/test split
    # {"train": ["frame_001", ...], "test": ["frame_002", ...]}
```

> **Lưu ý về confidence score trong MASt3R:**
> Confidence được lưu vào `normals[:,0]` của file `.ply`. Giá trị cao = điểm tin cậy hơn. CRE sắp xếp points giảm dần theo confidence trước khi lọc, đảm bảo giữ lại điểm chất lượng cao nhất trên mỗi bề mặt.

---

## Chạy training

```bash
python train.py \
  -s <dataset_path> \
  -m <output_path> \
  --eval \
  --iterations 30000
```

## Render + đánh giá

```bash
python render.py -m <output_path>
python metrics.py -m <output_path>
```
