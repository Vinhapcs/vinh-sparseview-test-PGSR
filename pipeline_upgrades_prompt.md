# Prompt: Replicate Advanced 3DGS Pipeline Upgrades

You are an expert AI coding assistant. Your task is to upgrade a vanilla 3D Gaussian Splatting (3DGS) codebase by implementing a series of advanced regularizations and view synthesis techniques (often found in sparse-view or geometry-enhanced 3DGS pipelines). 

Please follow these exact ablation steps to upgrade the pipeline:

## Step 1: Add New Hyperparameters
Modify `arguments/__init__.py` to include the following new optimization parameters in the `OptimizationParams` class:
```python
self.lambda_flatten = 1.0     # For PGSR (Planar Regularization)
self.lambda_depth = 0.1       # For Depth Regularization
self.lambda_normal = 0.05     # For Normal Map Regularization
```

## Step 2: Implement Advanced Losses
Modify `utils/loss_utils.py` to add the following functions:
1. **Confidence-Aware Pearson Correlation Depth Loss**:
   - Implement `confidence_aware_pearson_loss(pred_depth, gt_depth, confidence=None)`.
   - Formula: Calculate the Pearson correlation weighted by confidence, and return `1.0 - p_conf`.
2. **Confidence-Aware Normal Loss**:
   - Implement `confidence_aware_normal_loss(pred_normal, gt_normal, conf)`.
   - Normalize both normal maps to `[-1, 1]` on `dim=0` (C, H, W).
   - Compute Cosine Similarity (`torch.nn.functional.cosine_similarity`).
   - Loss formula: `1.0 - cosine_sim`, weighted by the confidence map.
3. **Fast SSIM**:
   - Create a `FusedSSIMMap` using `torch.autograd.Function` if `diff_gaussian_rasterization` supports `fusedssim`.
   - Implement `fast_ssim(img1, img2)`.

## Step 3: Normal Map Inference Tool (`normal.py`)
Create a new file `normal.py` at the root directory to automatically infer and generate normal maps:
- Automatically download the `omnidata_dpt_normal_v2.ckpt` from HuggingFace to a `weights/` folder.
- Clone the EPFL-VILAB Omnidata repository locally and import `DPTDepthModel`.
- Read the dataset images from the `images/` directory.
- Support reading dataset splits: if `train_split.npy` or `sparse/0/test.txt` exists, filter out test images. Also support an `--eval` flag for LLFF hold=8.
- For each image, inference the normal map using `DPTDepthModel`.
- Resize the prediction back to the scaled resolution and save as `.npy` in the `normals/` directory (shape `H x W x 3` or `3 x H x W`).

## Step 4: Upgrade Gaussian Model (`scene/gaussian_model.py`)
Enhance the Gaussian model to support normals and better quaternion constraints:
- **Unit Vector Fix**: Ensure quaternions are normalized by setting `self.rotation_activation = torch.nn.functional.normalize`.
- **Normal Extraction**: Implement a function (e.g., `get_normals_rgb()`) to extract normal vectors directly from the rotation quaternions of the Gaussians and map them to the RGB space for rendering.
- **PGSR Support**: Add scale regularization constraints to encourage Gaussians to become planar (flattening).

## Step 5: Incorporate Warping for Sparse Views (`utils/warp_utils.py`)
Create `utils/warp_utils.py` to handle cross-view warping:
- Implement `find_nearest_cameras(cameras, num_neighbors=2)`.
- Implement `slerp(q1, q2, t)` for quaternion interpolation.
- Implement `interpolate_cameras(cam1, cam2, t)` using SLERP for rotation and linear interpolation for translation.
- Implement `warp_image(src_cam, tgt_R, tgt_T, depth_map, conf_map)`:
  - Reproject pixels from `src_cam` to 3D space using `depth_map`.
  - Project them into the new target camera view.
  - Implement Z-buffer logic (`torch.argsort` on depth) to handle occlusions.

## Step 6: Integrate everything into the Training Loop (`train.py`)
Modify the `train.py` script to apply the regularizations during optimization:
1. **Load Priors**: Before the loop, scan the dataset directory for `normals/` and load `.npy` files into a `normal_priors` dictionary on CUDA. Do the same for depth if applicable.
2. **Normal Loss Pass**: Inside the training loop, if `opt.lambda_normal > 0`:
   - Extract Gaussian normals using `gaussians.get_normals_rgb()`.
   - Render a second pass using these normals as `override_color`.
   - Rescale the rendered normal from `[0, 1]` to `[-1, 1]`.
   - Resize the Ground Truth (GT) normal to match the render resolution.
   - Apply `viewpoint_cam.alpha_mask` to both rendered and GT normals to ignore background.
   - Calculate `normal_loss = confidence_aware_normal_loss(...)` and add `opt.lambda_normal * normal_loss` to the total loss.
3. **Depth & PGSR**:
   - Add depth loss calculation similarly if depth priors are available.
   - Add PGSR (Planar Regularization) loss using `opt.lambda_flatten` to penalize the smallest scale of the Gaussians.

## Step 7: Dataset Reader Fixes (`scene/dataset_readers.py`)
- Ensure that `test.txt` and `train_split.npy` are properly read to avoid leaking test images into the training pipeline (e.g., when initializing point clouds or normalizing NeRF poses).

Execute these steps precisely to complete the pipeline upgrade.
