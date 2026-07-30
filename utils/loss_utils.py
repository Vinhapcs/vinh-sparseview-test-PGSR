#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def align_depth_ls(pred, gt, mask=None):
    """
    Least-squares scale-shift alignment: finds scalar a, b such that gt ≈ a*pred + b.
    Operates only on valid (masked) pixels to avoid sky/invalid contamination.
    Returns the aligned pred depth (same shape as input pred).

    Args:
        pred: rendered depth tensor, any shape
        gt:   monocular GT depth tensor, same shape as pred
        mask: bool/float tensor, same shape. True/1 = valid pixel. None = all valid.
    """
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    valid = mask.bool().view(-1)

    pred_flat = pred.detach().view(-1)[valid].unsqueeze(1)   # (N, 1)
    gt_flat   = gt.detach().view(-1)[valid].unsqueeze(1)     # (N, 1)

    if pred_flat.numel() < 2:
        return pred  # not enough valid pixels — skip alignment

    A = torch.cat([pred_flat, torch.ones_like(pred_flat)], dim=1)  # (N, 2)
    try:
        result = torch.linalg.lstsq(A, gt_flat)
        a = result.solution[0].squeeze()
        b = result.solution[1].squeeze()
        a = torch.clamp(a, min=1e-3)   # depth scale must be positive
    except Exception:
        a = torch.tensor(1.0, device=pred.device)
        b = torch.tensor(0.0, device=pred.device)

    return a * pred + b

def confidence_aware_pearson_loss(pred_depth, gt_depth, confidence=None):
    if confidence is None:
        confidence = torch.ones_like(pred_depth)
    
    pred_depth = pred_depth.view(-1)
    gt_depth = gt_depth.view(-1)
    confidence = confidence.view(-1)
    
    mean_pred = torch.sum(pred_depth * confidence) / torch.clamp(torch.sum(confidence), min=1e-8)
    mean_gt = torch.sum(gt_depth * confidence) / torch.clamp(torch.sum(confidence), min=1e-8)
    
    pred_centered = pred_depth - mean_pred
    gt_centered = gt_depth - mean_gt
    
    cov = torch.sum(confidence * pred_centered * gt_centered)
    var_pred = torch.sum(confidence * pred_centered ** 2)
    var_gt = torch.sum(confidence * gt_centered ** 2)
    
    denominator = torch.sqrt(torch.clamp(var_pred * var_gt, min=1e-8))
    pearson_corr = cov / denominator
    
    loss = 1.0 - pearson_corr
    return loss

def confidence_aware_normal_loss(pred_normal, gt_normal, conf):
    pred_normal = torch.nn.functional.normalize(pred_normal, p=2, dim=0)
    gt_normal = torch.nn.functional.normalize(gt_normal, p=2, dim=0)
    
    # Cosine distance
    cosine_sim = torch.nn.functional.cosine_similarity(pred_normal, gt_normal, dim=0)
    loss_cos = (1.0 - cosine_sim)
    
    loss = loss_cos * conf.squeeze(0)
    return loss.mean()

try:
    from diff_gaussian_rasterization import fusedssim
    class FusedSSIMMap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, img1, img2):
            loss = fusedssim.FusedSSIM_forward(img1, img2)
            ctx.save_for_backward(img1, img2)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            img1, img2 = ctx.saved_tensors
            grad_img1, grad_img2 = fusedssim.FusedSSIM_backward(img1, img2, grad_output)
            return grad_img1, grad_img2
    
    def fast_ssim(img1, img2):
        return FusedSSIMMap.apply(img1, img2)
except Exception:
    def fast_ssim(img1, img2):
        return ssim(img1, img2)