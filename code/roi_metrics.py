# -*- coding: utf-8 -*-
"""
Lesion-region metrics for the SPIN paper ("ROI-PSNR").

ROI-PSNR = PSNR computed only over pixels inside the ground-truth
tumor-core + edema mask dilated by `dilate` pixels, on the same 8-bit
images TLP uses for global PSNR/SSIM/NMSE.

Usage inside test.py (after fake_B / real_B are converted to uint8 as in TLP):

    from roi_metrics import roi_mask, roi_psnr, global_metrics
    m = roi_mask(edema_region[0, 0].cpu().numpy(), tumor_region[0, 0].cpu().numpy(), dilate=5)
    r = roi_psnr(real_B_255, fake_B_255, m)          # float or nan if the slice has no lesion
    g = global_metrics(real_B_255, fake_B_255)        # dict(psnr, ssim, nmse)

Aggregate ROI-PSNR over slices that contain a lesion (m.sum() > 0) only, and
report the number of such slices alongside the mean.  Keep the same test
split (subjects 870..999 of data_list/Brats2021_selected.txt) as TLP.
"""
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim


def roi_mask(edema, tumor, dilate=5):
    """Binary ROI = (edema | tumor) dilated by `dilate` pixels (SciPy's default cross-shaped
    3x3 structuring element, applied `dilate` times)."""
    m = (edema > 0) | (tumor > 0)
    if dilate > 0 and m.any():
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m


def roi_psnr(gt_u8, pred_u8, mask, data_range=255.0):
    """PSNR restricted to `mask`. Returns np.nan when the mask is empty."""
    if mask.sum() == 0:
        return np.nan
    gt = gt_u8.astype(np.float64)[mask]
    pr = pred_u8.astype(np.float64)[mask]
    mse = np.mean((gt - pr) ** 2)
    if mse == 0:
        return np.inf
    return 10.0 * np.log10(data_range ** 2 / mse)


def global_metrics(gt_u8, pred_u8):
    """Same conventions as TLP's evaluation.py (8-bit images):
    PSNR/SSIM from skimage with uint8 data range, NMSE = MSE / var(GT)."""
    gt = gt_u8.astype(np.float64)
    pr = pred_u8.astype(np.float64)
    mse = np.mean((gt - pr) ** 2)
    psnr = np.inf if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    var = np.var(gt)
    nmse = mse / var if var != 0 else 0.0
    return dict(psnr=psnr, ssim=ssim(gt_u8, pred_u8, data_range=255), mse=mse, nmse=nmse)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, (192, 160), dtype=np.uint8)
    pred = np.clip(gt.astype(int) + rng.integers(-10, 11, gt.shape), 0, 255).astype(np.uint8)
    ed = np.zeros(gt.shape); tu = np.zeros(gt.shape); tu[80:110, 60:90] = 1
    m = roi_mask(ed, tu, 5)
    print("roi pixels:", int(m.sum()), "roi-psnr:", round(roi_psnr(gt, pred, m), 2), global_metrics(gt, pred))
