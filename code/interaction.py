# -*- coding: utf-8 -*-
"""
The simulated radiologist for interactive refinement (shared by train_ir.py and the evaluation scripts).

Given the model's current synthesis y and the truth gt, the oracle marks
  +1  the single worst region *inside* the lesion (missed / under-enhanced tumour), and
  -1  the single worst *over-enhanced* region outside the lesion (false enhancement),
each as one compact blob: pixels above a per-slice error quantile are dilated so they look
hand-drawn, and only the largest connected component of each sign is kept -- a radiologist
circles the worst spot, they do not paint the whole slice. At test time the same routine is
the oracle; the point of the experiment is what the *network* does with such marks, so both
the naive baseline and SPIN-IR receive exactly the same ones.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _largest_cc(mask):
    """mask: (H,W) bool numpy -> keep only its largest 8-connected component."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def simulate_correction(y, gt, ed, tu, q=0.90, r=3, largest_cc=True, drop_pos=0.0, drop_neg=0.0,
                        max_frac=None):
    """Returns (c, pos, neg): c in {-1,0,+1}, pos/neg the two blobs as {0,1} float maps.
    y, gt, ed, tu: (B,1,H,W) tensors on the same device.

    max_frac: if given, each mark is limited to this fraction of the slice. Without it the
    largest over-enhanced component outside the lesion can be very large on slices where the
    network over-enhances a whole region (median 11% of the slice, up to 64%, on the BraTS
    test set) -- not something a reader would draw. The cap is enforced by raising the error
    quantile for that sign until the resulting blob fits, so the mark stays on the worst
    pixels but becomes a spot rather than a territory."""
    err = (y - gt).abs()
    lesion = (ed + tu) > 0
    b, _, h, w = err.shape
    k = 2 * r + 1
    qs = [q] if max_frac is None else sorted({q, 0.95, 0.975, 0.99, 0.995, 0.998, 0.999})
    qs = [x for x in qs if x >= q]
    taus = torch.quantile(err.flatten(1), torch.tensor(qs, device=err.device, dtype=err.dtype), dim=1)  # (nq, B)
    over = (y > gt)

    def blob(sign_mask, i, qi):
        hi = err[i:i + 1] > taus[qi, i]
        m = (hi & sign_mask[i:i + 1]).float()
        m = F.max_pool2d(m, k, stride=1, padding=r)
        mn = m[0, 0].cpu().numpy() > 0
        if largest_cc:
            mn = _largest_cc(mn)
        return mn

    pn = np.zeros((b, h, w), dtype=bool); nn_ = np.zeros((b, h, w), dtype=bool)
    limit = None if max_frac is None else max_frac * h * w
    for i in range(b):
        for sign, store in ((lesion, pn), (~lesion & over, nn_)):
            for qi in range(len(qs)):
                m = blob(sign, i, qi)
                if limit is None or m.sum() <= limit:
                    break
            store[i] = m
    pos = torch.from_numpy(pn).to(y.device, y.dtype).unsqueeze(1)
    neg = torch.from_numpy(nn_).to(y.device, y.dtype).unsqueeze(1)
    if drop_pos > 0:
        pos = pos * (torch.rand(b, 1, 1, 1, device=y.device) > drop_pos).float()
    if drop_neg > 0:
        neg = neg * (torch.rand(b, 1, 1, 1, device=y.device) > drop_neg).float()
    c = (pos - neg).clamp(-1, 1)
    return c, pos, neg
