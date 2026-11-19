# -*- coding: utf-8 -*-
"""
Downstream endpoint: does a synthesised T1ce still support the read
that the contrast agent is given for?

A segmenter is trained on *real* T1ce to delineate the enhancing tumour (BraTS label 4) --
the tissue the contrast agent is administered to reveal -- and is then applied unchanged to
each model's synthesis. Dice against the ground-truth mask measures whether the synthetic
image carries the diagnostic content, which is the quantity a clinical reader cares about and
which PSNR does not report. Dice on the real T1ce is the ceiling this endpoint can reach.

The segmenter is a plain U-Net rather than nnU-Net: it is a fixed, frozen measuring device
applied identically to every method, so its absolute strength does not affect the comparison,
and the real-T1ce ceiling is reported alongside so the reader can calibrate it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = _block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2])
        self.bott = _block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2); self.d3 = _block(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2); self.d2 = _block(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2); self.d1 = _block(c[1], c[0])
        self.out = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d = self.d3(torch.cat([self.u3(b), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.out(d)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def dice_score(pred_bin, target_bin, eps=1e-8):
    """Per-slice Dice; returns NaN where the ground truth has no enhancing tumour, so that
    empty slices neither inflate the score (as a perfect empty prediction would) nor are
    silently counted as failures."""
    import numpy as np
    p, t = pred_bin.astype(bool), target_bin.astype(bool)
    if t.sum() == 0:
        return float("nan")
    return 2.0 * (p & t).sum() / (p.sum() + t.sum() + eps)
