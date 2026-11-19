# -*- coding: utf-8 -*-
"""
Safety analysis: can a *wrong* interaction create or erase a lesion?

A reader will sometimes mark the wrong place. Three misdirected interactions are simulated on
the BraTS 2021 test set and measured locally, where they act:
  sham prompt   the accurate prompt of another slice (the 'incorrect' regime), evaluated on the
                part S of it that lies on non-lesional brain tissue of this slice (|S| >= 100 px);
  sham +1       a +1 correction on that non-lesional region (largest component, <= 8% of slice);
  wrong -1      a -1 correction on the TRUE enhancing tumour (label 4, dilated 3 px, <= 8%).
For each: the mean intensity change inside the marked region (8-bit grey levels) against the same
model without the mark, and what the frozen enhancing-tumour segmenter (trained on real T1ce)
finds there -- new false-positive enhancing pixels for the sham marks, and the fraction of the
true enhancing tumour it still detects for the wrong -1. The effect of a *correct* mark of the
same kind is measured alongside, as the scale against which to read these numbers.

    python eval_safety.py --gpu_id 0 --out runs/_journal/safety
"""
import argparse, json, os, random

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from model.model_spin import SPIN
from model.FPG import FPG
from data.data_brats import LoadData, _path, _ids
from interaction import simulate_correction
from seg_unet import UNet

B = "."
ap = argparse.ArgumentParser()
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--start_num", type=int, default=870)
ap.add_argument("--end_num", type=int, default=1000)
ap.add_argument("--tlp_ckpt", default=f"{B}/tlp_official/G_T1T2.pth")
ap.add_argument("--spin_ckpt", default=f"{B}/runs/spin/G_8.pth")
ap.add_argument("--ir_ckpt", default=f"{B}/runs/spin_ir_cap8/G_6.pth")
ap.add_argument("--seg_ckpt", default=f"{B}/runs/_seg/best.pth")
ap.add_argument("--max_frac", type=float, default=0.08)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu_id", default="0")
ap.add_argument("--out", required=True, help="output prefix; writes <out>.npz and <out>.json")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
dev = "cuda"
ds = LoadData(args.dataroot, ["t1", "t2", "t1ce", "seg"], args.data_list, start_num=args.start_num,
              end_num=args.end_num, start_slice=27, end_slice=127)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
N = len(ds)
et = []
for sid in _ids(args.data_list, args.start_num, args.end_num):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(_path(args.dataroot, "brats2021", sid, "seg")))[:, 34:226, 40:-40]
    et += [seg[z] == 4 for z in range(27, 127)]
assert len(et) == N


def load(variant, ckpt, correction=False):
    g = SPIN(variant=variant, probability_of_losing_labels=1.0, device=dev, correction=correction)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
    g.load_state_dict(sd, strict=False)
    return g.to(dev).eval()


G_tlp, G_spin, G_ir = load("tlp", args.tlp_ckpt), load("spin", args.spin_ckpt), load("spin", args.ir_ckpt, True)
S = UNet().to(dev); S.load_state_dict(torch.load(args.seg_ckpt, map_location=dev)); S.eval()
fpg = FPG(probability_of_losing_labels=0.0, probability_of_dilatation=0.9,
          edema_operation_frequency=2, tumor_operation_frequency=5, device=dev)


def acc(ed, tu):
    p = ed * 0.5
    p[tu == 1] = 1.0
    return p


def gl(y):                       # grey levels, same mapping as the 8-bit metrics but without rounding
    return ((y[0, 0].float() + 1) / 2 * 255).cpu().numpy()


def seg_et(y):
    return (torch.sigmoid(S(y)) > 0.5)[0, 0].cpu().numpy()


def largest_cc(m):
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n <= 1:
        return m
    return lab == (1 + int(np.argmax(ndimage.sum(m, lab, index=np.arange(1, n + 1)))))


def capped_cc(m, frac):
    """largest component, eroded until it covers at most `frac` of the slice (None if it vanishes)."""
    m = largest_cc(m)
    for _ in range(30):
        if m.sum() <= frac * m.size:
            break
        m = largest_cc(ndimage.binary_erosion(m))
    return m if 0 < m.sum() <= frac * m.size else None


def t(m, sign=1.0):
    return sign * torch.from_numpy(m.astype(np.float32))[None, None].to(dev)


rows = []
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)   # coarse stream of eval_spin.py
with torch.no_grad():
    for i, b in enumerate(loader):
        t1, t2, gt = b["t1"][0].to(dev), b["t2"][0].to(dev), b["t1ce"][0].to(dev)
        ed, tu = b["seg"][0].to(dev), b["seg"][1].to(dev)
        p = fpg(ed, tu)
        zero = torch.zeros_like(p)
        brain = ((t1 > -0.999) | (t2 > -0.999))[0, 0].cpu().numpy()
        les = ((ed + tu) > 0)[0, 0].cpu().numpy()
        les5 = ndimage.binary_dilation(les, iterations=5) if les.any() else les
        o = ds[(i + N // 2) % N]["seg"]
        pw = acc(o[0].unsqueeze(0).to(dev), o[1].unsqueeze(0).to(dev))
        Sm = (pw[0, 0].cpu().numpy() > 0) & ~les5 & brain
        core = tu[0, 0].cpu().numpy() > 0
        r = {"i": i, "n_core": int(core.sum()), "n_et": int(et[i].sum()), "n_S": int(Sm.sum())}
        need_S, need_L = Sm.sum() >= 100, core.sum() >= 50
        if not (need_S or need_L):
            rows.append(r); continue
        y0 = G_tlp(t1, t2, prompt=zero); g0 = gl(y0); e0 = seg_et(y0); ggt = gl(gt)
        if need_L:                                   # a correct prompt on the true core, for scale
            pa = acc(ed, tu)
            r["gt_minus_y0_core"] = float((ggt - g0)[core].mean())
            r["y0_mae_core"] = float(np.abs(g0 - ggt)[core].mean())
            for nm, G in (("tlp", G_tlp), ("spin", G_spin)):
                g = gl(G(t1, t2, prompt=pa))
                r[f"{nm}_acc_dcore"] = float((g - g0)[core].mean())
                r[f"{nm}_acc_absd_core"] = float(np.abs(g - g0)[core].mean())
                r[f"{nm}_acc_mae_core"] = float(np.abs(g - ggt)[core].mean())
        if need_S:                                   # sham prompt on non-lesional tissue
            r["y0_fpS"] = int((e0 & Sm).sum()); r["y0_maeS"] = float(np.abs(g0 - ggt)[Sm].mean())
            r["gt_minus_y0_S"] = float((ggt - g0)[Sm].mean())
            for nm, G in (("tlp", G_tlp), ("spin", G_spin)):
                y = G(t1, t2, prompt=pw); g = gl(y); e = seg_et(y)
                r[f"{nm}_sham_dS"] = float((g - g0)[Sm].mean())
                r[f"{nm}_sham_absdS"] = float(np.abs(g - g0)[Sm].mean())
                r[f"{nm}_sham_fpS"] = int((e & Sm).sum()) - r["y0_fpS"]
                r[f"{nm}_sham_maeS"] = float(np.abs(g - ggt)[Sm].mean())
        if need_L:
            y1 = G_spin(t1, t2, prompt=p); g1 = gl(y1); e1 = seg_et(y1)
            c, pos, neg = simulate_correction(y1, gt, ed, tu, q=0.90, r=3, largest_cc=True, max_frac=args.max_frac)
            negm = neg[0, 0].cpu().numpy() > 0
            if negm.any():                           # a correct -1 (the oracle's), for scale
                g2 = gl(G_ir(t1, t2, prompt=p, correction=c))
                r["ok_neg_d"] = float((g2 - g1)[negm].mean()); r["ok_neg_absd"] = float(np.abs(g2 - g1)[negm].mean())
                r["ok_neg_gt_minus_y1"] = float((ggt - g1)[negm].mean())
            if need_S:                               # sham +1 on non-lesional tissue
                m = capped_cc(Sm, args.max_frac)
                if m is not None:
                    ys = G_ir(t1, t2, prompt=p, correction=t(m)); es = seg_et(ys)
                    r["sham_pos_n"] = int(m.sum())
                    r["sham_pos_d"] = float((gl(ys) - g1)[m].mean()); r["sham_pos_absd"] = float(np.abs(gl(ys) - g1)[m].mean())
                    r["sham_pos_fp"] = int((es & m).sum()) - int((e1 & m).sum())
            if et[i].sum() >= 30:                    # wrong -1 on the true enhancing tumour
                m = capped_cc(ndimage.binary_dilation(et[i], iterations=3), args.max_frac)
                if m is not None and (et[i] & m).sum() >= 10:
                    yw = G_ir(t1, t2, prompt=p, correction=t(m, -1.0)); gw = gl(yw); ew = seg_et(yw)
                    etm = et[i] & m
                    r["wrong_neg_n"] = int(m.sum())
                    r["wrong_neg_dET"] = float((gw - g1)[etm].mean()); r["wrong_neg_absdET"] = float(np.abs(gw - g1)[etm].mean())
                    r["et_minus_bg_y1"] = float(g1[etm].mean() - g1[brain & ~les5].mean())
                    r["et_minus_bg_gt"] = float(ggt[etm].mean() - ggt[brain & ~les5].mean())
                    r["det_y0"] = float((e0 & etm).sum() / etm.sum())
                    r["det_y1"] = float((e1 & etm).sum() / etm.sum())
                    r["det_wrong"] = float((ew & etm).sum() / etm.sum())
        rows.append(r)
        if i % 2000 == 0:
            print(f"  {i}/{N}", flush=True)

keys = sorted({k for r in rows for k in r if k != "i"})
A = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out + ".npz", **A)


def stat(v):
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()), "n": int(v.size)} if v.size else None


summ = {k: stat(A[k]) for k in keys if k not in ("n_core", "n_et", "n_S")}
for nm in ("tlp", "spin"):                        # share of slices where a sham prompt creates >= 20 px of new ET
    v = A[f"{nm}_sham_fpS"]; v = v[np.isfinite(v)]
    summ[f"{nm}_sham_newET20_frac"] = float((v >= 20).mean())
v = A["sham_pos_fp"]; v = v[np.isfinite(v)]; summ["sham_pos_newET20_frac"] = float((v >= 20).mean())
d1, dw = A["det_y1"], A["det_wrong"]; ok = np.isfinite(d1) & np.isfinite(dw) & (d1 > 0.2)
summ["wrong_neg_erased_frac"] = float((dw[ok] < 0.5 * d1[ok]).mean())    # lost >= half of its detected ET
summ["wrong_neg_n_slices_detected"] = int(ok.sum())
json.dump(summ, open(args.out + ".json", "w"), indent=1)
for k, v in summ.items():
    print(f"{k:28s} {v}")
print("saved", args.out + ".{npz,json}")
