# -*- coding: utf-8 -*-
"""
Figure: what a wrong mark does.

Row 1 -- a misplaced prompt: the accurate prompt of another slice lands on non-lesional tissue.
         Acquired T1ce, then TLP, SPIN-l1 and SPIN-adv given that prompt; the frozen segmenter's
         enhancing-tumour prediction is outlined in red.
Row 2 -- a wrong -1 on the true enhancing tumour: round 1 and round 2 of the l1 pair and of the
         adversarial pair, segmenter outlines as above.
Cases are chosen by rule, not by eye, from safety_adv.npz (eval_safety.py, one row per test slice):
row 1 is the slice with the median number of new enhancing pixels among the slices on which
SPIN-adv created at least 20; row 2 the slice with the median loss of detected enhancing tumour for
SPIN-IR-adv. The round-1 prompt of row 2 is the coarse prompt of the evaluation stream (replayed).

    python viz_safety.py --gpu_id 0 --out runs/_journal/fig_safety.pdf
"""
import argparse, os, random

import numpy as np
import SimpleITK as sitk
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
import matplotlib.pyplot as plt

from model.model_spin import SPIN
from model.FPG import FPG
from data.data_brats import LoadData, _path, _ids
from seg_unet import UNet

B = "."
ap = argparse.ArgumentParser()
ap.add_argument("--safety_npz", default=f"{B}/runs/_journal/safety_adv.npz")
ap.add_argument("--gpu_id", default="0")
ap.add_argument("--out", default=f"{B}/runs/_journal/fig_safety.pdf")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dump", default="", help="also save the panels (npz)")
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
dev = args.device
if dev == "cpu":
    torch.set_num_threads(16)
root, lst = args.dataroot, "./data_list/Brats2021_selected.txt"
ds = LoadData(root, ["t1", "t2", "t1ce", "seg"], lst, start_num=870, end_num=1000, start_slice=27, end_slice=127)
N = len(ds)
et = []
for sid in _ids(lst, 870, 1000):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(_path(root, "brats2021", sid, "seg")))[:, 34:226, 40:-40]
    et += [seg[z] == 4 for z in range(27, 127)]

# ---- choose the two cases by rule ----
A = np.load(args.safety_npz)
fp = A["spin_sham_fpS"]
ok = np.flatnonzero(np.isfinite(fp) & (fp >= 20))
row1 = int(ok[np.argsort(fp[ok])[len(ok) // 2]])
drop = A["det_y1"] - A["det_wrong"]
ok2 = np.flatnonzero(np.isfinite(drop))
row2 = int(ok2[np.argsort(drop[ok2])[len(ok2) // 2]])
print(f"row 1: slice {row1} (SPIN-adv new ET {fp[row1]:.0f} px, TLP {A['tlp_sham_fpS'][row1]:.0f}); "
      f"{len(ok)} slices with >=20 new px")
print(f"row 2: slice {row2} (SPIN-IR-adv detection {A['det_y1'][row2]:.2f} -> {A['det_wrong'][row2]:.2f})")


def load(variant, ckpt, correction=False):
    g = SPIN(variant=variant, probability_of_losing_labels=1.0, device=dev, correction=correction)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
    g.load_state_dict(sd, strict=False)
    return g.to(dev).eval()


G = {"TLP": load("tlp", f"{B}/tlp_official/G_T1T2.pth"),
     "SPIN-l1": load("spin", f"{B}/runs/spin/G_8.pth"),
     "SPIN-adv": load("spin", f"{B}/runs/spin_adv/G_8.pth")}
IR = {"l1": load("spin", f"{B}/runs/spin_ir_cap8/G_6.pth", True),
      "adv": load("spin", f"{B}/runs/spin_ir_adv/G_6.pth", True)}
S = UNet().to(dev); S.load_state_dict(torch.load(f"{B}/runs/_seg/best.pth", map_location=dev)); S.eval()
fpg = FPG(probability_of_losing_labels=0.0, probability_of_dilatation=0.9,
          edema_operation_frequency=2, tumor_operation_frequency=5, device=dev)


def t4(x):
    return x.unsqueeze(0).to(dev)


def acc(ed, tu):
    p = ed * 0.5
    p[tu == 1] = 1.0
    return p


def u8(y):
    return ((y[0, 0].float() + 1) / 2 * 255).clamp(0, 255).cpu().numpy()


def seg_et(y):
    return (torch.sigmoid(S(y)) > 0.5)[0, 0].cpu().numpy()


def largest_cc(m):                      # identical to eval_safety.py
    from scipy import ndimage
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n <= 1:
        return m
    return lab == (1 + int(np.argmax(ndimage.sum(m, lab, index=np.arange(1, n + 1)))))


def capped_cc(m, frac=0.08):            # identical to eval_safety.py
    from scipy import ndimage
    m = largest_cc(m)
    for _ in range(30):
        if m.sum() <= frac * m.size:
            break
        m = largest_cc(ndimage.binary_erosion(m))
    return m


with torch.no_grad():
    # replay the coarse-prompt stream of the evaluation up to row 2
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    p_row2 = None
    for j in range(row2 + 1):
        s_ = ds[j]["seg"]
        pj = fpg(t4(s_[0]), t4(s_[1]))
        if j == row2:
            p_row2 = pj

    # ---- row 1: misplaced prompt ----
    b = ds[row1]
    t1, t2, gt = t4(b["t1"][0]), t4(b["t2"][0]), t4(b["t1ce"][0])
    o = ds[(row1 + N // 2) % N]["seg"]
    pw = acc(t4(o[0]), t4(o[1]))
    r1 = [("T1w + misplaced prompt", u8(t1), None, pw[0, 0].cpu().numpy() > 0),
          ("Acquired T1ce", u8(gt), et[row1], None)]
    for name in ("TLP", "SPIN-l1", "SPIN-adv"):
        y = G[name](t1, t2, prompt=pw)
        r1.append((name, u8(y), seg_et(y), None))

    # ---- row 2: wrong -1 on the true enhancing tumour ----
    from scipy import ndimage
    b = ds[row2]
    t1, t2, gt = t4(b["t1"][0]), t4(b["t2"][0]), t4(b["t1ce"][0])
    mark = capped_cc(ndimage.binary_dilation(et[row2], iterations=3))
    c = -torch.from_numpy(mark.astype(np.float32))[None, None].to(dev)
    r2 = [("Acquired T1ce + wrong $-1$", u8(gt), et[row2], mark)]
    for key, name in (("l1", "SPIN-l1"), ("adv", "SPIN-adv")):
        y1 = G[name](t1, t2, prompt=p_row2)
        y2 = IR[key](t1, t2, prompt=p_row2, correction=c)
        r2.append((f"{name}, round 1", u8(y1), seg_et(y1), None))
        r2.append((f"SPIN-IR-{key}, after $-1$", u8(y2), seg_et(y2), None))

fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.3))
for row, panels in enumerate((r1, r2)):
    for k, (title, img, segm, extra) in enumerate(panels):
        ax = axes[row][k]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        if segm is not None and segm.any():
            ax.contour(segm.astype(float), levels=[0.5], colors="#ff3030", linewidths=0.6)
        if extra is not None and extra.any():
            ax.contour(extra.astype(float), levels=[0.5], colors="#ffd000" if row == 0 else "#3b8cff",
                       linewidths=0.7, linestyles="--")
        # paper naming: the l1-trained pair is SPIN / SPIN-IR, the adversarial pair SPIN-adv / SPIN-IR-adv
        ax.set_title(title.replace("SPIN-IR-l1", "SPIN-IR").replace("SPIN-l1", "SPIN"), fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
fig.subplots_adjust(wspace=0.03, hspace=0.16, left=0.005, right=0.995, top=0.93, bottom=0.01)
fig.savefig(args.out, dpi=400, bbox_inches="tight")
fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("wrote", args.out)
if args.dump:
    dump = {"rows": np.array([row1, row2])}
    for row, panels in enumerate((r1, r2)):
        for k, (title, img, segm, extra) in enumerate(panels):
            dump[f"r{row}_{k}_title"] = np.array(title)
            dump[f"r{row}_{k}_img"] = img
            dump[f"r{row}_{k}_et"] = segm if segm is not None else np.zeros_like(img, bool)
            dump[f"r{row}_{k}_mark"] = extra if extra is not None else np.zeros_like(img, bool)
    np.savez_compressed(args.dump, **dump)
    print("wrote", args.dump)
