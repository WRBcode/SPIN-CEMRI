# -*- coding: utf-8 -*-
"""
Figure: one interactive session on BraTS 2021 test slices.

Per row (one case):  T1w | prompt p | ground truth | TLP | SPIN | correction c | SPIN-IR
TLP and SPIN receive the same coarse prompt; the simulated reader then marks one +1 and one
-1 region from SPIN's error; SPIN-IR receives (p, c). The lesion box marks the ROI-PSNR region;
the number under each synthesis panel is its ROI-PSNR. Cases are chosen as the ones where
the correction helps most, plus one median case.

    python viz_qualitative.py --out runs/_journal/fig_qualitative.pdf
"""
import argparse
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
# Embed TrueType fonts; matplotlib emits Type 3 fonts by default in PDF.
matplotlib.rcParams["pdf.fonttype"] = 42          # TrueType
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"             # sans-serif throughout
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

from model.model_spin import SPIN
from model.FPG import FPG
from data import LoadData
from interaction import simulate_correction
from roi_metrics import roi_mask, roi_psnr

ap = argparse.ArgumentParser()
ap.add_argument("--tlp_ckpt", default="./tlp_official/G_T1T2.pth")
ap.add_argument("--spin_ckpt", default="./runs/spin/G_8.pth")
ap.add_argument("--ir_ckpt", default="./runs/spin_ir_cap8/G_6.pth",
                help="must be the same SPIN-IR checkpoint the tables report")
ap.add_argument("--max_frac", type=float, default=0.08,
                help="mark size cap; must match the one used for the refinement results (eval_journal.py)")
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--start_num", type=int, default=870)
ap.add_argument("--end_num", type=int, default=900)
ap.add_argument("--gpu_id", type=str, default="0")
ap.add_argument("--n_cases", type=int, default=2)
ap.add_argument("--max_cov", type=float, default=0.08,
                help="only show cases whose +1 and -1 marks each cover at most this fraction of the slice, "
                     "i.e. marks a reader would plausibly draw; the coverage statistics are printed for all cases")
ap.add_argument("--out", default="./runs/_journal/fig_qualitative.pdf")
ap.add_argument("--cases", default="", help="comma-separated slice indices: skip the ranking and show these")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dump", default="", help="also save the panels (npz)")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = args.device
if device == "cpu":
    torch.set_num_threads(16)
torch.manual_seed(0)
ds = LoadData(args.dataroot, ["t1", "t2", "t1ce", "seg"], args.data_list,
              start_num=args.start_num, end_num=args.end_num, start_slice=27, end_slice=127)


def load(variant, ckpt, correction=False):
    g = SPIN(variant=variant, probability_of_losing_labels=1.0, device=device, correction=correction)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
    for k in [k for k in list(sd) if k.endswith(("tau_A_raw", "tau_B_raw"))]:
        sd[k.replace("_raw", "")] = torch.nn.functional.softplus(sd.pop(k))
    g.load_state_dict(sd, strict=False)
    return g.to(device).eval()


G_tlp, G_spin, G_ir = load("tlp", args.tlp_ckpt), load("spin", args.spin_ckpt), load("spin", args.ir_ckpt, True)
fpg = FPG(probability_of_losing_labels=0.0, probability_of_dilatation=0.9,
          edema_operation_frequency=2, tumor_operation_frequency=5, device=device)


def u8(x):
    return ((np.squeeze(x) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)


# ---- rank cases by the benefit of the correction round ----
cand = []
with torch.no_grad():
    for i in ([int(v) for v in args.cases.split(",")] if args.cases else range(len(ds))):
        b = ds[i]
        if float(b["seg"][1].sum()) < 200:
            continue
        t1 = b["t1"][0].unsqueeze(0).to(device); t2 = b["t2"][0].unsqueeze(0).to(device)
        gt = b["t1ce"][0].unsqueeze(0).to(device)
        ed = b["seg"][0].unsqueeze(0).to(device); tu = b["seg"][1].unsqueeze(0).to(device)
        random.seed(1000 + i); torch.manual_seed(1000 + i); p = fpg(ed, tu)
        y_s = G_spin(t1, t2, prompt=p)
        c, pos, neg = simulate_correction(y_s, gt, ed, tu, q=0.90, r=3, largest_cc=True, max_frac=args.max_frac)
        cp, cn = float(pos.mean()), float(neg.mean())
        y_i = G_ir(t1, t2, prompt=p, correction=c)
        m = roi_mask(ed[0, 0].cpu().numpy(), tu[0, 0].cpu().numpy(), dilate=5)
        g8 = u8(gt)
        a, s, r = roi_psnr(g8, u8(G_tlp(t1, t2, prompt=p)), m), roi_psnr(g8, u8(y_s), m), roi_psnr(g8, u8(y_i), m)
        if all(np.isfinite(v) for v in (a, s, r)):
            cand.append((r - a, i, a, s, r, cp, cn))
covp, covn = np.array([x[5] for x in cand]), np.array([x[6] for x in cand])
print(f"mark coverage over {len(cand)} candidate slices (fraction of the slice):")
for nm, v in (("+1", covp), ("-1", covn)):
    print(f"  {nm}: mean {100*v.mean():.2f}%  median {100*np.median(v):.2f}%  p90 {100*np.percentile(v,90):.2f}%  "
          f"max {100*v.max():.1f}%  share>10%: {100*(v>0.10).mean():.1f}%")
ok = [x for x in cand if x[5] <= args.max_cov and x[6] <= args.max_cov]
ok.sort(reverse=True)
picks = cand if args.cases else [ok[0], ok[len(ok) // 2]][:args.n_cases]
print(f"{len(ok)}/{len(cand)} cases have both marks <= {args.max_cov:.0%}; showing best and median of those")
print("cases (dROI ir-tlp, idx, tlp, spin, ir, cov+, cov-):",
      [(round(d, 2), i, round(a, 2), round(s, 2), round(r, 2), round(cp, 3), round(cn, 3)) for d, i, a, s, r, cp, cn in picks])

cols = ["T1w", "prompt $p$", "T1ce (GT)", "TLP", "SPIN", "correction $c$", "SPIN-IR"]
fig, axes = plt.subplots(len(picks), len(cols), figsize=(len(cols) * 1.22, len(picks) * 1.5))
axes = np.atleast_2d(axes)
corr_cmap = ListedColormap(["#3b6cff", "#000000", "#ff4040"])       # -1 blue, 0 black, +1 red

dump = {}
with torch.no_grad():
    for row, (d, i, a, s, r, _, _) in enumerate(picks):
        b = ds[i]
        t1 = b["t1"][0].unsqueeze(0).to(device); t2 = b["t2"][0].unsqueeze(0).to(device)
        gt = b["t1ce"][0].unsqueeze(0).to(device)
        ed = b["seg"][0].unsqueeze(0).to(device); tu = b["seg"][1].unsqueeze(0).to(device)
        random.seed(1000 + i); torch.manual_seed(1000 + i); p = fpg(ed, tu)
        y_t = u8(G_tlp(t1, t2, prompt=p)); y_s_t = G_spin(t1, t2, prompt=p); y_s = u8(y_s_t)
        c, _, _ = simulate_correction(y_s_t, gt, ed, tu, q=0.90, r=3, largest_cc=True, max_frac=args.max_frac)
        y_i = u8(G_ir(t1, t2, prompt=p, correction=c))
        m = roi_mask(ed[0, 0].cpu().numpy(), tu[0, 0].cpu().numpy(), dilate=5)
        ys, xs = np.where(m)
        y0, y1 = max(ys.min() - 6, 0), min(ys.max() + 6, m.shape[0]); x0, x1 = max(xs.min() - 6, 0), min(xs.max() + 6, m.shape[1])
        panels = [u8(b["t1"][0]), np.squeeze(p.cpu().numpy()), u8(gt), y_t, y_s, np.squeeze(c.cpu().numpy()), y_i]
        for key, v in zip(("t1", "p", "gt", "tlp", "spin", "c", "ir"), panels):
            dump[f"case{row}_{key}"] = v
        dump[f"case{row}_roi"], dump[f"case{row}_box"] = m, np.array([x0, y0, x1, y1])
        dump[f"case{row}_psnr"], dump[f"case{row}_idx"] = np.array([a, s, r]), np.array(i)
        for k, (ax, im) in enumerate(zip(axes[row], panels)):
            if k == 5:
                ax.imshow(im, cmap=corr_cmap, vmin=-1, vmax=1)
            else:
                ax.imshow(im, cmap="gray", vmin=0, vmax=(1.0 if k == 1 else 255))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if k in (2, 3, 4, 6):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#39d353", lw=0.7))
            if row == 0:
                ax.set_title(cols[k], fontsize=6, pad=2)
        for k, v in ((3, a), (4, s), (6, r)):
            axes[row][k].set_xlabel(f"{v:.1f} dB", fontsize=5.5, labelpad=1)

fig.subplots_adjust(wspace=0.03, hspace=0.08, left=0.005, right=0.995, top=0.92, bottom=0.06)
fig.savefig(args.out, dpi=400, bbox_inches="tight")
fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("wrote", args.out)
if args.dump:
    np.savez_compressed(args.dump, **dump)
    print("wrote", args.dump)
