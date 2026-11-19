# -*- coding: utf-8 -*-
"""
One evaluation pass: every method on the same slices with the same
prompts, with per-slice metrics saved so the paired statistics can be computed per patient.

Methods (identical inputs on every slice):
  tlp_none       official TLP weights, no prompt (bit-identical to SPIN with no prompt)
  tlp_coarse     TLP, coarse FPG prompt
  spin_coarse    SPIN, the same coarse prompt
  tlp_acc        TLP, accurate prompt (0.5 edema, 1.0 core)
  spin_acc       SPIN, accurate prompt
  ir_naive       one correction round without the correction pathway: the +1 mark folded into
                 the prompt, the -1 mark lost (only on slices with >= 50 core pixels)
  ir_r2          SPIN-IR after one simulated correction of spin_coarse (marks capped at 8%)
  real           the acquired T1ce (Dice ceiling only)

The coarse prompt stream is the one eval_spin.py uses (Python `random` seeded once, one FPG call
per slice in loader order; the networks never touch `random` when given a prompt), so the
coarse-regime means reproduce eval_spin.py exactly -- checked against --expect below.

Metrics per slice: PSNR, SSIM, NMSE (TLP's evaluation.py conventions on 8-bit images),
ROI-PSNR inside the lesion mask dilated by 5 px, and enhancing-tumour Dice from a frozen U-Net
trained on real T1ce (NaN where the slice has no enhancing tumour).

    python eval_journal.py --gpu_id 0 --out runs/_journal/internal
    python eval_journal.py --cohort brats2020 --dataroot dataset/MICCAI_BraTS2020_TrainingData \
        --data_list ./data_list/Brats2020_heldout.txt --start_num 0 --end_num 50 \
        --out runs/_journal/heldout2020
"""
import argparse
import json
import os
import random
import time

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from model.model_spin import SPIN
from model.FPG import FPG
from data.data_brats import LoadData, _path, _ids
from roi_metrics import roi_mask, roi_psnr
from interaction import simulate_correction
from seg_unet import UNet, dice_score

B = "."
ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="brats2021")
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--start_num", type=int, default=870)
ap.add_argument("--end_num", type=int, default=1000)
ap.add_argument("--start_slice", type=int, default=27)
ap.add_argument("--end_slice", type=int, default=127)
ap.add_argument("--tlp_ckpt", default=f"{B}/tlp_official/G_T1T2.pth")
ap.add_argument("--spin_ckpt", default=f"{B}/runs/spin/G_8.pth")
ap.add_argument("--ir_ckpt", default=f"{B}/runs/spin_ir_cap8/G_6.pth")
ap.add_argument("--seg_ckpt", default=f"{B}/runs/_seg/best.pth")
ap.add_argument("--max_frac", type=float, default=0.08)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu_id", default="0")
ap.add_argument("--expect", default=f"{B}/runs/spin/eval_G8.json",
                help="eval_spin.py output whose coarse-regime SPIN means this run must reproduce "
                     "(only checked on the default BraTS 2021 test split)")
ap.add_argument("--no_ir", action="store_true",
                help="skip the correction round (ir_naive / ir_r2), e.g. before a matching SPIN-IR exists")
ap.add_argument("--out", required=True, help="output prefix; writes <out>.npz and <out>.json")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
dev = "cuda"
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

ds = LoadData(args.dataroot, ["t1", "t2", "t1ce", "seg"], args.data_list, start_num=args.start_num,
              end_num=args.end_num, start_slice=args.start_slice, end_slice=args.end_slice, cohort=args.cohort)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
N = len(ds)

# enhancing tumour (label 4) per slice, rebuilt from the raw seg: the loader merges labels 1 and 4
et = []
for sid in _ids(args.data_list, args.start_num, args.end_num):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(_path(args.dataroot, args.cohort, sid, "seg")))[:, 34:226, 40:-40]
    et += [seg[z] == 4 for z in range(args.start_slice, args.end_slice)]
assert len(et) == N, (len(et), N)


def load(variant, ckpt, correction=False, strict=False):
    g = SPIN(variant=variant, probability_of_losing_labels=1.0, device=dev, correction=correction)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
    for k in [k for k in list(sd) if k.endswith(("tau_A_raw", "tau_B_raw"))]:
        sd[k.replace("_raw", "")] = torch.nn.functional.softplus(sd.pop(k))
    missing, unexpected = g.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.endswith(("tau_A", "tau_B"))]
    if unexpected or (strict and bad):
        raise RuntimeError(f"{ckpt}: missing={bad} unexpected={unexpected}")
    return g.to(dev).eval()


G_tlp = load("tlp", args.tlp_ckpt)
G_spin = load("spin", args.spin_ckpt, strict=True)
G_ir = None if args.no_ir else load("spin", args.ir_ckpt, correction=True, strict=True)
S = UNet().to(dev); S.load_state_dict(torch.load(args.seg_ckpt, map_location=dev)); S.eval()
fpg = FPG(probability_of_losing_labels=0.0, probability_of_dilatation=0.9,
          edema_operation_frequency=2, tumor_operation_frequency=5, device=dev)


def accurate_prompt(ed, tu):
    p = ed * 0.5
    p[tu == 1] = 1.0
    return p


def u8(x):
    return ((np.squeeze(x) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)


def region_psnr(g, f, m):
    if m.sum() == 0:
        return np.nan
    d = np.mean((g.astype(np.float64)[m] - f.astype(np.float64)[m]) ** 2)
    return np.nan if d == 0 else 10 * np.log10(255 ** 2 / d)


METHODS = ["tlp_none", "tlp_coarse", "spin_coarse", "tlp_acc", "spin_acc", "ir_naive", "ir_r2"]
METRICS = ["psnr", "ssim", "nmse", "roi", "dice", "pos", "neg"]
R = {f"{m}.{k}": np.full(N, np.nan) for m in METHODS for k in METRICS}
R["real.dice"] = np.full(N, np.nan)
sid_z = []
n_core = np.zeros(N, int); n_et = np.zeros(N, int); n_roi = np.zeros(N, int)
cov_pos = np.full(N, np.nan); cov_neg = np.full(N, np.nan)
checked = False
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)   # as eval_spin.py, per regime
t0 = time.time()
with torch.no_grad():
    for i, b in enumerate(loader):
        sid_z.append(b["t1"][1][0])
        t1, t2 = b["t1"][0].to(dev), b["t2"][0].to(dev)
        gt = b["t1ce"][0].to(dev)
        ed, tu = b["seg"][0].to(dev), b["seg"][1].to(dev)
        p = fpg(ed, tu)                                   # same stream as eval_spin.py's coarse regime
        zero = torch.zeros_like(p)
        acc = accurate_prompt(ed, tu)
        outs = {"tlp_none": G_tlp(t1, t2, prompt=zero), "tlp_coarse": G_tlp(t1, t2, prompt=p),
                "spin_coarse": G_spin(t1, t2, prompt=p), "tlp_acc": G_tlp(t1, t2, prompt=acc),
                "spin_acc": G_spin(t1, t2, prompt=acc)}
        n_core[i] = int(tu.sum()); n_et[i] = int(et[i].sum())
        pos_np = neg_np = None
        if n_core[i] >= 50 and not args.no_ir:
            c, pos, neg = simulate_correction(outs["spin_coarse"], gt, ed, tu, q=0.90, r=3,
                                              largest_cc=True, max_frac=args.max_frac)
            outs["ir_naive"] = G_spin(t1, t2, prompt=torch.maximum(p, pos))
            outs["ir_r2"] = G_ir(t1, t2, prompt=p, correction=c)
            pos_np, neg_np = pos[0, 0].cpu().numpy() > 0, neg[0, 0].cpu().numpy() > 0
            cov_pos[i], cov_neg[i] = pos_np.mean(), neg_np.mean()
            if not checked:
                # the protocol's two invariants, on real data: no prompt => SPIN == TLP;
                # no correction => SPIN-IR == SPIN
                d0 = (G_spin(t1, t2, prompt=zero) - outs["tlp_none"]).abs().max().item()
                d1 = (G_ir(t1, t2, prompt=p) - outs["spin_coarse"]).abs().max().item()
                print(f"invariants: |SPIN(0)-TLP(0)|max={d0:.2e}  |IR(p,none)-SPIN(p)|max={d1:.2e}", flush=True)
                assert d0 < 1e-4 and d1 < 1e-4, "identity-at-zero invariant broken"
                checked = True
        g8 = u8(gt)
        rm = roi_mask(ed[0, 0].cpu().numpy(), tu[0, 0].cpu().numpy(), dilate=5)
        n_roi[i] = int(rm.sum())
        has_et = n_et[i] > 0
        if has_et:
            R["real.dice"][i] = dice_score((torch.sigmoid(S(gt)) > 0.5).cpu().numpy()[0, 0], et[i])
        for m, y in outs.items():
            f8 = u8(y)
            if not (f8 == g8).all():
                R[f"{m}.psnr"][i] = psnr(g8, f8)
            R[f"{m}.ssim"][i] = ssim(g8, f8)
            v = np.var(g8); R[f"{m}.nmse"][i] = mse(g8, f8) / v if v != 0 else 0.0
            r = roi_psnr(g8, f8, rm); R[f"{m}.roi"][i] = r if np.isfinite(r) else np.nan
            if has_et:
                R[f"{m}.dice"][i] = dice_score((torch.sigmoid(S(y)) > 0.5).cpu().numpy()[0, 0], et[i])
            if pos_np is not None:
                R[f"{m}.pos"][i] = region_psnr(g8, f8, pos_np); R[f"{m}.neg"][i] = region_psnr(g8, f8, neg_np)
        if i % 1000 == 0:
            print(f"  {i}/{N}  [{(time.time() - t0) / 60:.1f} min]", flush=True)

sid = np.array([s.rsplit("_", 1)[0] for s in sid_z]); zz = np.array([int(s.rsplit("_", 1)[1]) for s in sid_z])
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out + ".npz", sid=sid, z=zz, n_core=n_core, n_et=n_et, n_roi=n_roi,
                    cov_pos=cov_pos, cov_neg=cov_neg, **{k.replace(".", "__"): v for k, v in R.items()})

# slice-level means with eval_spin.py's conventions (NaN = excluded), and per-subject means
summ = {"n_slices": N, "n_subjects": int(len(set(sid))), "n_ir_slices": int((n_core >= 50).sum()),
        "n_et_slices": int((n_et > 0).sum()), "minutes": (time.time() - t0) / 60, "slice": {}, "subject": {}}
subs = sorted(set(sid))
for k, v in R.items():
    if np.isnan(v).all():
        continue
    summ["slice"][k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "n": int(np.isfinite(v).sum())}
    per = np.array([np.nanmean(v[sid == s]) if np.isfinite(v[sid == s]).any() else np.nan for s in subs])
    summ["subject"][k] = {"mean": float(np.nanmean(per)), "std": float(np.nanstd(per)), "n": int(np.isfinite(per).sum())}
if np.isfinite(cov_pos).any():
    summ["coverage"] = {"pos_mean": float(np.nanmean(cov_pos)), "neg_mean": float(np.nanmean(cov_neg)),
                        "pos_p90": float(np.nanpercentile(cov_pos, 90)), "neg_p90": float(np.nanpercentile(cov_neg, 90))}

if (args.cohort, args.start_num, args.end_num) == ("brats2021", 870, 1000) and os.path.exists(args.expect):
    ref = json.load(open(args.expect))["results"]["coarse"]
    got = {"psnr": summ["slice"]["spin_coarse.psnr"]["mean"], "ssim": summ["slice"]["spin_coarse.ssim"]["mean"],
           "roi_psnr": summ["slice"]["spin_coarse.roi"]["mean"]}
    summ["reproduces_eval_spin"] = {k: [got[k], ref[k]["mean"]] for k in got}
    print("reproduction check vs", args.expect, summ["reproduces_eval_spin"], flush=True)

json.dump(summ, open(args.out + ".json", "w"), indent=1)
print(f"\n{'method':12s} {'PSNR':>7s} {'SSIM':>7s} {'NMSE':>7s} {'ROI':>7s} {'Dice':>7s}   (slice means)")
for m in METHODS + ["real"]:
    row = [summ["slice"].get(f"{m}.{k}", {}).get("mean", np.nan) for k in ("psnr", "ssim", "nmse", "roi", "dice")]
    print(f"{m:12s} " + " ".join(f"{x:7.3f}" for x in row))
print("saved", args.out + ".{npz,json}")
