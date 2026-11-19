# -*- coding: utf-8 -*-
"""
Test-set evaluation for SPIN / TLP checkpoints under the four prompt regimes of the paper
(none, coarse, accurate, incorrect). Metrics follow TLP/evaluation.py exactly (uint8 images,
skimage PSNR/SSIM, NMSE = MSE / var(GT), PSNR skipped for identical images) plus
ROI-PSNR inside the dilated tumor+edema region (roi_metrics.py).

Run from the TLP repo root:
    python eval_spin.py --variant spin --ckpt runs/spin/G_8.pth --gpu_id 0 \
        --out runs/spin/eval_G8.json
For the official TLP weights use --variant tlp (tau_* keys are simply absent).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from model.model_spin import SPIN, VARIANTS
from model.FPG import FPG
from data import LoadData
from roi_metrics import roi_mask, roi_psnr

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="spin", choices=list(VARIANTS.keys()))
parser.add_argument("--single_input", action="store_true")
parser.add_argument("--modality", type=str, default="t1")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--dataroot", type=str, default="./dataset/BraTS2021")
parser.add_argument("--data_list", type=str, default="./data_list/Brats2021_selected.txt")
parser.add_argument("--start_num", type=int, default=870)
parser.add_argument("--end_num", type=int, default=1000)
parser.add_argument("--start_slice", type=int, default=27)
parser.add_argument("--end_slice", type=int, default=127)
parser.add_argument("--regimes", type=str, default="none,coarse,accurate,incorrect")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--roi_dilate", type=int, default=5)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--dump", action="store_true",
                    help="also save per-slice metrics to <out>.npz (NaN where a slice is excluded from a mean), "
                         "for paired per-patient statistics across runs; prompts are identical across runs")
parser.add_argument("--time_only", action="store_true", help="only measure inference time per slice")
parser.add_argument("--disable_pca", action="store_true",
                    help="zero the PCA gates at load time (control: shows how much the attention bias contributes)")
parser.add_argument("--tta", type=str, default="",
                    help="test-time augmentation: comma-separated subset of {flip_lr,flip_ud}; predictions "
                         "are averaged over the identity and each listed transform (brain slices are "
                         "approximately left-right symmetric, so flip_lr is the cheap one)")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = "cuda"

modalities = ["t1", "t2", "t1ce", "seg"] if not opt.single_input else [opt.modality, "t1ce", "seg"]
test_set = LoadData(opt.dataroot, modalities, opt.data_list, start_num=opt.start_num, end_num=opt.end_num,
                    start_slice=opt.start_slice, end_slice=opt.end_slice)
loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
N = len(test_set)
print(f"test slices: {N}")

G = SPIN(variant=opt.variant, single_input=opt.single_input, probability_of_losing_labels=1.0, device=device)
sd = torch.load(opt.ckpt, map_location="cpu")
if isinstance(sd, dict) and "G" in sd:      # last.pt
    sd = sd["G"]
# Legacy checkpoints (an earlier version) stored the PCA gate as softplus(tau_*_raw); convert.
legacy = [k for k in list(sd) if k.endswith("tau_A_raw") or k.endswith("tau_B_raw")]
for k in legacy:
    sd[k.replace("_raw", "")] = torch.nn.functional.softplus(sd.pop(k))
if legacy:
    print(f"converted legacy PCA gates {legacy} -> "
          f"{[float(sd[k.replace('_raw','')]) for k in legacy]}")
missing, unexpected = G.load_state_dict(sd, strict=False)
allowed_missing = {k for k in missing if k.endswith("tau_A") or k.endswith("tau_B")}
if unexpected or (set(missing) - allowed_missing):
    raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
if opt.disable_pca:
    for m in G.modules():
        if hasattr(m, "tau_A"):
            torch.nn.init.zeros_(m.tau_A); torch.nn.init.zeros_(m.tau_B)
    print("PCA gates zeroed (--disable_pca)")
gates = {k: float(v) for k, v in G.state_dict().items() if k.endswith(("tau_A", "tau_B"))}
print(f"loaded {opt.ckpt} (missing allowed: {sorted(allowed_missing)}; PCA gates {gates})")
G = G.to(device).eval()

fpg_coarse = FPG(probability_of_losing_labels=0.0, probability_of_dilatation=0.9,
                 edema_operation_frequency=2, tumor_operation_frequency=5, device=device)


def accurate_prompt(ed, tu):
    p = ed * 0.5
    p[tu == 1] = 1.0
    return p


def to_u8(x):
    return ((np.squeeze(x) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)


TTA_DIMS = {"flip_lr": [-1], "flip_ud": [-2]}
TTA = [TTA_DIMS[t] for t in opt.tta.split(",") if t.strip()] if opt.tta else []


def predict(*imgs, prompt):
    """Mean prediction over the identity and each requested flip (each flipped back)."""
    out = G(*imgs, prompt=prompt)
    for dims in TTA:
        f = [torch.flip(i, dims) for i in imgs]
        out = out + torch.flip(G(*f, prompt=torch.flip(prompt, dims)), dims)
    return out / (1 + len(TTA))


# fixed "wrong slice" mapping for the incorrect regime: slice i gets the GT prompt of slice (i + N//2) % N
shift = N // 2
results = {}
dump = {}
for regime in opt.regimes.split(","):
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    m = {"psnr": [], "ssim": [], "mse": [], "nmse": [], "mae": [], "roi_psnr": []}
    per = {k: np.full(N, np.nan) for k in ("psnr", "ssim", "nmse", "roi_psnr")}; sids = []
    t0 = time.time(); n_fwd = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            t1ce = batch["t1ce"][0].to(device)
            ed = batch["seg"][0].to(device); tu = batch["seg"][1].to(device)
            if regime == "none":
                p = torch.zeros_like(ed)
            elif regime == "coarse":
                p = fpg_coarse(ed, tu)
            elif regime == "accurate":
                p = accurate_prompt(ed, tu)
            elif regime == "incorrect":
                other = test_set[(i + shift) % N]["seg"]
                p = accurate_prompt(other[0].unsqueeze(0).to(device), other[1].unsqueeze(0).to(device))
            else:
                raise ValueError(regime)
            if opt.single_input:
                out = predict(batch[opt.modality][0].to(device), prompt=p)
            else:
                out = predict(batch["t1"][0].to(device), batch["t2"][0].to(device), prompt=p)
            torch.cuda.synchronize(); n_fwd += 1
            if opt.time_only:
                if n_fwd >= 1000:
                    break
                continue
            f8, r8 = to_u8(out), to_u8(t1ce)
            sids.append(batch["t1ce"][1][0])
            if not (f8 == r8).all():
                m["psnr"].append(psnr(r8, f8)); per["psnr"][i] = m["psnr"][-1]
            m["ssim"].append(ssim(r8, f8)); per["ssim"][i] = m["ssim"][-1]
            e = mse(r8, f8); m["mse"].append(e)
            var = np.var(r8); m["nmse"].append(e / var if var != 0 else 0.0); per["nmse"][i] = m["nmse"][-1]
            m["mae"].append(mae(r8, f8))
            rm = roi_mask(ed[0, 0].cpu().numpy(), tu[0, 0].cpu().numpy(), dilate=opt.roi_dilate)
            r = roi_psnr(r8, f8, rm)
            if np.isfinite(r):
                m["roi_psnr"].append(r); per["roi_psnr"][i] = r
            if i % 2000 == 0:
                print(f"  [{regime}] {i}/{N}", flush=True)
    dt = time.time() - t0
    if opt.time_only:
        results[regime] = {"ms_per_slice": 1000 * dt / n_fwd}
        print(f"{regime}: {1000*dt/n_fwd:.2f} ms/slice"); continue
    results[regime] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in m.items()}
    dump.update({f"{regime}__{k}": v for k, v in per.items()}); dump["sid"] = np.array([x.rsplit("_", 1)[0] for x in sids])
    r = results[regime]
    print(f"{regime:9s} PSNR {r['psnr']['mean']:.2f}±{r['psnr']['std']:.2f}  SSIM {r['ssim']['mean']:.3f}  "
          f"NMSE {r['nmse']['mean']:.3f}  MAE {r['mae']['mean']:.2f}  ROI-PSNR {r['roi_psnr']['mean']:.2f} (n={r['roi_psnr']['n']})  [{dt/60:.1f} min]",
          flush=True)

os.makedirs(os.path.dirname(os.path.abspath(opt.out)), exist_ok=True)
with open(opt.out, "w") as f:
    json.dump({"variant": opt.variant, "ckpt": opt.ckpt, "single_input": opt.single_input,
               "modality": opt.modality, "n_test_slices": N, "pca_gates": gates,
               "disable_pca": opt.disable_pca, "tta": opt.tta, "results": results}, f, indent=1)
if opt.dump:
    np.savez_compressed(os.path.splitext(opt.out)[0] + ".npz", **dump)
print("saved", opt.out)
