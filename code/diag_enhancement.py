# -*- coding: utf-8 -*-
"""
Why does SPIN (l1-trained prompt pathway) gain ROI-PSNR over TLP but lose enhancing-tumour Dice?
Measures, on test slices with enhancing tumour, how bright and how sharp each synthesis renders the
tissue classes, and what the frozen segmenter predicts:
  mean grey level in enhancing tumour (label 4), necrosis (label 1), edema (label 2), normal brain;
  ET contrast = ET - normal brain; mean gradient magnitude inside the lesion (sharpness);
  segmenter: predicted ET area / true ET area, precision, recall, Dice.
Same coarse-prompt stream as eval_spin.py / eval_journal.py.

    python diag_enhancement.py --gpu_id 0 --out runs/_journal/diag_enhancement.json
"""
import argparse, json, os, random
import numpy as np, SimpleITK as sitk, torch
from scipy import ndimage
from torch.utils.data import DataLoader
from model.model_spin import SPIN
from model.FPG import FPG
from data.data_brats import LoadData, _path, _ids
from seg_unet import UNet

B = "."
ap = argparse.ArgumentParser()
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--start_num", type=int, default=870)
ap.add_argument("--end_num", type=int, default=1000)
ap.add_argument("--extra", nargs="*", default=[], help="extra name=ckpt pairs (variant spin) to include")
ap.add_argument("--gpu_id", default="0")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
dev = "cuda"

ds = LoadData(args.dataroot, ["t1", "t2", "t1ce", "seg"], args.data_list, start_num=args.start_num,
              end_num=args.end_num, start_slice=27, end_slice=127)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
labs = []
for sid in _ids(args.data_list, args.start_num, args.end_num):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(_path(args.dataroot, "brats2021", sid, "seg")))[:, 34:226, 40:-40]
    labs += [seg[z] for z in range(27, 127)]
assert len(labs) == len(ds)


def load(variant, ckpt):
    g = SPIN(variant=variant, probability_of_losing_labels=1.0, device=dev)
    sd = torch.load(ckpt, map_location="cpu"); sd = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
    g.load_state_dict(sd, strict=False); return g.to(dev).eval()


models = {"tlp": load("tlp", f"{B}/tlp_official/G_T1T2.pth"), "spin": load("spin", f"{B}/runs/spin/G_8.pth")}
for e in args.extra:
    n, c = e.split("="); models[n] = load("spin", c)
S = UNet().to(dev); S.load_state_dict(torch.load(f"{B}/runs/_seg/best.pth", map_location=dev)); S.eval()
fpg = FPG(probability_of_losing_labels=0.0, probability_of_dilatation=0.9,
          edema_operation_frequency=2, tumor_operation_frequency=5, device=dev)


def gl(y):
    return ((y[0, 0].float() + 1) / 2 * 255).cpu().numpy()


def acc(ed, tu):
    p = ed * 0.5; p[tu == 1] = 1.0; return p


rows = {}
random.seed(0); np.random.seed(0); torch.manual_seed(0)
with torch.no_grad():
    for i, b in enumerate(loader):
        ed, tu = b["seg"][0].to(dev), b["seg"][1].to(dev)
        p = fpg(ed, tu)                                     # consume the stream on every slice
        lab = labs[i]; et = lab == 4
        if et.sum() < 30:
            continue
        t1, t2, gt = b["t1"][0].to(dev), b["t2"][0].to(dev), b["t1ce"][0].to(dev)
        les = lab > 0; brain = (t1[0, 0] > -0.999).cpu().numpy()
        normal = brain & ~ndimage.binary_dilation(les, iterations=5)
        outs = {"gt": gt, "none": models["tlp"](t1, t2, prompt=torch.zeros_like(p))}
        for n, G in models.items():
            outs[f"{n}_coarse"] = G(t1, t2, prompt=p)
            outs[f"{n}_acc"] = G(t1, t2, prompt=acc(ed, tu))
        for k, y in outs.items():
            g = gl(y)
            gy, gx = np.gradient(g)
            pred = (torch.sigmoid(S(y)) > 0.5)[0, 0].cpu().numpy()
            tp = (pred & et).sum()
            r = rows.setdefault(k, {m: [] for m in ("et", "ncr", "ed", "normal", "contrast", "grad_les",
                                                     "area_ratio", "precision", "recall", "dice")})
            r["et"].append(g[et].mean()); r["normal"].append(g[normal].mean() if normal.any() else np.nan)
            r["ncr"].append(g[lab == 1].mean() if (lab == 1).sum() >= 10 else np.nan)
            r["ed"].append(g[lab == 2].mean() if (lab == 2).sum() >= 10 else np.nan)
            r["contrast"].append(r["et"][-1] - r["normal"][-1])
            r["grad_les"].append(np.hypot(gy, gx)[les].mean())
            r["area_ratio"].append(pred.sum() / et.sum())
            r["precision"].append(tp / pred.sum() if pred.sum() else np.nan)
            r["recall"].append(tp / et.sum())
            r["dice"].append(2 * tp / (pred.sum() + et.sum()))
summ = {k: {m: float(np.nanmean(v)) for m, v in d.items()} | {"n": len(d["et"])} for k, d in rows.items()}
json.dump(summ, open(args.out, "w"), indent=1)
print(f"{'':12s}" + "".join(f"{m:>10s}" for m in ("et", "ncr", "ed", "normal", "contrast", "grad_les", "area_ratio", "precision", "recall", "dice")))
for k, d in summ.items():
    print(f"{k:12s}" + "".join(f"{d[m]:10.3f}" for m in ("et", "ncr", "ed", "normal", "contrast", "grad_les", "area_ratio", "precision", "recall", "dice")))
print("n slices:", summ["gt"]["n"], "saved", args.out)
