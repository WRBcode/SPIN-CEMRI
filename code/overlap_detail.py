# -*- coding: utf-8 -*-
"""
Split the BraTS 2020 subjects that the backbone never saw (check_overlap.py) into those that
reappear in our BraTS 2021 validation/test range and those that appear nowhere in the 2021 split.
Only the second group is an independent held-out set: the first would double-count internal
test patients. Writes data_list/Brats2020_heldout.txt (order as in Brats2020_selected.txt).
"""
import argparse, hashlib, json
import numpy as np, SimpleITK as sitk

ap = argparse.ArgumentParser()
ap.add_argument("--root21", default="./dataset/BraTS2021")
ap.add_argument("--root20", default="./dataset/MICCAI_BraTS2020_TrainingData")
ap.add_argument("--overlap", default="./data_list/cohort_overlap.json", help="output of check_overlap.py")
ap.add_argument("--out", default="./data_list/cohort_overlap_detail.json")
args = ap.parse_args()
ROOT21, ROOT20, OVERLAP, OUT = args.root21, args.root20, args.overlap, args.out
TRAIN_END, VAL_END = 850, 870


def ids(p):
    return [l.strip() for l in open(p) if l.strip()]


def vhash(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    return hashlib.md5(np.ascontiguousarray(a)).hexdigest()


i21, i20 = ids("./data_list/Brats2021_selected.txt"), ids("./data_list/Brats2020_selected.txt")
unseen = set(json.load(open(OVERLAP))["unseen_2020_ids"])
h21 = {}
for k in range(TRAIN_END, len(i21)):
    s = i21[k]
    h21.setdefault(vhash(f"{ROOT21}/BraTS2021_{s}/BraTS2021_{s}_t1.nii.gz"), []).append((k, s))
match = {s: h21.get(vhash(f"{ROOT20}/BraTS20_Training_{s}/BraTS20_Training_{s}_t1.nii"), [])
         for s in i20 if s in unseen}
in_val = [s for s, v in match.items() if any(k < VAL_END for k, _ in v)]
in_test = [s for s, v in match.items() if any(k >= VAL_END for k, _ in v)]
heldout = [s for s in i20 if s in match and not match[s]]
res = dict(n_unseen=len(match), n_in_2021_val=len(in_val), n_in_2021_test=len(in_test),
           n_heldout=len(heldout), heldout_ids=heldout,
           in_2021_val_test={s: v for s, v in match.items() if v})
print({k: v for k, v in res.items() if k not in ("heldout_ids", "in_2021_val_test")})
json.dump(res, open(OUT, "w"), indent=1)
with open("./data_list/Brats2020_heldout.txt", "w") as f:
    f.write("\n".join(heldout) + "\n")
print("saved", OUT, "and data_list/Brats2020_heldout.txt")
