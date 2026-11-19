# -*- coding: utf-8 -*-
"""
How much do the BraTS 2021 and BraTS 2020 cohorts share?

BraTS 2021 is a superset of earlier releases, so a model trained on the 2021 split may have
seen subjects that reappear in the 2020 cohort under a different identifier. Any claim that
2020 is an independent test set has to be checked. Instead of relying on an identifier
mapping, we check the pixels directly: both releases ship the same
co-registered, skull-stripped volumes, so a shared subject has a bit-identical T1 volume.
We hash each volume and intersect.
"""
import argparse, hashlib, json, os, sys
import numpy as np, SimpleITK as sitk

ap = argparse.ArgumentParser()
ap.add_argument("--root21", default="./dataset/BraTS2021")
ap.add_argument("--root20", default="./dataset/MICCAI_BraTS2020_TrainingData")
ap.add_argument("--list21", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--list20", default="./data_list/Brats2020_selected.txt")
ap.add_argument("--train_end", type=int, default=850, help="2021 subjects 0..train_end were seen by the backbone")
ap.add_argument("--out", default="./data_list/cohort_overlap.json")
args = ap.parse_args()

def ids(p): return [l.strip() for l in open(p) if l.strip()]

def vhash(path):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    return hashlib.md5(np.ascontiguousarray(a)).hexdigest()

i21, i20 = ids(args.list21), ids(args.list20)
print(f"hashing {len(i21)} BraTS2021 + {len(i20)} BraTS2020 T1 volumes", flush=True)
h21 = {}
for n, s in enumerate(i21):
    h21.setdefault(vhash(f"{args.root21}/BraTS2021_{s}/BraTS2021_{s}_t1.nii.gz"), []).append(s)
    if n % 200 == 0: print(f"  2021 {n}/{len(i21)}", flush=True)
h20 = {}
for n, s in enumerate(i20):
    h20.setdefault(vhash(f"{args.root20}/BraTS20_Training_{s}/BraTS20_Training_{s}_t1.nii"), []).append(s)
    if n % 100 == 0: print(f"  2020 {n}/{len(i20)}", flush=True)

train21 = set(i21[:args.train_end])
shared = sorted(set(h21) & set(h20))
pairs = [(h21[k], h20[k]) for k in shared]
seen = [(a, b) for a, b in pairs if any(x in train21 for x in a)]
unseen20 = [s for s in i20 if not any(s in b for a, b in seen)]

res = dict(n21=len(i21), n20=len(i20), n_shared_volumes=len(shared),
           n_2020_subjects_in_2021_train=len({b for a, bl in seen for b in bl}),
           n_2020_subjects_unseen_by_backbone=len(unseen20),
           unseen_2020_ids=unseen20)
print(json.dumps({k: v for k, v in res.items() if k != "unseen_2020_ids"}, indent=1))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print("saved", args.out)
