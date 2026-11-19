# -*- coding: utf-8 -*-
"""
SPIN-IR: interactive refinement with signed corrections.

Round 1 is SPIN as trained (frozen, run without gradient). We then simulate the radiologist
looking at the result: where the synthesis is worst *inside* the lesion they mark "you missed
this" (c=+1); where it is worst *outside* the lesion and over-enhanced they mark "this is not a
lesion" (c=-1). Round 2 re-runs the network with (p, c) and only the correction pathway
(pmb_corr_*) is trained on the round-2 loss. The prompt pathway and the backbone stay frozen,
so round 1 of SPIN-IR is bit-identical to SPIN, which itself is bit-identical to TLP without
a prompt.

    python train_ir.py --gpu_id 0 --init_backbone tlp_official/G_T1T2.pth --init_spin runs/spin/G_8.pth \
        --max_frac 0.08 --ckpt_dir runs/spin_ir_cap8/
"""
import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr

from model.model_spin import SPIN, freeze_backbone, count_params
from interaction import simulate_correction as _simulate
from data import LoadData
from utils import LambdaLR
from model.model_multi import Discriminator, weights_init_normal

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="spin")
ap.add_argument("--init_backbone", default="./tlp_official/G_T1T2.pth")
ap.add_argument("--init_spin", default="./runs/spin/G_8.pth",
                help="trained SPIN prompt pathway (round 1)")
ap.add_argument("--n_epochs", type=int, default=6)
ap.add_argument("--decay_epoch", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch_size", type=int, default=20)
ap.add_argument("--err_quantile", type=float, default=0.90,
                help="pixels above this per-slice error quantile are candidates for a correction")
ap.add_argument("--dilate", type=int, default=3, help="max-pool radius so a correction looks hand-drawn")
ap.add_argument("--p_drop_pos", type=float, default=0.3, help="drop the +1 corrections with this prob.")
ap.add_argument("--p_drop_neg", type=float, default=0.3, help="drop the -1 corrections with this prob.")
ap.add_argument("--max_frac", type=float, default=None,
                help="limit each simulated mark to this fraction of the slice (e.g. 0.08); None = uncapped")
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--ckpt_dir", default="./runs/spin_ir/")
ap.add_argument("--gpu_id", default="0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_cpu", type=int, default=8)
ap.add_argument("--gan", action="store_true",
                help="train the correction pathway with TLP's objective (LSGAN + lambda_id x L1, fresh PatchGAN "
                     "discriminator) instead of L1 alone")
ap.add_argument("--lambda_id", type=float, default=100.0, help="L1 weight when --gan is set (TLP: 100)")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = "cuda"
os.makedirs(args.ckpt_dir, exist_ok=True)
last_path = os.path.join(args.ckpt_dir, "last.pt")


def log(m):
    print(m, flush=True)
    with open(os.path.join(args.ckpt_dir, "log.txt"), "a") as f:
        f.write(m + "\n")


log(f"=== {datetime.datetime.now()} {vars(args)}")
mods = ["t1", "t2", "t1ce", "seg"]
tr = LoadData(args.dataroot, mods, args.data_list, start_num=0, end_num=850, start_slice=27, end_slice=127)
va = LoadData(args.dataroot, mods, args.data_list, start_num=850, end_num=870, start_slice=27, end_slice=127)
dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=4)
log(f"train {len(tr)} val {len(va)}")


def load_sd(path):
    sd = torch.load(path, map_location=device)
    sd = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
    for k in [k for k in list(sd) if k.endswith(("tau_A_raw", "tau_B_raw"))]:
        sd[k.replace("_raw", "")] = F.softplus(sd.pop(k))
    return sd


G = SPIN(variant=args.variant, probability_of_losing_labels=0.5, device=device, correction=True).to(device)
G.reset_prompt_pathway()
m1, u1 = G.load_state_dict(load_sd(args.init_backbone), strict=False)     # backbone
m2, u2 = G.load_state_dict(load_sd(args.init_spin), strict=False)         # trained prompt pathway
assert not u1 and not u2, (u1, u2)
left = [k for k in m2 if "pmb_corr" not in k]
assert not left, f"unexpected missing keys after init: {left[:5]}"
n_tr, n_fr = freeze_backbone(G, train_only="corr")
log(f"correction pathway trainable {n_tr} ({100*n_tr/(n_tr+n_fr):.2f}%), everything else frozen")

opt = torch.optim.Adam([p for p in G.parameters() if p.requires_grad], lr=args.lr, betas=(0.5, 0.999))
sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)
l1 = nn.L1Loss()
if args.gan:
    D = Discriminator(1).to(device); D.apply(weights_init_normal)
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    schD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)
    mse_gan = nn.MSELoss()
    log(f"adversarial term ON: LSGAN + {args.lambda_id} x L1 (fresh PatchGAN discriminator)")

start_epoch = 0
if os.path.isfile(last_path):
    ck = torch.load(last_path, map_location=device)
    G.load_state_dict(ck["G"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
    if args.gan and ck.get("D") is not None:
        D.load_state_dict(ck["D"]); optD.load_state_dict(ck["optD"]); schD.load_state_dict(ck["schD"])
    start_epoch = ck["epoch"] + 1
    log(f"AUTO-RESUME at epoch {start_epoch}")


def simulate_correction(y1, gt, ed, tu, drop_pos=args.p_drop_pos, drop_neg=args.p_drop_neg):
    c, _, _ = _simulate(y1, gt, ed, tu, q=args.err_quantile, r=args.dilate, largest_cc=True,
                        drop_pos=drop_pos, drop_neg=drop_neg, max_frac=args.max_frac)
    return c


def u8(x):
    return ((np.squeeze(x) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)


@torch.no_grad()
def validate():
    G.eval(); r1, r2 = [], []
    for bt in vl:
        t1 = bt["t1"][0].to(device); t2 = bt["t2"][0].to(device); gt = bt["t1ce"][0].to(device)
        ed = bt["seg"][0].to(device); tu = bt["seg"][1].to(device)
        if float(tu.sum()) < 50:
            continue
        p = ed * 0.5; p[tu == 1] = 1.0
        y1 = G(t1, t2, prompt=p)
        c = simulate_correction(y1, gt, ed, tu, drop_pos=0.0, drop_neg=0.0)
        y2 = G(t1, t2, prompt=p, correction=c)
        g = u8(gt); r1.append(psnr(g, u8(y1))); r2.append(psnr(g, u8(y2)))
    G.train()
    return float(np.mean(r1)), float(np.mean(r2))


for epoch in range(start_epoch, args.n_epochs):
    G.train(); t0 = time.time(); acc = []
    for i, bt in enumerate(dl):
        t1 = bt["t1"][0].to(device); t2 = bt["t2"][0].to(device); gt = bt["t1ce"][0].to(device)
        ed = bt["seg"][0].to(device); tu = bt["seg"][1].to(device)
        with torch.no_grad():
            p = G.fpg(ed, tu)                                  # coarse prompt, dropped w.p. 0.5
            y1 = G(t1, t2, prompt=p)                           # round 1 == SPIN, frozen
            c = simulate_correction(y1, gt, ed, tu)
        y2 = G(t1, t2, prompt=p, correction=c)                 # round 2, only pmb_corr_* learn
        if args.gan:
            pred_real = D(gt)
            valid, fake = torch.ones_like(pred_real), torch.zeros_like(pred_real)
            optD.zero_grad(set_to_none=True)
            loss_D = (mse_gan(pred_real, valid) + mse_gan(D(y2.detach()), fake)) / 2
            loss_D.backward(); optD.step()
            loss_l1 = l1(y2, gt)
            loss = mse_gan(D(y2), valid) + args.lambda_id * loss_l1
        else:
            loss_l1 = loss = l1(y2, gt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        acc.append(loss_l1.item())
        if i % 200 == 0:
            log(f"ep{epoch+1}/{args.n_epochs} it{i}/{len(dl)} l1_round2 {np.mean(acc[-200:]):.5f} "
                f"{(i+1)/max(time.time()-t0,1e-9):.2f} it/s")
    sch.step()
    if args.gan:
        schD.step()
    v1, v2 = validate()
    row = {"epoch": epoch + 1, "val_psnr_round1": v1, "val_psnr_round2": v2, "gain": v2 - v1,
           "train_l1": float(np.mean(acc)), "time_s": time.time() - t0}
    log("epoch " + json.dumps(row))
    with open(os.path.join(args.ckpt_dir, "history.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    torch.save(G.state_dict(), os.path.join(args.ckpt_dir, f"G_{epoch+1}.pth"))
    torch.save({"epoch": epoch, "G": G.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                "D": D.state_dict() if args.gan else None, "optD": optD.state_dict() if args.gan else None,
                "schD": schD.state_dict() if args.gan else None}, last_path + ".tmp")
    os.replace(last_path + ".tmp", last_path)
log("DONE")
