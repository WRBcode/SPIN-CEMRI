# -*- coding: utf-8 -*-
# The arguments, losses, optimiser schedule and training loop in this file are adapted from train.py
# of TLP (https://github.com/ChanghuiSu/TLP), which is distributed under the MIT License reproduced below.
#
#   MIT License
#
#   Copyright (c) 2025 Changhui Su
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
"""
Training script for the SPIN prompt pathways (SPIN, SPIN-adv and the injection-site variants).
Mirrors TLP/train.py (same data, losses, optimiser, schedule, FPG settings);
differences: --variant selects the prompt pathway, seeds are fixed, training-set
metrics are computed every --metric_every iterations instead of every iteration
(logging only, no effect on optimisation), and the run auto-resumes from last.pt.

Run from the TLP repo root. SPIN as in the paper (l1 objective, frozen TLP generator):
    python train_spin.py --variant spin --freeze_backbone --no_gan --init_ckpt tlp_official/G_T1T2.pth \
        --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --dataroot dataset/BraTS2021 --ckpt_dir runs/spin/
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from model.model_spin import SPIN, VARIANTS, count_params, freeze_backbone
from model.model_multi import Discriminator, weights_init_normal
from data import LoadData
from utils import LambdaLR

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="spin", choices=list(VARIANTS.keys()))
parser.add_argument("--single_input", action="store_true", help="one-branch model (T1->T1ce / T2->T1ce)")
parser.add_argument("--modality", type=str, default="t1", help="input modality for --single_input")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80)
parser.add_argument("--decay_epoch", type=int, default=40)
parser.add_argument("--dataroot", type=str, default="./dataset/BraTS2021")
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--num_refinement_blocks", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--img_height", type=int, default=192)
parser.add_argument("--img_width", type=int, default=160)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--checkpoint_interval", type=int, default=2)
parser.add_argument("--lambda_id", type=float, default=100.0)
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--filename", type=str, default="log.txt")
parser.add_argument("--metric_every", type=int, default=50)
parser.add_argument("--start_num_train", type=int, default=0)
parser.add_argument("--end_num_train", type=int, default=850)
parser.add_argument("--start_num_val", type=int, default=850)
parser.add_argument("--end_num_val", type=int, default=870)
parser.add_argument("--start_slice", type=int, default=27)
parser.add_argument("--end_slice", type=int, default=127)
parser.add_argument("--data_list", type=str, default="./data_list/Brats2021_selected.txt")
parser.add_argument("--no_resume", action="store_true")
parser.add_argument("--freeze_backbone", action="store_true",
                    help="train only the prompt pathway (PMB / PCA); every TLP weight stays as published")
parser.add_argument("--no_gan", action="store_true",
                    help="drop the adversarial term and train with the L1 reconstruction loss only "
                         "(SPIN in the paper; without this flag the objective is TLP's LSGAN + "
                         "lambda_id x L1, as for SPIN-adv)")
parser.add_argument("--init_ckpt", type=str, default="",
                    help="initialise the generator from a TLP/SPIN state_dict (e.g. the official G_T1T2.pth); "
                         "PMB / PCA parameters absent from the file stay zero-initialised")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)

os.makedirs(opt.ckpt_dir, exist_ok=True)
log_path = os.path.join(opt.ckpt_dir, opt.filename)
last_path = os.path.join(opt.ckpt_dir, "last.pt")


def log(msg):
    print(msg, flush=True)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


log(f"=== {datetime.datetime.now()}  {opt}")

modalities = ["t1", "t2", "t1ce", "seg"] if not opt.single_input else [opt.modality, "t1ce", "seg"]
train_set = LoadData(opt.dataroot, modalities, opt.data_list, start_num=opt.start_num_train,
                     end_num=opt.end_num_train, start_slice=opt.start_slice, end_slice=opt.end_slice)
val_set = LoadData(opt.dataroot, modalities, opt.data_list, start_num=opt.start_num_val,
                   end_num=opt.end_num_val, start_slice=opt.start_slice, end_slice=opt.end_slice)
dataloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
val_data = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=opt.n_cpu, drop_last=True)
log(f"train slices {len(train_set)}  val slices {len(val_set)}")

criterion_GAN = nn.MSELoss()
criterion_identity = nn.L1Loss()
device = "cuda"

G = SPIN(variant=opt.variant, single_input=opt.single_input, num_refinement_blocks=opt.num_refinement_blocks,
         probability_of_losing_labels=0.5, probability_of_dilatation=0.9, edema_operation_frequency=2,
         tumor_operation_frequency=5, device=device)
D = Discriminator(opt.channels)

if opt.freeze_backbone:
    n_tr, n_fr = freeze_backbone(G)
    log(f"FROZEN BACKBONE: training {n_tr} prompt-pathway params, {n_fr} frozen "
        f"({100*n_tr/(n_tr+n_fr):.2f}% trainable)")
    if n_tr == 0:
        raise SystemExit(f"variant '{opt.variant}' has no prompt-pathway parameters to train")

optimizer_G = torch.optim.Adam([p for p in G.parameters() if p.requires_grad],
                               lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

fake = torch.zeros((opt.batch_size, opt.channels, opt.img_height // 8 - 2, opt.img_width // 8 - 2)).to(device)
valid = torch.ones((opt.batch_size, opt.channels, opt.img_height // 8 - 2, opt.img_width // 8 - 2)).to(device)
G = G.to(device); D = D.to(device)
criterion_GAN.to(device); criterion_identity.to(device)

G.apply(weights_init_normal)
D.apply(weights_init_normal)
G.reset_prompt_pathway()          # keep PMB / PCA zero-initialised after TLP's conv re-init


def _modernise(sd):
    """Checkpoints of an earlier version store the PCA gate as softplus(tau_*_raw)."""
    for k in [k for k in list(sd) if k.endswith(("tau_A_raw", "tau_B_raw"))]:
        sd[k.replace("_raw", "")] = torch.nn.functional.softplus(sd.pop(k))
    return sd


if opt.init_ckpt:
    sd = torch.load(opt.init_ckpt, map_location=device)
    sd = _modernise(sd["G"] if isinstance(sd, dict) and "G" in sd else sd)
    missing, unexpected = G.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not (k.startswith("pmb_") or ".pmb_" in k
                                      or k.endswith(("tau_A", "tau_B")) or ".emb_" in k)]
    if unexpected or bad:
        raise RuntimeError(f"init_ckpt mismatch: unexpected={unexpected} missing={bad}")
    log(f"INIT from {opt.init_ckpt}: {len(sd)} tensors loaded, {len(missing)} prompt-pathway tensors left at zero-init")

start_epoch = opt.epoch
if not opt.no_resume and os.path.isfile(last_path):
    ck = torch.load(last_path, map_location=device)
    G.load_state_dict(_modernise(ck["G"])); D.load_state_dict(ck["D"])
    optimizer_G.load_state_dict(ck["optG"]); optimizer_D.load_state_dict(ck["optD"])
    lr_scheduler_G.load_state_dict(ck["schedG"]); lr_scheduler_D.load_state_dict(ck["schedD"])
    start_epoch = ck["epoch"] + 1
    log(f"AUTO-RESUME from {last_path}: continuing at epoch {start_epoch}")

log(f"variant={opt.variant} single_input={opt.single_input} params={count_params(G)/1e6:.3f}M")


def to_u8(x):
    return ((np.squeeze(x) + 1) / 2 * 255).cpu().detach().numpy().astype(np.uint8)


def forward(batch):
    t1ce = batch["t1ce"][0].to(device)
    ed = batch["seg"][0].to(device)
    tu = batch["seg"][1].to(device)
    if opt.single_input:
        out = G(batch[opt.modality][0].to(device), edema_region=ed, tumor_region=tu)
    else:
        out = G(batch["t1"][0].to(device), batch["t2"][0].to(device), ed, tu)
    return out, t1ce


prev_time = time.time()
for epoch in range(start_epoch, opt.n_epochs):
    G.train(); D.train()
    tr = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    t_epoch = time.time()
    for i, batch in enumerate(dataloader):
        fake_B, real_B = forward(batch)

        if i % opt.metric_every == 0:
            for idx in range(opt.batch_size):
                f8, r8 = to_u8(fake_B[idx]), to_u8(real_B[idx])
                if not (f8 == r8).all():
                    tr["psnr"].append(psnr(r8, f8))
                tr["ssim"].append(ssim(r8, f8)); tr["mse"].append(mse(r8, f8)); tr["mae"].append(mae(r8, f8))

        # ---- discriminator ----
        if opt.no_gan:
            loss_D = torch.zeros((), device=device)
        else:
            optimizer_D.zero_grad()
            loss_D = (criterion_GAN(D(real_B), valid) + criterion_GAN(D(fake_B.detach()), fake)) / 2
            loss_D.backward(); optimizer_D.step()

        # ---- generator ----
        optimizer_G.zero_grad()
        loss_identity = criterion_identity(fake_B, real_B)
        loss_GAN = torch.zeros((), device=device) if opt.no_gan else criterion_GAN(D(fake_B), valid)
        loss_G = loss_GAN + opt.lambda_id * loss_identity
        loss_G.backward(); optimizer_G.step()

        if i % 100 == 0:
            batches_left = (opt.n_epochs - epoch) * len(dataloader) - i
            it_s = 100 / max(time.time() - prev_time, 1e-6) if i else 0.0
            prev_time = time.time()
            print(f"[Epoch {epoch+1}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] D {loss_D.item():.4f} "
                  f"G {loss_G.item():.4f} (adv {loss_GAN.item():.4f}, l1 {loss_identity.item():.4f}) "
                  f"{it_s:.2f} it/s  ETA {datetime.timedelta(seconds=int(batches_left / max(it_s, 1e-3)))}", flush=True)

    lr_scheduler_G.step(); lr_scheduler_D.step()
    tr_line = f"Training: {epoch+1} {np.mean(tr['psnr']):.3f} {np.mean(tr['ssim']):.3f} {np.mean(tr['mse']):.3f} {np.mean(tr['mae']):.3f}"

    # ---- validation (as in TLP) ----
    G.eval()
    va = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    with torch.no_grad():
        for batch in val_data:
            fake_B, real_B = forward(batch)
            f8, r8 = to_u8(fake_B), to_u8(real_B)
            if not (f8 == r8).all():
                va["psnr"].append(psnr(r8, f8))
            va["ssim"].append(ssim(r8, f8)); va["mse"].append(mse(r8, f8)); va["mae"].append(mae(r8, f8))
    va_line = f"Validation: {epoch+1} {np.mean(va['psnr']):.3f} {np.mean(va['ssim']):.3f} {np.mean(va['mse']):.3f} {np.mean(va['mae']):.3f}"
    log(tr_line); log(va_line)
    log(f"epoch {epoch+1} took {datetime.timedelta(seconds=int(time.time() - t_epoch))}  lr {optimizer_G.param_groups[0]['lr']:.2e}")
    with open(os.path.join(opt.ckpt_dir, "history.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": epoch + 1, "train_psnr": float(np.mean(tr["psnr"])), "val_psnr": float(np.mean(va["psnr"])),
                            "val_ssim": float(np.mean(va["ssim"])), "val_mse": float(np.mean(va["mse"])),
                            "time_s": time.time() - t_epoch}) + "\n")

    # ---- checkpoints ----
    if (epoch + 1) % opt.checkpoint_interval == 0:
        torch.save(G.state_dict(), os.path.join(opt.ckpt_dir, f"G_{epoch+1}.pth"))
    torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(), "optG": optimizer_G.state_dict(),
                "optD": optimizer_D.state_dict(), "schedG": lr_scheduler_G.state_dict(),
                "schedD": lr_scheduler_D.state_dict()}, last_path + ".tmp")
    os.replace(last_path + ".tmp", last_path)

log("DONE")
