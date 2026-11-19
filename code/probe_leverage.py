# -*- coding: utf-8 -*-
"""
Where in the network does a modulation actually have leverage?

Runs on CPU against the *trained* TLP weights and *real* BraTS slices, and answers two
questions that decide the design of the prompt pathway:

  1. site sensitivity  -- if we perturb the feature map at site s by a relative amount eps,
     how much does the output move?  s in {enc1/2/3 of both branches, the cross-attention
     residual, dec1/2/3}.  This is the leverage a prompt-conditioned modulation at s can have.
  2. cross-attention contribution -- how much does the whole cross-attention residual
     contribute to the output at all?  If replacing it by the identity barely moves the
     output, no prompt mechanism placed there can matter.

Usage (from the TLP repo root, CPU, no GPU needed):
    python probe_leverage.py --ckpt tlp_official/G_T1T2.pth --n 4
"""
import argparse
import numpy as np
import torch

from model.model_spin import SPIN, PromptBiasedCrossAttention
from data import LoadData

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="./tlp_official/G_T1T2.pth")
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--start_num", type=int, default=870)
ap.add_argument("--end_num", type=int, default=871)
ap.add_argument("--start_slice", type=int, default=70)
ap.add_argument("--end_slice", type=int, default=90)
ap.add_argument("--n", type=int, default=4, help="slices to average over (picked with a lesion)")
ap.add_argument("--eps", type=float, default=0.05, help="relative perturbation size")
args = ap.parse_args()

torch.manual_seed(0)
ds = LoadData(args.dataroot, ["t1", "t2", "t1ce", "seg"], args.data_list,
              start_num=args.start_num, end_num=args.end_num,
              start_slice=args.start_slice, end_slice=args.end_slice)

G = SPIN(variant="tlp", probability_of_losing_labels=1.0, device="cpu")
sd = torch.load(args.ckpt, map_location="cpu")
missing, unexpected = G.load_state_dict(sd, strict=False)
assert not unexpected and not [k for k in missing if "pmb_" not in k and "tau" not in k and "emb_" not in k], (missing, unexpected)
G.eval()
print(f"loaded {args.ckpt}")

# pick slices that actually contain a lesion
idx = [i for i in range(len(ds)) if float(ds[i]["seg"][1].sum()) > 50][:args.n]
print(f"using {len(idx)} lesion-bearing slices of {len(ds)}")

sites = ["enc1", "enc2", "enc3", "xattn", "dec3", "dec2", "dec1"]
rel = {s: [] for s in sites}
xattn_share = []

for i in idx:
    b = ds[i]
    t1 = b["t1"][0].unsqueeze(0); t2 = b["t2"][0].unsqueeze(0)
    ed = b["seg"][0].unsqueeze(0); tu = b["seg"][1].unsqueeze(0)
    p = ed * 0.5
    p[tu == 1] = 1.0

    with torch.no_grad():
        y0 = G(t1, t2, prompt=p)
        scale = float(y0.abs().mean())

        # ---- 1) site sensitivity: multiply the site's features by (1 + eps * random sign pattern)
        for s in sites:
            hooks = []

            def make_hook(eps):
                def hook(mod, inp, out):
                    if isinstance(out, tuple):
                        return tuple(o * (1 + eps * torch.randn_like(o)) for o in out)
                    return out * (1 + eps * torch.randn_like(out))
                return hook

            target = {
                "enc1": [G.encoder_A.encoder_level1, G.encoder_B.encoder_level1],
                "enc2": [G.encoder_A.encoder_level2, G.encoder_B.encoder_level2],
                "enc3": [G.encoder_A.encoder_level3, G.encoder_B.encoder_level3],
                "xattn": [G.cross_attention],
                "dec3": [G.decoder_level3],
                "dec2": [G.decoder_level2],
                "dec1": [G.decoder_level1],
            }[s]
            torch.manual_seed(1234)                      # same noise pattern for every site
            for m in target:
                hooks.append(m.register_forward_hook(make_hook(args.eps)))
            y = G(t1, t2, prompt=p)
            for h in hooks:
                h.remove()
            rel[s].append(float((y - y0).abs().mean()) / scale)

        # ---- 2) how much does the cross-attention residual contribute at all?
        ca = G.cross_attention
        orig_forward = ca.forward
        ca.forward = lambda x_A, x_B, p3=None: (x_A, x_B)      # bypass: pure identity
        y_noca = G(t1, t2, prompt=p)
        ca.forward = orig_forward
        xattn_share.append(float((y_noca - y0).abs().mean()) / scale)

print(f"\n=== site sensitivity: relative output change for a {args.eps:.0%} feature perturbation")
for s in sites:
    m = np.mean(rel[s])
    print(f"  {s:6s} {m*100:7.3f}%   {'#' * int(round(m * 2000))}")
print(f"\n=== bypassing the cross-attention residual entirely changes the output by "
      f"{np.mean(xattn_share)*100:.3f}% of its mean magnitude")
print("    (if this is small, no prompt mechanism placed in the cross-attention can matter)")
