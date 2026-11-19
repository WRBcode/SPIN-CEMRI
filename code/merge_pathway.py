# -*- coding: utf-8 -*-
"""
Rebuild a full generator checkpoint from TLP's released weights and the pathway files in weights/.

The files in weights/ hold only the parameters that SPIN trains on top of the frozen TLP generator:
the prompt pathway (and, for SPIN-IR, the correction pathway). The backbone stays frozen during
training, so TLP's released weights plus these files give back the generator state_dict that
train_spin.py (G_8.pth) and train_ir.py (G_6.pth) save, which is what the evaluation scripts load
(--ckpt, --spin_ckpt, --ir_ckpt) and what train_ir.py expects as --init_spin. The merged
dictionary is loaded into a freshly built SPIN model with strict=True before it is written, so a
missing or unexpected key stops the script. The one-branch model (single_input) is detected from
the backbone file and the correction pathway from the pathway files.

Run from the TLP repo root (CPU is enough):
    python merge_pathway.py --backbone tlp_official/G_T1T2.pth \
        --pathway weights/spin_prompt_pathway_T1T2.pth --out runs/spin/G_8.pth
    python merge_pathway.py --backbone tlp_official/G_T1T2.pth \
        --pathway weights/spin_prompt_pathway_T1T2.pth weights/spin_ir_correction_pathway_T1T2.pth \
        --out runs/spin_ir_cap8/G_6.pth
"""
import argparse
import os

import torch

from model.model_spin import SPIN, VARIANTS

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, help="TLP generator weights: G_T1T2.pth, G_T1.pth or G_T2.pth")
ap.add_argument("--pathway", nargs="+", required=True, help="one or more pathway files from weights/")
ap.add_argument("--variant", default="spin", choices=list(VARIANTS.keys()))
ap.add_argument("--out", required=True)
args = ap.parse_args()


def load(path):
    sd = torch.load(path, map_location="cpu")
    return sd["G"] if isinstance(sd, dict) and "G" in sd else sd


sd = dict(load(args.backbone))
n_backbone = len(sd)
for path in args.pathway:
    extra = load(path)
    clash = sorted(set(extra) & set(sd))
    if clash:
        raise SystemExit(f"{path} would overwrite {len(clash)} tensors that are already set, e.g. {clash[:3]}")
    sd.update(extra)

single_input = not any(k.startswith("encoder_B.") for k in sd)      # TLP's one-branch generators (G_T1, G_T2)
correction = any(k.startswith("pmb_corr") for k in sd)              # SPIN-IR correction pathway present
G = SPIN(variant=args.variant, single_input=single_input, correction=correction, device="cpu")
G.load_state_dict(sd, strict=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(G.state_dict(), args.out)
print(f"wrote {args.out}: {n_backbone} backbone tensors + {len(sd) - n_backbone} pathway tensors "
      f"(variant={args.variant}, single_input={single_input}, correction={correction})")
