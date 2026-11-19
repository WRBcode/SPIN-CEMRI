# -*- coding: utf-8 -*-
# Parts of this file are adapted from model/model_multi.py of TLP (https://github.com/ChanghuiSu/TLP):
# the layout of the generator in SPIN.__init__ and SPIN.forward follows TLP's generator class, and
# PromptBiasedCrossAttention extends its Cross_attention block. TLP is distributed under the MIT
# License reproduced below. TLP's model_multi.py itself states: "This file is modified from
# [Restormer] (https://github.com/swz30/Restormer), [Text-IF] (https://github.com/XunpengYi/Text-IF)
# which is licensed under the MIT License."
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
SPIN: Scale-wise Prompt INjection on top of the public TLP code
(https://github.com/ChanghuiSu/TLP, MIT-licensed model files).

Drop this file into  TLP/model/  and use  `from model.model_spin import SPIN`.
It re-uses TLP's Encoder_A/Encoder_B, FusionBlock, Attention_spatial,
TransformerBlock, Upsample and FPG unchanged; only the prompt pathway is new.

Variants (the keys of VARIANTS below). The paper's SPIN is `spin`; `tlp`, `enc` and `dec` are
the other rows of its injection-site study ("Where should the prompt enter?"):

    name            inject_input  enc_pmb  dec_pmb  pca
    tlp                 True       False    False   None      == original TLP
    enc                 True       True     False   None
    dec                 True       False    True    None
    encdec              True       True     True    None
    spin                True       True     True    "bias"    <- SPIN in the paper
    spin2               True       True     True    "embed"
    spin2_noinput       False      True     True    "embed"

`pca` is the optional prompt term in the cross-modal attention (PromptBiasedCrossAttention):
"bias" adds a prompt-dependent logit bias scaled by two learnable scalars (tau_A, tau_B), which
vanishes without a prompt; "embed" adds learned prompt embeddings to queries and keys and is
not used in the paper.

`single_input=True` gives the one-branch model used for the T1->T1ce / T2->T1ce
tasks (mirrors TLP/model/model_single.py: one encoder, no LF/PCA, spatial
attention at level 3, plain skip concatenation in the decoder).

Everything new is zero-initialised, so every variant starts as the un-prompted
TLP network and the prompt pathway is learned as a residual correction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_multi import (Encoder_A, Encoder_B, FusionBlock, Attention_spatial,
                          TransformerBlock, Upsample)
from .FPG import FPG


VARIANTS = {
    "tlp":          dict(inject_input=True,  enc_pmb=False, dec_pmb=False, pca=None),
    "enc":          dict(inject_input=True,  enc_pmb=True,  dec_pmb=False, pca=None),
    "dec":          dict(inject_input=True,  enc_pmb=False, dec_pmb=True,  pca=None),
    "encdec":       dict(inject_input=True,  enc_pmb=True,  dec_pmb=True,  pca=None),
    # "bias": scalar-gated rank-one logit bias (first design; measured to be too weak, see PCA docstring)
    "spin":         dict(inject_input=True,  enc_pmb=True,  dec_pmb=True,  pca="bias"),
    # "embed": prompt as a segment embedding added to the cross-attention Q/K projections
    "spin2":        dict(inject_input=True,  enc_pmb=True,  dec_pmb=True,  pca="embed"),
    "spin2_noinput": dict(inject_input=False, enc_pmb=True, dec_pmb=True,  pca="embed"),
}


# --------------------------------------------------------------------------- #
#  Prompt modulation block  (Eq. 1):  F <- F * (1 + gamma(p_l)) + beta(p_l)
# --------------------------------------------------------------------------- #
class PMB(nn.Module):
    """Prompt modulation block:  F <- F * (1 + gamma(p_l)) + beta(p_l).

    Both convolutions are bias-free and GELU(0)=0, so g(0)=0 exactly: with no prompt the
    block is the identity for any weights, not just at initialisation. Biases would apply a
    prompt-independent channel-wise shift even when p=0, so they are omitted.
    """

    def __init__(self, channels, hidden=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * channels, kernel_size=3, padding=1, bias=False),
        )
        # zero-init the last layer -> the block is off at initialisation as well
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, feat, p_l):
        if p_l.shape[-2:] != feat.shape[-2:]:
            p_l = F.interpolate(p_l, size=feat.shape[-2:], mode="area")
        gamma, beta = self.head(p_l).chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


# --------------------------------------------------------------------------- #
#  Prompt-biased cross-modal attention
#  = TLP.Cross_attention  +  tau_i * pbar pbar^T  added to the logits
# --------------------------------------------------------------------------- #
class PromptBiasedCrossAttention(nn.Module):
    """TLP's cross-modal attention with an optional prompt pathway.

    mode=None      : plain cross-attention (the TLP block).
    mode="bias"    : A_i = softmax(Q_j K_i^T/sqrt(d) + tau_i * pbar pbar^T).
                     Measured on a real blob prompt (5% of the slice, 1.1% of level-3 tokens):
                     tau=1 shifts the output by 0.14% of its range and the gradient w.r.t. tau is
                     ~1e-7, three orders below a PMB conv weight. The bias only touches rows whose
                     query is inside the prompt, so it cannot move the network. It is part of the
                     `spin` variant as trained; `eval_spin.py --disable_pca` zeroes it as a control.
    mode="embed"   : prompt as a segment embedding added to the projections,
                     Q_i <- Q_i + W_q^i(p_3), K_i <- K_i + W_k^i(p_3), with zero-initialised 1x1
                     convs. The logit then carries q.k + q.W_k(p) + W_q(p).k + W_q(p).W_k(p): every
                     query row is modulated, content-dependently, and the gradients are ordinary
                     conv gradients. Off at initialisation, like every other block in SPIN.
    """

    def __init__(self, in_channel, n_head=1, norm_groups=16, mode=None):
        super().__init__()
        self.n_head = n_head
        self.mode = mode
        self.use_bias = mode is not None
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)
        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)
        # One learnable scalar per direction, used directly and initialised at zero:
        # the bias is off at init (so training starts exactly at the un-biased model) while the
        # gradient w.r.t. tau is O(1). An earlier version parameterised tau as softplus(raw) with
        # raw = -6 to keep tau >= 0; there softplus'(-6) ~ 2.5e-3 throttled the gradient and tau
        # stayed frozen at its initial value for the whole fine-tuning run. A negative tau is
        # allowed and simply means prompted tokens attend to each other *less*.
        self.tau_A = nn.Parameter(torch.zeros(1))
        self.tau_B = nn.Parameter(torch.zeros(1))
        if mode == "embed":
            # segment embeddings for the two branches' queries and keys; zero-init -> off at start
            self.emb_qA = nn.Conv2d(1, in_channel, 1)
            self.emb_kA = nn.Conv2d(1, in_channel, 1)
            self.emb_qB = nn.Conv2d(1, in_channel, 1)
            self.emb_kB = nn.Conv2d(1, in_channel, 1)
            for m in (self.emb_qA, self.emb_kA, self.emb_qB, self.emb_kB):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    @staticmethod
    def _attend(query, key, value, channel, bias):
        # query/key/value: (b, n_head, head_dim, h, w); bias: (b, 1, h, w, h*w) or None
        b, n, c, h, w = query.shape
        attn = torch.einsum("bnchw,bncyx->bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(b, n, h, w, -1)
        if bias is not None:
            attn = attn + bias
        attn = torch.softmax(attn, -1).view(b, n, h, w, h, w)
        return torch.einsum("bnhwyx,bncyx->bnchw", attn, value).contiguous()

    def forward(self, x_A, x_B, p3=None):
        b, c, h, w = x_A.shape
        n, hd = self.n_head, c // self.n_head
        xa = self.norm_A(x_A)
        qa, ka, va = self.qkv_A(xa).view(b, n, hd * 3, h, w).chunk(3, dim=2)
        xb = self.norm_B(x_B)
        qb, kb, vb = self.qkv_B(xb).view(b, n, hd * 3, h, w).chunk(3, dim=2)

        if self.use_bias and p3 is not None and p3.shape[-2:] != (h, w):
            p3 = F.interpolate(p3, size=(h, w), mode="area")

        if self.mode == "embed" and p3 is not None:
            emb = lambda m: m(p3).view(b, n, hd, h, w)
            qa = qa + emb(self.emb_qA); ka = ka + emb(self.emb_kA)
            qb = qb + emb(self.emb_qB); kb = kb + emb(self.emb_kB)

        bias_A = bias_B = None
        if self.mode == "bias" and p3 is not None:
            pbar = p3.flatten(1)                                   # (b, N)
            outer = pbar[:, :, None] * pbar[:, None, :]            # (b, N_q, N_k)
            outer = outer.view(b, 1, h, w, h * w)
            bias_A = self.tau_A * outer                            # queries from B, keys from A
            bias_B = self.tau_B * outer

        out_A = self._attend(qb, ka, va, c, bias_A)
        out_A = self.out_A(out_A.view(b, c, h, w)) + xa
        out_B = self._attend(qa, kb, vb, c, bias_B)
        out_B = self.out_B(out_B.view(b, c, h, w)) + xb
        return out_A, out_B


# --------------------------------------------------------------------------- #
#  SPIN generator
# --------------------------------------------------------------------------- #
class SPIN(nn.Module):
    def __init__(self, variant="spin", single_input=False, out_channels=1, dim=48,
                 num_blocks=(1, 2, 2), num_refinement_blocks=2, heads=(2, 4, 8),
                 ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias",
                 probability_of_losing_labels=0.5, probability_of_dilatation=0.9,
                 edema_operation_frequency=2, tumor_operation_frequency=5, device="cuda",
                 correction=False):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant = variant
        self.single_input = single_input
        self.inject_input = cfg["inject_input"]
        self.enc_pmb = cfg["enc_pmb"]
        self.dec_pmb = cfg["dec_pmb"]
        # "bias" | "embed" | None. A one-branch model has no cross-modal attention, so no PCA.
        # (Careful: `cfg["pca"] and not single_input` would collapse the mode string to a bool.)
        self.use_pca = None if single_input else cfg["pca"]

        self.fpg = FPG(probability_of_losing_labels=probability_of_losing_labels,
                       probability_of_dilatation=probability_of_dilatation,
                       edema_operation_frequency=edema_operation_frequency,
                       tumor_operation_frequency=tumor_operation_frequency, device=device)

        in_ch = 2 if self.inject_input else 1
        kw = dict(dim=dim, num_blocks=num_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                  bias=bias, LayerNorm_type=LayerNorm_type)
        dims = [dim, dim * 2, dim * 4]                      # level 1,2,3 widths (48, 96, 192)

        self.encoder_A = Encoder_A(inp_channels=in_ch, **kw)
        if self.enc_pmb:
            self.pmb_enc_A = nn.ModuleList([PMB(c) for c in dims])
        if not single_input:
            self.encoder_B = Encoder_B(inp_channels=in_ch, **kw)
            if self.enc_pmb:
                self.pmb_enc_B = nn.ModuleList([PMB(c) for c in dims])
            self.cross_attention = PromptBiasedCrossAttention(dim * 4, mode=self.use_pca)
            self.fusion_level3 = FusionBlock(in_dim=dim * 4)
            self.fusion_level2 = FusionBlock(in_dim=dim * 2)
            self.fusion_level1 = FusionBlock(in_dim=dim)
        if self.dec_pmb:
            # decoder widths: level3 = 4*dim, level2 = 2*dim, level1 = 2*dim (after skip concat)
            self.pmb_dec = nn.ModuleList([PMB(dim * 4), PMB(dim * 2), PMB(dim * 2)])

        # ---- interactive refinement: a second, independent modulation pathway driven by a
        # signed correction map c in {-1, 0, +1} ("you missed a lesion here" / "this is not a
        # lesion, suppress it"). It sits on top of the prompt pathway at the same sites and is
        # the identity at c=0, so round 1 (c=0) is bit-identical to the model without it.
        self.correction = correction
        if correction:
            self.pmb_corr_enc_A = nn.ModuleList([PMB(c) for c in dims]) if self.enc_pmb else None
            self.pmb_corr_enc_B = (nn.ModuleList([PMB(c) for c in dims])
                                   if (self.enc_pmb and not single_input) else None)
            self.pmb_corr_dec = (nn.ModuleList([PMB(dim * 4), PMB(dim * 2), PMB(dim * 2)])
                                 if self.dec_pmb else None)

        self.attention_spatial = Attention_spatial(dim * 4)
        tb = lambda d, hds: TransformerBlock(dim=d, num_heads=hds, ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias, LayerNorm_type=LayerNorm_type)
        self.decoder_level3 = nn.Sequential(*[tb(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[tb(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[tb(dim * 2, heads[0]) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[tb(dim * 2, heads[0]) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.tanh = nn.Tanh()

    # ---- helpers -----------------------------------------------------------
    def reset_prompt_pathway(self):
        """Re-zero the PMB heads and the PCA scalars. Call this AFTER TLP's
        `G.apply(weights_init_normal)`, which would otherwise overwrite the
        zero-initialised last conv of every PMB with N(0, 0.02) weights."""
        for m in self.modules():
            if isinstance(m, PMB):
                nn.init.zeros_(m.head[-1].weight)
            if isinstance(m, PromptBiasedCrossAttention):
                nn.init.zeros_(m.tau_A)
                nn.init.zeros_(m.tau_B)
                for name in ("emb_qA", "emb_kA", "emb_qB", "emb_kB"):
                    if hasattr(m, name):
                        e = getattr(m, name)
                        nn.init.zeros_(e.weight); nn.init.zeros_(e.bias)

    @staticmethod
    def prompt_pyramid(p):
        p1 = p
        p2 = F.avg_pool2d(p1, 2)
        p3 = F.avg_pool2d(p2, 2)
        return [p1, p2, p3]

    def _encode(self, enc, x, pyr, pmbs, cpyr=None, cpmbs=None):
        def mod(f, lvl):
            if pmbs is not None:
                f = pmbs[lvl](f, pyr[lvl])
            if cpmbs is not None and cpyr is not None:
                f = cpmbs[lvl](f, cpyr[lvl])
            return f
        f1 = mod(enc.encoder_level1(enc.patch_embed(x)), 0)
        f2 = mod(enc.encoder_level2(enc.down1_2(f1)), 1)
        f3 = mod(enc.encoder_level3(enc.down2_3(f2)), 2)
        return f3, f2, f1

    # ---- forward -----------------------------------------------------------
    def forward(self, inp_img_A, inp_img_B=None, edema_region=None, tumor_region=None, prompt=None,
                correction=None):
        """
        prompt: optional explicit prompt map in [0,1] (B,1,H,W). If None, FPG builds one
        from the masks (training / TLP-style evaluation). Pass torch.zeros_like(img) for the
        'no prompt' regime and the ground-truth map for the 'accurate' regime.
        correction: optional signed map in {-1,0,+1} (B,1,H,W) for the interactive-refinement
        pathway (only used when the model was built with correction=True); +1 = "enhance here",
        -1 = "suppress here". None or zeros leaves the output bit-identical to the plain model.
        For single_input=True, pass the single modality as inp_img_A and leave inp_img_B=None.
        """
        if prompt is None:
            prompt = self.fpg(edema_region, tumor_region)
        pyr = self.prompt_pyramid(prompt)
        use_corr = self.correction and correction is not None
        cpyr = self.prompt_pyramid(correction) if use_corr else None

        xa = torch.cat([inp_img_A, prompt], 1) if self.inject_input else inp_img_A
        pmbs_A = self.pmb_enc_A if self.enc_pmb else None
        cA = self.pmb_corr_enc_A if use_corr else None

        if self.single_input:
            f3, s2, s1 = self._encode(self.encoder_A, xa, pyr, pmbs_A, cpyr, cA)
        else:
            xb = torch.cat([inp_img_B, prompt], 1) if self.inject_input else inp_img_B
            cB = self.pmb_corr_enc_B if use_corr else None
            e3a, e2a, e1a = self._encode(self.encoder_A, xa, pyr, pmbs_A, cpyr, cA)
            e3b, e2b, e1b = self._encode(self.encoder_B, xb, pyr,
                                         self.pmb_enc_B if self.enc_pmb else None, cpyr, cB)
            e3a, e3b = self.cross_attention(e3a, e3b, pyr[2] if self.use_pca else None)
            f3 = self.fusion_level3(e3a, e3b)
            s2 = self.fusion_level2(e2a, e2b)
            s1 = self.fusion_level1(e1a, e1b)

        f3 = self.attention_spatial(f3)
        cD = self.pmb_corr_dec if (use_corr and self.dec_pmb) else None

        d3 = self.decoder_level3(f3)
        if self.dec_pmb:
            d3 = self.pmb_dec[0](d3, pyr[2])
        if cD is not None:
            d3 = cD[0](d3, cpyr[2])

        d2 = torch.cat([self.up3_2(d3), s2], 1)
        d2 = self.decoder_level2(self.reduce_chan_level2(d2))
        if self.dec_pmb:
            d2 = self.pmb_dec[1](d2, pyr[1])
        if cD is not None:
            d2 = cD[1](d2, cpyr[1])

        d1 = torch.cat([self.up2_1(d2), s1], 1)
        d1 = self.decoder_level1(d1)
        if self.dec_pmb:
            d1 = self.pmb_dec[2](d1, pyr[0])
        if cD is not None:
            d1 = cD[2](d1, cpyr[0])

        out = self.output(self.refinement(d1))
        return self.tanh(out)


def count_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


PROMPT_PATHWAY = ("pmb_", "tau_A", "tau_B", "emb_q", "emb_k")


def is_prompt_param(name):
    return name.startswith("pmb_") or any(t in name for t in PROMPT_PATHWAY)


def freeze_backbone(model, train_only=None):
    """Train only the prompt pathway; every weight inherited from TLP stays exactly as
    published. This isolates the effect of *where the prompt enters* from any change in the
    backbone, and makes the un-prompted output of every variant identical by construction.
    train_only="corr" additionally freezes the (already trained) prompt pathway and trains
    only the correction pathway, so that round 1 stays bit-identical to SPIN."""
    n_tr = n_fr = 0
    for name, p in model.named_parameters():
        if train_only == "corr":
            p.requires_grad = "pmb_corr" in name
        else:
            p.requires_grad = is_prompt_param(name)
        n_tr += p.numel() if p.requires_grad else 0
        n_fr += 0 if p.requires_grad else p.numel()
    return n_tr, n_fr


if __name__ == "__main__":
    # smoke test on CPU:  python -m model.model_spin   (run from the TLP repo root)
    torch.manual_seed(0)
    B, H, W = 2, 192, 160
    t1, t2 = torch.randn(B, 1, H, W), torch.randn(B, 1, H, W)
    edema = (torch.rand(B, 1, H, W) > 0.9).float()
    tumor = (torch.rand(B, 1, H, W) > 0.95).float()
    base = None
    for name in VARIANTS:
        g = SPIN(variant=name, device="cpu").eval()
        n = count_params(g)
        base = n if name == "tlp" else base
        with torch.no_grad():
            y = g(t1, t2, edema, tumor)
            y0 = g(t1, t2, prompt=torch.zeros(B, 1, H, W))
        print(f"multi  {name:13s} params={n/1e6:.3f}M (+{(n-base)/1e6:.3f}M vs tlp)  out={tuple(y.shape)} "
              f"range=({y.min():.2f},{y.max():.2f})  no-prompt ok={tuple(y0.shape)}")
    base = None
    for name in ("tlp", "spin"):
        g = SPIN(variant=name, single_input=True, device="cpu").eval()
        n = count_params(g)
        base = n if name == "tlp" else base
        with torch.no_grad():
            y = g(t1, edema_region=edema, tumor_region=tumor)
        print(f"single {name:13s} params={n/1e6:.3f}M (+{(n-base)/1e6:.3f}M vs tlp)  out={tuple(y.shape)}")
