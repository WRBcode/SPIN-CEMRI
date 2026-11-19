# -*- coding: utf-8 -*-
"""
Per-patient statistics and the patient-level figure (Supplementary Fig. S1 of the paper).

Inputs (one directory): internal.npz / heldout2020.npz (eval_journal.py), single_{t1,t2}_{tlp,spin}.npz
(eval_spin.py --dump), safety.json (eval_safety.py). Sections whose inputs are missing are skipped.

The unit of analysis is the patient: slices of one patient are not independent, so every slice
metric is first averaged within the patient (over the slices where it is defined), and methods
are compared on those per-patient means with the two-sided Wilcoxon signed-rank test. For each
comparison we report the mean and the median of the per-patient differences (median with a 95%
bootstrap CI, 10,000 patient-level resamples), the share of patients that improve, the
matched-pairs rank-biserial correlation, and p values Holm-adjusted within each table.

    python stats_journal.py --dir runs/_journal --out runs/_journal/stats.json --fig fig_patients.pdf
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fig", default="")
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--fig_suffix", default="", help="'_adv' plots the adversarially trained SPIN (internal_adv / heldout2020_adv)")
ap.add_argument("--spin_label", default="SPIN", help="name of the plotted SPIN variant in the figure")
args = ap.parse_args()
rng = np.random.default_rng(0)
LOWER_IS_BETTER = {"nmse"}


def load(name):
    p = os.path.join(args.dir, name)
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def per_patient(v, sid):
    subs = np.unique(sid)
    out = np.full(len(subs), np.nan)
    for k, s in enumerate(subs):
        x = v[sid == s]
        x = x[np.isfinite(x)]
        if x.size:
            out[k] = x.mean()
    return subs, out


def rank_biserial(d):
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(a, b, sid, metric, mask=None):
    """b vs a (b - a), per patient; `mask` restricts the slices first."""
    if mask is not None:
        a, b, sid = a[mask], b[mask], sid[mask]
    both = np.isfinite(a) & np.isfinite(b)
    _, pa = per_patient(np.where(both, a, np.nan), sid)
    _, pb = per_patient(np.where(both, b, np.nan), sid)
    ok = np.isfinite(pa) & np.isfinite(pb)
    pa, pb = pa[ok], pb[ok]
    d = pb - pa
    better = d < 0 if metric in LOWER_IS_BETTER else d > 0
    boot = np.median(d[rng.integers(0, d.size, (args.n_boot, d.size))], axis=1)
    p = stats.wilcoxon(pb, pa, zero_method="wilcox", alternative="two-sided").pvalue if np.any(d != 0) else 1.0
    return {"n_patients": int(d.size), "a_mean": float(pa.mean()), "b_mean": float(pb.mean()),
            "diff_mean": float(d.mean()), "diff_median": float(np.median(d)),
            "diff_median_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "share_improved": float(better.mean()),
            "rank_biserial": -rank_biserial(d) if metric in LOWER_IS_BETTER else rank_biserial(d),   # > 0 favours b
            "p": float(p)}


def holm(tab):
    keys = sorted(tab, key=lambda k: tab[k]["p"])
    m, run = len(keys), 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * tab[k]["p"]))
        tab[k]["p_holm"] = run
    return tab


def slice_means(D, methods, metrics):
    return {f"{m}.{k}": float(np.nanmean(D[f"{m}__{k}"])) for m in methods for k in metrics
            if f"{m}__{k}" in D and np.isfinite(D[f"{m}__{k}"]).any()}


def stratify(D, sid):
    """(4) lesion-size strata: ROI-PSNR gain of SPIN over TLP (coarse) by tertile of lesion area."""
    les = D["n_roi"] > 0
    q = np.percentile(D["n_roi"][les], [100 / 3, 200 / 3])
    strata = {"small": les & (D["n_roi"] <= q[0]), "medium": les & (D["n_roi"] > q[0]) & (D["n_roi"] <= q[1]),
              "large": les & (D["n_roi"] > q[1])}
    return {"size_tertiles_px": [float(x) for x in q],
            "by_size": holm({s: compare(D["tlp_coarse__roi"], D["spin_coarse__roi"], sid, "roi", m) for s, m in strata.items()}),
            "by_size_gain_over_none": {s: {"tlp": compare(D["tlp_none__roi"], D["tlp_coarse__roi"], sid, "roi", m)["diff_mean"],
                                           "spin": compare(D["tlp_none__roi"], D["spin_coarse__roi"], sid, "roi", m)["diff_mean"]}
                                       for s, m in strata.items()}}


res = {}
PRIMARY = ["roi", "psnr", "ssim", "nmse", "dice"]
COHORTS = ("internal", "heldout2020", "internal_adv", "heldout2020_adv")   # *_adv: SPIN/SPIN-IR trained with TLP's adversarial objective
for cohort in COHORTS:
    D = load(f"{cohort}.npz")
    if D is None:
        continue
    sid = D["sid"]
    R = {"n_slices": int(sid.size), "n_patients": int(np.unique(sid).size),
         "n_roi_slices": int(np.isfinite(D["tlp_none__roi"]).sum()), "n_et_slices": int((D["n_et"] > 0).sum()),
         "slice_means": slice_means(D, ["tlp_none", "tlp_coarse", "spin_coarse", "tlp_acc", "spin_acc",
                                        "ir_naive", "ir_r2", "real"], PRIMARY)}
    # (1) SPIN vs TLP given the same coarse prompt -- the paper's primary comparison
    R["spin_vs_tlp_coarse"] = holm({k: compare(D[f"tlp_coarse__{k}"], D[f"spin_coarse__{k}"], sid, k) for k in PRIMARY})
    R["spin_vs_tlp_accurate"] = holm({k: compare(D[f"tlp_acc__{k}"], D[f"spin_acc__{k}"], sid, k) for k in PRIMARY})
    # (2) what a coarse prompt buys each model (gain over the shared un-prompted output)
    R["tlp_coarse_vs_none"] = holm({k: compare(D[f"tlp_none__{k}"], D[f"tlp_coarse__{k}"], sid, k) for k in PRIMARY})
    R["spin_coarse_vs_none"] = holm({k: compare(D[f"tlp_none__{k}"], D[f"spin_coarse__{k}"], sid, k) for k in PRIMARY})
    # (3) interactive refinement, on the slices where a correction was simulated (absent in --no_ir runs)
    irm = np.isfinite(D["ir_r2__psnr"])
    if not irm.any():
        R["ir_slices"] = 0
        R.update(stratify(D, sid)); res[cohort] = R
        continue
    R["ir_slices"] = int(irm.sum())
    R["ir_slice_means"] = {f"{m}.{k}": float(np.nanmean(D[f"{m}__{k}"][irm]))
                           for m in ("spin_coarse", "ir_naive", "ir_r2") for k in ("psnr", "roi", "pos", "neg", "dice")}
    R["ir_vs_round1"] = holm({k: compare(D[f"spin_coarse__{k}"], D[f"ir_r2__{k}"], sid, k, irm)
                              for k in ("neg", "pos", "roi", "psnr", "dice")})
    R["ir_vs_naive"] = holm({k: compare(D[f"ir_naive__{k}"], D[f"ir_r2__{k}"], sid, k, irm)
                             for k in ("neg", "pos", "roi", "psnr", "dice")})
    R["naive_vs_round1"] = holm({k: compare(D[f"spin_coarse__{k}"], D[f"ir_naive__{k}"], sid, k, irm)
                                 for k in ("neg", "pos", "roi", "psnr", "dice")})
    R["coverage"] = {"pos_mean": float(np.nanmean(D["cov_pos"])), "neg_mean": float(np.nanmean(D["cov_neg"]))}
    R.update(stratify(D, sid))
    res[cohort] = R



for mod in ("t1", "t2"):
    T, S = load(f"single_{mod}_tlp.npz"), load(f"single_{mod}_spin.npz")
    if T is None or S is None:
        continue
    assert (T["sid"] == S["sid"]).all()
    sid = T["sid"]
    res[f"single_{mod}"] = {
        "slice_means": {"none." + k: float(np.nanmean(T[f"none__{k}"])) for k in ("psnr", "ssim", "nmse", "roi_psnr")}
        | {"tlp_coarse." + k: float(np.nanmean(T[f"coarse__{k}"])) for k in ("psnr", "ssim", "nmse", "roi_psnr")}
        | {"spin_coarse." + k: float(np.nanmean(S[f"coarse__{k}"])) for k in ("psnr", "ssim", "nmse", "roi_psnr")},
        "spin_vs_tlp_coarse": holm({k: compare(T[f"coarse__{k}"], S[f"coarse__{k}"], sid, k.replace("_psnr", ""))
                                    for k in ("roi_psnr", "psnr", "ssim", "nmse")})}

for name in ("diag_enhancement_full", "diag_enhancement_adv", "diag_enhancement"):   # most complete first
    dp = os.path.join(args.dir, f"{name}.json")
    if os.path.exists(dp):
        res["diag"] = json.load(open(dp)); res["diag_source"] = name
        break
for name in ("safety", "safety_adv"):
    sp = os.path.join(args.dir, f"{name}.json")
    if os.path.exists(sp):
        res[name] = json.load(open(sp))
# L1-trained vs adversarially trained SPIN on the same slices and prompts (paired, per patient)
for cohort in ("internal", "heldout2020"):
    A, V = load(f"{cohort}.npz"), load(f"{cohort}_adv.npz")
    if A is None or V is None:
        continue
    assert (A["sid"] == V["sid"]).all() and np.array_equal(A["tlp_coarse__psnr"], V["tlp_coarse__psnr"], equal_nan=True)
    res[f"{cohort}_adv_vs_l1"] = holm({k: compare(A[f"spin_coarse__{k}"], V[f"spin_coarse__{k}"], A["sid"], k)
                                       for k in PRIMARY})

json.dump(res, open(args.out, "w"), indent=1)


def fmt(c, k):
    x = c[k]
    return (f"{k:5s} n={x['n_patients']:3d}  {x['a_mean']:.4f} -> {x['b_mean']:.4f}  mean d={x['diff_mean']:+.4f}  "
            f"median d={x['diff_median']:+.4f} [{x['diff_median_ci95'][0]:+.4f},{x['diff_median_ci95'][1]:+.4f}]  "
            f"improved {100 * x['share_improved']:.0f}%  r={x['rank_biserial']:+.2f}  p_holm={x['p_holm']:.2g}")


def is_cmp_table(v):
    return isinstance(v, dict) and v and all(isinstance(x, dict) and "p" in x for x in v.values())


for cohort, R in res.items():
    if not isinstance(R, dict) or cohort.startswith(("safety", "diag")):
        continue
    print(f"\n===== {cohort}")
    tables = {cohort: R} if is_cmp_table(R) else {t: v for t, v in R.items() if is_cmp_table(v)}
    for tab, v in tables.items():
        print(f"-- {tab}")
        for k in v:
            print("   " + fmt(v, k))
print("\nsaved", args.out)

# ---------------- patient-level figure ----------------
if args.fig and f"internal{args.fig_suffix}" in res:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for cohort, mk in (("internal", "o"), ("heldout2020", "^")):
        D = load(f"{cohort}{args.fig_suffix}.npz")
        if D is None:
            continue
        sid = D["sid"]
        lab = "BraTS 2021 test (n=%d)" % len(np.unique(sid)) if cohort == "internal" else "BraTS 2020 held-out (n=%d)" % len(np.unique(sid))
        for j, k in enumerate(("roi", "dice")):
            a, b = D[f"tlp_coarse__{k}"], D[f"spin_coarse__{k}"]
            both = np.isfinite(a) & np.isfinite(b)
            _, pa = per_patient(np.where(both, a, np.nan), sid)
            _, pb = per_patient(np.where(both, b, np.nan), sid)
            ok = np.isfinite(pa) & np.isfinite(pb)
            ax[j].scatter(pa[ok], pb[ok], s=7, marker=mk, facecolors="none" if mk == "^" else "#1f5aa6",
                          edgecolors="#1f5aa6" if mk == "o" else "#c0392b", linewidths=0.6, alpha=0.8, label=lab)
    for j, (k, t) in enumerate((("roi", "ROI-PSNR (dB)"), ("dice", "ET Dice"))):
        lo, hi = ax[j].get_xlim()[0], ax[j].get_xlim()[1]
        lo, hi = min(lo, ax[j].get_ylim()[0]), max(hi, ax[j].get_ylim()[1])
        ax[j].plot([lo, hi], [lo, hi], "k--", lw=0.6); ax[j].set_xlim(lo, hi); ax[j].set_ylim(lo, hi)
        ax[j].set_aspect("equal", adjustable="box")
        ax[j].set_xlabel(f"TLP {t}"); ax[j].set_ylabel(f"{args.spin_label} {t}")
        ax[j].set_title("(a) lesion-region PSNR per patient" if j == 0 else "(b) enhancing-tumor Dice per patient", fontsize=7)
    ax[0].legend(frameon=False, fontsize=6, loc="lower right")
    R = res[f"internal{args.fig_suffix}"]["by_size_gain_over_none"]
    xs = np.arange(3)
    ax[2].bar(xs - 0.18, [R[s]["tlp"] for s in ("small", "medium", "large")], 0.36, color="#9fb7d9", label="TLP (input only)")
    ax[2].bar(xs + 0.18, [R[s]["spin"] for s in ("small", "medium", "large")], 0.36, color="#1f5aa6", label=args.spin_label)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(["small", "medium", "large"]); ax[2].set_xlabel("lesion area (tertile)")
    ax[2].set_ylabel("ROI-PSNR gain of a coarse prompt (dB)"); ax[2].legend(frameon=False, fontsize=6)
    ax[2].set_title("(c) prompt gain by lesion size", fontsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.fig, bbox_inches="tight"); fig.savefig(args.fig.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", args.fig)
