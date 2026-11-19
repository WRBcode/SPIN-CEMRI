# SPIN-CEMRI

Code, patient lists, cohort-matching scripts and trained pathway weights for the manuscript
*SPIN: Steering gadolinium-free brain MRI synthesis with scale-wise prompt injection and signed corrections*.

SPIN (scale-wise prompt injection) synthesizes contrast-enhanced T1-weighted brain MRI (T1ce) from non-contrast T1w and T2w slices and lets a reader steer the synthesis with an optional mark of the lesion, which is pooled into a multi-scale pyramid that modulates every encoder and decoder level of the frozen TLP generator through bias-free blocks that reduce exactly to the identity when no mark is given, so automatic synthesis stays bit-identical. A parallel pathway, SPIN-IR, accepts signed corrections after a first synthesis: +1 to revise a lesion region and −1 to suppress false enhancement.

## Contents

| Path | What it is |
|---|---|
| `code/model_spin.py` | SPIN generator: prompt modulation blocks, correction pathway, the variants of the injection-site study. Goes into `TLP/model/`. |
| `code/data_brats.py` | Loader for BraTS 2021 and BraTS 2020 with TLP's preprocessing. Goes into `TLP/data/`. |
| `code/interaction.py` | Simulated reader that places the +1 and −1 marks for SPIN-IR. |
| `code/roi_metrics.py` | ROI-PSNR (PSNR inside the lesion dilated by 5 pixels). |
| `code/seg_unet.py`, `code/train_seg.py` | 2D U-Net for the enhancing-tumor Dice endpoint and its training on acquired T1ce. |
| `code/train_spin.py`, `code/train_ir.py` | Training of the prompt pathways (SPIN, SPIN-adv, injection-site variants) and of the correction pathways (SPIN-IR, SPIN-IR-adv). |
| `code/eval_spin.py`, `code/eval_journal.py`, `code/eval_safety.py`, `code/diag_enhancement.py`, `code/probe_leverage.py` | Evaluation: prompt regimes and runtime, per-slice metrics of all methods on identical prompts, misdirected marks, conspicuity, feature-perturbation probe. |
| `code/stats_journal.py` | Patient-level statistics and the patient-level figure. |
| `code/viz_qualitative.py`, `code/viz_safety.py` | Rule-based choice of the slices shown in the session figure and the wrong-mark figure, with the numbers quoted for them. |
| `code/check_overlap.py`, `code/overlap_detail.py` | Cohort matching of BraTS 2020 against BraTS 2021 by T1w voxel hashes. |
| `code/merge_pathway.py` | Rebuilds full generator checkpoints from the TLP weights and the files in `weights/`. |
| `data_list/` | Patient lists and the outputs of the cohort matching. |
| `weights/` | Trained prompt and correction pathways (about 1 MB each). The TLP backbone weights are not included. |

## Requirements

- Python 3.9 or newer and a CUDA GPU (the experiments used one NVIDIA RTX A6000). `probe_leverage.py` and `merge_pathway.py` run on CPU.
- The packages in `requirements.txt`: PyTorch 2.3 or newer, torchvision, NumPy, SciPy, scikit-image, scikit-learn, SimpleITK, Pillow and einops (both used by TLP's code) and Matplotlib.

The scripts run inside a clone of the TLP repository, https://github.com/ChanghuiSu/TLP, because they import its generator, fuzzy prompt generator, data loader and learning-rate schedule. Use TLP's master branch (commit `4a52832` at the time of release).

```bash
git clone https://github.com/ChanghuiSu/TLP.git
git clone https://github.com/WRBcode/SPIN-CEMRI.git
cd TLP
pip install -r ../SPIN-CEMRI/requirements.txt
cp ../SPIN-CEMRI/code/model_spin.py model/
cp ../SPIN-CEMRI/code/data_brats.py data/
cp ../SPIN-CEMRI/code/{interaction,roi_metrics,seg_unet,merge_pathway,train_spin,train_ir,train_seg,eval_spin,eval_journal,eval_safety,diag_enhancement,probe_leverage,stats_journal,viz_qualitative,viz_safety,check_overlap,overlap_detail}.py .
cp ../SPIN-CEMRI/data_list/* data_list/
cp -r ../SPIN-CEMRI/weights .
```

All commands below are run from the `TLP` folder. `data_list/Brats2021_selected.txt` is the same file as TLP's own list, so copying it changes nothing.

## TLP backbone weights

SPIN keeps TLP's released generator frozen. Download the TLP weights from the Google Drive folder linked in the TLP README and place them as follows:

```
tlp_official/G_T1T2.pth    two-input generator (T1w + T2w), used by every experiment in the manuscript
tlp_official/G_T1.pth      one-input generators, only needed for the single-input pathways
tlp_official/G_T2.pth
```

## Data

- **BraTS 2021**: the training data of the RSNA-ASNR-MICCAI BraTS 2021 challenge (Task 1, 1,251 patients), available on Synapse (https://www.synapse.org/#!Synapse:syn25829067; see also https://www.med.upenn.edu/cbica/brats2021/).
- **BraTS 2020**: the training data of the BraTS 2020 challenge (`MICCAI_BraTS2020_TrainingData`, 369 patients), available after registration at https://www.med.upenn.edu/cbica/brats2020/data.html.

Keep the original folder and file names; the cohort matching compares voxel data, so the volumes must be the released ones. Expected layout (the default paths of the scripts point to these two folders):

```
TLP/dataset/BraTS2021/BraTS2021_00000/BraTS2021_00000_{t1,t2,t1ce,flair,seg}.nii.gz
TLP/dataset/BraTS2021/...
TLP/dataset/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_{t1,t2,t1ce,flair,seg}.nii
TLP/dataset/MICCAI_BraTS2020_TrainingData/...
```

The scripts read the BraTS 2021 volumes as `.nii.gz` files and the BraTS 2020 volumes as uncompressed `.nii` files, as in the copies used for the experiments. If your BraTS 2020 download holds `.nii.gz` files, decompress them first, for example with `gunzip dataset/MICCAI_BraTS2020_TrainingData/*/*.nii.gz`; this leaves the voxel data unchanged, so the matching is not affected.

### Patient lists

| File | Content |
|---|---|
| `data_list/Brats2021_selected.txt` | TLP's 1,000 quality-filtered BraTS 2021 patients in a fixed shuffled order. Rows 0–849 are the training set, 850–869 the validation set and 870–999 the test set (130 patients). Scripts select rows with `--start_num` / `--end_num`; the defaults of the training and evaluation scripts follow this split. |
| `data_list/Brats2020_selected.txt` | 330 quality-filtered BraTS 2020 training patients, the starting point of the cohort matching. |
| `data_list/Brats2020_heldout.txt` | The 50 BraTS 2020 patients whose T1w volume matches none of the 1,000 BraTS 2021 patients: the held-out cohort. |
| `data_list/cohort_overlap.json`, `data_list/cohort_overlap_detail.json` | Outputs of the two matching scripts below. |

From every volume, the loaders take the 100 axial slices 27 to 126, crop them to 192 × 160 pixels and scale each volume linearly to [−1, 1], as in TLP. The edema label defines the edema part of a prompt, and the necrotic and enhancing labels are merged into the tumor core.

### Held-out BraTS 2020 cohort

BraTS 2021 includes the patients of earlier releases under new identifiers, so the BraTS 2020 patients were matched by image data rather than by identifier:

```bash
python check_overlap.py    # MD5 of every T1w volume, 330 BraTS 2020 vs. the 1,000 BraTS 2021 patients
                           # -> data_list/cohort_overlap.json
python overlap_detail.py   # splits the BraTS 2020 patients not found in the training rows by where they occur
                           # -> data_list/cohort_overlap_detail.json, data_list/Brats2020_heldout.txt
```

Of the 330 patients, 280 have a bit-identical T1w volume in the BraTS 2021 list (240 in the training rows, 7 in the validation rows and 33 in the test rows) and 50 match none; these 50 form the held-out cohort. Running the two scripts on the released data rewrites the three files in `data_list/` with the same content.

## Released weights

The files in `weights/` hold only the parameters that SPIN trains on top of the frozen TLP generator. `merge_pathway.py` adds them to the TLP weights and writes the full generator checkpoints that the training and evaluation scripts load (it checks every key with a strict load):

| File | Model | Merge with | Rebuilt checkpoint |
|---|---|---|---|
| `spin_prompt_pathway_T1T2.pth` | SPIN, prompt pathway trained with the ℓ1 loss | `G_T1T2.pth` | `runs/spin/G_8.pth` |
| `spin_ir_correction_pathway_T1T2.pth` | SPIN-IR, correction pathway trained on top of SPIN (marks capped at 8% of the slice) | `G_T1T2.pth` + SPIN prompt pathway | `runs/spin_ir_cap8/G_6.pth` |
| `spinadv_prompt_pathway_T1T2.pth` | SPIN-adv, prompt pathway trained with TLP's adversarial objective (objective ablation) | `G_T1T2.pth` | `runs/spin_adv/G_8.pth` |
| `spinadv_ir_correction_pathway_T1T2.pth` | SPIN-IR-adv, correction pathway trained on top of SPIN-adv | `G_T1T2.pth` + SPIN-adv prompt pathway | `runs/spin_ir_adv/G_6.pth` |
| `spin_prompt_pathway_T1.pth`, `spin_prompt_pathway_T2.pth` | Single-input SPIN (T1w → T1ce, T2w → T1ce), same ℓ1 recipe on TLP's one-input generators; not part of the results reported in the manuscript | `G_T1.pth`, `G_T2.pth` | `runs/single_t1/G_8.pth`, `runs/single_t2/G_8.pth` |

```bash
python merge_pathway.py --backbone tlp_official/G_T1T2.pth --pathway weights/spin_prompt_pathway_T1T2.pth --out runs/spin/G_8.pth
python merge_pathway.py --backbone tlp_official/G_T1T2.pth --pathway weights/spin_prompt_pathway_T1T2.pth weights/spin_ir_correction_pathway_T1T2.pth --out runs/spin_ir_cap8/G_6.pth
python merge_pathway.py --backbone tlp_official/G_T1T2.pth --pathway weights/spinadv_prompt_pathway_T1T2.pth --out runs/spin_adv/G_8.pth
python merge_pathway.py --backbone tlp_official/G_T1T2.pth --pathway weights/spinadv_prompt_pathway_T1T2.pth weights/spinadv_ir_correction_pathway_T1T2.pth --out runs/spin_ir_adv/G_6.pth
python merge_pathway.py --backbone tlp_official/G_T1.pth --pathway weights/spin_prompt_pathway_T1.pth --out runs/single_t1/G_8.pth
python merge_pathway.py --backbone tlp_official/G_T2.pth --pathway weights/spin_prompt_pathway_T2.pth --out runs/single_t2/G_8.pth
```

The rebuilt checkpoints sit where the training scripts would have written them, so every evaluation command below works with either. They are loaded by `eval_spin.py` (`--ckpt`), `eval_journal.py` and `eval_safety.py` (`--spin_ckpt`, `--ir_ckpt`), `diag_enhancement.py`, `viz_qualitative.py` and `viz_safety.py`, and by `train_ir.py` (`--init_spin`). Without a prompt, the rebuilt SPIN generators return the TLP output bit for bit, and without a correction SPIN-IR returns the SPIN output; `eval_journal.py` checks both on the first test slice that receives a correction.

The two-input SPIN and SPIN-adv files also hold two scalars, `cross_attention.tau_A` and `cross_attention.tau_B`. They belong to the `spin` variant of `model_spin.py`, scale a prompt-dependent bias of the cross-modal attention logits and have no effect without a prompt.

## Training

The prompt pathways are trained for 8 epochs with the TLP generator frozen and the correction pathways for 6 epochs on top of a frozen prompt pathway, both with TLP's learning-rate schedule (constant, then linear decay from `--decay_epoch`). The last checkpoint of every run is the one used. These are the commands of the runs in the manuscript (the defaults of `--dataroot` and of the patient rows are already set):

```bash
# SPIN (l1 objective) and the single-site variants of the injection-site study
python train_spin.py --variant spin --freeze_backbone --no_gan --init_ckpt tlp_official/G_T1T2.pth \
    --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --ckpt_dir runs/spin/
python train_spin.py --variant enc --freeze_backbone --no_gan --init_ckpt tlp_official/G_T1T2.pth \
    --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --ckpt_dir runs/enc/
python train_spin.py --variant dec --freeze_backbone --no_gan --init_ckpt tlp_official/G_T1T2.pth \
    --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --ckpt_dir runs/dec/

# SPIN-adv: the same pathway with TLP's objective (least-squares GAN + 100 x l1)
python train_spin.py --variant spin --freeze_backbone --init_ckpt tlp_official/G_T1T2.pth \
    --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --ckpt_dir runs/spin_adv/

# SPIN-IR and SPIN-IR-adv
python train_ir.py --gpu_id 0 --init_backbone tlp_official/G_T1T2.pth --init_spin runs/spin/G_8.pth \
    --max_frac 0.08 --ckpt_dir runs/spin_ir_cap8/
python train_ir.py --gpu_id 0 --init_backbone tlp_official/G_T1T2.pth --init_spin runs/spin_adv/G_8.pth \
    --max_frac 0.08 --gan --ckpt_dir runs/spin_ir_adv/

# enhancing-tumor segmenter for the Dice endpoint, trained on acquired T1ce (best validation epoch is kept)
python train_seg.py --out runs/_seg

# single-input pathways (not used in the manuscript)
python train_spin.py --variant spin --single_input --modality t1 --freeze_backbone --no_gan \
    --init_ckpt tlp_official/G_T1.pth --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --ckpt_dir runs/single_t1/
python train_spin.py --variant spin --single_input --modality t2 --freeze_backbone --no_gan \
    --init_ckpt tlp_official/G_T2.pth --n_epochs 8 --decay_epoch 4 --checkpoint_interval 1 --ckpt_dir runs/single_t2/
```

`train_spin.py` and `train_ir.py` print the number of trainable parameters and their share of the model at the start of each run. The weights of the single-site variants (`enc`, `dec`) and of the segmenter are not released; the commands above retrain them. All three training scripts fix their random seeds, but GPU training is not bit-wise deterministic, so retrained models can differ slightly from the ones used for the manuscript.

## Evaluation

The commands below recompute the reported metrics on the BraTS 2021 test patients and on the 50 held-out BraTS 2020 patients. They need the checkpoints in `runs/` (trained or rebuilt) and the segmenter in `runs/_seg/best.pth`. Results that involve a retrained model (the `enc` and `dec` rows of the injection-site table, enhancing-tumor Dice and the other segmenter-based measures) can differ slightly from the reported values. Within each script all models receive the same prompts, and `eval_journal.py`, `eval_safety.py` and `diag_enhancement.py` replay the coarse-prompt stream of `eval_spin.py`.

```bash
# 1. Four prompt regimes (none, coarse, accurate, incorrect) per model: injection-site table, coarse vs.
#    accurate prompt, and the ROI-PSNR change under a misplaced prompt
python eval_spin.py --variant tlp  --ckpt tlp_official/G_T1T2.pth --out runs/tlp/eval.json
python eval_spin.py --variant spin --ckpt runs/spin/G_8.pth       --out runs/spin/eval_G8.json
python eval_spin.py --variant spin --ckpt runs/spin_adv/G_8.pth   --out runs/spin_adv/eval_G8.json
python eval_spin.py --variant enc  --ckpt runs/enc/G_8.pth        --out runs/enc/eval_G8.json
python eval_spin.py --variant dec  --ckpt runs/dec/G_8.pth        --out runs/dec/eval_G8.json

# 2. All methods on identical slices and prompts with per-slice metrics (PSNR, SSIM, NMSE, ROI-PSNR,
#    enhancing-tumor Dice, PSNR inside the +1 and -1 marks); the *_adv runs use SPIN-adv and SPIN-IR-adv
python eval_journal.py --out runs/_journal/internal
python eval_journal.py --cohort brats2020 --dataroot dataset/MICCAI_BraTS2020_TrainingData \
    --data_list data_list/Brats2020_heldout.txt --start_num 0 --end_num 50 --out runs/_journal/heldout2020
python eval_journal.py --spin_ckpt runs/spin_adv/G_8.pth --ir_ckpt runs/spin_ir_adv/G_6.pth \
    --expect runs/spin_adv/eval_G8.json --out runs/_journal/internal_adv
python eval_journal.py --cohort brats2020 --dataroot dataset/MICCAI_BraTS2020_TrainingData \
    --data_list data_list/Brats2020_heldout.txt --start_num 0 --end_num 50 \
    --spin_ckpt runs/spin_adv/G_8.pth --ir_ckpt runs/spin_ir_adv/G_6.pth --out runs/_journal/heldout2020_adv

# 3. Misdirected interactions: misplaced prompt, misplaced +1, wrong -1
python eval_safety.py --out runs/_journal/safety
python eval_safety.py --spin_ckpt runs/spin_adv/G_8.pth --ir_ckpt runs/spin_ir_adv/G_6.pth --out runs/_journal/safety_adv

# 4. Conspicuity: enhancing-tumor contrast, sharpness, segmenter recall, precision and area ratio
python diag_enhancement.py --extra adv=runs/spin_adv/G_8.pth --out runs/_journal/diag_enhancement_full.json

# 5. Patient-level statistics (per-patient means, Wilcoxon signed-rank test, bootstrap CI of the median
#    difference, rank-biserial r, Holm adjustment) and the patient-level figure
python stats_journal.py --dir runs/_journal --out runs/_journal/stats.json --fig runs/_journal/fig_patients.pdf

# 6. Inference time per slice over 1,000 test slices
python eval_spin.py --variant spin --ckpt runs/spin/G_8.pth --time_only --regimes none --end_num 880 --out runs/spin/time.json
python eval_spin.py --variant tlp --ckpt tlp_official/G_T1T2.pth --time_only --regimes none --end_num 880 --out runs/tlp/time.json

# 7. Feature-perturbation probe of the frozen backbone (5% Gaussian noise at one site at a time; CPU)
python probe_leverage.py --ckpt tlp_official/G_T1T2.pth

# 8. Slices of the session figure and of the wrong-mark figure, chosen by rule (needs safety_adv.npz from step 3)
python viz_qualitative.py --out runs/_journal/fig_qualitative.pdf
python viz_safety.py --out runs/_journal/fig_safety.pdf
```

Where the reported results come from:

| Result | Script and output |
|---|---|
| Comparison with baseline methods, TLP and SPIN rows (slice-level means) | `eval_journal.py`, `internal.json` |
| Patient-level comparison with TLP, held-out cohort, lesion-size tertiles | `stats_journal.py` on the outputs of `eval_journal.py` |
| Injection site, coarse vs. accurate prompt, misplaced prompt (ROI-PSNR change) | `eval_spin.py` |
| Interactive refinement and the prompt-only alternative | `eval_journal.py` (`ir_r2`, `ir_naive`) and `stats_journal.py` |
| Misdirected interactions | `eval_safety.py`, `safety.json` and `safety_adv.json` |
| Objective ablation | `eval_journal.py` (`*_adv`), `diag_enhancement.py`, `eval_safety.py` (`safety_adv`), `stats_journal.py` |
| Segmenter validation Dice | log of `train_seg.py` |
| Parameter counts | logs of `train_spin.py` and `train_ir.py` |
| Inference time | `eval_spin.py --time_only` |
| Feature-perturbation probe | `probe_leverage.py` |
| Slices and numbers of the session figure and the wrong-mark figure | `viz_qualitative.py`, `viz_safety.py` |

The rows of Hi-Net, MMgSN-Net, ResViT and DTF-Net in the comparison with baseline methods are the benchmark results that TLP reports on the identical split and metric code; they are not recomputed here. The single-input pathways can be evaluated with `eval_spin.py --single_input --modality t1` (or `t2`) and the matching checkpoint.

## Code from TLP

`model_spin.py`, `data_brats.py` and `train_spin.py` contain code adapted from TLP (https://github.com/ChanghuiSu/TLP), which is released under the MIT License; each of these files keeps TLP's copyright and license notice in its header. The other scripts import TLP's modules and follow its preprocessing and metric conventions (`data/data_2021.py`, `evaluation.py`).

## Citation

The manuscript is under review.

Ruibo Wang\*, Xuegang Hu\*, Changhui Su\*. SPIN: Steering gadolinium-free brain MRI synthesis with scale-wise prompt injection and signed corrections. (\*These authors contributed equally.)

```bibtex
@unpublished{wang2026spin,
  title  = {{SPIN}: Steering gadolinium-free brain {MRI} synthesis with scale-wise prompt injection and signed corrections},
  author = {Wang, Ruibo and Hu, Xuegang and Su, Changhui},
  note   = {Manuscript under review},
  year   = {2026}
}
```

SPIN builds on the generator, fuzzy prompt generator, patient list and released weights of TLP:

```bibtex
@inproceedings{li2026tlp,
  title     = {Interactive Gadolinium-Free {MRI} Synthesis: A Transformer with Localization Prompts},
  author    = {Li, Linhao and Su, Changhui and Guo, Yu and Zhang, Huimao and Liang, Dong and Shang, Kun},
  booktitle = {Pattern Recognition (ICPR 2026), Part IV, LNCS},
  publisher = {Springer},
  pages     = {418--432},
  year      = {2026},
  doi       = {10.1007/978-3-032-31663-9_28}
}
```

Please also cite the BraTS references requested on the data pages when you use the data.
