# -*- coding: utf-8 -*-
# The preprocessing in this file is adapted from data/data_2021.py of TLP
# (https://github.com/ChanghuiSu/TLP), which is distributed under the MIT License reproduced below.
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
BraTS loader covering both cohorts used in the paper.

Place this file in TLP/data/ (it is imported as data.data_brats). TLP's data/data_2021.py, which
train_spin.py and eval_spin.py use, is left untouched, so results computed with it stay
reproducible byte-for-byte; this module adds BraTS 2020, whose files are named differently
(`BraTS20_Training_269/BraTS20_Training_269_t1.nii`, uncompressed) from 2021
(`BraTS2021_00195/BraTS2021_00195_t1.nii.gz`). Preprocessing is identical: crop 240x240 to
192x160, per-volume scaling to [-1,1], slices 27..126, and the segmentation split into an
edema map (label 2) and a merged tumour-core map (labels 1 and 4), since ET and NCR cannot be
told apart before T1ce exists.
"""
import os

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

COHORTS = {
    # name:        (directory template, file template, suffix)
    "brats2021": ("BraTS2021_{id}", "BraTS2021_{id}_{mod}", ".nii.gz"),
    "brats2020": ("BraTS20_Training_{id}", "BraTS20_Training_{id}_{mod}", ".nii"),
}


def _path(root, cohort, subject_id, modality):
    d, f, ext = COHORTS[cohort]
    return os.path.join(root, d.format(id=subject_id),
                        f.format(id=subject_id, mod=modality) + ext)


def _ids(list_path, start, end):
    with open(list_path) as fh:
        ids = [l.strip() for l in fh if l.strip()]
    return ids[start:end]


def _load_modality(root, cohort, modality, list_path, start, end, s0, s1):
    out = []
    for sid in _ids(list_path, start, end):
        img = sitk.GetArrayFromImage(sitk.ReadImage(_path(root, cohort, sid, modality)))
        img = img[:, 34:226, 40:-40]
        lo, hi = img.min(), img.max()
        for z in range(s0, s1):
            sl = (img[z] - lo) / (hi - lo) * 2 - 1
            out.append([transforms.ToTensor()(sl).to(torch.float32), f"{sid}_{z}"])
    return out


def _load_seg(root, cohort, list_path, start, end, s0, s1):
    out, core = [], torch.tensor([1, 4])
    for sid in _ids(list_path, start, end):
        seg = sitk.GetArrayFromImage(sitk.ReadImage(_path(root, cohort, sid, "seg")))
        seg = seg[:, 34:226, 40:-40]
        for z in range(s0, s1):
            s = torch.from_numpy(seg[z].astype(np.float32)).unsqueeze(0)
            edema, tumour = torch.zeros_like(s), torch.zeros_like(s)
            edema[s == 2] = 1.0
            tumour[torch.isin(s, core)] = 1.0
            out.append([edema, tumour])
    return out


class LoadData(Dataset):
    def __init__(self, root, modalities, data_list, start_num=0, end_num=None,
                 start_slice=0, end_slice=155, cohort="brats2021"):
        super().__init__()
        assert cohort in COHORTS, f"unknown cohort {cohort}"
        self.modalities = modalities
        self.data = {m: _load_modality(root, cohort, m, data_list, start_num, end_num,
                                       start_slice, end_slice) for m in modalities}
        self.data["seg"] = _load_seg(root, cohort, data_list, start_num, end_num,
                                     start_slice, end_slice)
        print(f"num_Subjects: {len(self.data[self.modalities[0]])}", flush=True)

    def __getitem__(self, index):
        d = {m: self.data[m][index] for m in self.modalities}
        d["seg"] = self.data["seg"][index]
        return d

    def __len__(self):
        return len(self.data[self.modalities[0]])
