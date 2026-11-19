# -*- coding: utf-8 -*-
"""Train the enhancing-tumour segmenter on REAL T1ce (BraTS 2021 train split)."""
import argparse, json, os, time
import numpy as np, torch, SimpleITK as sitk
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from seg_unet import UNet, dice_loss, dice_score

ap = argparse.ArgumentParser()
ap.add_argument("--dataroot", default="./dataset/BraTS2021")
ap.add_argument("--data_list", default="./data_list/Brats2021_selected.txt")
ap.add_argument("--start_num", type=int, default=0); ap.add_argument("--end_num", type=int, default=850)
ap.add_argument("--val_start", type=int, default=850); ap.add_argument("--val_end", type=int, default=870)
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--gpu_id", default="0")
ap.add_argument("--out", default="./runs/_seg"); ap.add_argument("--n_cpu", type=int, default=8)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.manual_seed(0); np.random.seed(0)
os.makedirs(args.out, exist_ok=True)
def log(m):
    print(m, flush=True); open(os.path.join(args.out, "log.txt"), "a").write(m + "\n")

class T1ceSeg(Dataset):
    """real T1ce slice -> enhancing-tumour (label 4) mask, same crop/slices/scaling as synthesis."""
    def __init__(self, root, lst, a, b, s0=27, s1=127):
        ids = [l.strip() for l in open(lst) if l.strip()][a:b]
        self.x, self.y = [], []
        for sid in ids:
            img = sitk.GetArrayFromImage(sitk.ReadImage(f"{root}/BraTS2021_{sid}/BraTS2021_{sid}_t1ce.nii.gz"))[:, 34:226, 40:-40]
            seg = sitk.GetArrayFromImage(sitk.ReadImage(f"{root}/BraTS2021_{sid}/BraTS2021_{sid}_seg.nii.gz"))[:, 34:226, 40:-40]
            lo, hi = img.min(), img.max()
            for z in range(s0, s1):
                self.x.append(transforms.ToTensor()((img[z] - lo) / (hi - lo) * 2 - 1).to(torch.float32))
                self.y.append(torch.from_numpy((seg[z] == 4).astype(np.float32)).unsqueeze(0))
        log(f"  {len(self.x)} slices, {sum(1 for t in self.y if t.sum()>0)} with enhancing tumour")
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

log(f"=== {time.strftime('%F %T')} {vars(args)}")
tr = T1ceSeg(args.dataroot, args.data_list, args.start_num, args.end_num)
va = T1ceSeg(args.dataroot, args.data_list, args.val_start, args.val_end)
dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True, pin_memory=True)
vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=4)
dev = "cuda"; net = UNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
bce = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda")
best = -1.0
for ep in range(args.epochs):
    net.train(); t0 = time.time(); acc = []
    for i, (x, y) in enumerate(dl):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            lo = net(x); loss = bce(lo, y) + dice_loss(lo, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); acc.append(loss.item())
        if i % 300 == 0: log(f"  ep{ep+1}/{args.epochs} it{i}/{len(dl)} loss {np.mean(acc[-300:]):.4f} {(i+1)/max(time.time()-t0,1e-9):.2f} it/s")
    sch.step()
    net.eval(); ds = []
    with torch.no_grad():
        for x, y in vl:
            p = (torch.sigmoid(net(x.to(dev))) > 0.5).float().cpu().numpy()[0, 0]
            d = dice_score(p, y.numpy()[0, 0])
            if not np.isnan(d): ds.append(d)
    m = float(np.mean(ds))
    log(f"epoch {ep+1}: train_loss {np.mean(acc):.4f}  val_Dice(real T1ce) {m:.4f} (n={len(ds)})  [{(time.time()-t0)/60:.0f} min]")
    torch.save(net.state_dict(), os.path.join(args.out, "last.pth"))
    if m > best: best = m; torch.save(net.state_dict(), os.path.join(args.out, "best.pth")); log(f"  new best {best:.4f}")
log(f"DONE best val Dice on real T1ce = {best:.4f}")
