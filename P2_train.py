import os
import sys
import collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from PIL import Image

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from losses import FocalLoss, BinaryFocalLoss, EMDLoss
from dataset import XBDJointDataset
from P2_model import JointDamageNet

from dataset import TILE_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = "xbd"
TRAIN_SPLITS    = ["tier1", "tier3"]
VAL_SPLITS      = ["hold"]
BATCH_SIZE      = 8
NUM_WORKERS     = 4
LR              = 1e-4
ENCODER_LR_MULT = 0.5        # up from 0.3 — encoder needs more adaptation
EPOCHS          = 50
WARMUP_EPOCHS   = 5
FOCAL_GAMMA     = 2.0
LOC_LOSS_WEIGHT = 1.0
DMG_LOSS_WEIGHT = 1.0
EMD_LOSS_WEIGHT = 0.5        # ordinal penalty on top of focal
DROPOUT         = 0.3
DMG_CLASSES     = 5          # 0=bg 1=no-dmg 2=minor 3=major 4=destroyed
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH       = "best_model_p2.pth"
PHASE1_CKPT     = "best_model_p1.pth"
# ──────────────────────────────────────────────────────────────────────────────


def compute_dmg_weights(dataset):
    """Inverse-frequency class weights over tile-cropped post masks only."""
    counts = collections.Counter()
    for _, _, _, post_mask_path, r, c in dataset.tiles:
        arr = np.array(Image.open(post_mask_path))[r:r + TILE_SIZE, c:c + TILE_SIZE]
        for v in range(1, DMG_CLASSES):
            counts[v] += int((arr == v).sum())
    total   = sum(counts.values())
    weights = torch.tensor(
        [total / ((DMG_CLASSES - 1) * max(counts[v], 1)) for v in range(1, DMG_CLASSES)],
        dtype=torch.float32
    )
    weights = weights / weights.sum() * (DMG_CLASSES - 1)
    return torch.cat([torch.zeros(1), weights])  # index 0 = background (ignored)


def build_sample_weights(dataset):
    """Per-tile weights for WeightedRandomSampler.

    Tiles containing minority damage classes (2=minor, 3=major, 4=destroyed)
    are upsampled 3× relative to no-damage-only tiles.
    """
    weights = []
    for _, _, _, post_mask_path, r, c in dataset.tiles:
        tile = np.array(Image.open(post_mask_path))[r:r + TILE_SIZE, c:c + TILE_SIZE]
        has_minority = np.any((tile == 2) | (tile == 3))
        weights.append(3.0 if has_minority else 1.0)
    return weights


def compute_f1(tp, fp, fn):
    return [2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-8) for c in range(1, DMG_CLASSES)]


def run_epoch(model, loader, loc_criterion, dmg_criterion, emd_criterion,
              optimizer, scaler, train, epoch, num_epochs):
    model.train(train)
    phase = "Train" if train else "Val  "

    total_loc = total_dmg = total_emd = 0.0
    n         = 0
    tp = [0.0] * DMG_CLASSES
    fp = [0.0] * DMG_CLASSES
    fn = [0.0] * DMG_CLASSES

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{num_epochs} {phase}", leave=False)

    with torch.set_grad_enabled(train):
        for pre, post, loc_mask, dmg_mask in pbar:
            pre      = pre.to(DEVICE)
            post     = post.to(DEVICE)
            loc_mask = loc_mask.to(DEVICE)
            dmg_mask = dmg_mask.to(DEVICE)

            with autocast():
                loc_out, dmg_out = model(pre, post)
                loc_loss = loc_criterion(loc_out, loc_mask)
                dmg_loss = dmg_criterion(dmg_out, dmg_mask)
                emd_loss = emd_criterion(dmg_out, dmg_mask)
                loss     = (LOC_LOSS_WEIGHT * loc_loss
                            + DMG_LOSS_WEIGHT * dmg_loss
                            + EMD_LOSS_WEIGHT * emd_loss)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            bs    = pre.size(0)
            preds = dmg_out.argmax(1)
            mask  = dmg_mask > 0

            for c in range(1, DMG_CLASSES):
                pred_c = (preds == c) & mask
                true_c = dmg_mask == c
                tp[c] += float((pred_c &  true_c).sum())
                fp[c] += float((pred_c & ~true_c).sum())
                fn[c] += float((~pred_c & true_c & mask).sum())

            total_loc += loc_loss.item() * bs
            total_dmg += dmg_loss.item() * bs
            total_emd += emd_loss.item() * bs
            n         += bs

            f1s     = compute_f1(tp, fp, fn)
            mean_f1 = sum(f1s) / len(f1s)
            pbar.set_postfix(loc=f"{total_loc/n:.4f}", dmg=f"{total_dmg/n:.4f}",
                             emd=f"{total_emd/n:.4f}", mF1=f"{mean_f1:.3f}")

    f1s     = compute_f1(tp, fp, fn)
    mean_f1 = sum(f1s) / len(f1s)
    return total_loc / n, total_dmg / n, total_emd / n, f1s, mean_f1


def main():
    print(f"Device: {DEVICE}")

    train_ds = XBDJointDataset(TRAIN_SPLITS, BASE_DIR, augment=True)
    val_ds   = XBDJointDataset(VAL_SPLITS,   BASE_DIR, augment=False)
    print(f"Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}")

    print("Computing damage class weights...")
    dmg_weights = compute_dmg_weights(train_ds).to(DEVICE)
    print("Damage class weights:", [f"{w:.3f}" for w in dmg_weights.tolist()])

    print("Building weighted sampler for minority class upsampling...")
    sample_weights = build_sample_weights(train_ds)
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                           replacement=True)

    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = JointDamageNet(loc_classes=1, dmg_classes=DMG_CLASSES, dropout=DROPOUT).to(DEVICE)
    model.load_phase1_weights(PHASE1_CKPT, DEVICE)

    loc_criterion = BinaryFocalLoss(gamma=FOCAL_GAMMA)
    dmg_criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=dmg_weights, ignore_index=0)
    emd_criterion = EMDLoss(num_classes=DMG_CLASSES, ignore_index=0)

    optimizer = torch.optim.AdamW([
        {"params": model.low_level.parameters(),       "lr": LR * ENCODER_LR_MULT},
        {"params": model.high_level.parameters(),      "lr": LR * ENCODER_LR_MULT},
        {"params": model.dmg_reduce_high.parameters(), "lr": LR},
        {"params": model.dmg_reduce_low.parameters(),  "lr": LR},
        {"params": model.loc_head.parameters(),        "lr": LR},
        {"params": model.dmg_head.parameters(),        "lr": LR},
    ], weight_decay=1e-4)

    scaler = GradScaler()

    best_val_f1 = 0.0
    start_epoch = 1

    warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                   total_iters=WARMUP_EPOCHS)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine],
                                                       milestones=[WARMUP_EPOCHS])

    if os.path.exists(SAVE_PATH):
        print(f"Resuming from checkpoint: {SAVE_PATH}")
        ckpt        = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_f1 = ckpt.get("val_mean_f1", 0.0)
        start_epoch = ckpt["epoch"] + 1
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"Resumed at epoch {start_epoch}, best val mF1: {best_val_f1:.4f}")

    class_names = ["no-dmg", "minor", "major", "destr"]

    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loc, tr_dmg, tr_emd, tr_f1s, tr_mf1 = run_epoch(
            model, train_loader, loc_criterion, dmg_criterion, emd_criterion,
            optimizer, scaler, train=True, epoch=epoch, num_epochs=EPOCHS)

        vl_loc, vl_dmg, vl_emd, vl_f1s, vl_mf1 = run_epoch(
            model, val_loader, loc_criterion, dmg_criterion, emd_criterion,
            optimizer, scaler, train=False, epoch=epoch, num_epochs=EPOCHS)

        scheduler.step()

        improved = vl_mf1 > best_val_f1
        if improved:
            best_val_f1 = vl_mf1
            ckpt = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "val_mean_f1":          vl_mf1,
                "val_f1s":              vl_f1s,
            }
            torch.save(ckpt, SAVE_PATH)
            drive_roots = [
                os.path.expanduser("~/GoogleDrive/MyDrive"),
                os.path.expanduser("~/google-drive/MyDrive"),
                os.path.expanduser("~/Drive/MyDrive"),
                "/mnt/gdrive/MyDrive",
                "/media/gdrive/MyDrive",
            ]
            for root in drive_roots:
                drive_dest = os.path.join(root, "ATML_project", "Phase-2 weights", "best_expD_model.pth")
                if os.path.isdir(os.path.dirname(drive_dest)) or os.path.isdir(root):
                    os.makedirs(os.path.dirname(drive_dest), exist_ok=True)
                    torch.save(ckpt, drive_dest)
                    print(f"Saved to Drive: {drive_dest}")
                    break
            else:
                print("Drive not found — skipping Drive save.")

        status    = "\u2713" if improved else "\u2717"
        f1_detail = "  ".join(f"{n}={f:.3f}" for n, f in zip(class_names, vl_f1s))
        print(f"Epoch {epoch:02d} | "
              f"loc {vl_loc:.4f}  dmg {vl_dmg:.4f}  emd {vl_emd:.4f} | "
              f"val mF1 {vl_mf1:.4f} [{f1_detail}] {status}")

    print(f"\nBest val mF1: {best_val_f1:.4f} — saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
