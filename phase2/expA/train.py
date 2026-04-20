import os
import sys
import collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import FocalLoss, BinaryFocalLoss
from dataset import XBDJointDataset
from model import JointDamageNet

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = "xbd"
TRAIN_SPLITS    = ["tier1", "tier3"]
VAL_SPLITS      = ["hold"]
BATCH_SIZE      = 4
NUM_WORKERS     = 4
LR              = 1e-4
EPOCHS          = 40
FOCAL_GAMMA     = 2.0
LOC_LOSS_WEIGHT = 1.0   # λ for localization loss
DMG_LOSS_WEIGHT = 1.0   # λ for damage loss
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH       = "best_expA_model.pth"
PHASE1_CKPT     = "best_model.pth"
# ──────────────────────────────────────────────────────────────────────────────


def compute_dmg_weights(dataset):
    counts = collections.Counter()
    for _, _, _, post_mask_path in dataset.samples:
        from PIL import Image
        import numpy as np
        arr = np.array(Image.open(post_mask_path)).flatten()
        for v in range(1, 5):   # skip background
            counts[v] += (arr == v).sum()
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (4 * max(counts[c], 1)) for c in range(1, 5)], dtype=torch.float32
    )
    weights = weights / weights.sum() * 4
    return torch.cat([torch.zeros(1), weights])  # prepend 0 for background


def run_epoch(model, loader, loc_criterion, dmg_criterion, optimizer, train, epoch, num_epochs):
    model.train(train)
    total_loc, total_dmg, total_dmg_acc, n = 0.0, 0.0, 0.0, 0
    phase = "Train" if train else "Val"
    pbar  = tqdm(loader, desc=f"Epoch {epoch:02d}/{num_epochs} {phase}", leave=False)

    with torch.set_grad_enabled(train):
        for pre, post, loc_mask, dmg_mask in pbar:
            pre      = pre.to(DEVICE)
            post     = post.to(DEVICE)
            loc_mask = loc_mask.to(DEVICE)
            dmg_mask = dmg_mask.to(DEVICE)

            loc_out, dmg_out = model(pre, post)

            loc_loss = loc_criterion(loc_out, loc_mask)
            dmg_loss = dmg_criterion(dmg_out, dmg_mask)
            loss     = LOC_LOSS_WEIGHT * loc_loss + DMG_LOSS_WEIGHT * dmg_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = pre.size(0)
            total_loc     += loc_loss.item() * bs
            total_dmg     += dmg_loss.item() * bs
            # accuracy only on building pixels (dmg_mask > 0)
            mask           = dmg_mask > 0
            if mask.any():
                total_dmg_acc += (dmg_out.argmax(1)[mask] == dmg_mask[mask]).float().mean().item() * bs
            n             += bs

            pbar.set_postfix(loc=f"{total_loc/n:.4f}", dmg=f"{total_dmg/n:.4f}",
                             acc=f"{total_dmg_acc/n:.3f}")

    return total_loc / n, total_dmg / n, total_dmg_acc / n


def main():
    print(f"Device: {DEVICE}")

    train_ds = XBDJointDataset(TRAIN_SPLITS, BASE_DIR, augment=True)
    val_ds   = XBDJointDataset(VAL_SPLITS,   BASE_DIR, augment=False)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print("Computing damage class weights...")
    dmg_weights   = compute_dmg_weights(train_ds).to(DEVICE)
    print("Damage class weights:", dmg_weights.tolist())

    model = JointDamageNet(loc_classes=1, dmg_classes=5).to(DEVICE)
    model.load_phase1_weights(PHASE1_CKPT, DEVICE)

    loc_criterion = BinaryFocalLoss(gamma=FOCAL_GAMMA)
    dmg_criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=dmg_weights, ignore_index=0)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_dmg = float("inf")
    start_epoch   = 1

    if os.path.exists(SAVE_PATH):
        print(f"Resuming from checkpoint: {SAVE_PATH}")
        ckpt          = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_dmg  = ckpt["val_dmg_loss"]
        start_epoch   = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}, best val dmg loss: {best_val_dmg:.4f}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                                              last_epoch=start_epoch - 2)

    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loc, tr_dmg, tr_acc = run_epoch(model, train_loader, loc_criterion, dmg_criterion,
                                           optimizer, train=True,  epoch=epoch, num_epochs=EPOCHS)
        vl_loc, vl_dmg, vl_acc = run_epoch(model, val_loader,   loc_criterion, dmg_criterion,
                                           optimizer, train=False, epoch=epoch, num_epochs=EPOCHS)
        scheduler.step()

        improved = vl_dmg < best_val_dmg
        if improved:
            best_val_dmg = vl_dmg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dmg_loss": vl_dmg,
            }, SAVE_PATH)

        status = "\u2713" if improved else "\u2717"
        print(f"Epoch {epoch:02d} | "
              f"train loc {tr_loc:.4f} dmg {tr_dmg:.4f} acc {tr_acc:.3f} | "
              f"val loc {vl_loc:.4f} dmg {vl_dmg:.4f} acc {vl_acc:.3f} | {status}")

    print(f"\nBest val dmg loss: {best_val_dmg:.4f} — saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
