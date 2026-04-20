import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_simclr import XBDSimCLRDataset, MEAN, STD, TILE_SIZE
from model_simclr import SimCLREncoder, nt_xent_loss
import torchvision.transforms.functional as TF

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = "xbd"
TRAIN_SPLITS    = ["tier1", "tier3"]
VAL_SPLITS      = ["hold"]
BATCH_SIZE      = 64            # larger = more negatives = better; reduce if OOM
NUM_WORKERS     = 4
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
EPOCHS          = 50
TEMPERATURE     = 0.07
PROJ_DIM        = 128
KNN_K           = 5             # neighbours for KNN probe
KNN_TRAIN_N     = 1000          # training samples for KNN feature bank
KNN_VAL_N       = 500           # validation samples to evaluate
KNN_EVAL_FREQ   = 5             # run KNN probe every N epochs
SAVE_PATH       = "best_simclr.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def extract_features(model, dataset, indices, device, batch_size=64):
    """
    Extract 2048-d global encoder features from post-disaster tiles (no augmentation).
    Returns (features, labels) — labels are the dominant damage class per tile.
    Post images are used because they directly show the damage state.
    """
    features, labels = [], []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        imgs, batch_labels = [], []

        for idx in batch_idx:
            _, post_path, r, c = dataset.tiles[idx]
            arr  = np.array(Image.open(post_path).convert("RGB"))
            tile = Image.fromarray(arr[r:r + TILE_SIZE, c:c + TILE_SIZE])
            imgs.append(TF.normalize(TF.to_tensor(tile), MEAN, STD))
            batch_labels.append(dataset.get_tile_label(idx))

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad(), autocast("cuda"):
            feats = model.encode(batch_tensor)           # (B, 2048)
        features.append(F.normalize(feats, dim=1).cpu())
        labels.extend(batch_labels)

    return torch.cat(features, dim=0), torch.tensor(labels)


@torch.no_grad()
def evaluate_knn(model, train_ds, val_ds, device,
                 k=5, n_train=1000, n_val=500, batch_size=64):
    """
    KNN probe accuracy using frozen encoder features.

    - Builds a feature bank from n_train random training tiles.
    - Classifies n_val validation tiles by majority vote among k nearest neighbours.
    - Both sets filtered to tiles with a non-zero damage label.
    - Features extracted from post-disaster images (show damage state).
    """
    model.eval()

    train_idx = torch.randperm(len(train_ds))[:n_train].tolist()
    val_idx   = torch.randperm(len(val_ds))[:n_val].tolist()

    train_feats, train_labels = extract_features(model, train_ds, train_idx, device, batch_size)
    val_feats,   val_labels   = extract_features(model, val_ds,   val_idx,   device, batch_size)

    # Keep only tiles with a valid damage label
    tr_mask = train_labels > 0
    vl_mask = val_labels   > 0

    if tr_mask.sum() < k or vl_mask.sum() == 0:
        return 0.0

    tf = train_feats[tr_mask]
    tl = train_labels[tr_mask]
    vf = val_feats[vl_mask]
    vl = val_labels[vl_mask]

    # Cosine similarity (features are already L2-normalised)
    sim     = vf @ tf.T                    # (n_val, n_train)
    _, topk = sim.topk(k, dim=1)           # (n_val, k)
    nbr     = tl[topk]                     # (n_val, k) neighbour labels

    # Majority vote
    preds = torch.zeros(len(vf), dtype=torch.long)
    for i in range(len(vf)):
        vals, counts = nbr[i].unique(return_counts=True)
        preds[i] = vals[counts.argmax()]

    return (preds == vl).float().mean().item()


def main():
    print(f"Device: {DEVICE}")

    train_ds = XBDSimCLRDataset(TRAIN_SPLITS, BASE_DIR)
    val_ds   = XBDSimCLRDataset(VAL_SPLITS,   BASE_DIR)
    print(f"Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}")

    # drop_last=True keeps every batch full — NT-Xent needs consistent N
    loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model     = SimCLREncoder(proj_dim=PROJ_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler("cuda")

    best_val_acc = 0.0
    start_epoch  = 1

    if os.path.exists(SAVE_PATH):
        print(f"Resuming from {SAVE_PATH}")
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_acc = ckpt.get("val_knn_acc", 0.0)
        start_epoch  = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}, best val KNN acc: {best_val_acc:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # ── Training loop ─────────────────────────────────────────────────────
        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=False)

        for view1, view2 in pbar:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                z1   = model(view1)
                z2   = model(view2)
                loss = nt_xent_loss(z1, z2, TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs         = view1.size(0)
            total_loss += loss.item() * bs
            n          += bs
            pbar.set_postfix(loss=f"{total_loss / n:.4f}")

        scheduler.step()
        epoch_loss = total_loss / n

        # ── KNN validation probe (every KNN_EVAL_FREQ epochs) ─────────────────
        val_knn_acc = best_val_acc  # carry forward if not evaluating this epoch
        if epoch % KNN_EVAL_FREQ == 0 or epoch == EPOCHS:
            val_knn_acc = evaluate_knn(
                model, train_ds, val_ds, DEVICE,
                k=KNN_K, n_train=KNN_TRAIN_N, n_val=KNN_VAL_N,
            )

        improved = val_knn_acc > best_val_acc
        if improved:
            best_val_acc = val_knn_acc
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "val_knn_acc":          val_knn_acc,
                "train_loss":           epoch_loss,
            }, SAVE_PATH)

        status = "\u2713" if improved else " "
        knn_str = f"{val_knn_acc:.4f}" if epoch % KNN_EVAL_FREQ == 0 or epoch == EPOCHS else "  \u2014   "
        print(f"Epoch {epoch:03d} | loss {epoch_loss:.4f} "
              f"| knn_acc {knn_str} "
              f"| lr {scheduler.get_last_lr()[0]:.2e} | {status}")

    print(f"\nBest val KNN acc: {best_val_acc:.4f} — saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
