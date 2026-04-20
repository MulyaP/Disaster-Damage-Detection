import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import XBDSegDataset
from model import DeepLabV3Plus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def compute_tp_fp_fn(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp, fp, fn


def train():
    train_ds = XBDSegDataset(["tier1", "tier3"], augment=True)
    val_ds = XBDSegDataset(["hold"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4,
                            pin_memory=True)

    model = DeepLabV3Plus().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()

    best_score = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]",
                         leave=False)
        for images, masks in train_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            loss = bce(preds, masks) + dice_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        model.eval()
        total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  ",
                       leave=False)
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                tp, fp, fn = compute_tp_fp_fn(model(images), masks)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                # running batch-level display only
                b_iou = tp / (tp + fp + fn + 1e-6)
                b_dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
                val_bar.set_postfix(iou=f"{b_iou:.4f}", dice=f"{b_dice:.4f}")

        # global pixel-level metrics over entire val set
        val_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        val_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-6)
        combined = (val_iou + val_dice) / 2

        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}"
              f"  val_iou={val_iou:.4f}  val_dice={val_dice:.4f}")

        if combined > best_score:
            best_score = combined
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
            }, "best_model.pth")
            print(f"  -> saved best model (iou={val_iou:.4f}, dice={val_dice:.4f})")


if __name__ == "__main__":
    train()
