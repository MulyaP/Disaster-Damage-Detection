import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import XBDSegDataset
from model import DeepLabV3Plus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8


def compute_tp_fp_fn(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp, fp, fn


def evaluate(model, val_loader):
    total_tp, total_fp, total_fn = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(images)
            tp, fp, fn = compute_tp_fp_fn(preds, masks)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    val_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    val_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-6)

    print("Reproduced Validation Metrics")
    print("IoU :", val_iou)
    print("Dice:", val_dice)


if __name__ == "__main__":
    val_ds = XBDSegDataset(
        splits=["hold"],
        base_dir="/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd/",
        augment=False
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    print("Validation tiles:", len(val_ds))

    # ── ImageNet backbone ──────────────────────────────────────────────────────
    model = DeepLabV3Plus().to(DEVICE)
    checkpoint = torch.load(
        "/kaggle/input/models/draqa12/deeplabsv3-plus-xbd-best/pytorch/default/1/best_model.pth",
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded model from epoch:", checkpoint["epoch"])
    print("Saved IoU:", checkpoint["val_iou"])
    print("Saved Dice:", checkpoint["val_dice"])
    evaluate(model, val_loader)
