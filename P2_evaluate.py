"""
evaluate.py  —  expD test-set evaluation

Metrics reported:
  • Localization F1  (binary building segmentation)
  • Localization IoU
  • Class-wise damage F1  (no-dmg, minor, major, destroyed)
  • Damage F1  (mean of class-wise F1s, classes 1-4)
  • Overall score  = 0.3 * loc_f1 + 0.7 * dmg_f1

Sample predictions are saved to --sample-out (default: eval_samples/).

Usage:
  python phase2/expD/evaluate.py                                     # hold split
  python phase2/expD/evaluate.py --splits test                       # test split
  python phase2/expD/evaluate.py --ckpt path/to/model.pth --splits hold test
"""

import argparse
import os
import random
import sys
from typing import NamedTuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from P2_dataset import XBDJointDataset
from P2_model import JointDamageNet

# ── constants ──────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DMG_CLASSES = 5                                   # 0=bg 1=no-dmg 2=minor 3=major 4=destr
DMG_NAMES       = ["no-dmg", "minor", "major", "destr"]
DMG_COLORS      = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]
MEAN            = np.array([0.485, 0.456, 0.406])
STD             = np.array([0.229, 0.224, 0.225])


# ── results container ──────────────────────────────────────────────────────────
class Results(NamedTuple):
    loc_f1:   float
    loc_iou:  float
    cls_f1:   dict[str, float]
    dmg_f1:   float
    overall:  float


# ── metric helpers ─────────────────────────────────────────────────────────────
def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def _iou(tp: float, fp: float, fn: float) -> float:
    return tp / (tp + fp + fn + 1e-8)


# ── visualization ──────────────────────────────────────────────────────────────
def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _dmg_colormap(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(DMG_COLORS, start=1):
        out[mask == cls_idx] = color
    return out


def _save_samples(samples: list, out_dir: str, n: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    chosen = random.sample(samples, min(n, len(samples)))
    for i, (pre_t, post_t, loc_gt, dmg_gt, loc_prob, dmg_pred) in enumerate(chosen):
        pre_img       = _denorm(pre_t)
        post_img      = _denorm(post_t)
        loc_gt_vis    = (loc_gt.squeeze().numpy() > 0.5).astype(np.uint8) * 255
        loc_pred_vis  = (loc_prob.squeeze().numpy() > 0.5).astype(np.uint8) * 255
        dmg_gt_vis    = _dmg_colormap(dmg_gt.numpy())
        # Mask damage prediction to predicted building footprint so background
        # pixels (loc_prob ≤ 0.5) are shown as black (class 0) rather than
        # being assigned a spurious damage color by argmax.
        bldg_mask         = (loc_prob.squeeze().numpy() > 0.5)
        dmg_pred_masked   = dmg_pred.numpy().copy()
        dmg_pred_masked[~bldg_mask] = 0
        dmg_pred_vis      = _dmg_colormap(dmg_pred_masked)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, img, title in zip(
            axes.flat,
            [pre_img, post_img, loc_gt_vis, loc_pred_vis, dmg_gt_vis, dmg_pred_vis],
            ["Pre-disaster", "Post-disaster", "Loc GT", "Loc Pred", "Dmg GT", "Dmg Pred"],
        ):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        patches = [
            mpatches.Patch(color=np.array(c) / 255, label=name)
            for c, name in zip(DMG_COLORS, DMG_NAMES)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(os.path.join(out_dir, f"sample_{i:03d}.png"), dpi=100)
        plt.close(fig)

    print(f"Saved {len(chosen)} sample images → {out_dir}/")


# ── main evaluation ────────────────────────────────────────────────────────────
def evaluate(
    ckpt_path: str,
    base_dir: str,
    splits: list[str],
    batch_size: int = 8,
    num_workers: int = 4,
    sample_out: str = "eval_samples",
    n_samples: int = 8,
    dropout: float = 0.3,
) -> Results:
    print(f"Device      : {DEVICE}")
    print(f"Checkpoint  : {ckpt_path}")

    ds = XBDJointDataset(splits, base_dir=base_dir, augment=False)
    print(f"Tiles       : {len(ds)}")
    if len(ds) == 0:
        raise RuntimeError(
            f"No tiles found in splits={splits} under base_dir='{base_dir}'. "
            "Check --base-dir and --splits."
        )

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model = JointDamageNet(loc_classes=1, dmg_classes=NUM_DMG_CLASSES, dropout=dropout).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Checkpoint  : epoch={ckpt.get('epoch', '?')}  "
        f"val_mF1={ckpt.get('val_mean_f1', float('nan')):.4f}"
    )
    print()

    # accumulators
    loc_tp = loc_fp = loc_fn = 0.0
    dmg_tp = [0.0] * NUM_DMG_CLASSES
    dmg_fp = [0.0] * NUM_DMG_CLASSES
    dmg_fn = [0.0] * NUM_DMG_CLASSES
    samples_buf: list = []

    with torch.no_grad():
        for pre, post, loc_mask, dmg_mask in tqdm(loader, desc="Evaluating"):
            pre      = pre.to(DEVICE, non_blocking=True)
            post     = post.to(DEVICE, non_blocking=True)
            loc_mask = loc_mask.to(DEVICE, non_blocking=True)   # (B, 1, H, W) float
            dmg_mask = dmg_mask.to(DEVICE, non_blocking=True)   # (B, H, W) long

            with autocast():
                loc_out, dmg_out = model(pre, post)

            # localization: sigmoid threshold at 0.5
            loc_prob = torch.sigmoid(loc_out)
            loc_pred = (loc_prob > 0.5).squeeze(1)              # (B, H, W) bool
            loc_gt   = (loc_mask > 0.5).squeeze(1)              # (B, H, W) bool

            loc_tp += float((loc_pred &  loc_gt).sum())
            loc_fp += float((loc_pred & ~loc_gt).sum())
            loc_fn += float((~loc_pred & loc_gt).sum())

            # damage: argmax, evaluated only on building pixels (dmg_mask > 0)
            dmg_pred_cls = dmg_out.argmax(dim=1)                # (B, H, W) long
            bldg_px      = dmg_mask > 0

            for c in range(1, NUM_DMG_CLASSES):
                pred_c = (dmg_pred_cls == c) & bldg_px
                true_c = dmg_mask == c
                dmg_tp[c] += float((pred_c &  true_c).sum())
                dmg_fp[c] += float((pred_c & ~true_c).sum())
                dmg_fn[c] += float((~pred_c & true_c & bldg_px).sum())

            # buffer a few samples for visualization (keep on CPU)
            if len(samples_buf) < n_samples * 4:
                for b in range(pre.size(0)):
                    samples_buf.append((
                        pre[b].cpu(), post[b].cpu(),
                        loc_mask[b].cpu(), dmg_mask[b].cpu(),
                        loc_prob[b].cpu(), dmg_pred_cls[b].cpu(),
                    ))

    # ── compute metrics ────────────────────────────────────────────────────────
    loc_f1  = _f1(loc_tp, loc_fp, loc_fn)
    loc_iou = _iou(loc_tp, loc_fp, loc_fn)

    cls_f1 = [_f1(dmg_tp[c], dmg_fp[c], dmg_fn[c]) for c in range(1, NUM_DMG_CLASSES)]
    dmg_f1  = float(np.mean(cls_f1))
    overall = 0.3 * loc_f1 + 0.7 * dmg_f1

    # ── print report ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("  EVALUATION RESULTS  —  expD")
    print("=" * 55)
    print(f"  Loc  F1        : {loc_f1:.4f}")
    print(f"  Loc  IoU       : {loc_iou:.4f}")
    print("-" * 55)
    for name, f1 in zip(DMG_NAMES, cls_f1):
        print(f"  Dmg  F1  [{name:>8s}] : {f1:.4f}")
    print(f"  Dmg  F1  (mean)      : {dmg_f1:.4f}")
    print("-" * 55)
    print(f"  Overall score        : {overall:.4f}   (0.3·loc + 0.7·dmg)")
    print("=" * 55)

    if samples_buf:
        _save_samples(samples_buf, out_dir=sample_out, n=n_samples)

    return Results(
        loc_f1=loc_f1,
        loc_iou=loc_iou,
        cls_f1={n: f for n, f in zip(DMG_NAMES, cls_f1)},
        dmg_f1=dmg_f1,
        overall=overall,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the expD best model on the test/val set")
    p.add_argument(
        "--ckpt", default="best_model_p2.pth",
        help="Path to checkpoint  (default: best_expD_model.pth)",
    )
    p.add_argument(
        "--base-dir", default="xbd",
        help="Root of xbd dataset  (default: xbd)",
    )
    p.add_argument(
        "--splits", nargs="+", default=["hold"],
        help="Dataset split(s) to evaluate  (default: hold)",
    )
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument(
        "--sample-out", default="eval_samples",
        help="Directory for sample prediction images  (default: eval_samples/)",
    )
    p.add_argument("--n-samples",   type=int,   default=8,
                   help="Number of sample images to save")
    p.add_argument("--dropout",     type=float, default=0.3,
                   help="Dropout rate used during training (must match checkpoint)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        ckpt_path   = args.ckpt,
        base_dir    = args.base_dir,
        splits      = args.splits,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        sample_out  = args.sample_out,
        n_samples   = args.n_samples,
        dropout     = args.dropout,
    )
