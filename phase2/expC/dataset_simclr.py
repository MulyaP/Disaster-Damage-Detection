import os
import glob
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]
TILE_SIZE = 512


def _color_aug(img):
    """
    Independent per-view colour augmentation.
    Kept mild for aerial imagery where colour carries semantic meaning.
    """
    if random.random() < 0.8:
        img = TF.adjust_contrast(img,   random.uniform(0.6, 1.4))
        img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
        img = TF.adjust_saturation(img, random.uniform(0.6, 1.4))
        img = TF.adjust_hue(img,        random.uniform(-0.1, 0.1))
    if random.random() < 0.2:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    if random.random() < 0.3:
        img = TF.gaussian_blur(img, kernel_size=23,
                               sigma=random.uniform(0.1, 2.0))
    return img


class XBDSimCLRDataset(Dataset):
    """
    Positive pair = (pre tile, post tile) from the same scene location.

    Spatial augmentations (flip, rotate) are shared between pre and post
    so they remain spatially aligned. Colour augmentations are applied
    independently to each view so the encoder cannot trivially match them
    by low-level colour statistics alone.

    The contrastive objective forces the encoder to learn that pre and post
    images of the same location represent the same underlying scene structure,
    regardless of disaster-induced appearance changes.
    """
    def __init__(self, splits, base_dir="xbd", min_building_ratio=0.01):
        self.tiles = []  # (pre_img_path, post_img_path, row, col)

        for split in splits:
            img_dir  = os.path.join(base_dir, split, "images")
            mask_dir = os.path.join(base_dir, split, "masks")
            if not os.path.isdir(img_dir):
                continue

            for pre_img_path in glob.glob(os.path.join(img_dir, "*_pre_disaster.png")):
                stem          = os.path.basename(pre_img_path).replace("_pre_disaster.png", "")
                post_img_path = os.path.join(img_dir,  f"{stem}_post_disaster.png")
                pre_mask_path = os.path.join(mask_dir, f"{stem}_pre_disaster.png")

                if not all(os.path.exists(p) for p in [post_img_path, pre_mask_path]):
                    continue

                pre_mask = np.array(Image.open(pre_mask_path))
                h, w     = pre_mask.shape
                for r in range(0, h, TILE_SIZE):
                    for c in range(0, w, TILE_SIZE):
                        tile = pre_mask[r:r + TILE_SIZE, c:c + TILE_SIZE]
                        if tile.shape != (TILE_SIZE, TILE_SIZE):
                            continue
                        ratio = (tile > 0).sum() / (TILE_SIZE * TILE_SIZE)
                        if ratio >= min_building_ratio:
                            self.tiles.append((pre_img_path, post_img_path, r, c))

    def __len__(self):
        return len(self.tiles)

    def get_tile_label(self, idx):
        """
        Returns the dominant non-zero damage class for tile idx.
        Derived from the post-disaster mask (0=bg, 1-4=damage severity).
        Used by the KNN validation probe in train_simclr.py.
        """
        _, post_img_path, r, c = self.tiles[idx]
        post_mask_path = post_img_path.replace("/images/", "/masks/")
        if not os.path.exists(post_mask_path):
            return 0
        arr  = np.array(Image.open(post_mask_path))
        tile = arr[r:r + TILE_SIZE, c:c + TILE_SIZE]
        building = tile[tile > 0]
        if len(building) == 0:
            return 0
        vals, counts = np.unique(building, return_counts=True)
        return int(vals[counts.argmax()])

    def __getitem__(self, idx):
        pre_path, post_path, r, c = self.tiles[idx]

        def load_crop(path):
            arr = np.array(Image.open(path).convert("RGB"))
            return Image.fromarray(arr[r:r + TILE_SIZE, c:c + TILE_SIZE])

        pre_img  = load_crop(pre_path)
        post_img = load_crop(post_path)

        # ── Shared spatial augmentations (pre and post stay aligned) ──────────
        if random.random() > 0.5:
            pre_img  = TF.hflip(pre_img)
            post_img = TF.hflip(post_img)
        if random.random() > 0.5:
            pre_img  = TF.vflip(pre_img)
            post_img = TF.vflip(post_img)
        k = random.randint(0, 3)
        if k:
            pre_img  = TF.rotate(pre_img,  90 * k)
            post_img = TF.rotate(post_img, 90 * k)

        # ── Independent colour augmentations (each view looks different) ──────
        pre_img  = _color_aug(pre_img)
        post_img = _color_aug(post_img)

        to_t = lambda img: TF.normalize(TF.to_tensor(img), MEAN, STD)
        return to_t(pre_img), to_t(post_img)
