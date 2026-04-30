import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

TILE_SIZE = 512


class XBDSegDataset(Dataset):
    def __init__(self, splits, base_dir="xbd", augment=False, min_building_ratio=0.01):
        """
        min_building_ratio: tiles with fewer building pixels than this fraction
                            are discarded. Set to 0 to keep all tiles.
        """
        self.augment = augment
        self.tiles = []  # (img_path, mask_path, row, col)

        for split in splits:
            img_dir = os.path.join(base_dir, split, "images")
            mask_dir = os.path.join(base_dir, split, "masks")
            if not os.path.isdir(mask_dir):
                continue
            for fname in os.listdir(mask_dir):
                if "_rgb" in fname or not fname.endswith(".png"):
                    continue
                if "pre_disaster" not in fname:
                    continue
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, fname)
                if not os.path.exists(img_path):
                    continue

                mask = np.array(Image.open(mask_path))
                h, w = mask.shape
                for r in range(0, h, TILE_SIZE):
                    for c in range(0, w, TILE_SIZE):
                        tile = mask[r:r + TILE_SIZE, c:c + TILE_SIZE]
                        if tile.shape != (TILE_SIZE, TILE_SIZE):
                            continue
                        ratio = (tile > 0).sum() / (TILE_SIZE * TILE_SIZE)
                        if ratio >= min_building_ratio:
                            self.tiles.append((img_path, mask_path, r, c))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_path, mask_path, r, c = self.tiles[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        image = Image.fromarray(image[r:r + TILE_SIZE, c:c + TILE_SIZE])
        mask = Image.fromarray(mask[r:r + TILE_SIZE, c:c + TILE_SIZE])

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.normalize(TF.to_tensor(image),
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        return image, mask.unsqueeze(0)
