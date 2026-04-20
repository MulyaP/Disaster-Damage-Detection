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


class XBDJointDataset(Dataset):
    def __init__(self, splits, base_dir="xbd", augment=False, min_building_ratio=0.01):
        self.augment = augment
        self.tiles   = []  # (pre_img, post_img, pre_mask, post_mask, row, col)

        for split in splits:
            img_dir  = os.path.join(base_dir, split, "images")
            mask_dir = os.path.join(base_dir, split, "masks")
            if not os.path.isdir(img_dir):
                continue

            for pre_img_path in glob.glob(os.path.join(img_dir, "*_pre_disaster.png")):
                stem           = os.path.basename(pre_img_path).replace("_pre_disaster.png", "")
                post_img_path  = os.path.join(img_dir,  f"{stem}_post_disaster.png")
                pre_mask_path  = os.path.join(mask_dir, f"{stem}_pre_disaster.png")
                post_mask_path = os.path.join(mask_dir, f"{stem}_post_disaster.png")

                if not all(os.path.exists(p) for p in [post_img_path, pre_mask_path, post_mask_path]):
                    continue

                # Use pre_mask to filter tiles by building density
                pre_mask = np.array(Image.open(pre_mask_path))
                h, w     = pre_mask.shape
                for r in range(0, h, TILE_SIZE):
                    for c in range(0, w, TILE_SIZE):
                        tile = pre_mask[r:r + TILE_SIZE, c:c + TILE_SIZE]
                        if tile.shape != (TILE_SIZE, TILE_SIZE):
                            continue
                        ratio = (tile > 0).sum() / (TILE_SIZE * TILE_SIZE)
                        if ratio >= min_building_ratio:
                            self.tiles.append((pre_img_path, post_img_path,
                                               pre_mask_path, post_mask_path, r, c))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        pre_img_path, post_img_path, pre_mask_path, post_mask_path, r, c = self.tiles[idx]

        def crop(path, mode="RGB"):
            arr = np.array(Image.open(path).convert(mode) if mode else Image.open(path))
            return arr[r:r + TILE_SIZE, c:c + TILE_SIZE]

        pre_img   = Image.fromarray(crop(pre_img_path))
        post_img  = Image.fromarray(crop(post_img_path))
        pre_mask  = Image.fromarray(crop(pre_mask_path,  mode=None))
        post_mask = Image.fromarray(crop(post_mask_path, mode=None))

        if self.augment:
            if random.random() > 0.5:
                pre_img, post_img = TF.hflip(pre_img), TF.hflip(post_img)
                pre_mask, post_mask = TF.hflip(pre_mask), TF.hflip(post_mask)
            if random.random() > 0.5:
                pre_img, post_img = TF.vflip(pre_img), TF.vflip(post_img)
                pre_mask, post_mask = TF.vflip(pre_mask), TF.vflip(post_mask)
            k = random.randint(0, 3)
            if k:
                pre_img,  post_img  = TF.rotate(pre_img,  90*k), TF.rotate(post_img,  90*k)
                pre_mask, post_mask = TF.rotate(pre_mask, 90*k), TF.rotate(post_mask, 90*k)
            if random.random() > 0.5:
                pre_img  = TF.adjust_brightness(TF.adjust_contrast(pre_img,  random.uniform(0.8, 1.2)), random.uniform(0.8, 1.2))
                post_img = TF.adjust_brightness(TF.adjust_contrast(post_img, random.uniform(0.8, 1.2)), random.uniform(0.8, 1.2))

        pre_t  = TF.normalize(TF.to_tensor(pre_img),  MEAN, STD)
        post_t = TF.normalize(TF.to_tensor(post_img), MEAN, STD)

        loc_mask = torch.from_numpy(np.array(pre_mask)).float() / 255.0   # (H, W) in [0,1]
        dmg_mask = torch.from_numpy(np.array(post_mask)).long()            # (H, W) in {0,1,2,3,4}

        return pre_t, post_t, loc_mask.unsqueeze(0), dmg_mask
