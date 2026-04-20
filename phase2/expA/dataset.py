import os
import glob
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class XBDJointDataset(Dataset):
    def __init__(self, splits, base_dir="xbd", augment=False):
        self.augment = augment
        self.samples = []  # (pre_img, post_img, pre_mask, post_mask)

        for split in splits:
            img_dir  = os.path.join(base_dir, split, "images")
            mask_dir = os.path.join(base_dir, split, "masks")
            if not os.path.isdir(img_dir):
                continue

            for pre_img_path in glob.glob(os.path.join(img_dir, "*_pre_disaster.png")):
                stem          = os.path.basename(pre_img_path).replace("_pre_disaster.png", "")
                post_img_path  = os.path.join(img_dir,  f"{stem}_post_disaster.png")
                pre_mask_path  = os.path.join(mask_dir, f"{stem}_pre_disaster.png")
                post_mask_path = os.path.join(mask_dir, f"{stem}_post_disaster.png")

                if not all(os.path.exists(p) for p in [post_img_path, pre_mask_path, post_mask_path]):
                    continue

                self.samples.append((pre_img_path, post_img_path, pre_mask_path, post_mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_img_path, post_img_path, pre_mask_path, post_mask_path = self.samples[idx]

        pre_img  = Image.open(pre_img_path).convert("RGB")
        post_img = Image.open(post_img_path).convert("RGB")
        pre_mask = Image.open(pre_mask_path)   # 0 or 255
        post_mask = Image.open(post_mask_path)  # 0-4

        if self.augment:
            if random.random() > 0.5:
                pre_img, post_img = TF.hflip(pre_img), TF.hflip(post_img)
                pre_mask, post_mask = TF.hflip(pre_mask), TF.hflip(post_mask)
            if random.random() > 0.5:
                pre_img, post_img = TF.vflip(pre_img), TF.vflip(post_img)
                pre_mask, post_mask = TF.vflip(pre_mask), TF.vflip(post_mask)

        pre_t  = TF.normalize(TF.to_tensor(pre_img),  MEAN, STD)
        post_t = TF.normalize(TF.to_tensor(post_img), MEAN, STD)

        # pre mask: 255 -> 1 binary
        loc_mask = torch.from_numpy(np.array(pre_mask)).float() / 255.0  # (H, W) in [0,1]

        # post mask: values 0-4 directly as class indices
        dmg_mask = torch.from_numpy(np.array(post_mask)).long()           # (H, W) in {0,1,2,3,4}

        return pre_t, post_t, loc_mask.unsqueeze(0), dmg_mask
