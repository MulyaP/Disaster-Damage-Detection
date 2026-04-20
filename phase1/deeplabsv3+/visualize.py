import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import XBDSegDataset
from model import DeepLabV3Plus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def visualize_predictions(model, dataset, num_samples=5):
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 4))

    for i in range(num_samples):
        image, mask = dataset[i]

        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE))
            pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        pred_bin = (pred > 0.5).astype(np.float32)

        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask.squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_bin, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_random(model, dataset, num_samples=6):
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 4))

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]

        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE))
            pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        pred_bin = (pred > 0.5).astype(np.float32)

        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")

        axes[i, 1].imshow(mask.squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(pred_bin, cmap="gray")
        axes[i, 2].set_title("Prediction")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()
