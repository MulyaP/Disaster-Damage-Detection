# Satellite Disaster Damage Assessment

A two-phase deep learning pipeline for building localization and damage classification from pre/post-disaster satellite imagery, trained on the [xBD dataset](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset).

**Phase 1** segments building footprints in pre-disaster images. **Phase 2** jointly localizes buildings and classifies damage severity in post-disaster images.

---

## Method Overview

### Phase 1 — Building Localization

- **Model:** DeepLabV3+ with ResNet-50 backbone and Atrous Spatial Pyramid Pooling (ASPP)
- **Task:** Binary semantic segmentation (building vs. background) on pre-disaster imagery
- **Loss:** Dice loss + Binary Cross-Entropy
- **Metrics:** F1 score, IoU

### Phase 2 — Joint Damage Assessment

- **Model:** `JointDamageNet` — dual-output architecture built on the Phase 1 encoder
- **Task:** Simultaneous building localization and 5-class damage classification (background / no-damage / minor / major / destroyed) on post-disaster imagery
- **Loss:** Binary Focal Loss (localization) + Focal Loss (damage) + Earth Mover's Distance loss (ordinal penalty)
- **Metric:** `0.3 × localization_F1 + 0.7 × damage_F1`
- **Class imbalance:** Minority damage classes (minor/major/destroyed) are upsampled 3× during training

---

## Repository Structure

```
.
├── P1_train.py        # Phase 1 training script
├── P1_evaluate.py     # Phase 1 evaluation
├── P1_model.py        # DeepLabV3+ model definition
├── P1_dataset.py      # Phase 1 dataset loader
├── P1_visualize.py    # Visualization utilities
├── P2_train.py        # Phase 2 training script
├── P2_evaluate.py     # Phase 2 evaluation script
├── P2_model.py        # JointDamageNet model definition
├── P2_dataset.py      # Phase 2 dataset loader
├── P2_losses.py       # Focal, EMD, and combined loss functions
├── requirements.txt
└── xbd/               # Dataset directory (see setup below)
```

---

## Setup

### 1. Clone the repository

```bash
git clone -b submission https://github.com/MulyaP/NCSU-ATML-Project.git
cd NCSU-ATML-Project
```

### 2. Download and extract the xBD dataset

Download the xBD dataset from Kaggle: (https://www.kaggle.com/datasets/qianlanzz/xbd-dataset)

Extract the archive directly into the project directory so the layout looks like:

```
NCSU-ATML-Project/
└── xbd/
    ├── tier1/
    │   ├── images/
    │   └── targets/
    ├── tier3/
    │   ├── images/
    │   └── targets/
    ├── hold/
    │   ├── images/
    │   └── targets/
    └── test/
        ├── images/
        └── targets/
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** The project requires PyTorch 2.7.0 with CUDA 12.8. If you are using a different CUDA version or CPU-only, adjust the torch/torchvision/torchaudio lines in `requirements.txt` accordingly before installing. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right install command.

---

## Training

Run the two phases in order. Phase 2 loads the encoder weights saved by Phase 1, so Phase 1 must finish first.

### Phase 1 — Building Localization

```bash
python P1_train.py
```

- Trains for 30 epochs, batch size 8, learning rate 1e-4
- Saves the best checkpoint as `best_model.pth` in the working directory

### Phase 2 — Joint Damage Assessment

```bash
python P2_train.py
```

- Loads the Phase 1 encoder from `best_model.pth`
- Trains for 50 epochs with 5-epoch linear warmup and cosine annealing
- Uses mixed-precision training (requires a CUDA-capable GPU)
- Saves the best Phase 2 checkpoint as `best_joint_model.pth`

---

## Evaluation

To evaluate the Phase 2 model you can either train from scratch (above) or download the pre-trained weights.

### Download pre-trained weights

Download `best_joint_model.pth` from: https://drive.google.com/file/d/1_ULPMVJHhbIkzYo24qt2-F3RxP9ZBalk/view?usp=sharing

Place the file in the project root:

```
NCSU-ATML-Project/
└── best_joint_model.pth
```

### Run evaluation

```bash
python P2_evaluate.py
```

This evaluates on the test split by default and prints:

- Per-class damage F1 scores
- Overall localization F1 and IoU
- Combined score (`0.3 × loc_F1 + 0.7 × dmg_F1`)

Prediction visualizations are saved to `eval_samples/`.
