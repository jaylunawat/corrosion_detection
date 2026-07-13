# Concrete Defect Segmentation: Corrosion & Spalling

A multitask semantic segmentation model that detects **corrosion** and **spalling** in concrete surface images from a single shared encoder/decoder, with separate prediction heads per defect type.

## Overview

- **Architecture:** Shared UNet (`segmentation-models-pytorch`, MobileNetV2 encoder, ImageNet-pretrained) producing 64 shared feature channels, followed by two lightweight task-specific heads (conv → batch norm → ReLU → dropout → 1×1 conv → sigmoid) — one for corrosion, one for spalling.
- **Loss:** Weighted multitask Dice loss (corrosion weight 1.0, spalling weight 1.2).
- **Metrics tracked per task:** Dice score, F1, accuracy, precision, recall.
- **Params:** ~6.68M

## Dataset

[Corrosion and Spalling Concrete Defect Segmentation](https://www.kaggle.com/datasets) (Kaggle). Each sample is an image paired with an RGB label mask (`*_lab.png`) using:

| Class     | Color            |
|-----------|------------------|
| Corrosion | Yellow `[255, 255, 0]` |
| Spalling  | Red `[255, 0, 0]`      |

Expected directory layout:

```
data/
  train/
    image1.png
    image1_lab.png
    ...
  val/
    image1.png
    image1_lab.png
    ...
```

Download the dataset from Kaggle and place it locally, or point directly at your own copy — the notebook does not require the Kaggle environment.

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Set the dataset location before running the notebook:

```bash
export DATA_ROOT=/path/to/spalling_corrosion_patches
```

If `DATA_ROOT` isn't set, it defaults to the original Kaggle input path (`/kaggle/input/...`), so the notebook also runs unmodified on Kaggle.

## Usage

Open `corrosion-spalling.ipynb` and run all cells. It will:

1. Load and augment the train/val datasets.
2. Train the multitask UNet for 5 epochs (AdamW, cosine LR schedule, batch size 8).
3. Save the best checkpoint (lowest val loss) to `best_multitask_model.pth`.
4. Visualize predictions vs. ground truth on validation samples.
5. Plot per-task training/validation curves (loss, Dice, F1, accuracy, precision, recall).

## Results

Final run (5 epochs, GPU):

| Task      | Split | Dice  | F1    | Accuracy |
|-----------|-------|-------|-------|----------|
| Corrosion | Train | 0.363 | 0.363 | 0.809    |
| Corrosion | Val   | 0.400 | 0.400 | 0.867    |
| Spalling  | Train | 0.708 | 0.708 | 0.765    |
| Spalling  | Val   | 0.508 | 0.508 | 0.842    |

**Notes on current performance:**
- Corrosion segmentation is the weaker task — Dice stayed in the 0.3–0.4 range and did not fully converge in 5 epochs.
- Spalling shows a widening train/val gap (train Dice 0.71 vs. val Dice 0.51), suggesting overfitting.
- Val loss increased slightly over training while train loss decreased — worth addressing (e.g., more epochs with early stopping, stronger regularization, or more augmentation) before treating this as a finished model.

### Sample predictions

![Prediction samples](assets/prediction_samples.png)

### Training history

![Training history](assets/training_history.png)

## Project structure

```
.
├── corrosion-spalling.ipynb   # main notebook: data, model, training, evaluation
├── requirements.txt
├── assets/                    # sample output images used in this README
├── LICENSE
└── README.md
```

## Possible next steps

- Address the corrosion/spalling class imbalance and overfitting gap.
- Add early stopping and/or a validation-based LR scheduler.
- Try alternative encoders (ResNet, EfficientNet) for comparison.
- Refactor the notebook into `src/` modules (`dataset.py`, `model.py`, `train.py`) for reuse outside the notebook.

## License

[MIT](LICENSE)
