# Lab 1: Artifact Detection in AI-Generated Face Images

## What is this project about?

Modern image generation models (like Stable Diffusion, Midjourney, etc.) sometimes produce images with visible artifacts — weird hands, extra fingers, text overlays, pieces of face masks, strange tattoos, or eyes that look in wrong directions. These artifacts are especially common with human faces.

In this project, I built a binary classifier that automatically detects whether a generated face image contains artifacts or not. Think of it as a QA system that filters out "broken" images before they reach the user.

## Dataset

The dataset consists of 2000 AI-generated face images:

| Split | Total | Artifact (class 0) | Clean (class 1) | Ratio |
|-------|-------|-------------------|-----------------|-------|
| Train | 1800 | 180 | 1620 | 10% / 90% |
| Test | 200 | 20 | 180 | 10% / 90% |

All images are PNG files (~1024x1024), named as `image_<index>_<label>.png`.

**The main challenge here is the class imbalance** — there are 9x more clean images than artifact images. A naive model could just predict "clean" for everything and get 90% accuracy, but it would be completely useless. That's why I use **macro F1 score** as the primary metric — it treats both classes equally.

## My Approach

I tried two different model architectures and then combined them into an ensemble:

### Approach 1: EfficientNet-B0

- Pretrained on ImageNet, fine-tuned on our data
- Input size: 384x384
- This is a CNN-based model, so it's good at capturing local texture patterns — things like distorted fingers, blurred edges around masks, unnatural skin textures

### Approach 2: ViT-B/16 (Vision Transformer)

- Pretrained on ImageNet-21K, fine-tuned on our data
- Input size: 224x224
- Transformers look at the whole image globally through self-attention, so they're better at catching semantic-level artifacts — text overlays, eyes looking in different directions, compositional oddities

### Approach 3: Ensemble

- Weighted average of probabilities from both models
- Weights are proportional to each model's validation F1 score
- The idea is that EfficientNet and ViT make different kinds of mistakes, so combining them should help

## How I Handled Class Imbalance

This was honestly the trickiest part. I used three techniques together:

1. **WeightedRandomSampler** — during training, the DataLoader oversamples artifact images so the model sees roughly equal amounts of both classes in each epoch
2. **Weighted CrossEntropyLoss** — the loss function penalizes mistakes on artifacts 9x more than mistakes on clean images (weight = [9.0, 1.0])
3. **Threshold optimization** — instead of using the default 0.5 threshold, I sweep from 0.1 to 0.9 on the validation set and pick the threshold that maximizes macro F1. This turned out to be really important — the optimal thresholds ended up being 0.25 for EfficientNet and 0.15 for ViT, way below the default

## Training Details

- **Optimizer**: AdamW with weight decay 1e-4
- **Learning rate schedule**: Backbone frozen for 3 epochs (only train the classification head), then unfreeze everything with differential learning rates (backbone: 1e-5, head: 1e-3) and cosine annealing
- **Data augmentation**: horizontal flip, rotation, shift/scale, color jitter, Gaussian noise, Gaussian blur, coarse dropout (using albumentations)
- **Early stopping**: patience of 7 epochs, monitoring validation macro F1
- **Stratified split**: 85% train / 15% validation, preserving the 10/90 class ratio
- **Device**: Apple MPS (M-series GPU)

## Results

### Validation Set

| Model | Macro F1 | ROC-AUC | Optimal Threshold |
|-------|----------|---------|-------------------|
| EfficientNet-B0 | 0.7757 | 0.8839 | 0.25 |
| ViT-B/16 | 0.8574 | 0.9214 | 0.15 |
| Ensemble | 0.8678 | 0.9178 | 0.35 |

### Test Set

| Model | Macro F1 | ROC-AUC | Accuracy |
|-------|----------|---------|----------|
| EfficientNet-B0 | 0.7872 | 0.8600 | 92% |
| ViT-B/16 | **0.8302** | 0.8722 | 95% |
| Ensemble | 0.8183 | 0.8739 | 94% |

**Best model on test: ViT-B/16 with macro F1 = 0.8302**

### Confusion Matrices (Test Set)

**EfficientNet-B0:**
- Correctly detected 14/20 artifacts, missed 6
- 11 false alarms (clean images predicted as artifact)
- More aggressive at flagging artifacts but at the cost of more false positives

**ViT-B/16:**
- Correctly detected 11/20 artifacts, missed 9
- Only 1 false alarm
- Very precise — when it says "artifact", it's almost always right (92% precision)

**Ensemble:**
- Correctly detected 11/20 artifacts, missed 9
- Only 2 false alarms
- Similar to ViT but slightly more conservative

### Misclassified images: 11 out of 200 (for ensemble)

## What I Learned

1. **Threshold optimization matters a lot.** The default 0.5 threshold would have given much worse F1 scores. The optimal thresholds were 0.15-0.35, which makes sense — the model needs to be more "trigger-happy" about flagging artifacts since they're the minority class.

2. **ViT outperformed EfficientNet here.** I think this is because many artifacts in the dataset are semantic in nature (wrong eye direction, text, unusual compositions) rather than purely textural. ViT's global self-attention mechanism seems better suited for catching these.

3. **The ensemble didn't beat ViT on the test set.** On validation it was the best (0.8678 vs 0.8574), but on test ViT alone scored 0.8302 vs ensemble's 0.8183. This might be because with only 20 artifact images in the test set, a single misclassification swings the F1 quite a bit. The ensemble is probably more robust in general, but we'd need a larger test set to confirm.

4. **Class imbalance is hard.** Even with three techniques combined, the model still misses about half the artifacts. With only 180 training examples of artifacts, the model doesn't see enough variety. More data or more aggressive augmentation specifically targeting artifact patterns could help.

## Project Structure

```
lab_1/
├── config.py            # All hyperparameters and paths in one place
├── dataset.py           # Dataset class, augmentations, data loading
├── models.py            # EfficientNet-B0 and ViT-B/16 definitions
├── train.py             # Training loop with early stopping
├── evaluate.py          # Metrics, threshold optimization, plotting
├── inference.py         # Test set prediction + CSV export
├── ensemble.py          # Weighted ensemble logic
├── utils.py             # Seed setting, device detection, helpers
├── notebook.ipynb       # Main notebook — run this to reproduce everything
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── outputs/
    ├── checkpoints/     # Saved model weights (.pth)
    ├── predictions_efficientnet.csv
    ├── predictions_vit.csv
    └── predictions_ensemble.csv
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook (recommended)
jupyter notebook notebook.ipynb

# 3. Or train from command line
python train.py
```

## Possible Improvements

- **More augmentation for artifacts**: Mixup/CutMix between artifact images to synthetically increase the minority class
- **Test-time augmentation (TTA)**: Run inference on multiple augmented versions of each image and average predictions
- **Face-specific features**: Use a face landmark detector (like dlib or MediaPipe) to extract face regions and feed those as additional features — artifacts often occur around hands, eyes, or face boundaries
- **Larger models**: EfficientNet-B3/B4 or ViT-Large might capture more subtle artifacts, but would need more compute
- **Better ensemble**: Try stacking (train a small logistic regression on top of both models' predictions) instead of simple weighted averaging

## Requirements

- Python 3.11+
- PyTorch 2.0+
- timm (for pretrained models)
- albumentations (for augmentations)
- scikit-learn, matplotlib, seaborn, pandas, tqdm
