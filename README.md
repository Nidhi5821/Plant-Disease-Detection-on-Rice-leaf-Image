# Rice Leaf Disease Detection

Deep-learning classifier for three rice leaf diseases (**Bacterial blight, Brown spot, Leaf smut**) using TensorFlow/Keras. It compares a custom CNN with EfficientNetV2-B0 transfer learning under a leak-free evaluation setup.

Dataset: [Rice Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset), 4,684 images in 3 classes.

## Highlights
- **Leak-free evaluation.** Near-duplicate images (flipped or rotated copies of the same leaf) are detected with perceptual hashing and kept in the same split, so the test score measures generalisation rather than memorisation.
- **Two models compared:** a custom 6-block CNN and EfficientNetV2-B0 (ImageNet-pretrained, two-stage fine-tuning).
- **Full metrics:** per-class precision, recall and F1, macro F1, and confusion matrices.
- **Explainability:** Grad-CAM heatmaps show which leaf regions drive each prediction.
- **Mobile-ready:** INT8-quantized TFLite export (about 7× smaller), checked against the Keras model.

## Results

> Run `notebooks/02_improved_pipeline.ipynb` and paste the comparison table here.

| Model | Test accuracy | Macro F1 | Params |
|---|---|---|---|
| Custom CNN | – | – | 116k |
| EfficientNetV2-B0 (fine-tuned) | – | – | 5.9M |

The original notebook reported 98.6% test accuracy. That figure came from a split where test images leaked into training (see below), so it isn't comparable with the numbers above.

## Project structure
```
├── notebooks/
│   ├── 01_baseline_cnn_original.ipynb   # first version (kept for reference)
│   └── 02_improved_pipeline.ipynb       # current pipeline
├── scripts/
│   └── split_dataset.py                 # leak-free, duplicate-aware train/val/test split
├── requirements.txt
└── README.md
```
`data/` and `models/` are created locally and ignored by git.

## How to run
```bash
pip install -r requirements.txt

# 1. Download and unzip the Kaggle dataset, then create the split once
python scripts/split_dataset.py --src "path/to/rice leaf diseases dataset" --dst data/split

# 2. Open and run the notebook
jupyter notebook notebooks/02_improved_pipeline.ipynb
```
The split script prints how many duplicate groups it found per class and writes `data/split/split_report.csv`. A GPU (or Google Colab) is recommended for the EfficientNet stage.

## What was fixed from the first version
| Issue | Fix |
|---|---|
| The dataset was reshuffled every epoch before `take()`/`skip()`, so train, val and test overlapped | Fixed split on disk, created once |
| Augmented copies of a leaf could appear in both train and test | Duplicate-aware grouping with perceptual hashing |
| `Rescaling(1/300)` | `Rescaling(1/255)` |
| `predict()` ignored its `img` argument | Fixed |
| Accuracy only | Classification report, macro F1, confusion matrix |

## Roadmap
- [ ] Stratified 5-fold cross-validation by duplicate group
- [ ] Healthy class, out-of-distribution rejection and confidence calibration
- [ ] Cross-dataset evaluation on field images (e.g. Paddy Doctor)
- [ ] ViT / ConvNeXt / DINOv2 linear-probe comparison
- [ ] Lesion segmentation for disease-severity estimation
- [ ] Gradio demo on Hugging Face Spaces
