# Frost & Microstructure Classification

Transfer‑learning pipelines for:

- Binary **frost vs non‑frost** image classification  
- Multiclass **microstructure** image classification  

using CNN backbones (MobileNetV2, EfficientNetV2B0, ResNet50), Bayesian neural networks (BNNs), and **Score‑CAM** interpretability. An interactive demo is available on Hugging Face Spaces.

---

## 🔗 Interactive Demo

Try the trained models and visual explanations in your browser:

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/skkou/frost_classifier)

The app lets you upload images, see model predictions (CNN, BNN, MobileNetV2, EfficientNetV2B0, and ResNet50), and generate Score‑CAM and Grad-CAM heatmaps and overlays.

---

## Project Structure (High‑Level)

- **Data (in Google Drive)**
  - `frost/frost_non_frost/non_frost/` – binary class 0  
  - `frost/frost_non_frost/frost/` – binary class 1  
  - `frost/frost_non_frost/microstructures/` – one subfolder per microstructure class  

- **Models & outputs**
  - `frost/models_saved/` – binary models + metrics, plots, CAMs  
  - `frost/models_saved/multiclass/`, `multiclass2/` – multiclass models + outputs  

- **Hugging Face app**
  - `huggingface_space/app.py` – Gradio app code  
  - `huggingface_space/requirements.txt` – app dependencies  

---

## Models (Binary, Multiclass, BNN)

All models share:

- Stratified train/val/test splits (70/15/15, saved once and reused).  
- Strong data augmentation and class‑balanced weights.  
- Two‑phase training: K‑fold cross‑validation + final training on full train set.  
- Evaluation on a fixed test set with classification reports and confusion matrices.  
- Misclassification grids and CSVs (true/pred labels, confidences).

### Binary CNNs (Frost vs Non‑Frost)

- **MobileNetV2 / EfficientNetV2B0 / ResNet50** backbones with:
  - Pretrained ImageNet weights, last blocks unfrozen for fine‑tuning.  
  - Global average pooling + a small dense head with batch norm, dropout, and an output `sigmoid` neuron.  
  - Loss: `binary_crossentropy`, optimizer: Adam.

**ResNet50 binary code description:**

```python
"""
Transfer-learning ResNet50 pipeline for binary frost vs non-frost classification,
with cross-validation, detailed misclassification logging, and custom Score-CAM
explanations.

This codes:
1. Sets seeds, mounts Google Drive, loads and normalizes the frost/non-frost
   image dataset, constructs a stratified train/validation/test split (saving
   the test split for reuse), then applies data augmentation and computes class
   weights to counteract class imbalance.
2. Defines a ResNet50-based model (ImageNet backbone with its final blocks
   unfrozen) followed by L2-regularized dense layers and dropout with a sigmoid
   output for binary prediction, and runs stratified K-fold cross-validation to
   record per-fold metrics, training curves, and confusion matrices, which are
   saved to disk.
3. Trains a final ResNet50 model on the full augmented training set with
   checkpointing, learning-rate scheduling, and CSV logging, evaluates on the
   held-out test set, reports accuracy/precision/recall/F1, and saves summary
   metrics, cross-validation statistics, and misclassification counts to CSV,
   alongside plots of learning curves and the final confusion matrix.
4. Visualizes misclassified test images with true/predicted labels, confidence,
   and error type (false positive vs false negative), exports a sorted
   misclassification table, and implements a custom Score-CAM routine adapted
   to ResNet50 internals to generate triplets (original, heatmap, overlay) for
   representative correct and misclassified samples, highlighting which image
   regions most influence the model’s decisions.
"""
