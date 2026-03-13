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

### Multiclass CNNs (Blocky, Granular, Platy-Lenticular)

- **MobileNetV2 / EfficientNetV2B0 / ResNet50** backbones with:
  - Pretrained ImageNet weights, last blocks unfrozen for fine‑tuning.  
  - Global average pooling + a small dense head with batch norm, dropout, and an output `softmax` neuron.  
  - Loss: `categorical_crossentropy`, optimizer: Adam.


### Score-CAM analysis: 
Each model includes a Score-CAM analysis to see in which features each model in both binary and multiclass classification focusses on. 
In addition to this, we also checked the heatmaps into cumulative “long-tail” curves of activation 
mass vs. pixel fraction by sorting CAM intensities (file: 'long_tail.py'). 
