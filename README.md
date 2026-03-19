# Frost & Microstructure Classification

Transfer‑learning pipelines for:
* Binary frost vs non‑frost image classification
* Multiclass microstructure image classification

using CNN backbones (MobileNetV2, EfficientNetV2B0, ResNet50), Bayesian neural networks (BNNs), and Score‑CAM interpretability. An interactive demo is available on Hugging Face Spaces.

---

## 🔗 Interactive Demo

Try the trained models and visual explanations in your browser:

**https://huggingface.co/spaces/skkou/frost_classifier**

The app lets you upload images, see model predictions (CNN, BNN, MobileNetV2, EfficientNetV2B0, and ResNet50), and generate Score‑CAM and Grad-CAM heatmaps and overlays.

---

The app lets you upload images, see model predictions (CNN, BNN, MobileNetV2, EfficientNetV2B0, and ResNet50), and generate Score‑CAM and Grad-CAM heatmaps and overlays.

## Project Structure (High‑Level)

* **Data** (in Google Drive)
   * `frost/frost_non_frost/non_frost/` – binary class 0
   * `frost/frost_non_frost/frost/` – binary class 1
   * `frost/frost_non_frost/microstructures/` – one subfolder per microstructure class

* **Models & outputs**
   * `frost/models_saved/` – binary models + metrics, plots, CAMs
   * `frost/models_saved/multiclass/`, `multiclass2/` – multiclass models + outputs

* **Analysis**
   * `expert_models_comparisons/` – expert survey and human-AI alignment analysis
   * `statistical_analysis/` – Friedman-Nemenyi statistical testing and critical difference diagrams

* **Hugging Face app**
   * `huggingface_space/app.py` – Gradio app code
   * `huggingface_space/requirements.txt` – app dependencies

## Models (Binary, Multiclass, BNN)

All models share:
* Stratified train/val/test splits (70/15/15, saved once and reused).
* Strong data augmentation and class‑balanced weights.
* Two‑phase training: K‑fold cross‑validation + final training on full train set.
* Evaluation on a fixed test set with classification reports and confusion matrices.
* Misclassification grids and CSVs (true/pred labels, confidences).

### Binary CNNs (Frost vs Non‑Frost)

* **MobileNetV2 / EfficientNetV2B0 / ResNet50** backbones with:
   * Pretrained ImageNet weights, last blocks unfrozen for fine‑tuning.
   * Global average pooling + a small dense head with batch norm, dropout, and an output `sigmoid` neuron.
   * Loss: `binary_crossentropy`, optimizer: Adam.

### Multiclass CNNs (Blocky, Granular, Platy-Lenticular)

* **MobileNetV2 / EfficientNetV2B0 / ResNet50** backbones with:
   * Pretrained ImageNet weights, last blocks unfrozen for fine‑tuning.
   * Global average pooling + a small dense head with batch norm, dropout, and an output `softmax` neuron.
   * Loss: `categorical_crossentropy`, optimizer: Adam.

### Score-CAM Analysis

Each model includes Score-CAM analysis to visualize which features drive predictions in both binary and multiclass classification. In addition, we analyzed heatmap distributions using cumulative "long-tail" curves of activation mass vs. pixel fraction by sorting CAM intensities (see `long_tail.py`).

## Expert-Model Comparison

To validate model performance and quantify human interpretive variability, we conducted a blind survey of expert micromorphologists. This analysis reveals:

* **Expert agreement metrics**: Inter-rater reliability (Fleiss κ, Cohen's κ) and consensus patterns
* **Human-AI alignment**: Direct comparison of expert classifications vs. model predictions
* **Performance paradox**: High model accuracy despite low expert agreement, with near-zero CNN-expert error correlation

See [`expert_models_comparisons/README.md`](expert_models_comparisons/README.md) for full details on survey design, participant profiles, and human-AI alignment analysis.

## Statistical Analysis

Rigorous statistical comparison of model performance using:

* **Friedman test**: Non-parametric omnibus test across models
* **Nemenyi post-hoc test**: Pairwise comparisons with adjusted p-values
* **Critical difference diagrams**: Visual ranking of models with significance bars
* **Composite scoring**: Weighted combination of accuracy, F1, and inverted loss

Models are evaluated separately for binary and multiclass tasks using 5-fold cross-validation metrics.

See [`statistical_analysis/README.md`](statistical_analysis/README.md) for methodology, composite score formulations, and interpretation of critical difference values.

## Repository Contents

### Model Training Scripts

* `neu_cnn_binary.py` / `neu_cnn_multiclass.py` – Custom CNN architecture
* `bnn_binary.py` / `neu_bnn_multiclass.py` – Bayesian Neural Network with MC Dropout
* `neu_mobile_binary.py` / `neu_mobile_multiclass.py` – MobileNetV2 backbone
* `neu_efficient_binary.py` / `neu_efficient_multiclass.py` – EfficientNetV2B0 backbone
* `neu_resnet_binary.py` / `neu_resnet_multiclass.py` – ResNet50 backbone

### Analysis Scripts

* `long_tail.py` – Score-CAM activation mass distribution analysis
* `expert_models_comparisons/` – Survey and human-AI alignment scripts
* `statistical_analysis/` – Friedman-Nemenyi testing and critical difference diagrams

## Requirements

See `requirements.txt` for package dependencies. Key libraries:
* TensorFlow/Keras for model training
* scikit-learn for evaluation metrics
* scikit-posthocs for statistical testing
* matplotlib/seaborn for visualization
* Gradio for the interactive demo

# Dataset and results
The dataset and the results are publicly available on: https://doi.org/10.17605/OSF.IO/TYDSM 

## Contact
For questions about statistical methodology or interpretation:
skouki@ualg.pt

## Acknowledgments
This work is part of the ERC Starting Grant MATRIX project (nº101041245).

