# Expert-Model Comparison Analysis

This folder contains scripts and analyses for evaluating human expert performance on frost classification tasks and comparing expert classifications with CNN model predictions.

## Overview

To validate our automated classification approach and quantify the inherent subjectivity in micromorphological interpretation, we conducted a blind survey of expert micromorphologists. The analysis reveals critical insights into:

1. **Inter-rater reliability**: How consistently experts agree with each other
2. **Expert-ground truth alignment**: How well experts match reference labels
3. **Human-AI performance comparison**: Direct comparison of expert vs. model accuracy
4. **The performance paradox**: High model accuracy (94-99%) despite low expert agreement (Fleiss κ = 0.15-0.20) and near-zero CNN-expert error correlation

## Files

### Scripts

* **`survey.py`** – Expert survey analysis and participant profiling
  * Loads survey responses from expert micromorphologists
  * Creates Sankey diagrams showing participant expertise → years in field → frost experience
  * Analyzes response distributions for visibility and classification tasks
  * Computes inter-rater reliability metrics (Fleiss κ, Cohen's κ)
  * Generates comprehensive multi-panel figures showing:
    * (a) Radar plot comparing first screening vs classification task performance
    * (b) Cohen's κ boxplots by frost experience level
    * (c-d) Scatter plots: uncertainty vs. performance with Spearman correlations

* **`human_ai_alignment_2.py`** – Human-AI alignment dataset construction
  * Builds rich alignment dataset combining expert survey ratings, model predictions, and consensus metrics
  * Decodes binary (Frost/Non-Frost) and multiclass (Blocky/Granular/Platy-Lenticular) ground truth
  * Computes expert consensus metrics:
    * Majority voting (decisive experts only vs. including "Unsure")
    * Agreement percentages
    * Entropy measures of disagreement
  * Loads predictions from all 5 models (BNN, Custom CNN, MobileNetV2, EfficientNet, ResNet50)
  * Derives model consensus and alignment metrics:
    * Average model confidence and disagreement (std dev)
    * Majority model prediction and agreement %
    * Expert vs. model correctness comparison
  * Categorizes images by difficulty: Easy_Both, Hard_Human, Hard_AI, Hard_Both
  * Outputs:
    * `human_ai_alignment_analysis.csv` – Wide format (one row per image)
    * `human_ai_alignment_long.csv` – Long format (one row per rater per image)
    * `expert_metadata.csv` – Expert experience and expertise profiles

* **`ml_frost_binary_multiclass_performance_plots__1_.py`** – Performance visualization
  * Creates publication-quality figures comparing model performance:
    * Stacked bar charts of median misclassifications per class (binary and multiclass)
    * Grouped bar plots of precision, recall, and F1 scores across models and classes
    * Box plots showing performance distributions across cross-validation folds
  * Uses consistent colorblind-friendly palette across all visualizations
  * Includes error bars (standard deviation) for cross-validation metrics
  * Generates separate panels for binary (a) and multiclass (b) tasks

## Key Findings

### Expert Agreement

* **Low inter-rater reliability**: Fleiss κ = 0.15-0.20 for classification tasks, indicating substantial interpretive variability even among experienced micromorphologists
* **Uncertainty correlates with performance**: Negative Spearman correlation between "Unsure" responses and agreement with reference labels (r ≈ -0.6 to -0.8, p < 0.05)
* **Experience matters but doesn't eliminate disagreement**: Experts with "Significant experience" show higher Cohen's κ but still substantial disagreement

### Human-AI Comparison

* **The performance paradox**: CNNs achieve 94-99% accuracy despite expert agreement being much lower
* **Near-zero error correlation**: Where experts disagree with ground truth, models show no systematic pattern of making the same errors (r ≈ 0)
* **Complementary failure modes**: Models and experts make errors on different images, suggesting potential for human-in-the-loop workflows

### Image Difficulty Categories

Images are classified into four categories based on expert agreement and model confidence:

1. **Easy_Both**: High expert agreement (>80%) + high model confidence (>90%)
2. **Hard_Human**: Low expert agreement (<60%) but high model confidence
3. **Hard_AI**: High expert agreement but low model confidence or high BNN uncertainty
4. **Hard_Both**: Low agreement and low confidence

This categorization enables targeted analysis of where human and machine interpretations diverge.

## Survey Design

### Participants

* **N = 19** expert micromorphologists
* Expertise levels:
  * Beginner/Early Career: ~30%
  * Intermediate: ~35%
  * Advanced/Expert: ~35%
* Years in field: 1-5 (25%), 6-10 (30%), 11-20 (25%), >20 (20%)
* Frost experience:
  * Little to no experience: ~20%
  * Some experience: ~50%
  * Significant experience: ~30%

### Tasks

1. **First Screening Task** (`vis_img`): Binary visibility assessment of frost features
   * Question: "Is frost visible in this image?"
   * Options: Yes / No / Unsure

2. **Classification Task** (`class_img`): Detailed frost microstructure classification
   * 14 deliberately challenging images
   * Question: "What frost microstructure do you see?"
   * Options: Non-Frost / Granular / Platy-Lenticular / Blocky / Unsure

### Survey Methodology

* **Blind survey**: Experts did not see model predictions
* **Reference labels**: Ground truth provided by lead researcher (ID=20 in dataset)
* **Challenging subset**: Images selected to maximize interpretive difficulty, ensuring variability in responses

## Data Outputs

### Wide Format (`human_ai_alignment_analysis.csv`)

One row per image with columns:
* Ground truth labels (binary and multiclass)
* Per-expert predictions (`expert{ID}_binary`, `expert{ID}_multiclass`)
* Expert consensus metrics (majority vote, agreement %, entropy)
* Per-model predictions and probabilities for all 5 models
* Model consensus metrics (average confidence, majority prediction, agreement %)
* Alignment metrics (expert correctness, model correctness, consensus match)
* Difficulty category

### Long Format (`human_ai_alignment_long.csv`)

One row per rater (human or model) per image, suitable for:
* Cohen's κ calculations between any two raters
* Fleiss κ calculations across all raters
* Inter-rater reliability analyses
* Stratified analyses by expertise level or model type

### Expert Metadata (`expert_metadata.csv`)

* Expert ID
* Expertise level (Beginner/Intermediate/Advanced)
* Years of experience in micromorphology
* Self-reported frost experience level

## Visualization Outputs

### Combined Figure (Survey Analysis)

Four-panel figure saved as `combined_figure_2x2.tif` (330 dpi):

* **(a) Radar plot**: Comparing first screening vs. classification task across 5 dimensions:
  * Agreement with reference labels
  * Agreement among participants
  * Disagreement with reference labels
  * Disagreement among participants
  * Unsure responses

* **(b) Cohen's κ boxplot**: Inter-rater agreement stratified by frost experience level

* **(c) Scatter plot (visibility task)**: Uncertainty (% unsure) vs. performance (% agreement with reference)
  * Includes Spearman correlation coefficient and p-value
  * Color-coded by frost experience level
  * Trend line showing negative correlation

* **(d) Scatter plot (classification task)**: Same as (c) but for the classification task

### Sankey Diagram

Interactive Plotly visualization showing flow of participants:
* Expertise level → Years in field → Frost experience
* Color-coded by group (expertise: blues, years: greens, frost: reds)
* Title: "Responders' Profiles (Expertise → Years → Frost Experience)"

### Performance Comparison Figures

Saved as `combined_misclassifications_stacked_median.tif` (300 dpi):

* **(a) Binary misclassifications**: Stacked bar chart showing median errors for Non-Frost and Frost classes across all models
* **(b) Multiclass misclassifications**: Stacked bar chart for Blocky, Granular, and Platy-Lenticular errors

Additional figures include:
* Precision, recall, F1 bar plots with error bars
* Box plots of performance distributions across CV folds

## Dependencies

```python
numpy
pandas
seaborn
matplotlib
plotly
scipy
google-colab  # for Google Drive mounting
```

## Usage

### 1. Expert Survey Analysis

```python
# Run survey.py to generate participant profiles and agreement metrics
# Outputs: Sankey diagram, combined 2x2 figure, Fleiss/Cohen's kappa values

python survey.py
```

### 2. Human-AI Alignment Dataset

```python
# Run human_ai_alignment_2.py to build comprehensive alignment dataset
# Outputs: 
#   - human_ai_alignment_analysis.csv (wide format)
#   - human_ai_alignment_long.csv (long format)
#   - expert_metadata.csv

python human_ai_alignment_2.py
```

### 3. Performance Visualization

```python
# Run ml_frost_binary_multiclass_performance_plots__1_.py
# Outputs: Combined stacked bar charts, precision/recall/F1 plots

python ml_frost_binary_multiclass_performance_plots__1_.py
```

## Key Metrics Explained

### Fleiss' Kappa (κ)

* Measures inter-rater reliability across all experts simultaneously
* κ < 0: No agreement (worse than chance)
* 0 ≤ κ < 0.20: Slight agreement
* 0.20 ≤ κ < 0.40: Fair agreement
* 0.40 ≤ κ < 0.60: Moderate agreement
* 0.60 ≤ κ < 0.80: Substantial agreement
* κ ≥ 0.80: Almost perfect agreement

Our results (κ = 0.15-0.20) indicate **slight agreement**, revealing significant interpretive challenges even among experts.

### Cohen's Kappa

* Measures agreement between two raters (pairwise)
* Used to compare individual experts against ground truth
* Same interpretation scale as Fleiss κ

### Entropy

* Measures disorder/uncertainty in expert responses per image
* Higher entropy = more disagreement
* Computed two ways:
  1. Including "Unsure" as a category
  2. Excluding "Unsure" (decisive responses only)

### Composite Difficulty Score

Images categorized based on:
* Expert agreement threshold: 60-80%
* Model confidence threshold: 90%
* BNN uncertainty metrics (epistemic uncertainty from MC Dropout)

## Interpretation Guidelines

### The Performance Paradox

The apparent contradiction between high model accuracy and low expert agreement is explained by:

1. **Label noise is real but bounded**: Ground truth labels have uncertainty, but not complete randomness
2. **CNNs learn robust features**: Despite noisy labels, models extract stable visual patterns
3. **Different error distributions**: Experts and models fail on different images, suggesting complementary strengths
4. **Augmentation not automation**: Results support a human-in-the-loop workflow where models handle clear cases and experts review uncertain predictions

### Implications for Archaeological Micromorphology

* **Classification is reductive**: Collapsing rich visual information into discrete labels loses interpretive nuance
* **Automation should augment, not replace**: Use models to screen large datasets, flag uncertain cases for expert review
* **Inter-rater reliability matters**: Low κ values highlight need for explicit protocols and calibration exercises in micromorphology
* **Model interpretability is essential**: Score-CAM visualizations help experts understand *what* models see, not just *what* they predict

## Citation

If you use this survey methodology or alignment analysis in your research, please cite:

```
[Your citation here once published]
```

## Contact

For questions about survey design, expert recruitment, or alignment methodology:
[Your contact information]

## Acknowledgments

Special thanks to the 19 expert micromorphologists who participated in the blind survey.
