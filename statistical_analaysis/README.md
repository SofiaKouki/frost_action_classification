# Statistical Analysis: Friedman-Nemenyi Testing

This folder contains rigorous statistical comparisons of model performance across binary and multiclass frost classification tasks using non-parametric methods suitable for cross-validation results.

## Overview

Model performance is evaluated using **Friedman's test** (non-parametric omnibus test) followed by **Nemenyi post-hoc pairwise comparisons**. This approach is appropriate for:

* Comparing multiple models across the same folds (repeated measures design)
* Non-normal performance distributions
* Small sample sizes (K=5 folds)
* Avoiding assumptions of homoscedasticity

Results are visualized using **critical difference (CD) diagrams**, which show model rankings and identify statistically significant performance differences.

## File

### Script

* **`ml_frost_statistical_evaluation.py`** – Complete statistical pipeline
  * Loads binary and multiclass cross-validation results from Excel
  * Computes composite performance scores for each fold and model
  * Runs Friedman test to detect overall differences
  * Applies Nemenyi post-hoc test for pairwise comparisons
  * Generates combined critical difference diagram (binary + multiclass)
  * Saves results as high-resolution TIFF (300 dpi)

## Methodology

### 1. Composite Performance Score

Models are ranked using a **weighted composite score** combining multiple metrics:

#### Binary Classification

```
Composite = 0.4 × Accuracy + 0.4 × F1_avg + 0.2 × (1 - Loss)
```

Where:
* `Accuracy` = Overall classification accuracy
* `F1_avg` = Mean of F1 scores for Frost and Non-Frost classes
* `(1 - Loss)` = Inverted loss (higher is better)

**Rationale**: 
* Accuracy and F1 weighted equally (0.4 each) to balance overall performance with class-specific performance
* Inverted loss (0.2) provides additional signal about model confidence
* Sum of weights = 1.0

#### Multiclass Classification

```
Composite = 0.4 × Accuracy + 0.4 × Macro_F1 + 0.2 × (1 - Loss)
```

Where:
* `Accuracy` = Overall classification accuracy
* `Macro_F1` = Mean of F1 scores for Blocky, Granular, and Platy-Lenticular classes
* `(1 - Loss)` = Inverted loss

**Rationale**: Same weighting scheme as binary, but using macro-averaged F1 to treat all three classes equally.

### 2. Friedman Test

**Null hypothesis (H₀)**: All models have the same median performance across folds.

**Test statistic**: 
```
χ²_F = (12N / k(k+1)) × Σ(R_j² - k(k+1)²/4)
```

Where:
* N = number of folds (5)
* k = number of models (5: BNN, Custom CNN, MobileNet, EfficientNet, ResNet)
* R_j = sum of ranks for model j across folds

**Decision rule**: Reject H₀ if p < 0.05

**Why Friedman?**
* Non-parametric (no normality assumption)
* Accounts for within-fold dependencies (repeated measures)
* Robust to outliers
* Appropriate for small sample sizes

### 3. Nemenyi Post-Hoc Test

If Friedman test rejects H₀, pairwise comparisons are conducted using the **Nemenyi test**.

**Test statistic**:
```
CD = q_α √(k(k+1) / 6N)
```

Where:
* q_α = studentized range statistic at significance level α (default: 0.05)
* k = number of models
* N = number of folds

**Decision rule**: Two models are significantly different if their rank difference > CD.

**Why Nemenyi?**
* Controls familywise error rate across all pairwise comparisons
* Does not require equal sample sizes (though we have them)
* Conservative (less likely to detect false differences)

### 4. Average Ranks

Models are ranked within each fold (1 = best, k = worst). Average ranks across folds are computed and used for:
* Ordering models in the CD diagram
* Computing pairwise rank differences
* Comparing against the critical difference threshold

## Critical Difference Diagrams

### Interpretation Guide

The CD diagram visualizes:

1. **Horizontal axis**: Rank scale from 1 (best) to k (worst)
2. **Colored dots**: Model positions at their average rank
3. **Vertical lines**: Connect each model to its label below the axis
4. **Thick horizontal bars**: Connect models that are NOT significantly different (p > 0.05)
5. **CD value**: Shown in bottom-left box (minimum rank difference for significance)

**Example interpretation**:

```
Rank: 1 ────────────────────────────── 5
      ●━━━━━━━━●         ●       ●     ●
    Model_A  Model_B  Model_C Model_D Model_E
    (1.2)    (2.1)    (3.0)   (3.8)   (4.9)

CD = 1.5
```

* Model_A ranks best (avg rank = 1.2)
* Model_A and Model_B are NOT significantly different (rank diff = 0.9 < CD = 1.5)
* Model_A IS significantly better than Model_C, D, E (rank diff > CD)
* Model_E ranks worst but may not be significantly worse than Model_D (depends on CD)

### Color Scheme

Models are assigned unique colors from the **colorblind-friendly seaborn palette**:
* Colors remain consistent across binary and multiclass panels
* High contrast for clear visual separation
* White edge on scatter points for visibility

### Layout

The combined figure contains two vertically stacked diagrams:
* **(a) Binary classification** (top panel)
* **(b) Multiclass classification** (bottom panel)

Each panel shows:
* Rank axis with tick marks at integer ranks (1, 2, 3, ...)
* Model positions as colored dots
* Non-significant pairs connected by thick bars below the axis
* Model labels with average ranks in parentheses
* CD value annotated in a yellow box (bottom-left)

## Results

### Binary Classification

**Friedman Test**:
```
Friedman statistic: [value from output]
p-value: [value from output]
```

**Interpretation**: [If p < 0.05] There are significant differences among models. [If p ≥ 0.05] No significant differences detected.

**Average Ranks** (best to worst):
```
[Model rankings from output, e.g.:]
1. ResNet50: 1.2
2. EfficientNetV2B0: 2.0
3. MobileNetV2: 2.8
4. Custom_CNN: 3.5
5. BNN: 5.5
```

**Critical Difference (α=0.05)**: [CD value, e.g., 1.85]

**Pairwise Comparisons**:
* Models connected by bars are NOT significantly different
* Disconnected models have rank differences > CD (significant)

**Key Finding**: [Summarize which models cluster together and which are significantly different]

### Multiclass Classification

**Friedman Test**:
```
Friedman statistic: [value from output]
p-value: [value from output]
```

**Average Ranks** (best to worst):
```
[Model rankings from output]
```

**Critical Difference (α=0.05)**: [CD value]

**Key Finding**: [Summarize findings for multiclass task]

### Binary vs. Multiclass Comparison

* **Rank stability**: [Do models maintain relative rankings across tasks?]
* **CD values**: [Compare CD thresholds - are differences easier to detect in one task?]
* **Top performers**: [Which models consistently rank highest?]

## Statistical Power and Limitations

### Power

With K=5 folds and k=5 models:
* **Minimum detectable rank difference** ≈ CD value
* **Effect size**: Large differences (>2 rank units) are reliably detected
* **Moderate differences** (1-2 ranks) may or may not reach significance

### Limitations

1. **Small sample size**: K=5 provides limited power; differences must be large and consistent to achieve significance
2. **Conservative post-hoc test**: Nemenyi controls familywise error rate, reducing false positives but potentially missing true differences
3. **Composite score weights**: Results depend on chosen weights (0.4/0.4/0.2); sensitivity analysis could explore alternative schemes
4. **Independence assumption**: Assumes folds are independent (reasonable for stratified K-fold CV)

### Recommendations

* **CD is a *minimum* detectable difference**: Rank differences smaller than CD may still reflect real performance differences, but sample size is insufficient to confirm with 95% confidence
* **Consider practical significance**: Even if not statistically significant, a 0.5-rank difference might be meaningful in practice
* **Use CD diagrams alongside raw metrics**: Don't rely solely on significance testing—inspect precision/recall/F1 values directly

## Dependencies

```python
pandas
numpy
matplotlib
seaborn
scipy
scikit-posthocs  # For Nemenyi test
google-colab  # For Google Drive mounting
```

Install `scikit-posthocs`:
```bash
pip install scikit-posthocs
```

## Usage

### Running the Analysis

```python
# Ensure Excel file is accessible in Google Drive
# Path: /content/drive/My Drive/SI_1_ML_Results_Binary_Multiclass.xlsx

# Run the script
python ml_frost_statistical_evaluation.py
```

### Expected Output

1. **Console output**:
   ```
   BINARY CLASSIFICATION RESULTS
   ============================================================
   Friedman statistic: X.XXXX, p-value: X.XXXXXX
   
   Average Ranks:
   [Model rankings]
   
   Binary Critical Difference (α=0.05): X.XXXX
   
   MULTICLASS CLASSIFICATION RESULTS
   ============================================================
   [Similar output for multiclass]
   ```

2. **Saved figure**: `combined_critical_difference_diagram.tif` (300 dpi, TIFF format)

### Customization

**Adjust composite score weights**:
```python
# In calculate_composite_score_binary():
composite = (0.4 * accuracy + 0.4 * f1_avg + 0.2 * inverted_loss)

# Modify weights (must sum to 1.0), e.g.:
composite = (0.5 * accuracy + 0.3 * f1_avg + 0.2 * inverted_loss)
```

**Change significance level**:
```python
# Default α = 0.05
cd_binary = plot_cd_subplot(..., alpha=0.05)

# Use α = 0.01 for stricter threshold:
cd_binary = plot_cd_subplot(..., alpha=0.01)
```

**Modify figure size**:
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # Default

# Adjust width or height:
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
```

## Excel File Format

The script expects `SI_1_ML_Results_Binary_Multiclass.xlsx` with two sheets:

### Sheet: `ml_results_binary`

Columns:
* `Model`: Model name with fold suffix (e.g., `ResNet50_f1`, `MobileNetV2_test`)
* `Accuracy`: Overall accuracy (0-1)
* `Loss`: Training loss
* `F1 Score Frost`: F1 for Frost class
* `F1 Score Non frost`: F1 for Non-Frost class
* `Misclassification Non Frost`: Error count for Non-Frost
* `Misclassifications Frost`: Error count for Frost

**Important**: Only rows where `Model` ends in `_f1`, `_f2`, ..., `_f5` are used for statistical testing. Test set results (`_test`) are excluded.

### Sheet: `ml_results_multiclass`

Columns:
* `Model`: Model name with fold suffix
* `Accuracy`: Overall accuracy (0-1)
* `Loss`: Training loss
* `F1-Score Blocky`: F1 for Blocky class
* `F1-Score Granular`: F1 for Granular class
* `F1-Score Platy-Lenticular`: F1 for Platy-Lenticular class
* `Misclassifications Blocky`: Error count for Blocky
* `Misclassifications Granular`: Error count for Granular
* `Misclassifications Platy-Lenticular`: Error count for Platy-Lenticular

## Theoretical Background

### Why Non-Parametric Methods?

Cross-validation metrics often violate parametric assumptions:

1. **Non-normality**: Performance distributions may be skewed or have outliers
2. **Small sample size**: K=5 provides limited data for normality tests
3. **Dependency structure**: Folds share training data (though this is mitigated by stratification)

Friedman-Nemenyi is the standard approach for comparing classifiers across CV folds (see Demšar 2006, JMLR).

### Rank-Based vs. Value-Based Testing

**Ranks** (used here):
* Robust to outliers and non-normality
* Focus on relative ordering, not absolute differences
* Appropriate when "which model is better" matters more than "how much better"

**Values** (e.g., paired t-test):
* More powerful when assumptions hold
* Detect smaller differences
* Requires normality and equal variances

**Trade-off**: We sacrifice some power for robustness.

### Critical Difference Interpretation

CD represents the **minimum average rank difference** required for significance at α = 0.05 with Nemenyi correction.

**Key insight**: CD depends on:
* Number of models k (more models → larger CD)
* Number of folds N (more folds → smaller CD)
* Significance level α (stricter α → larger CD)

For our setup (k=5, N=5, α=0.05), CD ≈ 1.8-2.0 (exact value computed from studentized range).

## References

1. **Demšar, J. (2006)**. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
   * Seminal paper establishing Friedman-Nemenyi as the standard for ML classifier comparisons

2. **Friedman, M. (1937)**. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675-701.

3. **Nemenyi, P. (1963)**. Distribution-free multiple comparisons. *PhD thesis*, Princeton University.

4. **García, S., & Herrera, F. (2008)**. An extension on "statistical comparisons of classifiers over multiple data sets" for all pairwise comparisons. *Journal of Machine Learning Research*, 9, 2677-2694.

## Citation

If you use this statistical methodology in your research, please cite:

```
[Your citation here once published]

And cite the methodological paper:
Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. 
Journal of Machine Learning Research, 7, 1-30.
```
