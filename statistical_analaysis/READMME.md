'''Statistical comparison and performance plots
This part of the project contains two main analysis components:
A. ml_frost_statistical_evaluation.py
Performs a Friedman–Nemenyi comparison of all binary and multiclass classifiers using cross‑validation results stored in SI_1_ML_Results_Binary_Multiclass.xlsx.
​
For the binary task, it:
Reads the ml_results_binary sheet, trims to the main metric and misclassification columns, and extracts base model names and fold IDs from the Model column (e.g. Custom_CNN_f1, BNN_f3).
​Computes a composite performance score per fold and model that combines accuracy, the mean F1 of the Frost and Non‑frost classes, and inverted loss (1 − loss) with user‑defined weights.
Builds a Fold × Model table of composite scores, runs the Friedman test, and then applies the Nemenyi post‑hoc test to obtain pairwise p‑values and average ranks for all models.
For the multiclass task, it repeats the same pipeline on the ml_results_multiclass sheet, using a composite score based on accuracy, macro F1 over Blocky/Granular/Platy‑lenticular, and inverted loss.
Finally, it generates a combined critical difference (CD) diagram:

Top panel: binary models.
Bottom panel: multiclass models.
Models are placed on a rank axis, colored with a colorblind‑safe palette, and non‑significantly different models are connected by thick bars below the axis.
The script saves the CD figure as a 300 dpi .tif and prints Friedman statistics, p‑values, average ranks, and CD values for both tasks.
​

B. ml_frost_binary_multiclass_performance_plots.ipynb
Explores and visualizes the per‑class performance and training behavior of all models using the same Excel file.
It includes cells that:
Inspect the structure of ml_results_binary and ml_results_multiclass (sheet names, column layout, example rows) to confirm that accuracy, loss, per‑class precision/recall/F1, and misclassification counts are correctly stored before analysis.
Aggregate mean ± std of precision, recall, and F1 for each class and model (binary: Frost vs Non‑frost; multiclass: Blocky, Granular, Platy‑lenticular) across CV folds and produce multi‑panel bar plots comparing models on a 0–100% scale with error bars.
Summarize misclassification counts per class and model and visualize them as stacked bar charts for binary and multiclass tasks.
Load the training_dynamics_* sheets, which store training/validation accuracy and loss at fixed checkpoints, summarize them across folds, and plot learning curves (accuracy and loss vs training progress) for each model, with shaded bands showing variability.
Together, these two files provide the full statistical comparison (ranks and significance) and visual diagnostics (performance plots and learning curves) used to compare the CNN and BNN architectures on both binary frost detection and multiclass microstructure classification.'''
