# Analysis of Feature Selection Methods and Their Impact on Classification

This project analyzes the impact of different feature selection methods on the classification performance of models using multiple datasets. The feature selection methods include:

- **Variance Threshold**: Removing features with low variance.
- **Correlation-based**: Removing features with high correlation.
- **Feature Importance**: Using a Random Forest classifier to rank features by importance.
- **Recursive Feature Elimination (RFE)**: Recursively removing features to select the best subset.
- **SelectKBest**: Selecting the top k features based on statistical tests (ANOVA F-value).

## Objective

The objective of this project is to evaluate and compare the performance of these feature selection methods using various datasets and assess their impact on classification performance using Random Forest as the classifier. The datasets used include:

- **Mice Protein**: Protein expression data for mice.
- **Wine Quality**: Data for red wine quality prediction.
- **Breast Cancer**: Data for breast cancer diagnosis.

## Features

- **Feature Selection Methods**: Evaluate different feature selection techniques.
- **Cross-validation**: Performance evaluation using cross-validation with varying splits.
- **Random Forest Classifier**: Used as the model for classification.
- **Execution Time Measurement**: Measures the execution time for each feature selection method.
- **Visualization**: Visualize the impact of feature selection on accuracy using plots.

## Requirements

- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

You can install the necessary libraries using the following:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. **Prepare the Datasets**:
   - The datasets (`mice-protein.csv`, `winequality-red.csv`, `breast_cancer.csv`) should be in the same directory as the script.

2. **Run the Script**:
   - The script will load the datasets, perform preprocessing (imputation, scaling), and apply feature selection methods.
   - It will evaluate the accuracy of each feature selection method using cross-validation and plot the results.

3. **View Results**:
   - After execution, the script displays the average accuracy for each feature selection method on each dataset and plots the accuracy across different values of `k` (number of splits for cross-validation).

### Key Functions:
- `evaluate_model_cv(X, y, model, n_splits_list)`: Evaluates the classifier using cross-validation with different values of `k`.
- `remove_constant_features(X, feature_names)`: Removes features with zero variance.
- `remove_high_corr_features(X, threshold=0.8)`: Removes highly correlated features.
- `impute_missing_values(X)`: Imputes missing values in the dataset.
- `measure_execution_time(func, *args)`: Measures the execution time of a function.
- `evaluate_feature_selection_methods(datasets, n_splits_list)`: Evaluates the performance of different feature selection methods on multiple datasets.

### Example Output:

The script generates visualizations and outputs, including:

1. **Accuracy Comparison**: Plots showing the average accuracy for each feature selection method across different datasets.
2. **Feature Analysis**: Lists the selected features for each feature selection method.
3. **Execution Times**: Displays the execution time for each feature selection method.

## Results Example:

### Accuracy Comparison:

- **Mice Protein**: The plot compares the accuracy of each feature selection method using the Mice Protein dataset.
- **Wine Quality**: The plot compares the accuracy of each feature selection method using the Wine Quality dataset.
- **Breast Cancer**: The plot compares the accuracy of each feature selection method using the Breast Cancer dataset.

### Selected Features:

For each dataset, the script provides a list of features selected by each method:

- **Variance Threshold**: Selected features with high variance.
- **Correlation-based**: Features that are not highly correlated.
- **Feature Importance**: Top features based on Random Forest feature importance.
- **RFE**: Features selected by recursive feature elimination.
- **SelectKBest**: Top k features based on statistical significance.

### Execution Times:

The script also outputs the execution time for each feature selection method, allowing you to compare their efficiency.
