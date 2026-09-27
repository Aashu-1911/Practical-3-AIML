# Performance Evaluation of Regression Model

## 1. Overview

This repository contains TY B.Tech AIML Practical Assignment-03 on regression model development and Gradient Descent analysis. The work compares simple, multiple, and polynomial regression models and evaluates their predictions using standard regression metrics.

## 2. Case Study

The case study is a healthcare application: **Diabetes Disease Progression Prediction**. This is a supervised learning regression problem in which patient measurements are used to predict a quantitative disease-progression score.

## 3. Models Implemented

- Simple Linear Regression using BMI as the predictor
- Multiple Linear Regression using the available diabetes features
- Polynomial Regression using BMI with degrees 2 and 3
- Linear Regression optimized with Gradient Descent implemented from scratch

No unrelated ensemble or neural-network models are included.

## 4. Evaluation Metrics

The models are evaluated using:

- **Mean Squared Error (MSE):** average squared prediction error
- **Root Mean Squared Error (RMSE):** square root of MSE, in the target variable's units
- **Mean Absolute Error (MAE):** average absolute prediction error
- **R2:** proportion of target variance explained by the model

## 5. Gradient Descent

The scratch implementation uses batch Gradient Descent for linear regression. During training it:

1. Computes predictions and the mean squared cost function.
2. Calculates the gradients for the weights and bias.
3. Updates the parameters using the learning rate.
4. Records the cost after every epoch.

The source code includes convergence tracking and experiments comparing learning rates `0.001`, `0.01`, `0.05`, and `0.1`, using 1000 epochs.

## 6. Dataset

The project uses the built-in diabetes dataset provided by scikit-learn:

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
```

No separate dataset file is required or committed. The dataset is loaded automatically when the program runs.

## 7. Project Structure

```text
Practical-3/
├── README.md
├── requirements.txt
├── .gitignore
├── practical_assignment_03.ipynb
├── src/
│   └── regression_assignment.py
├── results/
│   ├── results_summary.csv
│   ├── sample_output.csv
│   ├── learning_rate_results.csv
│   └── plots/
│       ├── target_distribution.png
│       ├── bmi_vs_target.png
│       ├── correlation_matrix.png
│       ├── actual_vs_predicted_simple.png
│       ├── actual_vs_predicted_multiple.png
│       ├── simple_regression_line.png
│       ├── polynomial_regression.png
│       ├── gradient_descent_convergence.png
│       └── learning_rate_comparison.png
└── screenshots/
    └── README.md
```

The notebook in the working folder is currently named `Untitled1.ipynb`; it may be renamed to `practical_assignment_03.ipynb` for submission.

## 8. Installation

Create and activate a virtual environment, then install the required packages:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Running the Project

Run the extracted Python implementation from the repository root:

```bash
python src/regression_assignment.py
```

The program loads the dataset, trains the models, prints the model comparison, and writes the CSV result files and plots to `results/`.

The notebook can be opened and executed from top to bottom in Jupyter or Google Colab. Open the notebook directly in [Google Colab](https://colab.research.google.com/drive/1zShdNaoa62i2cDxKchCgthZOL54E1pzj#scrollTo=93-hT1V24cKi), then run the cells in order.

## 10. Results

The source program calculates the results at runtime and writes them to `results/results_summary.csv`. The file contains these columns:

```text
Model,MSE,RMSE,MAE,R2
```

The included model rows are:

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression Degree 2
- Polynomial Regression Degree 3
- Gradient Descent Linear Regression

Metric values should be read from the generated CSV after running the program; no values are manually hard-coded in this repository.

## 11. Sample Output

A small set of actual test predictions is written to `results/sample_output.csv` with the columns:

```text
Actual_Value,Predicted_Value,Model
```

These predictions are generated directly from the fitted models.

## 12. Graphs

The program saves the main analysis plots in `results/plots/`, including:

- Target distribution
- BMI versus disease progression
- Feature correlation matrix
- Simple and multiple regression predictions
- Polynomial regression comparison
- Gradient Descent convergence
- Learning-rate comparison

## 13. Reproducibility

The experiment uses an 80/20 train-test split with `random_state=42`. Standardization for Gradient Descent is fitted using the training data and then applied to the test data. The built-in scikit-learn dataset makes the experiment reproducible without downloading or storing a dataset file.

## 14. Limitations

- The experiment uses a relatively small built-in dataset.
- Model performance depends on the selected train-test split.
- Polynomial models use BMI as a single predictor and may not represent all feature interactions.
- Gradient Descent results depend on the learning rate, feature scaling, and number of epochs.
- This educational experiment is not a clinical diagnostic system.

## 15. Academic Context

Prepared for **TY B.Tech AIML Practical Assignment-03**.

## Screenshots

Screenshots are intentionally not fabricated. Add notebook execution, dataset/output, model-comparison, and Gradient Descent screenshots manually in `screenshots/` as required by the assignment document.
