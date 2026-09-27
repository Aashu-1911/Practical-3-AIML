from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 1000
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1]
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


class LinearRegressionGD:
    """Linear regression optimized with batch Gradient Descent."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.epochs):
            predictions = np.dot(X, self.weights) + self.bias
            error = predictions - y
            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error) / n_samples
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.cost_history.append(np.mean(error**2) / 2)

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def evaluate_model(model_name, y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, predictions),
        "R2": r2_score(y_true, predictions),
    }


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    df = X.copy()
    df["target"] = y

    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=30)
    plt.xlabel("Disease Progression")
    plt.ylabel("Frequency")
    plt.title("Distribution of Diabetes Disease Progression")
    save_plot(PLOTS_DIR / "target_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["bmi"], df["target"])
    plt.xlabel("BMI")
    plt.ylabel("Disease Progression")
    plt.title("BMI vs Diabetes Disease Progression")
    save_plot(PLOTS_DIR / "bmi_vs_target.png")

    plt.figure(figsize=(10, 8))
    plt.imshow(df.corr(), cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Correlation Matrix")
    save_plot(PLOTS_DIR / "correlation_matrix.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_bmi = X[["bmi"]]
    X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(
        X_bmi, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    simple_model = LinearRegression().fit(X_train_bmi, y_train_bmi)
    y_pred_simple = simple_model.predict(X_test_bmi)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_bmi, y_pred_simple)
    min_value = min(y_test_bmi.min(), y_pred_simple.min())
    max_value = max(y_test_bmi.max(), y_pred_simple.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual Disease Progression")
    plt.ylabel("Predicted Disease Progression")
    plt.title("Actual vs Predicted - Simple Linear Regression")
    save_plot(PLOTS_DIR / "actual_vs_predicted_simple.png")

    sorted_indices = np.argsort(X_test_bmi["bmi"].values)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test_bmi["bmi"], y_test_bmi, label="Actual")
    plt.plot(
        X_test_bmi["bmi"].values[sorted_indices],
        y_pred_simple[sorted_indices],
        linestyle="--",
        label="Regression Line",
    )
    plt.xlabel("BMI")
    plt.ylabel("Disease Progression")
    plt.title("Simple Linear Regression: BMI vs Disease Progression")
    plt.legend()
    save_plot(PLOTS_DIR / "simple_regression_line.png")

    multiple_model = LinearRegression().fit(X_train, y_train)
    y_pred_multiple = multiple_model.predict(X_test)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_multiple)
    min_value = min(y_test.min(), y_pred_multiple.min())
    max_value = max(y_test.max(), y_pred_multiple.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual Disease Progression")
    plt.ylabel("Predicted Disease Progression")
    plt.title("Actual vs Predicted - Multiple Linear Regression")
    save_plot(PLOTS_DIR / "actual_vs_predicted_multiple.png")

    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train_bmi)
    X_test_poly2 = poly2.transform(X_test_bmi)
    poly2_model = LinearRegression().fit(X_train_poly2, y_train_bmi)
    y_pred_poly2 = poly2_model.predict(X_test_poly2)

    poly3 = PolynomialFeatures(degree=3)
    X_train_poly3 = poly3.fit_transform(X_train_bmi)
    X_test_poly3 = poly3.transform(X_test_bmi)
    poly3_model = LinearRegression().fit(X_train_poly3, y_train_bmi)
    y_pred_poly3 = poly3_model.predict(X_test_poly3)

    bmi_range = np.linspace(X_bmi["bmi"].min(), X_bmi["bmi"].max(), 200).reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_bmi["bmi"], y_test_bmi, label="Actual Test Data")
    plt.plot(bmi_range, simple_model.predict(bmi_range), linestyle="--", label="Linear")
    plt.plot(
        bmi_range,
        poly2_model.predict(poly2.transform(bmi_range)),
        linestyle="--",
        label="Polynomial Degree 2",
    )
    plt.plot(
        bmi_range,
        poly3_model.predict(poly3.transform(bmi_range)),
        linestyle="--",
        label="Polynomial Degree 3",
    )
    plt.xlabel("BMI")
    plt.ylabel("Disease Progression")
    plt.title("Comparison of Linear and Polynomial Regression")
    plt.legend()
    save_plot(PLOTS_DIR / "polynomial_regression.png")

    scaler = StandardScaler()
    X_train_gd = scaler.fit_transform(X_train)
    X_test_gd = scaler.transform(X_test)
    gd_model = LinearRegressionGD(learning_rate=0.01, epochs=EPOCHS).fit(
        X_train_gd, y_train
    )
    y_pred_gd = gd_model.predict(X_test_gd)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), gd_model.cost_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Convergence")
    plt.grid(True)
    save_plot(PLOTS_DIR / "gradient_descent_convergence.png")

    plt.figure(figsize=(9, 6))
    for learning_rate in LEARNING_RATES:
        model = LinearRegressionGD(learning_rate=learning_rate, epochs=EPOCHS).fit(
            X_train_gd, y_train
        )
        plt.plot(model.cost_history, label=f"Learning Rate = {learning_rate}")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Effect of Learning Rate on Gradient Descent")
    plt.legend()
    plt.grid(True)
    save_plot(PLOTS_DIR / "learning_rate_comparison.png")

    result_rows = [
        evaluate_model("Simple Linear Regression", y_test_bmi, y_pred_simple),
        evaluate_model("Multiple Linear Regression", y_test, y_pred_multiple),
        evaluate_model("Polynomial Regression Degree 2", y_test_bmi, y_pred_poly2),
        evaluate_model("Polynomial Regression Degree 3", y_test_bmi, y_pred_poly3),
        evaluate_model("Gradient Descent Linear Regression", y_test, y_pred_gd),
    ]
    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS_DIR / "results_summary.csv", index=False)

    sample_output = pd.concat(
        [
            pd.DataFrame(
                {
                    "Actual_Value": y_test_bmi.to_numpy()[:3],
                    "Predicted_Value": y_pred_simple[:3],
                    "Model": "Simple Linear Regression",
                }
            ),
            pd.DataFrame(
                {
                    "Actual_Value": y_test.to_numpy()[:3],
                    "Predicted_Value": y_pred_multiple[:3],
                    "Model": "Multiple Linear Regression",
                }
            ),
            pd.DataFrame(
                {
                    "Actual_Value": y_test.to_numpy()[:3],
                    "Predicted_Value": y_pred_gd[:3],
                    "Model": "Gradient Descent Linear Regression",
                }
            ),
        ],
        ignore_index=True,
    )
    sample_output.to_csv(RESULTS_DIR / "sample_output.csv", index=False)

    learning_rate_rows = []
    for learning_rate in LEARNING_RATES:
        model = LinearRegressionGD(learning_rate=learning_rate, epochs=EPOCHS).fit(
            X_train_gd, y_train
        )
        predictions = model.predict(X_test_gd)
        learning_rate_rows.append(
            {
                "Learning Rate": learning_rate,
                "Final Cost": model.cost_history[-1],
                **evaluate_model("Gradient Descent", y_test, predictions),
            }
        )
    pd.DataFrame(learning_rate_rows).drop(columns="Model").to_csv(
        RESULTS_DIR / "learning_rate_results.csv", index=False
    )

    print(results.to_string(index=False))
    print(f"\nWrote results and plots to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()