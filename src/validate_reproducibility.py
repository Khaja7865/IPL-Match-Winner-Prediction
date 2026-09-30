import os
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join("models", "logistic_regression.pkl")
OUTPUT_DIR = "outputs"


def evaluate_once():
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv"))
    y_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "y_test.csv")
    ).squeeze("columns")

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    return accuracy_score(y_test, predictions)


def main():
    print("[INFO] Running reproducibility check...")

    execution_1 = evaluate_once()
    execution_2 = evaluate_once()

    is_reproducible = execution_1 == execution_2

    result = {
        "model": "Logistic Regression",
        "execution_1_accuracy": execution_1,
        "execution_2_accuracy": execution_2,
        "is_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED"
    }

    print(result)

    if not is_reproducible:
        raise RuntimeError("Reproducibility check failed.")

    print("[SUCCESS] Reproducibility validation passed.")


if __name__ == "__main__":
    main()