import os
import pandas as pd

OUTPUT_DIR = "outputs"


def main():
    errors = []

    print("[INFO] Validating Lab 5 outputs...")

    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
        "X_pipeline.csv",
        "y_pipeline.csv"
    ]

    for filename in required_files:
        path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(path):
            errors.append(f"Missing file: {filename}")

    if not errors:
        X_train = pd.read_csv(
            os.path.join(OUTPUT_DIR, "X_train.csv")
        )
        X_test = pd.read_csv(
            os.path.join(OUTPUT_DIR, "X_test.csv")
        )
        y_train = pd.read_csv(
            os.path.join(OUTPUT_DIR, "y_train.csv")
        )
        y_test = pd.read_csv(
            os.path.join(OUTPUT_DIR, "y_test.csv")
        )

        predictions_path = os.path.join(
            OUTPUT_DIR, "predictions.csv"
        )

        if os.path.exists(predictions_path):
            predictions = pd.read_csv(predictions_path)
            prediction_count = len(predictions)
        else:
            prediction_count = 0
            errors.append("Missing file: predictions.csv")

        if len(X_train) != len(y_train):
            errors.append("X_train and y_train row counts differ.")

        if len(X_test) != len(y_test):
            errors.append("X_test and y_test row counts differ.")

        if prediction_count != len(X_test):
            errors.append(
                "Prediction count does not match X_test rows."
            )

    result = {
        "status": "PASSED" if not errors else "FAILED",
        "X_train_shape": list(X_train.shape) if not errors else [],
        "X_test_shape": list(X_test.shape) if not errors else [],
        "predictions": prediction_count if not errors else 0,
        "errors": errors
    }

    print(result)

    if errors:
        raise RuntimeError("Lab 5 output validation failed.")

    print("[SUCCESS] Lab 5 output validation passed.")


if __name__ == "__main__":
    main()