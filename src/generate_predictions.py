import os
import pandas as pd
import joblib

MODEL_PATH = os.path.join(
    "models", "logistic_regression.pkl"
)

OUTPUT_DIR = "outputs"


def main():
    print("[INFO] Generating Lab 5 predictions...")

    X_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "X_test.csv")
    )

    y_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "y_test.csv")
    ).squeeze("columns")

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "actual": y_test,
        "prediction": predictions,
        "probability_team1_wins": probabilities
    })

    output_path = os.path.join(
        OUTPUT_DIR, "predictions.csv"
    )

    results.to_csv(output_path, index=False)

    print(f"[INFO] Predictions generated: {len(results)}")
    print(f"[INFO] Saved: {output_path}")
    print("[SUCCESS] Prediction generation completed.")


if __name__ == "__main__":
    main()