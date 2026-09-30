import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl"
}


def main():

    X_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "X_test.csv")
    )

    y_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "y_test.csv")
    ).squeeze("columns")

    results = []

    for name, filename in MODELS.items():

        path = os.path.join(MODEL_DIR, filename)

        print(f"[INFO] Evaluating {name}...")

        model = joblib.load(path)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Accuracy",
        ascending=False
    )

    print("\nMODEL EVALUATION")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_path = os.path.join(
        OUTPUT_DIR,
        "lab3_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    best_model_name = results_df.iloc[0]["Model"]

    print("\nBest model:", best_model_name)

    print(
        "\n[SUCCESS] Evaluation completed."
    )
    print(
        f"[INFO] Results saved to: {results_path}"
    )


if __name__ == "__main__":
    main()