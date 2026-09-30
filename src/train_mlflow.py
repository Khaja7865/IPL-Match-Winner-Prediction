import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_FILE = os.path.join(MODEL_DIR, "logistic_regression.pkl")


def main():
    print("[INFO] Loading test data...")
    
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, "X_test.csv"))
    y_test = pd.read_csv(
        os.path.join(OUTPUT_DIR, "y_test.csv")
    ).squeeze("columns")

    print("[INFO] Loading Logistic Regression model...")
    model = joblib.load(MODEL_FILE)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    mlflow.set_experiment("IPL_Match_Winner_Prediction")

    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["numpy.dtype"]
)
        print("\nMLFLOW RESULTS")
        print("=" * 50)
        print(f"Accuracy : {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall   : {recall:.6f}")
        print(f"F1       : {f1:.6f}")
        print(f"ROC-AUC  : {roc_auc:.6f}")

    print("\n[SUCCESS] Lab 4 MLflow tracking completed.")


if __name__ == "__main__":
    main()