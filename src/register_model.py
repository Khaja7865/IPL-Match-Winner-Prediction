import os
import joblib
import mlflow
import mlflow.sklearn


MODEL_PATH = os.path.join(
    "models", "logistic_regression.pkl"
)


def main():
    print("[INFO] Loading Logistic Regression model...")

    model = joblib.load(MODEL_PATH)

    mlflow.set_experiment(
        "IPL_Match_Winner_Prediction"
    )

    with mlflow.start_run(
        run_name="Logistic_Regression_Registry"
    ):
        mlflow.log_param(
            "model",
            "Logistic Regression"
        )

        model_info = mlflow.sklearn.log_model(
    model,
    name="ipl_logistic_regression",
    skops_trusted_types=["numpy.dtype"]
        )

        print("[INFO] Model logged to MLflow.")
        print(
            "[INFO] Model URI:",
            model_info.model_uri
        )

    print(
        "\n[SUCCESS] Lab 6 model registration completed."
    )


if __name__ == "__main__":
    main()