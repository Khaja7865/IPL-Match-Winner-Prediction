import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


INPUT_X = os.path.join("outputs", "X_ipl.csv")
INPUT_Y = os.path.join("outputs", "y_ipl.csv")

MODEL_DIR = "models"

CATEGORICAL_FEATURES = [
    "season",
    "city",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision"
]

NUMERICAL_FEATURES = [
    "team1_win_rate",
    "team2_win_rate"
]


def build_preprocessor():

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES)
    ])


def main():

    print("[INFO] Loading preprocessed data...")

    X = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y).squeeze("columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        print(f"[INFO] Training {name}...")

        pipeline.fit(X_train, y_train)

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        path = os.path.join(MODEL_DIR, filename)

        joblib.dump(pipeline, path)

        print(f"[INFO] Saved: {path}")

    # Save train/test data for evaluation
    X_train.to_csv(
        os.path.join("outputs", "X_train.csv"),
        index=False
    )

    X_test.to_csv(
        os.path.join("outputs", "X_test.csv"),
        index=False
    )

    y_train.to_csv(
        os.path.join("outputs", "y_train.csv"),
        index=False
    )

    y_test.to_csv(
        os.path.join("outputs", "y_test.csv"),
        index=False
    )

    print("[INFO] Train shape:", X_train.shape)
    print("[INFO] Test shape:", X_test.shape)

    print("[SUCCESS] IPL models trained and saved.")


if __name__ == "__main__":
    main()