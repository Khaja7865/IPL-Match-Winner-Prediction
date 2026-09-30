import os
import pandas as pd

DATA_PATH = os.path.join(
    "Dataset", "Raw data", "matches (1).csv"
)


def main():
    errors = []

    print("[INFO] Validating IPL dataset...")

    if not os.path.exists(DATA_PATH):
        errors.append(f"Dataset not found: {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            errors.append("Dataset is empty.")

        required_columns = [
            "season",
            "city",
            "date",
            "venue",
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "winner"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            errors.append(
                f"Missing columns: {missing_columns}"
            )

        if df["winner"].notna().sum() == 0:
            errors.append("No valid winner records found.")

    result = {
        "status": "PASSED" if not errors else "FAILED",
        "rows": len(df) if os.path.exists(DATA_PATH) else 0,
        "columns": len(df.columns) if os.path.exists(DATA_PATH) else 0,
        "errors": errors
    }

    print(result)

    if errors:
        raise RuntimeError("Data validation failed.")

    print("[SUCCESS] Data validation passed.")


if __name__ == "__main__":
    main()