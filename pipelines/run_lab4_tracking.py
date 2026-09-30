import subprocess
import sys


def run_step(description, command):
    print(f"\n[INFO] {description}")
    subprocess.run(command, check=True)


def main():
    run_step(
        "Running src/preprocess.py",
        [sys.executable, "src/preprocess.py"]
    )

    run_step(
        "Running src/train.py",
        [sys.executable, "src/train.py"]
    )

    run_step(
        "Running src/train_mlflow.py",
        [sys.executable, "src/train_mlflow.py"]
    )

    run_step(
        "Running src/validate_reproducibility.py",
        [sys.executable, "src/validate_reproducibility.py"]
    )

    print("\n[SUCCESS] Lab 4 tracking pipeline completed successfully.")


if __name__ == "__main__":
    main()