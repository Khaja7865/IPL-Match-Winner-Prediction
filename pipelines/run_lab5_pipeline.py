import subprocess
import sys


def run_step(description, command):
    print(f"\n[INFO] {description}")
    subprocess.run(command, check=True)


def main():
    run_step(
        "Running src/validate_data.py",
        [sys.executable, "src/validate_data.py"]
    )

    run_step(
        "Running src/preprocess_pipeline.py",
        [sys.executable, "src/preprocess_pipeline.py"]
    )

    run_step(
        "Running src/train.py",
        [sys.executable, "src/train.py"]
    )

    run_step(
        "Running src/generate_predictions.py",
        [sys.executable, "src/generate_predictions.py"]
    )

    run_step(
        "Running src/validate_outputs.py",
        [sys.executable, "src/validate_outputs.py"]
    )

    print("\n[SUCCESS] Lab 5 end-to-end pipeline completed successfully.")


if __name__ == "__main__":
    main()