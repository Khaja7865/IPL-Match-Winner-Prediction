import subprocess
import sys

def run_step(description, command):
    print(f"\n[INFO] {description}")
    result = subprocess.run(command, check=True)
    return result.returncode

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
        "Running src/evaluate.py",
        [sys.executable, "src/evaluate.py"]
    )

    print("\n[SUCCESS] Lab 3 baseline pipeline completed successfully.")

if __name__ == "__main__":
    main()