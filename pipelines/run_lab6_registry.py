import subprocess
import sys

def run_step(script):
    print(f"\n[INFO] Running: {script}")
    result = subprocess.run(
        [sys.executable, script],
        check=True
    )
    return result

def main():
    print("=" * 60)
    print("IPL LAB 6 - MODEL REGISTRATION PIPELINE")
    print("=" * 60)

    run_step("src/register_model.py")

    print("\n[SUCCESS] Lab 6 model registration pipeline completed successfully.")

if __name__ == "__main__":
    main()