# Set-Location "e:\collage project prompt engeeniering"; & "e:/collage project prompt engeeniering/.venv/Scripts/python.exe" run_project.py


"""
run_project.py
--------------
One-click script to run the entire Prompt Quality Analyzer project.
  1. Generates the dataset (if not already present)
  2. Trains the ML model
  3. Starts the Flask web server

Run: python run_project.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

DATASET_PATH = os.path.join(ROOT, "dataset", "prompt_dataset.csv")
MODEL_PATH = os.path.join(ROOT, "models", "prompt_model.pkl")


def run_script(name, path):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run([PYTHON, path], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed (exit code {result.returncode}).")
        sys.exit(result.returncode)


def main():
    print("=" * 60)
    print("  PROMPT QUALITY ANALYZER — Full Project Runner")
    print("=" * 60)

    # Step 1 — Generate dataset
    if os.path.exists(DATASET_PATH):
        print(f"\nDataset already exists at: {DATASET_PATH}")
        print("Skipping generation. Delete the file to regenerate.")
    else:
        run_script("Generate Dataset", os.path.join(ROOT, "generate_dataset.py"))

    # Step 2 — Train model
    if os.path.exists(MODEL_PATH):
        print(f"\nTrained model already exists at: {MODEL_PATH}")
        print("Skipping training. Delete models/ folder to retrain.")
    else:
        run_script("Train Model", os.path.join(ROOT, "train_model.py"))

    # Step 3 — Start web server
    print(f"\n{'=' * 60}")
    print("  STEP: Start Web Server")
    print(f"{'=' * 60}")
    print("\n  Open your browser at: http://127.0.0.1:8000")
    print("  Press Ctrl+C to stop the server.\n")
    subprocess.run([PYTHON, os.path.join(ROOT, "app.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
