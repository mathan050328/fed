"""
Creates a clean, portable zip archive of the project excluding virtual environment and cache folders.
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "federated_disease_prediction_project.zip"

EXCLUDE_DIRS = {".venv", ".pytest_cache", "__pycache__", ".git", ".system_generated"}
EXCLUDE_FILES = {"federated_disease_prediction_project.zip"}


def make_archive():
    print(f"Creating zip archive at: {ZIP_PATH}")
    count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(BASE_DIR)
                zipf.write(file_path, arcname)
                count += 1

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {count} files into {ZIP_PATH.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    make_archive()
