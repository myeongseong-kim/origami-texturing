"""Filesystem paths shared by notebooks, scripts, and tests."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
EXTERNAL_DIR = ROOT_DIR / "external"
INPUTS_DIR = ROOT_DIR / "inputs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUT_DIR = ROOT_DIR / "outputs"


def ensure_project_dirs() -> None:
    """Create writable project directories used during experiments."""

    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EXTERNAL_DIR, INPUTS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
