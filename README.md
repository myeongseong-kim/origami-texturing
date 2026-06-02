# origami-texturing

Python package + notebook workspace for origami texturing experiments.

## Project Layout

```text
origami-texturing/
|-- data/
|   |-- raw/              # original input files, kept out of git
|   `-- processed/        # generated/cleaned data, kept out of git
|-- external/             # local third-party libraries, kept out of git
|-- inputs/               # source images for OpenAI image-editing runs
|-- notebooks/
|   `-- setup.ipynb       # import smoke test and starter notebook
|-- outputs/              # generated figures, meshes, textures, logs
|-- src/
|   `-- origami_texturing/
|       |-- __init__.py
|       |-- config.py
|       `-- paths.py
|-- tests/
|   `-- test_paths.py
|-- pyproject.toml
`-- requirements.txt
```

## Setup

Create a virtual environment, install the package in editable mode, then start Jupyter.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
jupyter lab
```

## Notebook Usage

After `python -m pip install -e .`, notebooks can import project code without modifying
`sys.path`:

```python
from origami_texturing import paths

paths.DATA_DIR
```

Use `inputs/` for source images passed to OpenAI image-editing runs, `data/raw/` for
original datasets, `data/processed/` for derived data, and `outputs/` for generated
experiment results. These folders contain `.gitkeep` files so the layout is versioned,
while generated contents remain ignored by git.

Use `external/` for local third-party libraries, SDKs, or source checkouts that need
to be referenced from a local path. The directory itself is versioned, while its
contents are ignored by git by default.
