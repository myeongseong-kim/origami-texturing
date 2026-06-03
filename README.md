# origami-texturing

Python package + notebook workspace for origami texturing experiments.

## Project Layout

```text
origami-texturing/
|-- data/
|   |-- models/
|   |-- raw/
|   `-- processed/
|-- external/
|   `-- dinov3/             # facebookresearch/dinov3 submodule
|-- inputs/
|-- notebooks/
|   `-- setup.ipynb
|-- outputs/
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

This project is written and tested with Python 3.12.10.

Create a virtual environment, install the package in editable mode, and register
the project notebook kernel.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name ".venv"
```

VS Code can open and run `.ipynb` notebooks directly after this setup. To use
the browser-based JupyterLab UI instead, start it separately:

```powershell
jupyter lab
```

## Notebook Usage

After `python -m pip install -e .`, notebooks can import project code without modifying
`sys.path`:

```python
from origami_texturing import paths

paths.DATA_DIR
```

Use `inputs/` for experiment inputs such as source images, crease patterns, masks,
or texture references. Store pretrained model weights in `data/models/`, original
datasets in `data/raw/`, derived or cleaned datasets in `data/processed/`, and
generated artifacts in `outputs/`. These folders contain `.gitkeep` files so the
layout is versioned, while generated contents remain ignored by git.

Do not commit pretrained model files to this repository. `data/models/.gitkeep` is
tracked only to preserve the directory; model weight files such as `.pth`, `.pt`,
`.ckpt`, `.onnx`, and `.safetensors` are ignored by git.

Use `external/` for local third-party libraries, SDKs, or source checkouts that need
to be referenced from a local path. The directory itself is versioned, while its
contents are ignored by git by default.

## CUDA PyTorch

`requirements.txt` includes `torch` and `torchvision`, but those package names alone
do not guarantee a CUDA-enabled PyTorch install. If `model.cuda()` fails with an
error such as "Torch not compiled with CUDA enabled", first check whether PyTorch
was installed as a CPU-only build:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If PyTorch reports a CPU-only build, reinstall `torch` and `torchvision` using the
CUDA install command from the official PyTorch installer. Select the CUDA option
that matches your NVIDIA driver:

<https://docs.pytorch.org/get-started/locally/>

Verify that the GPU is visible:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Restart the Jupyter kernel after changing the PyTorch install.

## External Submodules

This project keeps third-party research code under `external/` as git
submodules. The source trees are not copied into this repository directly;
instead, this repository tracks the exact upstream commits that the experiments
are expected to use.

Current submodules:

- `external/dinov3`: `https://github.com/facebookresearch/dinov3.git`, tracking
  upstream `main`.

Clone this repository with submodules to download this project and its external
research code:

```powershell
git clone --recurse-submodules <repo-url>
```

If the repository is already cloned, initialize submodules with:

```powershell
git submodule update --init --recursive
```

To move a submodule to a newer upstream `main` commit, update the submodule and
commit the changed submodule pointer:

```powershell
git submodule update --remote external/dinov3
git add external/dinov3
