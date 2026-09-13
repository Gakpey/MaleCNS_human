# MaleCNS_human Python Development Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/MaleCNS_human/blob/main/notebooks/01_starter_colab.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Python project environment for **MaleCNS_human** analysis, configured to work seamlessly across local development and **Google Colab** (CPU/GPU/TPU).

---

## 📁 Directory Structure

```
MaleCNS_human/
├── .gitignore              # Ignores bytecode, virtualenvs, datasets, models, and checkpoints
├── README.md               # Documentation and execution guides
├── pyproject.toml          # Modern Python packaging configuration (pip install -e .)
├── requirements.txt        # Package dependencies list
├── colab_setup.py          # Automated Google Colab environment setup script
├── notebooks/
│   └── 01_starter_colab.ipynb # Starter Colab notebook
├── src/
│   └── malecns/            # Main modular Python library
│       ├── __init__.py     # Package initialization and exports
│       ├── config.py       # Auto-detects Colab vs Local paths & GPU/CPU device
│       └── utils.py        # Utility functions (Drive mounting, random seed setting)
├── data/
│   ├── raw/                # Unprocessed input data
│   └── processed/          # Cleaned/processed datasets
├── models/                 # Saved model weights (.pth, .pt, .pkl)
└── results/                # Output figures, plots, and analysis tables
```

---

## 🚀 Running on Google Colab

1. **Open Notebook in Colab**: Click the **Open In Colab** badge above or upload `notebooks/01_starter_colab.ipynb` to Google Colab.
2. **Execute Setup Cell**: The starter notebook automatically detects the Colab runtime and handles installation:
   ```python
   import sys
   IS_COLAB = "google.colab" in sys.modules

   if IS_COLAB:
       # Mount Google Drive if using Drive datasets
       from google.colab import drive
       drive.mount('/content/drive')
       
       # Install requirements & local package
       !pip install -q -r requirements.txt
       !pip install -e .
   ```
3. **GPU Accelerator**: Go to **Runtime > Change runtime type** in Colab and select **T4 GPU** or **A100 GPU** if acceleration is needed.

---

## 💻 Running Locally

### 1. Clone & Navigate to Repository
```bash
cd /home/joshua/Documents/Work/MaleCNS_human
```

### 2. Create Virtual Environment & Install Package
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

### 3. Launch JupyterLab or Notebook
```bash
jupyter lab
# or
jupyter notebook
```

---

## 🛠️ Usage in Python Scripts or Notebooks

```python
from malecns.config import setup_environment, get_device
from malecns.utils import set_seed, mount_google_drive

# 1. Mount Google Drive if in Colab (no-op if local)
mount_google_drive()

# 2. Set reproducible seed
set_seed(42)

# 3. Initialize project paths & compute device
paths = setup_environment()
device = get_device()

print(f"Raw data path: {paths['raw_data']}")
print(f"Results path:  {paths['results']}")
```

---

## ⚡ Key Features

- **Colab & Local Dual Compatibility**: Works seamlessly without needing path refactoring.
- **Dynamic Path Management**: Automatically resolves paths relative to `/content/MaleCNS_human` in Colab or local root directory.
- **Reproducibility**: Pre-packaged seed setter for Python, NumPy, and PyTorch.
