import sys
import os
from pathlib import Path

# Detect Google Colab execution environment
IS_COLAB = "google.colab" in sys.modules

def get_project_root() -> Path:
    """
    Returns the root directory of the project.
    Works dynamically whether running locally or in Google Colab.
    """
    if IS_COLAB:
        # In Google Colab, default to /content/MaleCNS_human or current working directory
        content_dir = Path("/content/MaleCNS_human")
        if content_dir.exists():
            return content_dir
        return Path.cwd()
    else:
        # Local execution: root directory relative to this config file
        return Path(__file__).resolve().parent.parent.parent

def get_device():
    """
    Determines available computing device (CUDA GPU, Apple MPS, or CPU).
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device Config] Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device Config] Using Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            print("[Device Config] Using CPU")
        return device
    except ImportError:
        print("[Device Config] PyTorch not installed. Defaulting to CPU string.")
        return "cpu"

def setup_environment(gdrive_folder: str = "MaleCNS_human"):
    """
    Automates path configurations and directory creation for Colab and local setups.
    """
    root = get_project_root()
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    results_dir = root / "results"
    models_dir = root / "models"

    for d in [data_raw, data_processed, results_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Environment Ready] Project root set to: {root}")
    return {
        "root": root,
        "raw_data": data_raw,
        "processed_data": data_processed,
        "results": results_dir,
        "models": models_dir,
        "is_colab": IS_COLAB
    }
