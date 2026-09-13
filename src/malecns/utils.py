import os
import random
import sys
import numpy as np

def set_seed(seed: int = 42):
    """
    Sets global seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"[Seed] Global seed set to: {seed}")

def mount_google_drive(mount_point: str = "/content/drive"):
    """
    Safely mounts Google Drive if running within a Google Colab notebook environment.
    """
    if "google.colab" in sys.modules:
        from google.colab import drive
        print("[Google Colab] Mounting Google Drive...")
        drive.mount(mount_point)
        print(f"[Google Colab] Google Drive mounted successfully at {mount_point}")
    else:
        print("[Local Environment] Not running in Google Colab. Drive mounting skipped.")
