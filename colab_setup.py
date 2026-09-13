"""
Google Colab Setup Script
-------------------------
Run this script in Google Colab to clone the repository, mount Google Drive,
install dependencies, and set up the Python path.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing command: {res.stderr}")
    else:
        print(res.stdout)

def setup_colab_env(repo_url: str = None, mount_drive: bool = True):
    if "google.colab" not in sys.modules:
        print("[Setup] Not running in Google Colab. Exiting Colab setup routine.")
        return

    print("==================================================")
    print("🚀 Initializing Google Colab Development Environment")
    print("==================================================")

    # 1. Mount Google Drive if requested
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("✔ Google Drive mounted at /content/drive")
        except Exception as e:
            print(f"⚠️ Drive mount warning: {e}")

    # 2. Clone repo if passed or ensure working directory
    if repo_url:
        target_dir = "/content/MaleCNS_human"
        if not os.path.exists(target_dir):
            print(f"Cloning repository from {repo_url}...")
            run_command(f"git clone {repo_url} {target_dir}")
        os.chdir(target_dir)
        print(f"✔ Working directory changed to: {os.getcwd()}")
    else:
        # Check current dir
        if Path("/content/MaleCNS_human").exists():
            os.chdir("/content/MaleCNS_human")
            print(f"✔ Changed directory to /content/MaleCNS_human")

    # 3. Add src/ to sys.path
    src_path = str(Path.cwd() / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✔ Added {src_path} to sys.path")

    # 4. Install local package in editable mode
    if Path("pyproject.toml").exists() or Path("setup.py").exists():
        print("Installing project packages...")
        run_command(f"{sys.executable} -m pip install -e .")

    print("\n✅ Google Colab environment successfully configured!")

if __name__ == "__main__":
    setup_colab_env()
