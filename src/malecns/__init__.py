"""
MaleCNS_human Python Package
----------------------------
Core connectome neural dynamics, physics-based 3D humanoid simulation,
and live WebGL rendering utilities optimized for Google Colab and local execution.
"""

from .config import IS_COLAB, get_device, get_project_root, setup_environment
from .utils import mount_google_drive, set_seed
from .connectome import MaleCNSNetwork
from .mapping import ConnectomeToHumanoidMapper
from .simulation import HumanoidPhysicsSim
from .render import LiveWebGLViewer, VideoRenderer

__version__ = "0.1.0"
__all__ = [
    "IS_COLAB",
    "get_device",
    "get_project_root",
    "setup_environment",
    "mount_google_drive",
    "set_seed",
    "MaleCNSNetwork",
    "ConnectomeToHumanoidMapper",
    "HumanoidPhysicsSim",
    "LiveWebGLViewer",
    "VideoRenderer",
]
