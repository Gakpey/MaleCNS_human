"""
Connectome-to-Humanoid Motor Adapter Module
-------------------------------------------
Translates descending motor neuron (DN) firing rates from the MaleCNS connectome
into target joint angles, angular velocities, and motor torques for a 3D Humanoid physics body.
"""

import numpy as np
from typing import Dict, List, Tuple

class ConnectomeToHumanoidMapper:
    """
    Maps MaleCNS Descending Neuron firing rates to 3D Humanoid body degrees of freedom (DoFs).
    """
    
    # Standard Humanoid Joint Specifications and Motion Limits (in radians)
    HUMANOID_JOINTS = [
        {"name": "neck_pitch", "min": -0.5, "max": 0.5, "default": 0.0},
        {"name": "neck_yaw", "min": -0.8, "max": 0.8, "default": 0.0},
        {"name": "right_shoulder_pitch", "min": -1.5, "max": 1.5, "default": 0.0},
        {"name": "right_shoulder_roll", "min": -1.2, "max": 0.5, "default": 0.2},
        {"name": "right_elbow", "min": 0.0, "max": 2.2, "default": 0.5},
        {"name": "left_shoulder_pitch", "min": -1.5, "max": 1.5, "default": 0.0},
        {"name": "left_shoulder_roll", "min": -0.5, "max": 1.2, "default": -0.2},
        {"name": "left_elbow", "min": 0.0, "max": 2.2, "default": 0.5},
        {"name": "right_hip_pitch", "min": -0.8, "max": 1.0, "default": 0.0},
        {"name": "right_hip_roll", "min": -0.4, "max": 0.4, "default": 0.0},
        {"name": "right_knee", "min": -2.0, "max": 0.0, "default": -0.3},
        {"name": "right_ankle", "min": -0.5, "max": 0.5, "default": 0.0},
        {"name": "left_hip_pitch", "min": -0.8, "max": 1.0, "default": 0.0},
        {"name": "left_hip_roll", "min": -0.4, "max": 0.4, "default": 0.0},
        {"name": "left_knee", "min": -2.0, "max": 0.0, "default": -0.3},
        {"name": "left_ankle", "min": -0.5, "max": 0.5, "default": 0.0},
    ]

    def __init__(self, num_descending_neurons: int = 16, gain: float = 1.0, smoothing: float = 0.2):
        self.num_dns = num_descending_neurons
        self.num_joints = len(self.HUMANOID_JOINTS)
        self.gain = gain
        self.smoothing = smoothing
        
        # Projection Matrix mapping DNs -> Humanoid Joints (16x16 or NxM)
        np.random.seed(42)
        # Construct weight matrix connecting specific motor circuits (flight/walking/turning) to limbs
        self.projection_matrix = self._build_projection_matrix()
        
        # Filtered target joint angles
        self.current_joint_targets = np.array([j["default"] for j in self.HUMANOID_JOINTS])

    def _build_projection_matrix(self) -> np.ndarray:
        """
        Creates a structured mapping from fly motor neuron groups to human body joint targets:
        - DN_0..3: Head/Neck movement (visual tracking / gaze orientation)
        - DN_4..7: Right Arm & Left Arm (flight wing stroke / upper limb gesture)
        - DN_8..11: Gait / Leg Flexion (walking leg rhythm)
        - DN_12..15: Turning / Balance (asymmetric left vs right motor bias)
        """
        P = np.zeros((self.num_joints, self.num_dns))
        
        # Neck / Head
        P[0, 0] = 0.8  # neck_pitch
        P[1, 1] = 0.8  # neck_yaw
        
        # Arms (Upper Body)
        P[2, 2] = 1.2  # right_shoulder_pitch
        P[3, 3] = 0.8  # right_shoulder_roll
        P[4, 2] = 0.9  # right_elbow
        P[5, 4] = 1.2  # left_shoulder_pitch
        P[6, 5] = -0.8 # left_shoulder_roll
        P[7, 4] = 0.9  # left_elbow
        
        # Legs & Walking Gait
        P[8, 8] = 0.9   # right_hip_pitch
        P[10, 8] = -1.0 # right_knee
        P[11, 8] = 0.4  # right_ankle
        
        P[12, 9] = 0.9  # left_hip_pitch
        P[14, 9] = -1.0 # left_knee
        P[15, 9] = 0.4  # left_ankle
        
        # Steering / Turning asymmetry
        P[9, 12] = 0.5  # right_hip_roll (turning)
        P[13, 13] = -0.5 # left_hip_roll (turning)
        
        # Normalize projection matrix
        return P

    def map_dn_to_joints(self, dn_firing_rates: np.ndarray) -> np.ndarray:
        """
        Transforms fly descending neuron (DN) activation rates [0, 1] into target joint angles.
        
        Args:
            dn_firing_rates: Vector of descending neuron firing rates [num_descending_neurons]
            
        Returns:
            target_joint_angles: Target angle in radians for each humanoid joint [num_joints]
        """
        # Center activation around zero (-0.5 to +0.5 baseline)
        centered_rates = (dn_firing_rates[:self.num_dns] - 0.5) * 2.0
        
        # Raw projected angles
        raw_angles = np.dot(self.projection_matrix, centered_rates) * self.gain
        
        # Apply motion limits for each joint
        bounded_targets = np.zeros(self.num_joints)
        for i, joint_spec in enumerate(self.HUMANOID_JOINTS):
            default_val = joint_spec["default"]
            target_val = default_val + raw_angles[i]
            bounded_targets[i] = np.clip(target_val, joint_spec["min"], joint_spec["max"])
            
        # Low-pass exponential smoothing to prevent abrupt physics joint snaps
        self.current_joint_targets = (
            (1.0 - self.smoothing) * self.current_joint_targets + self.smoothing * bounded_targets
        )
        
        return self.current_joint_targets.copy()

    def get_joint_names(self) -> List[str]:
        """Returns ordered list of joint names."""
        return [j["name"] for j in self.HUMANOID_JOINTS]
