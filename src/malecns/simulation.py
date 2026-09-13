"""
3D Humanoid Physics Simulation Module
--------------------------------------
Headless 3D physics environment powered by PyBullet / MuJoCo.
Loads a rigged 3D Humanoid model, configures ground plane, gravity, camera,
and applies mapped joint position targets at each timestep.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

class HumanoidPhysicsSim:
    """
    Manages 3D Humanoid physics simulation environment in PyBullet.
    """
    def __init__(self, render_mode: str = "DIRECT", time_step: float = 1.0 / 240.0):
        self.render_mode = render_mode
        self.time_step = time_step
        self.physics_client = None
        self.humanoid_id = None
        self.plane_id = None
        self.joint_indices = {}
        self.joint_names = []
        
        self.camera_distance = 2.5
        self.camera_yaw = 45.0
        self.camera_pitch = -20.0
        self.camera_target = [0.0, 0.0, 1.0]
        
        self._init_physics()

    def _init_physics(self):
        """Initializes PyBullet physics engine in DIRECT (headless) mode."""
        try:
            import pybullet as p
            import pybullet_data
            
            # Connect in DIRECT (headless for Colab) or GUI mode
            connection_mode = p.DIRECT if self.render_mode.upper() == "DIRECT" else p.GUI
            self.physics_client = p.connect(connection_mode)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.time_step)
            
            # Load ground plane
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Load Humanoid 3D URDF model
            humanoid_start_pos = [0, 0, 1.25]
            humanoid_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            self.humanoid_id = p.loadURDF(
                "humanoid/humanoid.urdf",
                humanoid_start_pos,
                humanoid_start_orientation,
                useFixedBase=False
            )
            
            # Discover and catalogue available joints
            num_joints = p.getNumJoints(self.humanoid_id)
            for j in range(num_joints):
                info = p.getJointInfo(self.humanoid_id, j)
                joint_name = info[1].decode("utf-8")
                joint_type = info[2]
                # Filter revolute or spherical controllable joints
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL]:
                    self.joint_indices[joint_name] = j
                    self.joint_names.append(joint_name)
                    
            print(f"[Physics Sim] 3D Humanoid loaded successfully! Controllable Joints: {len(self.joint_indices)}")
            
        except ImportError:
            print("[Physics Sim Warning] pybullet module not found. Falling back to synthetic simulation mode.")
            self.physics_client = None

    def apply_joint_targets(self, target_angles: np.ndarray, max_force: float = 150.0):
        """
        Applies joint motor target positions to the 3D Humanoid joints.
        
        Args:
            target_angles: Target joint position vector [num_joints]
            max_force: Maximum motor torque limit
        """
        if self.physics_client is None:
            return
            
        import pybullet as p
        
        num_targets = len(target_angles)
        for i, (j_name, j_idx) in enumerate(self.joint_indices.items()):
            if i < num_targets:
                target_pos = target_angles[i]
                p.setJointMotorControl2(
                    bodyUniqueId=self.humanoid_id,
                    jointIndex=j_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=max_force
                )

    def step(self):
        """Steps physics engine forward by one delta time step."""
        if self.physics_client is not None:
            import pybullet as p
            p.stepSimulation()

    def get_state(self) -> Dict:
        """Returns 3D Humanoid position, orientation, and joint values."""
        if self.physics_client is None:
            return {"position": [0, 0, 1.0], "orientation": [0, 0, 0, 1]}
            
        import pybullet as p
        pos, orient = p.getBasePositionAndOrientation(self.humanoid_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.humanoid_id)
        
        return {
            "position": list(pos),
            "orientation": list(orient),
            "linear_velocity": list(linear_vel),
            "angular_velocity": list(angular_vel)
        }

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Renders off-screen RGB frame of the 3D physics environment.
        
        Returns:
            rgb_array: NumPy array of shape (height, width, 3) containing RGB image pixel values
        """
        if self.physics_client is None:
            # Fallback synthetic frame generator
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        import pybullet as p
        
        # Dynamically follow humanoid body position
        if self.humanoid_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
            camera_target = [pos[0], pos[1], max(pos[2], 0.5)]
        else:
            camera_target = self.camera_target

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / float(height),
            nearVal=0.1,
            farVal=100.0
        )
        
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            lightDirection=[1, 1, 2],
            lightColor=[1, 1, 1],
            lightDistance=5.0,
            shadow=1,
            renderer=p.ER_TINY_RENDERER
        )
        
        # PyBullet returns RGBA uint8 buffer -> extract RGB
        rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
        return rgb_array.astype(np.uint8)

    def close(self):
        """Cleanly disconnects physics engine."""
        if self.physics_client is not None:
            import pybullet as p
            p.disconnect()
            self.physics_client = None
