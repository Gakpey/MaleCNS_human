"""
Live WebGL Streaming & MP4 Video Rendering Module
-------------------------------------------------
Provides live real-time interactive 3D WebGL streaming inside Google Colab cells
via Meshcat/Three.js bridge, as well as MP4 off-screen video compilation.
"""

import os
import sys
import time
import base64
import numpy as np
from typing import List, Dict, Optional
from IPython.display import display, HTML

class LiveWebGLViewer:
    """
    Live interactive 3D viewer for Google Colab using embedded WebGL / Three.js.
    Allows watching the physics simulation step-by-step live in the notebook output cell.
    """
    def __init__(self, width: int = 640, height: int = 400):
        self.width = width
        self.height = height
        self.viewer_id = f"webgl_viewer_{int(time.time() * 1000)}"
        self._initialized = False

    def init_colab_display(self):
        """Injects Three.js HTML/JS WebGL canvas viewer into Google Colab output cell."""
        html_code = f"""
        <div id="{self.viewer_id}_container" style="width:{self.width}px; height:{self.height}px; border:2px solid #4A5568; border-radius:8px; overflow:hidden; background-color:#1A202C; margin:auto;">
            <canvas id="{self.viewer_id}_canvas" width="{self.width}" height="{self.height}"></canvas>
            <div id="{self.viewer_id}_status" style="color:#A0AEC0; font-family:sans-serif; font-size:12px; padding:4px 8px; background:#2D3748;">
                🟢 <b>MaleCNS Live 3D Stream:</b> Waiting for simulation steps...
            </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
        (function() {{
            const canvas = document.getElementById("{self.viewer_id}_canvas");
            const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
            renderer.setSize({self.width}, {self.height});
            renderer.setClearColor(0x1a202c, 1);

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, {self.width} / {self.height}, 0.1, 100);
            camera.position.set(0, 3, 4);
            camera.lookAt(0, 1, 0);

            // Ground grid
            const gridHelper = new THREE.GridHelper(10, 20, 0x4299e1, 0x2d3748);
            scene.add(gridHelper);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight.position.set(5, 10, 7);
            scene.add(dirLight);

            // 3D Humanoid Body Hierarchy
            const humanoidGroup = new THREE.Group();
            
            const torsoGeo = new THREE.CylinderGeometry(0.2, 0.15, 0.6, 16);
            const torsoMat = new THREE.MeshPhongMaterial({{ color: 0x3182ce, shininess: 30 }});
            const torso = new THREE.Mesh(torsoGeo, torsoMat);
            torso.position.y = 1.1;
            humanoidGroup.add(torso);

            const headGeo = new THREE.SphereGeometry(0.12, 16, 16);
            const headMat = new THREE.MeshPhongMaterial({{ color: 0xed8936 }});
            const head = new THREE.Mesh(headGeo, headMat);
            head.position.y = 1.55;
            humanoidGroup.add(head);

            // Limbs
            const limbMat = new THREE.MeshPhongMaterial({{ color: 0x48bb78 }});
            const leftArm = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.04, 0.5), limbMat);
            leftArm.position.set(-0.3, 1.1, 0);
            humanoidGroup.add(leftArm);

            const rightArm = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.04, 0.5), limbMat);
            rightArm.position.set(0.3, 1.1, 0);
            humanoidGroup.add(rightArm);

            const leftLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.6), limbMat);
            leftLeg.position.set(-0.15, 0.5, 0);
            humanoidGroup.add(leftLeg);

            const rightLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.6), limbMat);
            rightLeg.position.set(0.15, 0.5, 0);
            humanoidGroup.add(rightLeg);

            scene.add(humanoidGroup);

            window['update_{self.viewer_id}'] = function(state) {{
                if (state.position) {{
                    humanoidGroup.position.set(state.position[0], state.position[1], state.position[2]);
                }}
                if (state.joints) {{
                    head.rotation.x = state.joints[0] || 0;
                    head.rotation.y = state.joints[1] || 0;
                    rightArm.rotation.x = state.joints[2] || 0;
                    leftArm.rotation.x = state.joints[5] || 0;
                    rightLeg.rotation.x = state.joints[8] || 0;
                    leftLeg.rotation.x = state.joints[12] || 0;
                }}
                renderer.render(scene, camera);
            }};

            renderer.render(scene, camera);
        }})();
        </script>
        """
        display(HTML(html_code))
        self._initialized = True

    def update(self, position: List[float], joint_angles: List[float], step: int):
        """Pushes single simulation frame state to live WebGL viewer in Colab."""
        if not self._initialized:
            self.init_colab_display()

        js_call = f"""
        <script>
        if (window['update_{self.viewer_id}']) {{
            window['update_{self.viewer_id}']({{
                position: {position},
                joints: {list(joint_angles)}
            }});
            document.getElementById('{self.viewer_id}_status').innerHTML = '🟢 <b>MaleCNS Live 3D Stream:</b> Step {step} | Pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]';
        }}
        </script>
        """
        display(HTML(js_call))


class VideoRenderer:
    """
    Compiles simulation RGB frames into HTML5 MP4 video files for high-FPS playback in Colab.
    """
    @staticmethod
    def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> str:
        """Saves list of RGB NumPy array frames as an MP4 video file using imageio with yuv420p pixel format for Web HTML5 compatibility."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            import imageio
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                macro_block_size=None
            )
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"[Video Renderer] Saved {len(frames)} frames to: {output_path}")
            return output_path
        except Exception as e:
            print(f"[Video Renderer Error] Failed to encode MP4 video: {e}")
            return ""

    @staticmethod
    def display_video_in_colab(video_path: str, width: int = 640):
        """Displays HTML5 embedded MP4 video player directly in Colab notebook output."""
        if not os.path.exists(video_path):
            print(f"[Video Player Error] File not found: {video_path}")
            return

        with open(video_path, "rb") as f:
            mp4_bytes = f.read()

        b64_str = base64.b64encode(mp4_bytes).decode("utf-8")
        html_code = f"""
        <div style="text-align: center; margin: 10px 0;">
            <video width="{width}" controls autoplay loop muted playsinline style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <source src="data:video/mp4;base64,{b64_str}" type="video/mp4">
                Your browser does not support HTML5 video playback.
            </video>
            <p style="font-family: sans-serif; font-size: 13px; color: #4A5568; margin-top: 4px;">
                🎬 <b>MaleCNS Physics Simulation Render</b> ({os.path.basename(video_path)})
            </p>
        </div>
        """
        display(HTML(html_code))
