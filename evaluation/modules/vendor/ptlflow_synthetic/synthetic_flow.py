"""Generate synthetic optical flow fields from game actions using the Longuet-Higgins formulation.

Each action (keyboard/mouse) maps to camera rotation and translation velocities.
The resulting flow field is computed analytically from the physics model:
    - Rotation flow: depth-independent, spatially non-uniform (quadratic terms)
    - Translation flow: depth-dependent, uses depth estimation from Depth Anything v2

Simultaneous actions are linearly superposed (valid because the motion field equations
are linear in angular velocity and translation velocity).

Usage:
    from synthetic_flow import SyntheticFlowGenerator

    gen = SyntheticFlowGenerator(calibration, frame_shape=(352, 640))
    flow = gen.generate_flow(keyboard_vec, mouse_vec)  # HxWx2
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def load_calibration(path: Union[str, Path]) -> dict:
    """Load calibration coefficients from a JSON file."""
    with open(path) as f:
        return json.load(f)


class SyntheticFlowGenerator:
    """Generate synthetic optical flow from game actions using the Longuet-Higgins model.

    Parameters
    ----------
    calibration : dict
        Calibration coefficients with keys:
        alpha_yaw, alpha_pitch, alpha_turn, beta_fwd, beta_strafe, focal_length
    frame_shape : tuple
        (H, W) of the output flow field
    use_depth : bool
        If True and depth_model is loaded, use depth estimation for translation flow.
        Otherwise use constant depth Z=1 everywhere.
    """

    def __init__(
        self,
        calibration: dict,
        frame_shape: Tuple[int, int],
        use_depth: bool = True,
    ):
        self.alpha_yaw = calibration["alpha_yaw"]
        self.alpha_pitch = calibration["alpha_pitch"]
        self.alpha_turn = calibration["alpha_turn"]
        self.beta_fwd = calibration["beta_fwd"]
        self.beta_strafe = calibration["beta_strafe"]
        self.f = calibration["focal_length"]

        self.H, self.W = frame_shape
        self.use_depth = use_depth

        # Depth model (lazy-loaded)
        self._depth_model = None
        self._depth_processor = None

        # Pre-compute pixel coordinate grids relative to principal point
        cx, cy = self.W / 2.0, self.H / 2.0
        xs = np.arange(self.W, dtype=np.float64) - cx
        ys = np.arange(self.H, dtype=np.float64) - cy
        self.x_grid, self.y_grid = np.meshgrid(xs, ys)  # HxW each

        # Pre-compute terms used in rotation flow (quadratic in spatial coords)
        f = self.f
        self.xy_over_f = self.x_grid * self.y_grid / f
        self.f_plus_x2_over_f = f + self.x_grid ** 2 / f
        self.f_plus_y2_over_f = f + self.y_grid ** 2 / f

    def load_depth_model(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        """Load Depth Anything v2 model from HuggingFace."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._depth_processor = AutoImageProcessor.from_pretrained(model_name)
        self._depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self._depth_model.eval()
        if torch.cuda.is_available():
            self._depth_model = self._depth_model.cuda()

    @property
    def has_depth_model(self) -> bool:
        return self._depth_model is not None

    def _compute_rotation_flow(
        self, omega_x: float, omega_y: float, omega_z: float
    ) -> np.ndarray:
        """Compute depth-independent rotation flow field (HxWx2).

        Uses Longuet-Higgins & Prazdny formulation:
            u_R = (xy/f)*wx - (f + x^2/f)*wy + y*wz
            v_R = (f + y^2/f)*wx - (xy/f)*wy - x*wz
        """
        u = (self.xy_over_f * omega_x
             - self.f_plus_x2_over_f * omega_y
             + self.y_grid * omega_z)
        v = (self.f_plus_y2_over_f * omega_x
             - self.xy_over_f * omega_y
             - self.x_grid * omega_z)
        flow = np.stack([u, v], axis=-1)
        return flow

    def _compute_translation_flow(
        self, Tx: float, Ty: float, Tz: float, depth_map: np.ndarray
    ) -> np.ndarray:
        """Compute depth-dependent translation flow field (HxWx2).

        Uses:
            u_T = (-f*Tx + x*Tz) / Z(x,y)
            v_T = (-f*Ty + y*Tz) / Z(x,y)

        Parameters
        ----------
        depth_map : np.ndarray, shape HxW
            Depth values (positive). Normalized so median=1.0.
        """
        f = self.f
        inv_depth = 1.0 / np.clip(depth_map, 1e-3, None)
        u = (-f * Tx + self.x_grid * Tz) * inv_depth
        v = (-f * Ty + self.y_grid * Tz) * inv_depth
        flow = np.stack([u, v], axis=-1)
        return flow

    @torch.no_grad()
    def estimate_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR frame using Depth Anything v2.

        Returns
        -------
        np.ndarray, shape HxW, float64
            Depth map normalized so that median = 1.0.
            Higher values = farther from camera.
        """
        if self._depth_model is None:
            raise RuntimeError("Depth model not loaded. Call load_depth_model() first.")

        import cv2 as cv
        from PIL import Image

        # Convert BGR to RGB PIL Image
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = self._depth_processor(images=pil_img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self._depth_model(**inputs)
        predicted_depth = outputs.predicted_depth  # 1xHxW

        # Interpolate to original resolution
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Model outputs inverse depth (higher = closer), invert to get depth
        # Add small epsilon to avoid division by zero
        depth = 1.0 / (prediction + 1e-6)

        # Normalize so median = 1.0
        median_depth = np.median(depth)
        if median_depth > 1e-6:
            depth = depth / median_depth

        return depth

    def generate_flow(
        self,
        keyboard: np.ndarray,
        mouse: np.ndarray,
        frame_bgr: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        return_depth: bool = False,
    ):
        """Generate HxWx2 synthetic flow for one frame's actions.

        Parameters
        ----------
        keyboard : np.ndarray, shape (6,)
            [W, S, A, D, left, right] key states (0 or 1).
        mouse : np.ndarray, shape (2,)
            [pitch, yaw] mouse values.
        frame_bgr : optional np.ndarray
            BGR frame for depth estimation. Only used if use_depth=True
            and depth model is loaded.
        depth_map : optional np.ndarray, shape HxW
            Pre-computed depth map. If provided, skips depth estimation.
        return_depth : bool
            If True, return (flow, depth_map) tuple.

        Returns
        -------
        np.ndarray or tuple
            If return_depth=False: HxWx2 flow array.
            If return_depth=True: (HxWx2 flow, HxW depth) tuple.
            Depth is None if no depth was computed.
        """
        # --- Map actions to angular velocities ---
        omega_y = self.alpha_yaw * mouse[1]           # mouse yaw -> wy
        omega_x = self.alpha_pitch * mouse[0]          # mouse pitch -> wx
        omega_y += self.alpha_turn * (keyboard[5] - keyboard[4])  # turn keys

        # No roll in Minecraft
        omega_z = 0.0

        # --- Map actions to translation velocities ---
        Tz = self.beta_fwd * (keyboard[0] - keyboard[1])     # W - S = net forward
        Tx = self.beta_strafe * (keyboard[3] - keyboard[2])   # D - A = net rightward
        Ty = 0.0  # no vertical translation from these actions

        # --- Compute rotation flow (always available, no depth needed) ---
        flow = self._compute_rotation_flow(omega_x, omega_y, omega_z)

        # --- Compute translation flow ---
        Z = None
        has_translation = abs(Tx) > 1e-8 or abs(Ty) > 1e-8 or abs(Tz) > 1e-8
        if has_translation:
            if depth_map is not None:
                Z = depth_map
            elif self.use_depth and self.has_depth_model and frame_bgr is not None:
                Z = self.estimate_depth(frame_bgr)
            else:
                Z = np.ones((self.H, self.W), dtype=np.float64)
            flow = flow + self._compute_translation_flow(Tx, Ty, Tz, Z)

        flow = flow.astype(np.float32)
        if return_depth:
            return flow, Z
        return flow

    def generate_flow_sequence(
        self,
        actions: dict,
        frames: Optional[List[np.ndarray]] = None,
        return_depths: bool = False,
    ):
        """Generate synthetic flow for all frame transitions.

        Parameters
        ----------
        actions : dict
            {"keyboard": (T, 6), "mouse": (T, 2)} action arrays.
        frames : optional list of np.ndarray
            BGR frames for depth estimation. Length should be >= T.
        return_depths : bool
            If True, also return list of depth maps.

        Returns
        -------
        list of np.ndarray, or (list, list) if return_depths=True
            List of (T-1) flow fields, each HxWx2. Each flow[i] corresponds
            to the motion from frame i to frame i+1, driven by action[i].
            If return_depths=True, also returns list of depth maps (or None).
        """
        keyboard = actions["keyboard"]
        mouse_arr = actions["mouse"]
        T = len(keyboard)

        # We generate T-1 flows (matching frame pairs)
        n_flows = T - 1
        flows = []
        depths = []
        for i in range(n_flows):
            frame = frames[i] if frames is not None else None
            result = self.generate_flow(
                keyboard[i], mouse_arr[i], frame_bgr=frame,
                return_depth=return_depths,
            )
            if return_depths:
                flow, depth = result
                flows.append(flow)
                depths.append(depth)
            else:
                flows.append(result)

        if return_depths:
            return flows, depths
        return flows
