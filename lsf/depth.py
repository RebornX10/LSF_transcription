"""Depth estimation and dual-pixel handling.

What "dual-pixel depth" actually is
-----------------------------------
A dual-pixel (DP) sensor splits every photosite into two halves, giving two
slightly different sub-images from which a *disparity* (depth) map can be
computed — the trick behind Google Pixel "Portrait mode". DP data is exposed by
the *camera stack* (e.g. Android Camera2 `RAW`/`DEPTH` streams, or a depth file
embedded in the capture), **not** by a generic UVC webcam through OpenCV.

So in practice:

  * On a phone / device that publishes a depth or DP stream, feed that map into
    `DepthEstimator.from_stream(...)` and we sample it at each landmark.
  * On a normal laptop webcam there is no DP data. We *detect that*, and
    optionally fall back to a monocular depth model (MiDaS) if `torch` is
    installed. Otherwise we transparently use MediaPipe's own relative z.

Either way, the recogniser keeps working; depth only refines the per-landmark
z so movements toward/away from the camera are measured more accurately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .landmarks import LandmarkFrame


@dataclass
class DepthInfo:
    """Where a frame's depth came from and the map itself (if any)."""

    source: str                      # "dual_pixel" | "monocular" | "mediapipe"
    available: bool                  # True if a real depth map backs this frame
    depth_map: Optional[np.ndarray]  # (H, W) float32, larger = farther, or None


def detect_dual_pixel_support(capture=None) -> tuple[bool, str]:
    """Best-effort check for a usable dual-pixel / depth stream.

    OpenCV's VideoCapture cannot surface dual-pixel data, so for a plain webcam
    this returns False with an explanation. The hook exists so a device backend
    that *does* provide depth can report support here.
    """
    if capture is None:
        return False, "No capture device supplied."

    # Some depth cameras expose CAP_PROP_OPENNI_* / a second stream; a UVC
    # webcam does not. We treat presence of a non-trivial depth generator as
    # support. This stays False for ordinary webcams.
    try:
        import cv2

        has_depth = capture.get(cv2.CAP_PROP_OPENNI_REGISTRATION) not in (-1, 0)
        if has_depth:
            return True, "Depth-capable capture detected."
    except Exception:
        pass
    return (
        False,
        "Dual-pixel/depth stream not exposed by this camera via OpenCV; "
        "using monocular fallback or MediaPipe z.",
    )


class DepthEstimator:
    """Produces a per-frame depth map and refines landmark z values.

    Priority order:
      1. An externally supplied dual-pixel/depth map (best).
      2. MiDaS monocular estimate, if `torch` + `timm` are installed and enabled.
      3. None — landmarks keep MediaPipe's relative z.
    """

    def __init__(self, *, enable_monocular: bool = False) -> None:
        self._supports_dp = False
        self._dp_reason = "uninitialised"
        self._midas = None
        self._midas_transform = None
        self._enable_monocular = enable_monocular
        if enable_monocular:
            self._try_load_midas()

    def probe(self, capture) -> None:
        """Inspect a capture device for dual-pixel support."""
        self._supports_dp, self._dp_reason = detect_dual_pixel_support(capture)

    @property
    def status(self) -> str:
        if self._supports_dp:
            return "dual_pixel"
        if self._midas is not None:
            return "monocular"
        return "mediapipe"

    @property
    def reason(self) -> str:
        return self._dp_reason

    # -- depth map sources ----------------------------------------------------
    def _try_load_midas(self) -> None:
        """Load MiDaS small from torch.hub. Silently no-ops if torch missing."""
        try:
            import torch  # noqa: F401

            self._midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self._midas.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._midas_transform = transforms.small_transform
        except Exception as exc:  # pragma: no cover - optional path
            self._midas = None
            self._dp_reason = f"Monocular depth unavailable: {exc}"

    def _monocular_depth(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        if self._midas is None:
            return None
        import torch

        with torch.no_grad():
            batch = self._midas_transform(rgb_frame)
            pred = self._midas(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        # MiDaS returns inverse depth (larger = closer); invert so larger=farther.
        depth = depth.max() - depth
        return depth

    def depth_for_frame(
        self,
        rgb_frame: np.ndarray,
        external_depth: Optional[np.ndarray] = None,
    ) -> DepthInfo:
        """Return depth info for a frame, choosing the best available source."""
        if external_depth is not None:
            return DepthInfo("dual_pixel", True, external_depth.astype(np.float32))
        mono = self._monocular_depth(rgb_frame) if self._midas is not None else None
        if mono is not None:
            return DepthInfo("monocular", True, mono)
        return DepthInfo("mediapipe", False, None)

    # -- landmark refinement --------------------------------------------------
    @staticmethod
    def refine_landmarks(frame: LandmarkFrame, depth: DepthInfo) -> LandmarkFrame:
        """Overwrite each landmark's z with a metric depth sample when we have one.

        Landmark x/y are normalised [0, 1]; we map them to pixel coordinates to
        sample the depth map. Values are normalised to ~[0, 1] so the feature
        scale matches MediaPipe's native z range.
        """
        if not depth.available or depth.depth_map is None:
            return frame

        dmap = depth.depth_map
        h, w = dmap.shape[:2]
        dmin, dmax = float(dmap.min()), float(dmap.max())
        span = (dmax - dmin) or 1.0

        def sample(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            out = arr.copy()
            xs = np.clip((arr[:, 0] * w).astype(int), 0, w - 1)
            ys = np.clip((arr[:, 1] * h).astype(int), 0, h - 1)
            out[:, 2] = (dmap[ys, xs] - dmin) / span
            return out

        return LandmarkFrame(
            pose=sample(frame.pose),
            face=sample(frame.face),
            left_hand=sample(frame.left_hand),
            right_hand=sample(frame.right_hand),
            raw=frame.raw,
        )
