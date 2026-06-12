"""
YOLO face detector — direct face bounding boxes via Ultralytics YOLO.

Uses a dedicated face-detection model (face_yolov8n.pt) that outputs
precise face boxes for front, profile, and tilted faces in one pass.
MPS (Metal) accelerated on Apple Silicon.
"""

import os
from collections import namedtuple

import cv2
import numpy as np
import Quartz

# ---------------------------------------------------------------------------
# CGRect-compatible named tuples
# ---------------------------------------------------------------------------
_CGPoint = namedtuple("CGPoint", ["x", "y"])
_CGSize = namedtuple("CGSize", ["width", "height"])
CGRect = namedtuple("CGRect", ["origin", "size"])


def _make_rect(x, y, w, h):
    return CGRect(origin=_CGPoint(x=x, y=y), size=_CGSize(width=w, height=h))


# ---------------------------------------------------------------------------
# CGImage → numpy helper
# ---------------------------------------------------------------------------


def _cgimage_to_bgr(cg_image):
    """Convert a CGImage to a numpy BGR array."""
    width = Quartz.CGImageGetWidth(cg_image)
    height = Quartz.CGImageGetHeight(cg_image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)

    provider = Quartz.CGImageGetDataProvider(cg_image)
    cf_data = Quartz.CGDataProviderCopyData(provider)

    raw = np.frombuffer(bytes(cf_data), dtype=np.uint8)
    raw = raw[:height * bytes_per_row].reshape(height, bytes_per_row)
    raw = raw[:, :width * 4].reshape(height, width, 4)
    bgr = raw[:, :, :3].copy()  # BGRA → BGR (little-endian)
    return bgr


# ---------------------------------------------------------------------------
# Path to the face model
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_FACE_MODEL = os.path.join(_MODEL_DIR, "face_yolov8n.pt")


# ---------------------------------------------------------------------------
# YOLO face detector
# ---------------------------------------------------------------------------


class YOLODetector:
    """Face detector — direct, precise face boxes from a YOLO face model.

    Parameters
    ----------
    model_path : str
        Path to a YOLO face-detection .pt file.
    conf : float
        Confidence threshold (0.0 – 1.0).
    iou : float
        NMS IoU threshold.
    imgsz : int
        Inference image size in pixels.
    """

    def __init__(self, model_path=None, conf=0.25, iou=0.45, imgsz=640):
        from ultralytics import YOLO

        if model_path is None:
            model_path = _FACE_MODEL

        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._model = YOLO(model_path, task="detect")

        # Apple Silicon GPU
        try:
            self._model.to("mps")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def detect(self, image):
        """Detect faces.  Accepts a CGImage or numpy BGR array.

        Returns a list of normalised CGRects ready for the overlay view.
        """
        # ---- Image → numpy BGR ----
        if isinstance(image, np.ndarray):
            bgr = image
        else:
            bgr = _cgimage_to_bgr(image)

        # ---- YOLO inference (handles resize internally, optimized) ----
        results = self._model(
            bgr,
            verbose=False,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            half=True,   # FP16 — faster on MPS
        )

        h, w = bgr.shape[:2]
        faces = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Pixel coords (top-left origin) → normalised (bottom-left origin)
                nx = float(x1) / w
                ny = float(h - y2) / h   # flip Y
                nw = float(x2 - x1) / w
                nh = float(y2 - y1) / h
                faces.append(_make_rect(nx, ny, nw, nh))

        return faces
