"""
Face tracker — runs YOLO every N frames, OpenCV KCF in between.

This gives smooth 30 FPS overlay updates while doing the heavy
detection work only 7-8× per second.
"""

import cv2
import numpy as np

from detector_yolo import _make_rect


class FaceTracker:
    """Multi-face tracker: YOLO detection + per-face KCF tracking.

    Parameters
    ----------
    yolo_detector : YOLODetector
        The face detection model.
    detect_interval : int
        Run full YOLO detection every N frames (default 4).
    """

    def __init__(self, yolo_detector, detect_interval=4):
        self._yolo = yolo_detector
        self._interval = detect_interval
        self._tick = 0
        self._trackers = []       # list of cv2.Tracker
        self._bboxes = []         # pixel bboxes (x_tl, y_tl, w, h)

    # ------------------------------------------------------------------
    def update(self, bgr):
        """Update trackers and return face boxes as normalised CGRects.

        Every *detect_interval* calls a full YOLO detection is performed
        and trackers are re-initialised.  Intermediate calls only update
        the existing KCF trackers (very cheap).

        Parameters
        ----------
        bgr : numpy.ndarray
            The screen-capture image (BGR).

        Returns
        -------
        list of CGRect
        """
        h, w = bgr.shape[:2]
        self._tick += 1

        # ---- Full YOLO detection ------------------------------------
        if self._tick % self._interval == 1 or not self._trackers:
            yolo_faces = self._yolo.detect(bgr)

            self._trackers = []
            self._bboxes = []

            for face in yolo_faces:
                # normalised (bottom-left) → pixel (top-left)
                px = int(face.origin.x * w)
                py_bl = int(face.origin.y * h)
                pw = int(face.size.width * w)
                ph = int(face.size.height * h)
                py_tl = h - py_bl - ph

                t = cv2.TrackerKCF_create()
                try:
                    t.init(bgr, (px, py_tl, pw, ph))
                    self._trackers.append(t)
                    self._bboxes.append((px, py_tl, pw, ph))
                except Exception:
                    pass

            return yolo_faces

        # ---- Tracking-only frame ------------------------------------
        alive_trackers = []
        new_bboxes = []
        for t in self._trackers:
            ok, bb = t.update(bgr)
            if ok:
                alive_trackers.append(t)
                new_bboxes.append(tuple(int(v) for v in bb))
        self._trackers = alive_trackers
        self._bboxes = new_bboxes

        # pixel (top-left) → normalised (bottom-left)
        result = []
        for (x, y_tl, fw, fh) in self._bboxes:
            nx = float(x) / w
            ny = float(h - y_tl - fh) / h
            nw = float(fw) / w
            nh = float(fh) / h
            result.append(_make_rect(nx, ny, nw, nh))

        return result
