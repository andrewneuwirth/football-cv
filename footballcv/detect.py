"""DETECT stage: run a person detector over a play's clip.mp4.

Writes detections.json per the film-lab CONTRACT:

    {"play_id": "p001", "model": "...", "conf_threshold": 0.3,
     "frames": [{"i": 0, "boxes": [[x1, y1, x2, y2, conf], ...]}, ...]}

Person class only. Every frame appears, even with empty ``boxes``. Boxes are in
native clip pixels (origin top-left, floats).

Model selection happens at import time: RF-DETR when the ``rfdetr`` package is
installed in the venv, otherwise torchvision ``fasterrcnn_resnet50_fpn_v2`` with
default COCO weights filtered to label == 1 (person). Device is MPS when
available, else CPU. Frames are letterboxed so the long side equals ``imgsz``
before inference and boxes are mapped back to native pixels.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import numpy as np

from footballcv.paths import play_dir

# ---------------------------------------------------------------- backend pick
# Import-time model selection per the contract: prefer rfdetr, fall back to
# torchvision. Only the *availability* check runs at import time; the actual
# (heavy) model build is deferred to PersonDetector.__init__.

try:  # pragma: no cover - depends on the environment
    import rfdetr  # noqa: F401

    BACKEND = "rfdetr"
    # rf-detr logs scary-looking DINOv2 warnings while constructing the
    # architecture; the pretrained checkpoint loads right after, so they are
    # noise (and clutter the dashboard's run-log tail).
    import logging as _logging

    _logging.getLogger("rfdetr").setLevel(_logging.ERROR)
    _logging.getLogger("rf-detr").setLevel(_logging.ERROR)
except Exception:  # pragma: no cover
    BACKEND = "torchvision"

_PERSON_CLASS_ID = 1  # COCO category id for "person" (both backends)


def _require_detect_deps() -> None:
    """Fail early with an actionable message if the detection extra is missing."""
    import importlib.util

    missing = [m for m in ("torch", "ultralytics")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            "person detection needs the optional 'detect' extra "
            f"(missing: {', '.join(missing)}). Install it with:\n"
            "    pip install -e '.[detect]'"
        )


def pick_device() -> str:
    """mps if torch.backends.mps.is_available() else cpu."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- letterbox math
# Pure functions so the round-trip math is unit-testable without a model.


def letterbox_params(width: int, height: int, imgsz: int) -> tuple[float, int, int, int, int]:
    """Compute letterbox geometry for a (width x height) image into imgsz x imgsz.

    The image is scaled so its long side equals ``imgsz``, then centered on a
    square canvas with integer padding.

    Returns (scale, pad_x, pad_y, new_w, new_h) where
    ``letterboxed = native * scale + pad``.
    """
    if width <= 0 or height <= 0 or imgsz <= 0:
        raise ValueError(f"bad letterbox dims: {width}x{height} -> {imgsz}")
    scale = imgsz / max(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    return scale, pad_x, pad_y, new_w, new_h


def letterbox_image(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """Resize ``img`` so its long side is ``imgsz`` and pad to a square canvas.

    Returns (letterboxed_image, scale, pad_x, pad_y).
    """
    h, w = img.shape[:2]
    scale, pad_x, pad_y, new_w, new_h = letterbox_params(w, h, imgsz)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((imgsz, imgsz, img.shape[2]), 114, dtype=img.dtype)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def boxes_to_letterbox(boxes: np.ndarray, scale: float, pad_x: float, pad_y: float) -> np.ndarray:
    """Map [x1,y1,x2,y2] boxes from native pixels into letterboxed pixels."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
    return boxes


def boxes_to_native(
    boxes: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Map [x1,y1,x2,y2] boxes from letterboxed pixels back to native pixels.

    Inverse of :func:`boxes_to_letterbox`; results are clipped to the native
    image bounds.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, float(width))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, float(height))
    return boxes


# ---------------------------------------------------------------- detector


class PersonDetector:
    """Person detector wrapper: BGR frame in, native-pixel [x1,y1,x2,y2,conf] out.

    Backend is chosen at import time (module-level ``BACKEND``): rfdetr when
    installed, else torchvision fasterrcnn_resnet50_fpn_v2 (COCO, person only).
    """

    def __init__(self, conf: float = 0.3, imgsz: int = 1920, backend: str | None = None):
        _require_detect_deps()
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.backend = backend or BACKEND
        self.device = pick_device()
        if self.backend == "rfdetr":
            self._build_rfdetr()
        else:
            self._build_torchvision()

    # -- builders ----------------------------------------------------------

    def _build_rfdetr(self) -> None:
        import rfdetr as _rfdetr

        # Prefer the current names; RFDETRBase is deprecated in newer releases
        # (RFDETRMedium is its successor) but keep it as a fallback.
        for attr, name in (
            ("RFDETRMedium", "rfdetr-medium"),
            ("RFDETRBase", "rfdetr-base"),
            ("RFDETRNano", "rfdetr-nano"),
        ):
            cls = getattr(_rfdetr, attr, None)
            if cls is not None:
                break
        else:  # pragma: no cover - rfdetr always exports at least one of these
            raise ImportError("rfdetr installed but no known model class found")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._model = cls(device=self.device)
            except TypeError:  # older/newer rfdetr without a device kwarg
                self._model = cls()
            # fp16 + compiled inference path; ~free speedup when supported.
            # torch.compile is flaky on MPS, so fall back through the combos.
            import torch as _torch

            for kwargs in (
                {"compile": True, "dtype": _torch.float16},
                {"compile": False, "dtype": _torch.float16},
                {"compile": False, "dtype": _torch.float32},
            ):
                try:
                    self._model.optimize_for_inference(**kwargs)
                    break
                except Exception:
                    continue
        self.model_name = name

    def _build_torchvision(self) -> None:
        import torch
        import torchvision

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        model.eval()
        self._torch = torch
        self._model = model.to(self.device)
        self.model_name = "torchvision-fasterrcnn_resnet50_fpn_v2"

    # -- inference ---------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> list[list[float]]:
        """Detect people in one BGR frame; boxes returned in native pixels."""
        h, w = frame_bgr.shape[:2]
        lb, scale, pad_x, pad_y = letterbox_image(frame_bgr, self.imgsz)
        rgb = np.ascontiguousarray(lb[:, :, ::-1])
        if self.backend == "rfdetr":
            xyxy, confs = self._infer_rfdetr(rgb)
        else:
            xyxy, confs = self._infer_torchvision(rgb)
        if len(xyxy) == 0:
            return []
        native = boxes_to_native(xyxy, scale, pad_x, pad_y, w, h)
        return [
            [float(x1), float(y1), float(x2), float(y2), float(c)]
            for (x1, y1, x2, y2), c in zip(native, confs)
        ]

    def _infer_rfdetr(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                det = self._model.predict(rgb, threshold=self.conf, include_source_image=False)
            except TypeError:  # older rfdetr without include_source_image
                det = self._model.predict(rgb, threshold=self.conf)
        if det is None or det.xyxy is None or len(det.xyxy) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        xyxy = np.asarray(det.xyxy, dtype=np.float64)
        confs = np.asarray(det.confidence, dtype=np.float64)
        class_ids = np.asarray(det.class_id)
        keep = (class_ids == _PERSON_CLASS_ID) & (confs >= self.conf)
        return xyxy[keep], confs[keep]

    def _infer_torchvision(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            out = self._model([tensor])[0]
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy().astype(np.float64)
        boxes = out["boxes"].cpu().numpy().astype(np.float64)
        keep = (labels == _PERSON_CLASS_ID) & (scores >= self.conf)
        return boxes[keep], scores[keep]


# ---------------------------------------------------------------- stage entry


def run(play_id: str, data_dir: Path, conf: float = 0.3, imgsz: int = 1920) -> None:
    """Run person detection over data/plays/<play_id>/clip.mp4 -> detections.json."""
    pdir = play_dir(play_id, data_dir)
    clip = pdir / "clip.mp4"
    if not clip.exists():
        raise FileNotFoundError(f"no clip.mp4 for play '{play_id}' at {clip}")

    detector = PersonDetector(conf=conf, imgsz=imgsz)
    print(f"detect[{play_id}]: model={detector.model_name} device={detector.device} "
          f"conf={conf} imgsz={imgsz}", flush=True)

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {clip}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frames: list[dict] = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = detector.detect(frame)
            frames.append({"i": i, "boxes": boxes})
            if i % 30 == 0:
                total = f"/{n_total}" if n_total else ""
                print(f"detect[{play_id}]: frame {i}{total} ({len(boxes)} people)", flush=True)
            i += 1
    finally:
        cap.release()

    out = {
        "play_id": play_id,
        "model": detector.model_name,
        "conf_threshold": conf,
        "frames": frames,
    }
    out_path = pdir / "detections.json"
    out_path.write_text(json.dumps(out))
    print(f"detect[{play_id}]: wrote {out_path} ({i} frames)", flush=True)
