"""
MedSAM plugin v2 — GL-based interactive overlay.

The overlay widget floats transparently on top of the existing GLWidget so
the user draws directly on the live medical image.  Coordinate mapping is
delegated entirely to the GLWidget's own to_real_world / fromRealWorld
methods, which already handle zoom, pan and rotation — no manual axis
flip or transpose needed.

Segmentation results are fed back through the GLWidget's segChanged signal,
which is already wired into the main window's 3D coordinate mapping pipeline
(permute_axis → updateSegmentation).

The original SliceCanvas version is preserved as medsam_v1_slicecanvas.py.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor

from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .medsam_schema import get_schema, _MODEL_REGISTRY, _DEFAULT_MODEL

try:
    from segment_anything import sam_model_registry
    from skimage import transform as sk_transform
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

_MODEL_CACHE: dict = {}
_MODE_MAP = ["bbox", "point_pos", "point_neg"]
_AXIS_TO_WINDOW = {"Axial": "axial", "Coronal": "coronal", "Sagittal": "sagittal"}

# Starting mini-batch size for the image encoder.
# SAM ViT-B has global attention layers: (B, 12, 4096, 4096) float32 ≈ B×800 MB.
# fp16 autocast halves that.  The encoder automatically retries with half the batch
# on OOM so this is a starting hint, not a hard limit.
_ENCODE_BATCH = 8


# ---------------------------------------------------------------------------
# Weights directory (platform-appropriate, no admin needed)
# ---------------------------------------------------------------------------

def _weights_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", home), "melage", "weights")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "melage", "weights")
    xdg = os.environ.get("XDG_DATA_HOME", os.path.join(home, ".local", "share"))
    return os.path.join(xdg, "melage", "weights")


def _weights_path(model_key: str = _DEFAULT_MODEL) -> str:
    filename = _MODEL_REGISTRY[model_key]["filename"]
    return os.path.join(_weights_dir(), filename)


# ---------------------------------------------------------------------------
# Transparent overlay widget
# ---------------------------------------------------------------------------

class MedSamOverlay(QtWidgets.QWidget):
    """
    Transparent child of a GLWidget.  The user draws bounding boxes or places
    points here; coordinate conversion uses the parent GL widget's own
    to_real_world / fromRealWorld, so zoom/pan/rotation are handled for free.
    """

    bbox_changed = pyqtSignal(list)          # [col_min, row_min, col_max, row_max]
    point_added  = pyqtSignal(float, float, bool)  # col, row, is_positive

    def __init__(self, gl_widget):
        super().__init__(gl_widget)
        self.gl = gl_widget
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.resize(gl_widget.size())
        gl_widget.installEventFilter(self)  # keep size in sync

        self.mode:   str        = "bbox"
        self.bbox:   list|None  = None    # [col_min, row_min, col_max, row_max]
        self.points: list       = []      # [(col, row, is_positive)]
        self._drag_start        = None

    # ------------------------------------------------------------------
    # Event filter: resize overlay when GL widget resizes
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.gl and event.type() == QtCore.QEvent.Resize:
            self.resize(self.gl.size())
        return False

    # ------------------------------------------------------------------
    # Coordinate helpers (delegated to GL widget)
    # ------------------------------------------------------------------

    def _to_image(self, sx: int, sy: int):
        return self.gl.to_real_world(sx, sy)   # → (col, row) in imSlice space

    def _to_screen(self, col: float, row: float):
        return self.gl.fromRealWorld(col, row)  # → (sx, sy) in widget pixels

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        col, row = self._to_image(event.x(), event.y())
        if self.mode == "bbox":
            self._drag_start = (col, row)
            self.bbox = [col, row, col, row]
        elif self.mode == "point_pos":
            self.points.append((col, row, True))
            self.point_added.emit(col, row, True)
        else:
            self.points.append((col, row, False))
            self.point_added.emit(col, row, False)
        self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "bbox" and self._drag_start and (event.buttons() & Qt.LeftButton):
            col, row = self._to_image(event.x(), event.y())
            x0, y0 = self._drag_start
            self.bbox = [min(x0, col), min(y0, row), max(x0, col), max(y0, row)]
            self.update()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode == "bbox" and self._drag_start:
            self._drag_start = None
            if self.bbox and self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]:
                self.bbox_changed.emit(self.bbox)
        else:
            event.ignore()

    def wheelEvent(self, event):
        # Forward scroll to GL widget so zoom still works while overlay is active
        self.gl.wheelEvent(event)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def clear_prompts(self):
        self.bbox = None
        self.points = []
        self.update()

    def has_prompt(self) -> bool:
        return self.bbox is not None or bool(self.points)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            sx1, sy1 = self._to_screen(x1, y1)
            sx2, sy2 = self._to_screen(x2, y2)
            left   = int(min(sx1, sx2))
            top    = int(min(sy1, sy2))
            width  = int(abs(sx2 - sx1))
            height = int(abs(sy2 - sy1))
            p.setPen(QPen(QColor(255, 220, 0), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(left, top, width, height)
            p.setPen(QColor(255, 220, 0))
            p.drawText(left + 3, top - 4,
                       f"({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})")

        for col, row, is_pos in self.points:
            sx, sy = self._to_screen(col, row)
            color = QColor(50, 220, 50) if is_pos else QColor(220, 50, 50)
            p.setPen(QPen(QColor("white"), 1))
            p.setBrush(color)
            p.drawEllipse(QtCore.QPoint(int(sx), int(sy)), 6, 6)

        hints = {
            "bbox":      "drag to draw box",
            "point_pos": "click  ✚  positive",
            "point_neg": "click  ✖  negative",
        }
        p.setPen(QColor("#ddd"))
        p.drawText(6, 16, hints.get(self.mode, ""))
        p.end()


# ---------------------------------------------------------------------------
# Background download thread
# ---------------------------------------------------------------------------

class _DownloadThread(QtCore.QThread):
    progress  = pyqtSignal(int)
    succeeded = pyqtSignal(str)
    failed    = pyqtSignal(str)

    def __init__(self, dest: str, urls: list[str], parent=None):
        super().__init__(parent)
        self.dest = dest
        self.urls = urls
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import requests
        os.makedirs(os.path.dirname(self.dest), exist_ok=True)
        tmp = self.dest + ".tmp"
        last_error = "No URL."
        for url in self.urls:
            if self._cancelled:
                return
            try:
                with requests.get(url, stream=True, timeout=60,
                                  allow_redirects=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    done = 0
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65_536):
                            if self._cancelled:
                                return
                            if chunk:
                                f.write(chunk)
                                done += len(chunk)
                                if total:
                                    self.progress.emit(int(done / total * 100))
                os.replace(tmp, self.dest)
                self.succeeded.emit(self.dest)
                return
            except Exception as exc:
                last_error = str(exc)
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        self.failed.emit(last_error)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _get_model(checkpoint: str, device: str, arch: str = "vit_b"):
    key = (checkpoint, device, arch)
    if key not in _MODEL_CACHE:
        model = sam_model_registry[arch](checkpoint=checkpoint)
        model.to(device)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _preprocess(slc: np.ndarray) -> np.ndarray:
    """
    Prepare a 2D slice for MedSAM, returning float32 [0, 1] RGB (H, W, 3)
    at 1024×1024.

    Two paths depending on the source dtype:

    uint8  — npImage was already windowed by MELAGE (CT HU window, MRI
             percentile normalisation).  Divide directly by 255; do NOT
             re-normalise per-slice or the carefully chosen window is lost.

    float  — raw get_fdata() output (CT: [-1024…3000 HU], MRI: arbitrary
             positive range).  Percentile clip first (1 %–99 %) to suppress
             metal artefacts and MRI bias-field spikes that would otherwise
             compress the clinically relevant range into a few grey levels.
    """
    if slc.dtype == np.uint8:
        norm = slc.astype(np.float32) / 255.0
    else:
        p1, p99 = np.percentile(slc, [1, 99])
        clipped = np.clip(slc.astype(np.float64), p1, p99)
        lo, hi  = float(clipped.min()), float(clipped.max())
        norm    = ((clipped - lo) / max(hi - lo, 1e-8)).astype(np.float32)

    rgb = np.ascontiguousarray(np.stack([norm, norm, norm], axis=-1))
    return sk_transform.resize(
        rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)   # [0, 1] — ready for image_encoder directly


def _preprocess_parallel(slices: list[np.ndarray]) -> list[np.ndarray]:
    """
    Preprocess a list of 2D slices in parallel across CPU cores.
    Returns a list of (1024, 1024, 3) uint8 arrays in the same order.
    """
    workers = min(len(slices), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_preprocess, slices))


def _encode_batch(model, imgs_1024: list[np.ndarray],
                  device: str, batch_size: int = _ENCODE_BATCH) -> torch.Tensor:
    """
    Run the image encoder on N preprocessed 1024×1024 float32 [0,1] arrays.

    OOM prevention strategy (SAM ViT-B global attention is the culprit):
      - fp16 autocast on CUDA: (B, 12, 4096, 4096) tensor drops from
        B×800 MB (float32) to B×400 MB (float16).
      - empty_cache() between batches: returns fragmented memory.
      - Automatic retry: on OutOfMemoryError the batch is halved and
        retried until batch_size == 1 before re-raising.
    """
    from contextlib import nullcontext
    autocast_ctx = (torch.autocast("cuda", dtype=torch.float16)
                    if device == "cuda" else nullcontext())

    all_emb = []
    start = 0
    current_batch = batch_size

    while start < len(imgs_1024):
        chunk = imgs_1024[start : start + current_batch]
        arr = np.stack([img.transpose(2, 0, 1) for img in chunk]).astype(np.float32)
        t = torch.from_numpy(arr).to(device, non_blocking=True)

        try:
            with torch.inference_mode(), autocast_ctx:
                emb = model.image_encoder(t)
            all_emb.append(emb.float())   # back to float32 for the mask decoder
            del t
            if device == "cuda":
                torch.cuda.empty_cache()
            start += current_batch

        except torch.cuda.OutOfMemoryError:
            del t
            torch.cuda.empty_cache()
            if current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            print(f"[MedSAM] OOM — retrying with encode batch = {current_batch}")

    return torch.cat(all_emb, dim=0)   # (N, C, H, W), float32, on device


def _build_prompt_tensors(bbox, points, orig_h: int, orig_w: int, device: str):
    """Return (box_torch, pts_torch, lbl_torch) ready for prompt_encoder."""
    box_torch = None
    if bbox is not None:
        col_min, row_min, col_max, row_max = bbox
        b = np.array([
            col_min / orig_w * 1024, row_min / orig_h * 1024,
            col_max / orig_w * 1024, row_max / orig_h * 1024,
        ], dtype=np.float32)
        box_torch = torch.as_tensor(b, device=device).unsqueeze(0).unsqueeze(0)

    pts_torch = lbl_torch = None
    if points:
        coords = np.array(
            [[p[0] / orig_w * 1024, p[1] / orig_h * 1024] for p in points],
            dtype=np.float32,
        )
        labels = np.array([int(p[2]) for p in points], dtype=np.int64)
        pts_torch = torch.as_tensor(coords, device=device).unsqueeze(0)
        lbl_torch = torch.as_tensor(labels, device=device).unsqueeze(0)

    return box_torch, pts_torch, lbl_torch


def _decode_embedding(model, embedding: torch.Tensor,
                      box_torch, pts_torch, lbl_torch,
                      orig_h: int, orig_w: int,
                      mask_prompt: torch.Tensor | None = None
                      ) -> tuple[np.ndarray, torch.Tensor]:
    """
    Run the mask decoder for a single pre-computed embedding.

    mask_prompt : (1, 1, 256, 256) low-res logits from the *previous* slice —
                  pass None for the first/reference slice.  SAM was trained to
                  accept its own previous output as a prompt, so providing the
                  prior slice's logits gives the model shape information that a
                  plain bounding box cannot carry.

    Returns
    -------
    binary_mask   : (H, W) uint8
    low_res_logits: (1, 1, 256, 256) float32 tensor on the same device —
                    pass back as mask_prompt for the next slice.
    """
    with torch.no_grad():
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(pts_torch, lbl_torch) if pts_torch is not None else None,
            boxes=box_torch,
            masks=mask_prompt,   # None → no prior mask; tensor → propagation
        )
        low_res_logits, iou_preds = model.mask_decoder(
            image_embeddings=embedding.unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
    pred = F.interpolate(
        torch.sigmoid(low_res_logits), size=(orig_h, orig_w),
        mode="bilinear", align_corners=False,
    )
    binary_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    iou_score   = float(iou_preds.squeeze().cpu())   # model's own confidence (0→1)
    return binary_mask, low_res_logits.detach(), iou_score


def _mask_to_logits(mask: np.ndarray, device: str, logit_scale: float = 10.0) -> torch.Tensor:
    """
    Convert a binary (H, W) mask into (1, 1, 256, 256) low-res logits suitable
    as ``mask_prompt`` for SAM's mask decoder.

    SAM's decoder expects its *own* previous low-res output as a shape prior
    (sigmoid(logits) ~ 1 inside the mask, ~0 outside). A user-drawn binary
    mask doesn't come with logits, so we synthesise them: resize to 256x256
    (nearest-neighbour, to keep the mask binary) and map {0, 1} -> {-scale,
    +scale}, which after sigmoid gives values close to 0/1 — the same shape
    of input the decoder normally consumes from itself during propagation.
    """
    resized = sk_transform.resize(
        mask.astype(np.float32), (256, 256), order=0,
        preserve_range=True, anti_aliasing=False,
    )
    logits = np.where(resized > 0.5, logit_scale, -logit_scale).astype(np.float32)
    return torch.from_numpy(logits).to(device).unsqueeze(0).unsqueeze(0)


def _bbox_from_mask(mask: np.ndarray, pad_frac: float = 0.08) -> list | None:
    """
    Derive [col_min, row_min, col_max, row_max] from a binary mask.
    Returns None if the mask is empty.
    Padding = pad_frac × the bbox side length on each side (min 3 px).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return None
    r_min, r_max = int(np.where(rows)[0][[0, -1]].tolist()[0]), \
                   int(np.where(rows)[0][[0, -1]].tolist()[1])
    c_min, c_max = int(np.where(cols)[0][[0, -1]].tolist()[0]), \
                   int(np.where(cols)[0][[0, -1]].tolist()[1])
    pad_r = max(int((r_max - r_min) * pad_frac), 3)
    pad_c = max(int((c_max - c_min) * pad_frac), 3)
    H, W = mask.shape
    return [
        max(0,     c_min - pad_c),   # col_min
        max(0,     r_min - pad_r),   # row_min
        min(W - 1, c_max + pad_c),   # col_max
        min(H - 1, r_max + pad_r),   # row_max
    ]


def _infer(model, img_1024: np.ndarray,
           bbox, points, orig_h: int, orig_w: int, device: str) -> np.ndarray:
    """Single-slice inference (used when segmenting one slice at a time)."""
    from contextlib import nullcontext
    t = torch.from_numpy(img_1024.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    autocast_ctx = (torch.autocast("cuda", dtype=torch.float16)
                    if device == "cuda" else nullcontext())
    with torch.inference_mode(), autocast_ctx:
        embedding = model.image_encoder(t).float()
    del t
    if device == "cuda":
        torch.cuda.empty_cache()
    box_torch, pts_torch, lbl_torch = _build_prompt_tensors(
        bbox, points, orig_h, orig_w, device)
    mask, _, _iou = _decode_embedding(model, embedding.squeeze(0),
                                      box_torch, pts_torch, lbl_torch, orig_h, orig_w)
    return mask


# ---------------------------------------------------------------------------
# Plugin dialog
# ---------------------------------------------------------------------------

class MedSamLogic(DynamicDialog):
    completed = pyqtSignal(object)   # kept for mainwindow compat; unused by this plugin

    def __init__(self, data_context, parent=None):
        super().__init__(parent)
        self.data_context  = data_context or {}
        self._main_window  = parent
        self._overlay: MedSamOverlay | None = None
        self._download_thread = None
        self._stop_requested = False
        self._medsam_active: bool = False  # overlay inactive until user enables

        self.create_main_ui(schema=get_schema(), default_items=False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setSizeGripEnabled(True)

        for widget_id, widget_obj in self.widgets.items():
            setattr(self, widget_id, widget_obj)

        if self.data_context:
            self.combo_view.clear()
            self.combo_view.addItems(list(self.data_context.keys()))
            self._select_loaded_view()

        self.combo_view.currentIndexChanged.connect(self._reattach_overlay)
        self.combo_axis.currentIndexChanged.connect(self._reattach_overlay)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_toggle_active.clicked.connect(self._on_toggle_active)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_download.clicked.connect(self.on_btn_download_clicked)
        self.btn_apply.clicked.connect(self.on_btn_apply_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        self._refresh_weights_status()
        self._refresh_device_label()
        self.check_cuda.stateChanged.connect(self._refresh_device_label)
        self._refresh_active_ui()

    # ------------------------------------------------------------------
    # GL widget lookup
    # ------------------------------------------------------------------

    def _get_gl_widget(self, view_name: str, axis_label: str):
        """Find the GLWidget for the requested view/axis."""
        if self._main_window is None:
            return None
        target = _AXIS_TO_WINDOW.get(axis_label, axis_label.lower())
        start, end = (1, 7) if view_name == "view 1" else (7, 13)

        for i in range(start, end):
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is not None and hasattr(w, "currentWidnowName"):
                if w.currentWidnowName.lower() == target:
                    return w

        # Fallback: first widget in the view that has an image
        for i in range(start, end):
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is not None and getattr(w, "imSlice", None) is not None:
                return w
        return None

    # ------------------------------------------------------------------
    # Data context (mirrors NNILogic — see nninteractive.py)
    # ------------------------------------------------------------------

    def update_data_context(self, data_context):
        """Called by the main window whenever a new image/video is loaded."""
        self.data_context = data_context or {}

        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        if self.data_context:
            self.combo_view.addItems(list(self.data_context.keys()))
            self._select_loaded_view()
        self.combo_view.blockSignals(False)

        self._reattach_overlay()

    def _select_loaded_view(self):
        """Point combo_view at a view that actually has an image loaded."""
        current = self.combo_view.currentText()
        if self.data_context.get(current) is not None:
            return
        for i in range(self.combo_view.count()):
            view_name = self.combo_view.itemText(i)
            if self.data_context.get(view_name) is not None:
                self.combo_view.setCurrentIndex(i)
                return

    def _refresh_data_context(self):
        """Pull the current readers directly from the main window — used as a
        fallback when ``self.data_context`` is stale or empty."""
        main_window = getattr(self, "_main_window", None)
        if main_window is None or not hasattr(main_window, "get_current_image_data"):
            return
        self.update_data_context(main_window.get_current_image_data())

    # ------------------------------------------------------------------
    # Overlay lifecycle
    # ------------------------------------------------------------------

    def _reattach_overlay(self):
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None

        if not self._medsam_active:
            return   # overlay only exists while MedSAM is active

        view_name  = self.combo_view.currentText()
        axis_label = self.combo_axis.currentText()
        gl = self._get_gl_widget(view_name, axis_label)
        if gl is None:
            return

        overlay = MedSamOverlay(gl)
        overlay.mode = _MODE_MAP[self.combo_mode.currentIndex()]
        overlay.show()
        self._overlay = overlay

    def _on_mode_changed(self):
        if self._overlay:
            self._overlay.mode = _MODE_MAP[self.combo_mode.currentIndex()]

    def _on_clear(self):
        if self._overlay:
            self._overlay.clear_prompts()

    def _on_stop_clicked(self):
        """Request that an in-progress propagation stop after the current slice/frame."""
        self._stop_requested = True
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Activate / deactivate (mirrors NNILogic — see nninteractive.py)
    # ------------------------------------------------------------------

    def _on_toggle_active(self):
        self._medsam_active = not self._medsam_active
        if self._medsam_active:
            # `data_context`/`combo_view` may be stale or empty if
            # `update_data_context` was never pushed to this widget instance
            # (e.g. created after the image was loaded) — refresh before
            # looking up the GL widget, otherwise `_get_gl_widget` can pick
            # the wrong view (or none at all) and the overlay never attaches.
            self._refresh_data_context()
        self._refresh_active_ui()
        self._reattach_overlay()

    def _refresh_active_ui(self):
        """Update button label/style and control enabled-state to reflect active/inactive."""
        if self._medsam_active:
            self.btn_toggle_active.setText("■  Deactivate")
            self.btn_toggle_active.setStyleSheet(
                "QPushButton { background-color: #8B0000; color: white; "
                "font-weight: bold; padding: 6px; border-radius: 4px; }")
        else:
            self.btn_toggle_active.setText("▶  Activate")
            self.btn_toggle_active.setStyleSheet(
                "QPushButton { background-color: #1a5c1a; color: white; "
                "font-weight: bold; padding: 6px; border-radius: 4px; }")

        for wid in (self.combo_mode, self.combo_axis, self.combo_scope,
                    self.btn_clear, self.btn_apply,
                    self.spin_min_area, self.spin_iou_threshold):
            wid.setEnabled(self._medsam_active)

    def _force_refresh_widgets(self, reader, view_name: str):
        """Directly refresh GL widgets after an in-place npSeg edit (used for
        video frames, where segChanged has no per-frame handling)."""
        from melage.utils.utils import setSliceSeg
        candidates = [1, 2, 3, 11] if view_name == "view 1" else [4, 5, 6, 12]
        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is None:
                continue
            try:
                setSliceSeg(w, reader.npSeg)
                if hasattr(w, "makeObject"):
                    w.makeObject()
                w.update()
            except Exception as e:
                print(f"[MedSAM] refresh error widget {i}: {e}")

    def closeEvent(self, event):
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def _refresh_device_label(self):
        if not self.check_cuda.isChecked():
            self.lbl_device.setText("(CPU mode)")
            return
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            self.lbl_device.setText(f"✔ {name}")
            self.lbl_device.setStyleSheet("color: green;")
        else:
            self.lbl_device.setText("✘ CUDA unavailable")
            self.lbl_device.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _refresh_weights_status(self):
        path = _weights_path()
        if os.path.isfile(path):
            size_gb = os.path.getsize(path) / 1e9
            self.lbl_weights_status.setText(f"✔ Found ({size_gb:.1f} GB)  —  {path}")
            self.btn_download.setEnabled(False)
            self.btn_apply.setEnabled(True)
        else:
            self.lbl_weights_status.setText(f"✘ Not found — will be saved to:\n{path}")
            self.btn_download.setEnabled(True)
            self.btn_apply.setEnabled(False)

    def on_btn_download_clicked(self):
        if not SAM_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "pip install git+https://github.com/facebookresearch/segment-anything.git",
            )
            return
        self.btn_download.setEnabled(False)
        self.progress_download.setValue(0)
        self.lbl_weights_status.setText("Downloading…")
        self._download_thread = _DownloadThread(_weights_path(), parent=self)
        self._download_thread.progress.connect(self.progress_download.setValue)
        self._download_thread.succeeded.connect(self._on_download_succeeded)
        self._download_thread.failed.connect(self._on_download_failed)
        self._download_thread.start()

    def _on_download_succeeded(self, path):
        self.progress_download.setValue(100)
        self._refresh_weights_status()
        QMessageBox.information(self, "Done", f"Weights saved to:\n{path}")

    def _on_download_failed(self, error):
        self._refresh_weights_status()
        self.progress_download.setValue(0)
        QMessageBox.critical(
            self, "Download failed",
            f"{error}\n\nManual download:\nhttps://zenodo.org/records/10689643\n"
            f"Save as: {_weights_path()}",
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def on_btn_apply_clicked(self):
        if not SAM_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "pip install git+https://github.com/facebookresearch/segment-anything.git",
            )
            return

        overlay = self._overlay
        if overlay is None:
            self._refresh_data_context()
            overlay = self._overlay
        if overlay is None:
            QMessageBox.warning(self, "No image", "No image loaded.")
            return

        scope = self.combo_scope.currentText()

        if scope != "Propagate current label mask" and not overlay.has_prompt():
            QMessageBox.warning(
                self, "No prompt",
                "Draw a bounding box by dragging, or click to place points on the image.",
            )
            return

        gl = overlay.gl
        if scope != "Propagate current label mask" and getattr(gl, "imSlice", None) is None:
            QMessageBox.warning(self, "No image", "The selected view has no image loaded.")
            return

        checkpoint = _weights_path()
        if not os.path.isfile(checkpoint):
            QMessageBox.critical(self, "Weights missing", "Please download the weights first.")
            return

        view_name = self.combo_view.currentText()
        reader    = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "npImage"):
            QMessageBox.warning(self, "No reader", "Cannot access image data.")
            return

        bbox        = overlay.bbox
        points      = overlay.points
        window_name = gl.currentWidnowName
        colorInd    = gl.colorInd if gl.colorInd != 9876 else 1
        is_video    = getattr(reader, "isChunkedVideo", False)
        ref_idx     = reader.current_frame if is_video else gl.sliceNum

        wants_cuda = self.check_cuda.isChecked()
        if wants_cuda and not torch.cuda.is_available():
            QMessageBox.warning(
                self, "CUDA not available",
                "GPU was requested but torch.cuda.is_available() returned False.\n\n"
                "The most common cause is that the CPU-only build of PyTorch is installed.\n\n"
                "To install the CUDA build (example for CUDA 12.1):\n"
                "  pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu121\n\n"
                "Falling back to CPU for this run.",
            )
        device = "cuda" if wants_cuda and torch.cuda.is_available() else "cpu"
        print(f"[MedSAM] device = {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

        try:
            from melage.utils.utils import getCurrentSlice, permute_axis
        except ImportError as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        def _emit(mask: np.ndarray, sn: int) -> None:
            """Write one 2D mask back into the 3D label volume."""
            if not np.any(mask):
                return
            y_idx, x_idx = np.where(mask > 0)
            wi = np.column_stack([x_idx, y_idx, np.full_like(x_idx, sn)])
            wi, _ = permute_axis(wi, None, window_name)
            gl.segChanged.emit(wi.astype("int"), window_name, colorInd, sn)

        def _get_slice(sn: int):
            slc, _, _ = getCurrentSlice(gl, reader.npImage, reader.npSeg, sn)
            return slc

        def _get_slice_any(sn: int):
            """Like `_get_slice`, but for video readers `sn` is a frame index
            into `reader.video_im` (grayscale-collapsed RGB frame)."""
            if is_video:
                frame = reader.video_im.get_frame(sn)
                if frame.ndim == 3:
                    frame = frame.mean(axis=-1)
                return frame.astype(np.float32)
            return _get_slice(sn)

        def _get_label_mask(sn: int) -> np.ndarray:
            """Binary mask of `colorInd` at slice/frame `sn`."""
            if is_video:
                seg = reader.seg_ims.get_frame(sn)
            else:
                _, seg, _ = getCurrentSlice(gl, reader.npImage, reader.npSeg, sn)
            return (seg == colorInd).astype(np.uint8)

        def _emit_any(mask: np.ndarray, sn: int) -> None:
            """Like `_emit`, but writes video frames directly into
            `reader.seg_ims` (segChanged has no per-frame video handling)."""
            if not np.any(mask):
                return
            if is_video:
                seg_frame = reader.seg_ims.get_frame(sn).copy()
                seg_frame[seg_frame == colorInd] = 0
                seg_frame[mask > 0] = colorInd
                reader.commit_frame_segmentation_changes(seg_frame, sn)
                if sn == reader.current_frame:
                    reader.npSeg = seg_frame
                    self._force_refresh_widgets(reader, view_name)
            else:
                _emit(mask, sn)

        from contextlib import nullcontext
        ac = (torch.autocast("cuda", dtype=torch.float16)
              if device == "cuda" else nullcontext())

        def _encode_slice(slc):
            t = torch.from_numpy(
                _preprocess(slc).transpose(2, 0, 1)[None]).to(device)
            with torch.inference_mode(), ac:
                emb = model.image_encoder(t).float()
            del t
            if device == "cuda":
                torch.cuda.empty_cache()
            return emb.squeeze(0)

        try:
            self.btn_apply.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._stop_requested = False
            self.progress_bar.setValue(5)
            QtWidgets.QApplication.processEvents()

            model = _get_model(checkpoint, device)
            self.progress_bar.setValue(10)

            # ── SCOPE: current slice only ──────────────────────────────────
            if scope == "Current slice":
                slc = _get_slice(ref_idx)
                if slc is None:
                    return
                orig_h, orig_w = slc.shape
                mask = _infer(model, _preprocess(slc), bbox, points,
                              orig_h, orig_w, device)
                _emit(mask, ref_idx)
                self.progress_bar.setValue(100)
                segs = 1

            # ── SCOPE: all slices, same fixed box (fast batch encoding) ────
            elif scope == "All slices — fixed box":
                n = gl.imDepth
                if n > 1:
                    reply = QMessageBox.question(
                        self, "Apply to all slices?",
                        f"The same prompt will be applied to all {n} slices.\n"
                        "Best when the target stays in roughly the same position.",
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        return

                raw_slices, valid_sn, orig_sizes = [], [], []
                for sn in range(n):
                    slc = _get_slice(sn)
                    if slc is not None:
                        raw_slices.append(slc)
                        valid_sn.append(sn)
                        orig_sizes.append(slc.shape)

                self.lbl_device.setText(
                    self.lbl_device.text() + "  |  preprocessing…")
                QtWidgets.QApplication.processEvents()
                imgs_1024 = _preprocess_parallel(raw_slices)

                self.progress_bar.setValue(25)
                self.lbl_device.setText(
                    self.lbl_device.text().split("|")[0].strip() + "  |  encoding…")
                QtWidgets.QApplication.processEvents()
                embeddings = _encode_batch(model, imgs_1024, device)

                self.progress_bar.setValue(60)
                self.lbl_device.setText(
                    self.lbl_device.text().split("|")[0].strip() + "  |  decoding…")
                total = len(valid_sn)
                for step, (sn, emb, (orig_h, orig_w)) in enumerate(
                        zip(valid_sn, embeddings, orig_sizes)):
                    box_t, pts_t, lbl_t = _build_prompt_tensors(
                        bbox, points, orig_h, orig_w, device)
                    mask, _, _iou = _decode_embedding(model, emb, box_t, pts_t, lbl_t,
                                                     orig_h, orig_w)
                    _emit(mask, sn)
                    self.progress_bar.setValue(
                        60 + int((step + 1) / total * 35))
                    if step % 5 == 0:
                        QtWidgets.QApplication.processEvents()

                self._refresh_device_label()
                segs = total

            # ── SCOPE: propagate from reference slice (bbox/point prompt) ──
            elif scope == "All slices — propagate":
                min_area      = int(self.spin_min_area.value())
                iou_threshold = float(self.spin_iou_threshold.value()) / 100.0

                # ── Reference slice: user's bbox/points prompt ─────────────
                ref_slc = _get_slice(ref_idx)
                if ref_slc is None:
                    return
                orig_h, orig_w = ref_slc.shape
                box_t, pts_t, lbl_t = _build_prompt_tensors(
                    bbox, points, orig_h, orig_w, device)
                ref_mask, ref_logits, ref_iou = _decode_embedding(
                    model, _encode_slice(ref_slc),
                    box_t, pts_t, lbl_t, orig_h, orig_w,
                    mask_prompt=None)

                if np.sum(ref_mask) < min_area:
                    QMessageBox.warning(
                        self, "Empty reference mask",
                        f"The reference slice ({ref_idx}) produced a mask with "
                        f"fewer than {min_area} pixels.\n"
                        "Try a tighter bounding box or a different slice.",
                    )
                    return

                _emit(ref_mask, ref_idx)
                print(f"[MedSAM] ref slice {ref_idx}: "
                      f"area={np.sum(ref_mask)} px, IoU={ref_iou:.2f}")
                self.progress_bar.setValue(20)

                all_indices = sorted(range(gl.imDepth))
                ref_pos    = all_indices.index(ref_idx)
                forward    = all_indices[ref_pos + 1:]
                backward   = list(reversed(all_indices[:ref_pos]))
                segs       = 1
                total_dirs = len(forward) + len(backward)

                def _propagate_one_direction(indices, start_mask, start_logits):
                    """
                    Propagate slice-by-slice using BOTH prompts:
                      • bbox derived from previous mask  → spatial anchor
                        (without this, sparse_prompt_embeddings is empty and
                         SAM's IoU estimate collapses immediately)
                      • mask_prompt = previous logits    → shape prior

                    The IoU score + area threshold decide when to stop:
                    when the structure disappears the box lands on plain tissue,
                    the mask prior finds nothing to latch onto → IoU drops.
                    """
                    nonlocal segs
                    prev_mask   = start_mask
                    prev_logits = start_logits
                    pad_frac    = 0.08   # 8 % padding on derived bbox

                    for sn in indices:
                        if self._stop_requested:
                            break

                        # Derive bbox from the previous slice's binary mask
                        derived_bbox = _bbox_from_mask(prev_mask, pad_frac)
                        if derived_bbox is None:
                            break   # previous mask was empty

                        slc = _get_slice(sn)
                        if slc is None:
                            break
                        oh, ow = slc.shape
                        emb = _encode_slice(slc)

                        # bbox for spatial location + mask prompt for shape
                        box_t2, _, _ = _build_prompt_tensors(
                            derived_bbox, [], oh, ow, device)
                        mask, logits, iou = _decode_embedding(
                            model, emb,
                            box_t2, None, None,
                            oh, ow,
                            mask_prompt=prev_logits)

                        print(f"[MedSAM] slice {sn}: "
                              f"area={np.sum(mask)} px, IoU={iou:.2f}")

                        # Stop when model is not confident OR mask too small
                        if iou < iou_threshold or np.sum(mask) < min_area:
                            break

                        _emit(mask, sn)
                        prev_mask   = mask
                        prev_logits = logits
                        segs += 1

                        pct = 20 + int(segs / max(total_dirs, 1) * 75)
                        self.progress_bar.setValue(min(pct, 95))
                        QtWidgets.QApplication.processEvents()

                _propagate_one_direction(forward,  ref_mask, ref_logits)
                _propagate_one_direction(backward, ref_mask, ref_logits)
                self.progress_bar.setValue(100)

            # ── SCOPE: propagate from the currently drawn label mask ───────
            elif scope == "Propagate current label mask":
                min_area      = int(self.spin_min_area.value())
                iou_threshold = float(self.spin_iou_threshold.value()) / 100.0

                ref_mask = _get_label_mask(ref_idx)
                if not np.any(ref_mask):
                    QMessageBox.warning(
                        self, "No label mask",
                        "No segmentation found for the active label on the "
                        "current slice/frame.\nDraw or select a label mask "
                        "first, then run propagation.",
                    )
                    return

                # ── Warm-up pass: run the model on the reference slice itself,
                # using a bbox derived from the user's mask (+ the mask as a
                # shape prior). This yields true model logits/IoU as the
                # starting point for propagation, instead of relying solely
                # on the synthetic logits from `_mask_to_logits`.
                ref_slc = _get_slice_any(ref_idx)
                if ref_slc is None:
                    return
                ref_oh, ref_ow = ref_slc.shape
                ref_bbox = _bbox_from_mask(ref_mask, 0.08)
                ref_box_t, _, _ = _build_prompt_tensors(ref_bbox, [], ref_oh, ref_ow, device)
                ref_mask_model, ref_logits, ref_iou = _decode_embedding(
                    model, _encode_slice(ref_slc),
                    ref_box_t, None, None,
                    ref_oh, ref_ow,
                    mask_prompt=_mask_to_logits(ref_mask, device))

                print(f"[MedSAM] reference {'frame' if is_video else 'slice'} "
                      f"{ref_idx}: warm-up area={np.sum(ref_mask_model)} px, "
                      f"IoU={ref_iou:.2f}")

                if np.any(ref_mask_model) and ref_iou >= iou_threshold:
                    ref_mask = ref_mask_model
                else:
                    print(f"[MedSAM] warm-up pass produced a low-confidence "
                          f"mask (IoU={ref_iou:.2f}) — falling back to the "
                          f"drawn mask for propagation.")
                    ref_logits = _mask_to_logits(ref_mask, device)

                n_indices   = reader.seg_ims.shape[2] if is_video else gl.imDepth
                all_indices = list(range(n_indices))
                ref_pos     = all_indices.index(ref_idx)
                forward     = all_indices[ref_pos + 1:]
                backward    = list(reversed(all_indices[:ref_pos]))
                segs        = 0
                total_dirs  = len(forward) + len(backward)

                def _propagate_mask_direction(indices, start_mask, start_logits):
                    """Same idea as `_propagate_one_direction`, but seeded from
                    a user-drawn mask instead of a SAM-decoded reference mask,
                    and writing through `_get_slice_any`/`_emit_any` so it
                    works for both 3-D slices and video frames."""
                    nonlocal segs
                    prev_mask   = start_mask
                    prev_logits = start_logits
                    pad_frac    = 0.08

                    for sn in indices:
                        if self._stop_requested:
                            break

                        derived_bbox = _bbox_from_mask(prev_mask, pad_frac)
                        if derived_bbox is None:
                            break   # previous mask was empty

                        slc = _get_slice_any(sn)
                        if slc is None:
                            break
                        oh, ow = slc.shape
                        emb = _encode_slice(slc)

                        box_t2, _, _ = _build_prompt_tensors(
                            derived_bbox, [], oh, ow, device)
                        mask, logits, iou = _decode_embedding(
                            model, emb,
                            box_t2, None, None,
                            oh, ow,
                            mask_prompt=prev_logits)

                        print(f"[MedSAM] {'frame' if is_video else 'slice'} {sn}: "
                              f"area={np.sum(mask)} px, IoU={iou:.2f}")

                        if iou < iou_threshold or np.sum(mask) < min_area:
                            break

                        _emit_any(mask, sn)
                        prev_mask   = mask
                        prev_logits = logits
                        segs += 1

                        pct = int(segs / max(total_dirs, 1) * 95)
                        self.progress_bar.setValue(min(pct, 95))
                        self.lbl_device.setText(
                            self.lbl_device.text().split("|")[0].strip()
                            + f"  |  {'frame' if is_video else 'slice'} {sn}")
                        QtWidgets.QApplication.processEvents()

                self.progress_bar.setValue(5)
                _propagate_mask_direction(forward,  ref_mask, ref_logits)
                _propagate_mask_direction(backward, ref_mask, ref_logits)
                self.progress_bar.setValue(100)

            QMessageBox.information(
                self, "Done",
                f"MedSAM segmented {segs} slice(s) on {view_name} ({window_name}).",
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            self.btn_apply.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._stop_requested = False
            self._refresh_device_label()


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

class MedSamPlugin(MelagePlugin):
    @property
    def name(self) -> str:
        return "MedSAM"

    @property
    def description(self) -> str:
        return ("Interactive bounding-box / point-prompted segmentation "
                "for medical images (Ma et al., Nature Communications 2024).")

    @property
    def reference(self) -> str:
        return ('Ma et al. '
                '<a href="https://doi.org/10.1038/s41467-024-44824-z">'
                'Segment anything in medical images.</a> '
                'Nature Communications, 2024.')

    @property
    def category(self) -> str:
        return "Deep Learning"

    def get_widget(self, data_context=None, parent=None):
        return MedSamLogic(data_context, parent)
