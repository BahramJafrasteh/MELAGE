"""
SAM 2 plugin — GL-based interactive overlay with native video-memory propagation.

This mirrors the nnInteractive / MedSAM sidebar pattern: a transparent
overlay widget floats on top of the active GLWidget so the user draws a
bounding box / points directly on the live image, with activate/deactivate
toggling exactly like the other plugins.

Unlike MedSAM (which re-runs the model slice-by-slice as isolated 2-D
images), the "Propagate current label mask" scope here uses SAM 2's native
video predictor: every slice/frame is exported once into a short-lived
sequence, the reference mask is registered as a prompt, and
``propagate_in_video`` lets the model carry its own memory-attention state
from slice to slice — the same engine SAM 2 uses for video object tracking.
"""

import os
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QColor

from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .sam2_schema import get_schema, _MODEL_REGISTRY, _DEFAULT_MODEL

from melage.plugins.medsam.medsam import (
    MedSamOverlay, _DownloadThread, _AXIS_TO_WINDOW, _MODE_MAP, _weights_dir,
)

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

_IMAGE_PREDICTOR_CACHE: dict = {}
_VIDEO_PREDICTOR_CACHE: dict = {}

# colorInd values used in npSeg — map 1:1 with label index (mirrors nninteractive.py)
_COLORINDS = [1, 2, 3, 4, 5, 6, 7, 8]

# QColor per label index (0-based), used for the label-status indicator
_LABEL_QCOLORS = [
    QColor(255,  60,  60),   # 1 Red
    QColor( 60, 220,  60),   # 2 Green
    QColor( 60, 100, 255),   # 3 Blue
    QColor(255, 220,  50),   # 4 Yellow
    QColor(220,  60, 220),   # 5 Magenta
    QColor( 50, 220, 220),   # 6 Cyan
    QColor(255, 140,  50),   # 7 Orange
    QColor(160,  60, 255),   # 8 Purple
]

_MAX_LABELS = len(_COLORINDS)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def _weights_path(model_key: str = _DEFAULT_MODEL) -> str:
    filename = _MODEL_REGISTRY[model_key]["filename"]
    return os.path.join(_weights_dir(), filename)


# ---------------------------------------------------------------------------
# Predictor cache
# ---------------------------------------------------------------------------

def _get_image_predictor(checkpoint: str, config: str, device: str) -> "SAM2ImagePredictor":
    key = (checkpoint, device)
    if key not in _IMAGE_PREDICTOR_CACHE:
        model = build_sam2(config, checkpoint, device=device)
        _IMAGE_PREDICTOR_CACHE[key] = SAM2ImagePredictor(model)
    return _IMAGE_PREDICTOR_CACHE[key]


def _get_video_predictor(checkpoint: str, config: str, device: str):
    key = (checkpoint, device)
    if key not in _VIDEO_PREDICTOR_CACHE:
        _VIDEO_PREDICTOR_CACHE[key] = build_sam2_video_predictor(
            config, checkpoint, device=device)
    return _VIDEO_PREDICTOR_CACHE[key]


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _slice_to_rgb_uint8(slc: np.ndarray) -> np.ndarray:
    """Convert a 2-D slice/frame (any dtype) to an (H, W, 3) uint8 RGB array."""
    if slc.ndim == 3:
        slc = slc.mean(axis=-1)
    if slc.dtype == np.uint8:
        norm = slc
    else:
        p1, p99 = np.percentile(slc, [1, 99])
        clipped = np.clip(slc.astype(np.float64), p1, p99)
        lo, hi = float(clipped.min()), float(clipped.max())
        norm = (((clipped - lo) / max(hi - lo, 1e-8)) * 255.0).astype(np.uint8) \
            if hi > lo else np.zeros_like(slc, dtype=np.uint8)
    return np.ascontiguousarray(np.stack([norm, norm, norm], axis=-1))


def _write_jpeg(args: tuple):
    path, frame_rgb = args
    Image.fromarray(frame_rgb).save(path, format="JPEG", quality=90,
                                     subsampling=0, optimize=False)


def _get_fast_temp_dir() -> str:
    """Prefer /dev/shm (RAM disk) for the exported frame sequence."""
    ram_disk = "/dev/shm"
    if os.path.isdir(ram_disk):
        return tempfile.mkdtemp(dir=ram_disk, prefix="melage_sam2_")
    return tempfile.mkdtemp(prefix="melage_sam2_")


# ---------------------------------------------------------------------------
# Plugin dialog
# ---------------------------------------------------------------------------

class Sam2Logic(DynamicDialog):
    completed = pyqtSignal(object)   # kept for mainwindow compat; unused by this plugin

    def __init__(self, data_context, parent=None):
        super().__init__(parent)
        self.data_context  = data_context or {}
        self._main_window  = parent
        self._overlay: MedSamOverlay | None = None
        self._download_thread = None
        self._stop_requested = False
        self._sam2_active: bool = False  # overlay inactive until user enables
        self._label_idx: int = 0         # 0-based index into _COLORINDS

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
        self.combo_model.currentIndexChanged.connect(self._refresh_weights_status)
        self.btn_toggle_active.clicked.connect(self._on_toggle_active)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_new_label.clicked.connect(self._on_new_label)
        self.btn_clear_label.clicked.connect(self._on_clear_label)
        self.btn_download.clicked.connect(self.on_btn_download_clicked)
        self.btn_apply.clicked.connect(self.on_btn_apply_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        self._refresh_weights_status()
        self._refresh_device_label()
        self._refresh_label_status()
        self.check_cuda.stateChanged.connect(self._refresh_device_label)
        self._refresh_active_ui()

    # ------------------------------------------------------------------
    # GL widget lookup (mirrors MedSamLogic — see medsam.py)
    # ------------------------------------------------------------------

    def _get_gl_widget(self, view_name: str, axis_label: str):
        if self._main_window is None:
            return None
        target = _AXIS_TO_WINDOW.get(axis_label, axis_label.lower())
        # Covers both the Tab-0 three-panel widgets (1-3 / 4-6) and the
        # Tab-1/Tab-2 fullscreen widgets (11 / 12) used by the Mutual view —
        # otherwise the overlay can attach to a hidden widget that never
        # receives mouse events while the user draws on 11/12.
        candidates = [1, 2, 3, 11] if view_name == "view 1" else [4, 5, 6, 12]

        # Prefer a *visible* widget whose orientation matches the requested axis.
        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is not None and w.isVisible() and hasattr(w, "currentWidnowName"):
                if w.currentWidnowName.lower() == target:
                    return w

        # Fall back to any visible widget with an image loaded.
        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is not None and w.isVisible() and getattr(w, "imSlice", None) is not None:
                return w

        # Last resort: any widget with an image loaded, regardless of visibility.
        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is not None and getattr(w, "imSlice", None) is not None:
                return w
        return None

    # ------------------------------------------------------------------
    # Data context (mirrors NNILogic / MedSamLogic)
    # ------------------------------------------------------------------

    def update_data_context(self, data_context):
        self.data_context = data_context or {}

        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        if self.data_context:
            self.combo_view.addItems(list(self.data_context.keys()))
            self._select_loaded_view()
        self.combo_view.blockSignals(False)

        self._reattach_overlay()

    def _select_loaded_view(self):
        current = self.combo_view.currentText()
        if self.data_context.get(current) is not None:
            return
        for i in range(self.combo_view.count()):
            view_name = self.combo_view.itemText(i)
            if self.data_context.get(view_name) is not None:
                self.combo_view.setCurrentIndex(i)
                return

    def _refresh_data_context(self):
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

        if not self._sam2_active:
            return   # overlay only exists while SAM 2 is active

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

    # ------------------------------------------------------------------
    # Label management (mirrors NNILogic — see nninteractive.py)
    # ------------------------------------------------------------------

    def _refresh_label_status(self):
        colorInd = _COLORINDS[self._label_idx % _MAX_LABELS]
        color    = _LABEL_QCOLORS[self._label_idx % _MAX_LABELS]
        self.lbl_label_status.setText(f"Label {self._label_idx + 1}")
        self.lbl_label_status.setToolTip(f"colorInd = {colorInd}")
        self.lbl_label_status.setStyleSheet(
            f"color: rgb({color.red()},{color.green()},{color.blue()});"
        )

    def _on_new_label(self):
        if self._label_idx >= _MAX_LABELS - 1:
            QMessageBox.information(self, "Max labels",
                                    f"Maximum {_MAX_LABELS} labels supported.")
            return
        self._label_idx += 1
        self._refresh_label_status()
        if self._overlay is not None:
            self._overlay.clear_prompts()

    def _on_clear_label(self):
        """Erase the current label's mask from npSeg on the active view."""
        colorInd  = _COLORINDS[self._label_idx % _MAX_LABELS]
        view_name = self.combo_view.currentText()
        reader    = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "npSeg"):
            return
        if getattr(reader, "isChunkedVideo", False):
            seg_frame = reader.seg_ims.get_frame(reader.current_frame).copy()
            seg_frame[seg_frame == colorInd] = 0
            reader.commit_frame_segmentation_changes(seg_frame, reader.current_frame)
            reader.npSeg = seg_frame
        else:
            reader.npSeg[reader.npSeg == colorInd] = 0
        self._force_refresh_widgets(reader, view_name)
        if self._overlay is not None:
            self._overlay.clear_prompts()

    def _ensure_label_visible(self, colorInd: int):
        """
        Make sure colorInd appears in every relevant GL widget's colorInds list
        so makeObject() renders it immediately — no manual Color Settings toggle needed.

        Also ticks the matching row in the tree_colors panel (signals blocked so
        changeColorPen's side-effects — changing the active drawing color — don't fire).
        """
        if self._main_window is None:
            return
        view_name  = self.combo_view.currentText()
        candidates = [1, 2, 3, 11] if view_name == "view 1" else [4, 5, 6, 12]

        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is None:
                continue
            if 9876 not in w.colorInds and colorInd not in w.colorInds:
                w.colorInds = list(w.colorInds) + [colorInd]

        try:
            tree = getattr(self._main_window, "tree_colors", None)
            if tree is None:
                return
            source = tree.model().sourceModel()
            root   = source.invisibleRootItem()
            source.blockSignals(True)
            for i in range(root.rowCount()):
                item = root.child(i)
                if item is None:
                    continue
                try:
                    if int(float(item.text())) == colorInd:
                        if item.checkState() != Qt.Checked:
                            item.setCheckState(Qt.Checked)
                        break
                except (ValueError, TypeError):
                    pass
            source.blockSignals(False)
        except Exception as e:
            print(f"[SAM2] _ensure_label_visible tree sync: {e}")

    def _on_stop_clicked(self):
        """Request that an in-progress propagation stop after the current frame."""
        self._stop_requested = True
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Activate / deactivate (mirrors NNILogic / MedSamLogic)
    # ------------------------------------------------------------------

    def _on_toggle_active(self):
        self._sam2_active = not self._sam2_active
        if self._sam2_active:
            self._refresh_data_context()
        self._refresh_active_ui()
        self._reattach_overlay()

    def _refresh_active_ui(self):
        if self._sam2_active:
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
                    self.btn_clear, self.btn_new_label, self.btn_clear_label,
                    self.btn_apply, self.check_limit_range, self.spin_prop_range):
            wid.setEnabled(self._sam2_active)

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
                print(f"[SAM2] refresh error widget {i}: {e}")

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
        model_key = self.combo_model.currentText()
        path = _weights_path(model_key)
        if os.path.isfile(path):
            size_gb = os.path.getsize(path) / 1e9
            self.lbl_weights_status.setText(f"✔ Found ({size_gb:.1f} GB)  —  {path}")
            self.btn_download.setEnabled(False)
            self.btn_apply.setEnabled(self._sam2_active)
        else:
            self.lbl_weights_status.setText(f"✘ Not found — will be saved to:\n{path}")
            self.btn_download.setEnabled(True)
            self.btn_apply.setEnabled(False)

    def on_btn_download_clicked(self):
        if not SAM2_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git",
            )
            return
        model_key = self.combo_model.currentText()
        self.btn_download.setEnabled(False)
        self.progress_download.setValue(0)
        self.lbl_weights_status.setText("Downloading…")
        self._download_thread = _DownloadThread(
            _weights_path(model_key), [_MODEL_REGISTRY[model_key]["url"]], parent=self)
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
        model_key = self.combo_model.currentText()
        QMessageBox.critical(
            self, "Download failed",
            f"{error}\n\nManual download:\n{_MODEL_REGISTRY[model_key]['url']}\n"
            f"Save as: {_weights_path(model_key)}",
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def on_btn_apply_clicked(self):
        if not SAM2_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git",
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

        if scope == "Current slice" and not overlay.has_prompt():
            QMessageBox.warning(
                self, "No prompt",
                "Draw a bounding box by dragging, or click to place points on the image.",
            )
            return

        gl = overlay.gl
        if scope != "Propagate current label mask" and getattr(gl, "imSlice", None) is None:
            QMessageBox.warning(self, "No image", "The selected view has no image loaded.")
            return

        model_key  = self.combo_model.currentText()
        config     = _MODEL_REGISTRY[model_key]["config"]
        checkpoint = _weights_path(model_key)
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
        colorInd    = _COLORINDS[self._label_idx % _MAX_LABELS]
        is_video    = getattr(reader, "isChunkedVideo", False)
        ref_idx     = reader.current_frame if is_video else gl.sliceNum

        wants_cuda = self.check_cuda.isChecked()
        if wants_cuda and not torch.cuda.is_available():
            QMessageBox.warning(
                self, "CUDA not available",
                "GPU was requested but torch.cuda.is_available() returned False.\n\n"
                "Falling back to CPU for this run.",
            )
        device = "cuda" if wants_cuda and torch.cuda.is_available() else "cpu"
        print(f"[SAM2] device = {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

        try:
            from melage.utils.utils import getCurrentSlice, permute_axis
        except ImportError as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        def _emit(mask: np.ndarray, sn: int) -> None:
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
            if is_video:
                frame = reader.video_im.get_frame(sn)
                if frame.ndim == 3:
                    frame = frame.mean(axis=-1)
                return frame.astype(np.float32)
            return _get_slice(sn)

        def _get_label_mask(sn: int) -> np.ndarray:
            if is_video:
                seg = reader.seg_ims.get_frame(sn)
            else:
                _, seg, _ = getCurrentSlice(gl, reader.npImage, reader.npSeg, sn)
            return (seg == colorInd).astype(np.uint8)

        def _emit_any(mask: np.ndarray, sn: int) -> None:
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

        try:
            self.btn_apply.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._stop_requested = False
            self.progress_bar.setValue(5)
            QtWidgets.QApplication.processEvents()

            self._ensure_label_visible(colorInd)

            from contextlib import nullcontext
            ac = (torch.autocast("cuda", dtype=torch.bfloat16)
                  if device == "cuda" else nullcontext())

            # ── SCOPE: current slice only (single-image predictor) ────────
            if scope == "Current slice":
                slc = _get_slice(ref_idx)
                if slc is None:
                    return
                rgb = _slice_to_rgb_uint8(slc)

                predictor = _get_image_predictor(checkpoint, config, device)
                self.progress_bar.setValue(20)

                pt_coords = pt_labels = None
                if points:
                    pt_coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
                    pt_labels = np.array([int(p[2]) for p in points], dtype=np.int64)
                box_arr = np.array(bbox, dtype=np.float32) if bbox is not None else None

                with torch.inference_mode(), ac:
                    predictor.set_image(rgb)
                    self.progress_bar.setValue(60)
                    masks, scores, _ = predictor.predict(
                        point_coords=pt_coords, point_labels=pt_labels,
                        box=box_arr, multimask_output=False)

                mask = (masks[0] > 0).astype(np.uint8)
                _emit(mask, ref_idx)
                print(f"[SAM2] slice {ref_idx}: area={int(mask.sum())} px, "
                      f"score={float(scores[0]):.2f}")
                self.progress_bar.setValue(100)
                segs = 1

            # ── SCOPE: refine from existing segmentation mask ─────────────
            elif scope == "Refine from existing mask (current slice)":
                ref_mask = _get_label_mask(ref_idx)
                if not np.any(ref_mask):
                    QMessageBox.warning(
                        self, "No label mask",
                        "No segmentation found for the active label on the "
                        "current slice/frame.\nDraw a mask for this label "
                        "first, or switch to 'Current slice' and draw a "
                        "bounding box / points.",
                    )
                    return

                y_idx, x_idx = np.where(ref_mask > 0)
                box_arr = np.array(
                    [x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()],
                    dtype=np.float32,
                )

                slc = _get_slice_any(ref_idx)
                if slc is None:
                    return
                rgb = _slice_to_rgb_uint8(slc)

                predictor = _get_image_predictor(checkpoint, config, device)
                self.progress_bar.setValue(20)

                pt_coords = pt_labels = None
                if points:
                    pt_coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
                    pt_labels = np.array([int(p[2]) for p in points], dtype=np.int64)

                with torch.inference_mode(), ac:
                    predictor.set_image(rgb)
                    self.progress_bar.setValue(60)
                    masks, scores, _ = predictor.predict(
                        point_coords=pt_coords, point_labels=pt_labels,
                        box=box_arr, multimask_output=False)

                mask = (masks[0] > 0).astype(np.uint8)
                _emit_any(mask, ref_idx)
                print(f"[SAM2] refine slice {ref_idx}: area={int(mask.sum())} px, "
                      f"score={float(scores[0]):.2f}")
                self.progress_bar.setValue(100)
                segs = 1

            # ── SCOPE: propagate from the current label mask (video memory) ─
            elif scope == "Propagate current label mask":
                ref_mask = _get_label_mask(ref_idx)
                if not np.any(ref_mask):
                    QMessageBox.warning(
                        self, "No label mask",
                        "No segmentation found for the active label on the "
                        "current slice/frame.\nDraw or select a label mask "
                        "first, then run propagation.",
                    )
                    return

                n_indices = reader.seg_ims.shape[2] if is_video else gl.imDepth

                # ── Limit the exported range around the reference slice so
                # large volumes/videos don't stall on exporting every frame.
                if self.check_limit_range.isChecked():
                    span = int(self.spin_prop_range.value())
                    start_z = max(0, ref_idx - span)
                    end_z   = min(n_indices, ref_idx + span + 1)
                else:
                    start_z, end_z = 0, n_indices
                ref_local = ref_idx - start_z
                sub_len   = end_z - start_z

                # ── Export the (sub-)sequence once so SAM 2's video predictor
                # can build its memory bank across slices/frames.
                temp_dir = _get_fast_temp_dir()
                try:
                    self.lbl_device.setText(
                        self.lbl_device.text().split("|")[0].strip() + "  |  exporting frames…")
                    QtWidgets.QApplication.processEvents()

                    tasks = []
                    for local_i, sn in enumerate(range(start_z, end_z)):
                        slc = _get_slice_any(sn)
                        rgb = _slice_to_rgb_uint8(slc) if slc is not None \
                            else np.zeros((8, 8, 3), dtype=np.uint8)
                        tasks.append((os.path.join(temp_dir, f"{local_i:05d}.jpg"), rgb))
                        if local_i % 30 == 0:
                            QtWidgets.QApplication.processEvents()

                    workers = min(4, max(1, sub_len))
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(_write_jpeg, t) for t in tasks]
                        for done, _ in enumerate(futures):
                            futures[done].result()
                            if done % 10 == 0 or done == len(futures) - 1:
                                self.progress_bar.setValue(
                                    5 + int((done + 1) / sub_len * 20))
                                QtWidgets.QApplication.processEvents()

                    self.progress_bar.setValue(25)

                    predictor = _get_video_predictor(checkpoint, config, device)
                    self.progress_bar.setValue(35)

                    with torch.inference_mode(), ac:
                        inference_state = predictor.init_state(video_path=temp_dir)
                        predictor.add_new_mask(
                            inference_state, frame_idx=ref_local, obj_id=1,
                            mask=(ref_mask > 0))
                        self.progress_bar.setValue(45)

                        segs = 0
                        total = max(sub_len - 1, 1)

                        def _propagate(reverse: bool):
                            nonlocal segs
                            for out_idx, _, out_logits in predictor.propagate_in_video(
                                    inference_state, start_frame_idx=ref_local, reverse=reverse):
                                if self._stop_requested:
                                    break
                                if out_idx == ref_local:
                                    continue
                                sn = out_idx + start_z
                                mask = (out_logits[0] > 0).cpu().numpy().squeeze().astype(np.uint8)
                                if mask.any():
                                    _emit_any(mask, sn)
                                    print(f"[SAM2] {'frame' if is_video else 'slice'} "
                                          f"{sn}: area={int(mask.sum())} px")
                                segs += 1
                                self.progress_bar.setValue(min(45 + int(segs / total * 50), 95))
                                self.lbl_device.setText(
                                    self.lbl_device.text().split("|")[0].strip()
                                    + f"  |  {'frame' if is_video else 'slice'} {sn}")
                                QtWidgets.QApplication.processEvents()

                        _propagate(reverse=False)
                        if not self._stop_requested:
                            _propagate(reverse=True)

                        predictor.reset_state(inference_state)

                    self.progress_bar.setValue(100)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            QMessageBox.information(
                self, "Done",
                f"SAM 2 segmented {segs} slice(s) on {view_name} ({window_name}).",
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

class Sam2Plugin(MelagePlugin):
    @property
    def name(self) -> str:
        return "SAM 2"

    @property
    def description(self) -> str:
        return ("Interactive bounding-box / point-prompted segmentation with "
                "native video-memory propagation across slices/frames "
                "(Ravi et al., Meta AI 2024).")

    @property
    def reference(self) -> str:
        return ('Ravi et al. '
                '<a href="https://ai.meta.com/sam2/">'
                'SAM 2: Segment Anything in Images and Videos.</a> '
                'Meta AI, 2024.')

    @property
    def category(self) -> str:
        return "Deep Learning"

    def get_widget(self, data_context=None, parent=None):
        return Sam2Logic(data_context, parent)
