"""
nnInteractive plugin for MELAGE.

Wraps nnInteractiveInferenceSession to provide 3D-aware interactive
segmentation directly on the existing GL canvas.

Workflow
--------
1. Draw positive/negative clicks on any slice of the live viewer.
2. nnInteractive propagates the segmentation through the full 3D volume
   after each click (no manual propagation needed).
3. Press "New Label →" to start segmenting the next structure with a
   fresh session; the previous label remains in npSeg.

Multi-label is handled through MELAGE's existing colorInd system:
each label occupies a distinct integer in npSeg and is rendered with
its own colour.
"""

import os
import sys
import numpy as np
import torch

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor

from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .nninteractive_schema import get_schema

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    NNI_AVAILABLE = True
except ImportError:
    NNI_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HF_REPO      = "nnInteractive/nnInteractive"
_SENTINEL_FILE = "plans.json"   # file that proves the model is fully downloaded

# QColor per label index (0-based) for the overlay dots
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

# colorInd values used in npSeg — map 1:1 with label index
_COLORINDS = [1, 2, 3, 4, 5, 6, 7, 8]

_MAX_LABELS = len(_COLORINDS)


# ---------------------------------------------------------------------------
# Weights location
# ---------------------------------------------------------------------------

def _weights_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", home), "melage", "nninteractive")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "melage", "nninteractive")
    xdg = os.environ.get("XDG_DATA_HOME", os.path.join(home, ".local", "share"))
    return os.path.join(xdg, "melage", "nninteractive")


def _model_folder() -> str | None:
    """
    Find the nnU-Net model folder inside the download directory.

    snapshot_download places files either directly in local_dir or inside a
    named subdirectory (e.g. nnInteractive_v1.0/).  Search up to two levels
    deep for the folder that contains plans.json — that is the folder to pass
    to initialize_from_trained_model_folder.
    """
    base = _weights_dir()
    if not os.path.isdir(base):
        return None
    sentinel = _SENTINEL_FILE

    # Level 0 — the download root itself
    if os.path.isfile(os.path.join(base, sentinel)):
        return base

    # Level 1 — direct children
    try:
        for name in sorted(os.listdir(base)):
            child = os.path.join(base, name)
            if not os.path.isdir(child):
                continue
            if os.path.isfile(os.path.join(child, sentinel)):
                return child
            # Level 2 — grandchildren
            try:
                for sub in sorted(os.listdir(child)):
                    grandchild = os.path.join(child, sub)
                    if os.path.isdir(grandchild) and \
                            os.path.isfile(os.path.join(grandchild, sentinel)):
                        return grandchild
            except OSError:
                pass
    except OSError:
        pass
    return None


def _weights_ready() -> bool:
    return _model_folder() is not None


# ---------------------------------------------------------------------------
# Session cache — one session per (device, model_folder) pair
# ---------------------------------------------------------------------------

_SESSION_CACHE: dict = {}


def _get_session(device: str) -> "nnInteractiveInferenceSession":
    folder = _model_folder()
    if folder is None:
        raise RuntimeError(
            "nnInteractive model weights not found.\n"
            "Click 'Download' in the plugin to fetch them."
        )
    key = (device, folder)
    if key not in _SESSION_CACHE:
        # Free any reserved-but-unallocated CUDA blocks before loading the
        # ~3 GB model so MedSAM (or other plugins) can still share the GPU.
        if "cuda" in device:
            torch.cuda.empty_cache()
        print(f"[nnInteractive] loading model from {folder}")
        session = nnInteractiveInferenceSession(
            device=torch.device(device),
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=min(os.cpu_count() or 4, 8),
            do_autozoom=True,
        )
        session.initialize_from_trained_model_folder(folder)
        _SESSION_CACHE[key] = session
    return _SESSION_CACHE[key]


# ---------------------------------------------------------------------------
# Background download thread (snapshot_download is blocking)
# ---------------------------------------------------------------------------

class _DownloadThread(QtCore.QThread):
    progress  = pyqtSignal(int)
    succeeded = pyqtSignal(str)
    failed    = pyqtSignal(str)

    def run(self):
        if not HF_AVAILABLE:
            self.failed.emit(
                "huggingface_hub is not installed.\n"
                "pip install huggingface_hub"
            )
            return
        try:
            self.progress.emit(5)
            # Download every file in the repo; local_dir_use_symlinks=False
            # ensures real files land in _weights_dir() rather than just
            # symlinks that may break when the HF cache is cleared.
            path = snapshot_download(
                repo_id=_HF_REPO,
                local_dir=_weights_dir(),
                local_dir_use_symlinks=False,
            )
            self.progress.emit(90)

            # Verify the model folder is actually present
            folder = _model_folder()
            if folder is None:
                self.failed.emit(
                    f"Download completed to:\n{path}\n\n"
                    f"But '{_SENTINEL_FILE}' was not found inside it.\n"
                    "The HuggingFace repo structure may have changed.\n"
                    f"Please check {path} and locate plans.json manually."
                )
                return

            self.progress.emit(100)
            self.succeeded.emit(folder)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Transparent overlay on the GL canvas
# ---------------------------------------------------------------------------

class _NNIOverlay(QtWidgets.QWidget):
    """
    Transparent child of a GLWidget.
    Records positive/negative point clicks and draws them on the canvas.
    Uses to_real_world / fromRealWorld so zoom/pan/rotation are handled
    automatically by the GL widget.
    """

    interaction_added = pyqtSignal(float, float, bool)   # col, row, is_positive

    def __init__(self, gl_widget):
        super().__init__(gl_widget)
        self.gl = gl_widget
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.resize(gl_widget.size())
        gl_widget.installEventFilter(self)

        self.mode        = "point_pos"   # "point_pos" | "point_neg"
        self.label_color = _LABEL_QCOLORS[0]

        # [(col, row, sliceNum, window_name, is_positive), …]
        self.interactions: list = []

    def eventFilter(self, obj, event):
        if obj is self.gl and event.type() == QtCore.QEvent.Resize:
            self.resize(self.gl.size())
        return False

    def _to_image(self, sx: int, sy: int):
        return self.gl.to_real_world(sx, sy)

    def _to_screen(self, col: float, row: float):
        return self.gl.fromRealWorld(col, row)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        col, row = self._to_image(event.x(), event.y())
        is_pos   = (self.mode == "point_pos")
        print(f"[NNIOverlay] click  col={col:.1f}  row={row:.1f}  "
              f"slice={self.gl.sliceNum}  win={self.gl.currentWidnowName}  "
              f"positive={is_pos}")
        self.interactions.append((
            col, row,
            self.gl.sliceNum,
            self.gl.currentWidnowName,
            is_pos,
        ))
        self.interaction_added.emit(col, row, is_pos)
        self.update()

    def wheelEvent(self, event):
        self.gl.wheelEvent(event)   # forward scroll → zoom

    def clear_interactions(self):
        self.interactions = []
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cur_sn  = self.gl.sliceNum
        cur_win = self.gl.currentWidnowName

        for col, row, sn, win, is_pos in self.interactions:
            if sn != cur_sn or win != cur_win:
                continue
            sx, sy = self._to_screen(col, row)
            color = (self.label_color
                     if is_pos else QColor(220, 60, 60))
            outer = QColor("white") if is_pos else QColor(40, 40, 40)
            p.setPen(QPen(outer, 2))
            p.setBrush(color)
            p.drawEllipse(QtCore.QPoint(int(sx), int(sy)), 7, 7)
            # + / × glyph
            p.setPen(QPen(QColor("white"), 2))
            s = 4
            cx, cy = int(sx), int(sy)
            if is_pos:
                p.drawLine(cx - s, cy, cx + s, cy)
                p.drawLine(cx, cy - s, cx, cy + s)
            else:
                p.drawLine(cx - s, cy - s, cx + s, cy + s)
                p.drawLine(cx + s, cy - s, cx - s, cy + s)

        hint = "click ✚ include" if self.mode == "point_pos" else "click ✖ exclude"
        p.setPen(QColor("#ccc"))
        p.drawText(6, 16, hint)
        p.end()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Plugin dialog
# ---------------------------------------------------------------------------

class NNILogic(DynamicDialog):
    completed = pyqtSignal(object)   # kept for mainwindow compat

    def __init__(self, data_context, parent=None):
        super().__init__(parent)
        self.data_context  = data_context or {}
        self._main_window  = parent
        self._download_thread = None

        # Session state
        self._session          = None
        self._image_data       = None   # (nX, nY, nZ) raw float, used for set_image
        self._target_buffer    = None   # bool (nX, nY, nZ), updated in-place by session
        self._image_hash       = None   # tracks if image has changed
        self._label_idx        = 0      # 0-based index into _COLORINDS
        self._device           = "cpu"
        self._overlays: list        = []   # one overlay per visible GL widget
        self._interaction_log: list = []   # [(col,row,sn,win,is_pos)] — for Segment replay
        self._nni_active: bool      = False  # overlays inactive until user enables

        self.create_main_ui(schema=get_schema(), default_items=False)
        self.setAttribute(Qt.WA_DeleteOnClose)

        for widget_id, widget_obj in self.widgets.items():
            setattr(self, widget_id, widget_obj)

        if self.data_context:
            self.combo_view.clear()
            self.combo_view.addItems(list(self.data_context.keys()))
            self._select_loaded_view()

        self.combo_view.currentIndexChanged.connect(self._on_view_changed)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_toggle_active.clicked.connect(self._on_toggle_active)
        self.btn_new_label.clicked.connect(self._on_new_label)
        self.btn_clear_label.clicked.connect(self._on_clear_label)
        self.btn_download.clicked.connect(self.on_btn_download_clicked)
        self.btn_apply.clicked.connect(self.on_btn_apply_clicked)
        self.check_cuda.stateChanged.connect(self._refresh_device_label)

        self._refresh_weights_status()
        self._refresh_device_label()
        self._refresh_label_status()
        self._refresh_active_ui()
        self.btn_apply.setEnabled(False)

        # Connect a lightweight side-effect to MELAGE's actionUndo so that
        # when the standard Undo() restores the display we also reset the NNI
        # session (preventing stale interactions from surfacing on the next click).
        # The standard Undo() / Redo() handle all display concerns; we only
        # clean up internal NNI state.
        if self._main_window is not None:
            try:
                self._main_window.actionUndo.triggered.connect(
                    self._nni_undo_side_effect)
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # GL widget lookup — returns ALL active widgets for the selected view
    # ------------------------------------------------------------------

    def _get_gl_widgets(self, view_name: str) -> list:
        """Return every active GLWidget for the view.

        Covers both the Tab-0 three-panel widgets (1-3 / 4-6) and the
        Tab-1/Tab-2 fullscreen widgets (11 / 12) so overlays are attached
        regardless of which tab is currently visible.
        """
        if self._main_window is None:
            return []
        candidates = [1, 2, 3, 11] if view_name == "view 1" else [4, 5, 6, 12]
        result = []
        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is not None and getattr(w, "imSlice", None) is not None:
                result.append(w)
        return result

    # ------------------------------------------------------------------
    # Overlay — one per axis view so the user can click on any orientation
    # ------------------------------------------------------------------

    def _reattach_overlay(self):
        for ov in self._overlays:
            ov.close()
        self._overlays = []

        if not self._nni_active:
            return   # overlays only exist when NNI is active

        mode = "point_pos" if self.combo_mode.currentIndex() == 0 else "point_neg"
        for gl in self._get_gl_widgets(self.combo_view.currentText()):
            ov = _NNIOverlay(gl)
            ov.label_color = _LABEL_QCOLORS[self._label_idx % _MAX_LABELS]
            ov.mode = mode
            ov.interaction_added.connect(
                lambda col, row, is_pos, _gl=gl: self._on_interaction(col, row, is_pos, _gl)
            )
            ov.show()
            self._overlays.append(ov)

    # ------------------------------------------------------------------
    # Activate / deactivate
    # ------------------------------------------------------------------

    def _on_toggle_active(self):
        self._nni_active = not self._nni_active
        self._refresh_active_ui()
        if self._nni_active:
            self._reattach_overlay()
        else:
            for ov in self._overlays:
                ov.close()
            self._overlays = []

    def _refresh_active_ui(self):
        """Update button label and control enabled-state to reflect active/inactive."""
        if self._nni_active:
            self.btn_toggle_active.setText("■  Deactivate")
            self.btn_toggle_active.setStyleSheet(
                "QPushButton { background-color: #8B0000; color: white; "
                "font-weight: bold; padding: 6px; border-radius: 4px; }")
        else:
            self.btn_toggle_active.setText("▶  Activate")
            self.btn_toggle_active.setStyleSheet(
                "QPushButton { background-color: #1a5c1a; color: white; "
                "font-weight: bold; padding: 6px; border-radius: 4px; }")
        # Enable/disable interaction controls
        for wid in (self.combo_mode, self.check_live, self.btn_new_label,
                    self.btn_clear_label, self.btn_apply):
            wid.setEnabled(self._nni_active)

    def _on_view_changed(self):
        self._image_hash = None   # force set_image on next interaction
        self._reattach_overlay()

    def _on_mode_changed(self):
        mode = "point_pos" if self.combo_mode.currentIndex() == 0 else "point_neg"
        for ov in self._overlays:
            ov.mode = mode

    # ------------------------------------------------------------------
    # Weights / device status
    # ------------------------------------------------------------------

    def _refresh_weights_status(self):
        folder = _model_folder()
        if folder is not None:
            short = os.path.basename(folder) or os.path.basename(os.path.dirname(folder))
            self.lbl_weights_status.setText(f"✔ {short}")
            self.lbl_weights_status.setToolTip(folder)
            self.btn_download.setEnabled(False)
        else:
            self.lbl_weights_status.setText("✘ Not found")
            self.lbl_weights_status.setToolTip(
                f"Will be saved to:\n{_weights_dir()}"
            )
            self.btn_download.setEnabled(True)

    def _refresh_segment_button(self):
        """Enable Segment only when at least one point has been placed."""
        self.btn_apply.setEnabled(bool(self._interaction_log) and self._nni_active)

    def _refresh_device_label(self):
        if not self.check_cuda.isChecked():
            self.lbl_device.setText("(CPU)")
            return
        if torch.cuda.is_available():
            self.lbl_device.setText(f"✔ {torch.cuda.get_device_name(0)}")
            self.lbl_device.setStyleSheet("color: green;")
        else:
            self.lbl_device.setText("✘ CUDA unavailable")
            self.lbl_device.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Label management
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
        # Commit current label, reset session for the next object
        self._reset_session_for_new_label()
        self._label_idx += 1
        self._refresh_label_status()
        color = _LABEL_QCOLORS[self._label_idx % _MAX_LABELS]
        for ov in self._overlays:
            ov.label_color = color
            ov.clear_interactions()
        self._interaction_log.clear()
        self._refresh_segment_button()

    def _on_clear_label(self):
        """Erase the current label from npSeg and reset session interactions."""
        colorInd = _COLORINDS[self._label_idx % _MAX_LABELS]
        view_name = self.combo_view.currentText()
        reader = self.data_context.get(view_name)
        if reader is not None and hasattr(reader, "npSeg"):
            reader.npSeg[reader.npSeg == colorInd] = 0
            # segChanged with an empty array is silently ignored by updateSegmentation,
            # so force a direct widget refresh instead (same fix as empty-buffer undo).
            self._force_refresh_widgets(reader, view_name)
        if self._session is not None:
            try:
                self._session.reset_interactions()
                self._target_buffer = np.zeros_like(self._image_data, dtype=bool)
                self._session.set_target_buffer(self._target_buffer)
            except Exception:
                pass
        for ov in self._overlays:
            ov.clear_interactions()
        self._interaction_log.clear()
        self._refresh_segment_button()

    def _nni_undo_side_effect(self):
        """
        Connected to actionUndo.triggered alongside the standard Undo().
        Standard Undo() handles the display (restores npSeg from the undo stack).
        We only reset the NNI session state so that the next click produces a
        fresh inference rather than adding on top of stale interactions.
        """
        if self._session is not None:
            try:
                self._session.reset_interactions()
                if self._target_buffer is not None:
                    self._target_buffer[:] = False
            except Exception:
                pass
        # Also remove the most-recent recorded interaction and its overlay dot
        if self._interaction_log:
            removed = self._interaction_log.pop()
            col_r, row_r, sn_r, win_r, _ = removed
            for ov in self._overlays:
                for i, item in enumerate(ov.interactions):
                    if (abs(item[0] - col_r) < 0.5 and
                            abs(item[1] - row_r) < 0.5 and
                            item[2] == sn_r and item[3] == win_r):
                        ov.interactions.pop(i)
                        ov.update()
                        break
        self._refresh_segment_button()

    def _reset_session_for_new_label(self):
        """Keep the session loaded; reset interactions + buffer for the new object."""
        if self._session is not None and self._target_buffer is not None:
            try:
                self._session.reset_interactions()
                self._target_buffer = np.zeros_like(self._image_data, dtype=bool)
                self._session.set_target_buffer(self._target_buffer)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def on_btn_download_clicked(self):
        if not NNI_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "nnInteractive is not installed.\n\n"
                "pip install nnInteractive",
            )
            return
        self.btn_download.setEnabled(False)
        self.progress_download.setValue(10)
        self.lbl_weights_status.setText("Downloading from HuggingFace…")
        self._download_thread = _DownloadThread(parent=self)
        self._download_thread.progress.connect(self.progress_download.setValue)
        self._download_thread.succeeded.connect(self._on_download_succeeded)
        self._download_thread.failed.connect(self._on_download_failed)
        self._download_thread.start()

    def _on_download_succeeded(self, path):
        self.progress_download.setValue(100)
        self._refresh_weights_status()
        QMessageBox.information(self, "Done", f"nnInteractive weights ready at:\n{path}")

    def _on_download_failed(self, error):
        self._refresh_weights_status()
        self.progress_download.setValue(0)
        QMessageBox.critical(
            self, "Download failed",
            f"{error}\n\nManual download:\n"
            "pip install huggingface_hub && "
            f"python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{_HF_REPO}', local_dir='{_weights_dir()}')\"",
        )

    # ------------------------------------------------------------------
    # Session / image setup
    # ------------------------------------------------------------------

    def _ensure_session(self, device: str) -> bool:
        """Initialise the session and load the image if not already done."""
        view_name = self.combo_view.currentText()
        reader    = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "im"):
            # `update_data_context` is only pushed to us by the main window
            # when an image/video is (re)loaded; if this widget instance
            # missed that notification (e.g. it was created after the load,
            # or the push was dropped) self.data_context can be stale/empty
            # even though an image IS currently loaded. Pull the live state
            # directly from the main window before giving up.
            self._refresh_data_context()
            view_name = self.combo_view.currentText()
            reader    = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "im"):
            QMessageBox.warning(self, "No image", "No image loaded.")
            return False

        # (Re-)initialise session if device changed
        if self._session is None or self._device != device:
            self._device  = device
            self._session = None   # release old session
            self._session = _get_session(device)
            self._image_hash = None  # force set_image

        # Set image when it changes (hash by id).
        # IMPORTANT: use reader.npImage (not reader.im.get_fdata()) so that
        # the image and the target_buffer are in the same coordinate space as
        # reader.npSeg — they may differ if the NIfTI was reoriented on load.
        img_id = id(reader.npImage)
        if img_id != self._image_hash:
            image_data = reader.npImage.astype(np.float32)
            if getattr(reader, "isChunkedVideo", False):
                # A video frame is a single 2D (RGB) image (H, W[, 3]) — there
                # is no real depth axis. Collapse to grayscale and add a
                # length-1 "Z" axis so it matches the (H, W) shape of npSeg /
                # the click coordinates' Z=0 (see _run_interaction).
                if image_data.ndim == 3:
                    image_data = image_data.mean(axis=-1)
                image_data = image_data[:, :, np.newaxis]
            elif image_data.ndim == 4:
                image_data = image_data[..., 0]

            self._image_data    = image_data
            # target_buffer MUST match npSeg shape so boolean indexing works
            self._target_buffer = np.zeros(image_data.shape[:3], dtype=bool)

            # Spacing from the NIfTI header (best-effort)
            try:
                spacing = [float(reader.im.header.get_zooms()[i]) for i in range(3)]
            except Exception:
                spacing = [1.0, 1.0, 1.0]

            self._session.set_image(
                image_data[np.newaxis, ...],   # (1, X, Y, Z)
                image_properties={"spacing": spacing},
            )
            self._session.set_target_buffer(self._target_buffer)
            self._image_hash = img_id
            print(f"[nnInteractive] set_image  shape={image_data.shape}  "
                  f"spacing={spacing}  "
                  f"range=[{image_data.min():.1f}, {image_data.max():.1f}]")

        return True

    # ------------------------------------------------------------------
    # Interaction handling
    # ------------------------------------------------------------------

    def _on_interaction(self, col: float, row: float, is_positive: bool, gl):
        """Called immediately when the user clicks on any axis canvas."""
        self._refresh_segment_button()

        # Always record the click so undo side-effect can track it
        self._interaction_log.append(
            (col, row, gl.sliceNum, gl.currentWidnowName, is_positive)
        )

        if not self.check_live.isChecked():
            return   # inference deferred to the Segment button
        if not NNI_AVAILABLE:
            QMessageBox.critical(self, "Missing dependency", "pip install nnInteractive")
            return
        if not _weights_ready():
            QMessageBox.warning(
                self, "Weights not downloaded",
                "Click 'Download' to get the model weights first,\n"
                "or turn off 'Live update' and use the Segment button manually.",
            )
            return
        self._run_interaction(col, row, is_positive, gl)

    def _run_interaction(self, col: float, row: float, is_positive: bool, gl):
        try:
            from melage.utils.utils import permute_axis

            device = ("cuda" if self.check_cuda.isChecked()
                      and torch.cuda.is_available() else "cpu")

            # Return fragmented reserved blocks before inference so concurrent
            # models (e.g. MedSAM) can still allocate on the same GPU.
            if device == "cuda":
                torch.cuda.empty_cache()

            # Session + image setup (raises on error → caught below)
            if not self._ensure_session(device):
                return

            view_name = self.combo_view.currentText()
            reader    = self.data_context.get(view_name)

            # Convert 2D canvas (col, row, sliceNum) → 3D voxel (X, Y, Z)
            pt_2d    = np.array([[int(col), int(row), gl.sliceNum]])
            pt_3d, _ = permute_axis(pt_2d, None, gl.currentWidnowName)
            coords   = pt_3d[0].tolist()
            if getattr(reader, "isChunkedVideo", False):
                # Video frames are a single (H, W, 1) "volume" (see
                # _ensure_session) — Z is always 0.
                coords[2] = 0
            print(f"[nnInteractive] point {'+' if is_positive else '-'}  "
                  f"coords={coords}  device={device}")

            self.btn_apply.setEnabled(False)
            self.progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()

            self._session.add_point_interaction(
                coordinates=tuple(int(c) for c in coords),
                include_interaction=is_positive,
            )

            buf_sum = int(np.asarray(self._target_buffer).sum())
            print(f"[nnInteractive] target_buffer sum after prediction: {buf_sum}")
            self.progress_bar.setValue(80)

            self._emit_segmentation(gl)
            self.progress_bar.setValue(100)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "nnInteractive error", str(exc))
        finally:
            self.btn_apply.setEnabled(True)

    def _push_nni_undo(self, reader_name: str, buf: np.ndarray, colorInd: int, reader) -> None:
        """
        Inject an undo snapshot into MELAGE's undo stack BEFORE writing to npSeg.

        Stores only the DELTA (pixels newly added by this NNI call), so that each
        Undo step removes only the contribution of the last NNI click rather than
        wiping the entire label.  prevCol == col makes MELAGE's logic:
          • Undo → both equal → restore colour = 0 → erases the delta pixels ✓
          • Redo → both equal → restore colour = colorInd → re-draws them ✓
        """
        mw = self._main_window
        if mw is None:
            return

        # Delta: only pixels that NNI is newly adding (already-colorInd pixels are
        # cleared then rewritten to the same value, so they are not a net change).
        prev_mask  = reader.npSeg == colorInd
        added_mask = buf & ~prev_mask
        if not np.any(added_mask):
            return

        new_coords  = np.column_stack(np.where(added_mask)).astype(int)
        is_view1    = (reader_name == "readView1")
        idx_list    = [1, 2, 3, 11] if is_view1 else [4, 5, 6, 12]
        widgets     = [w for i in idx_list
                       if (w := getattr(mw, f"openGLWidget_{i}", None)) is not None]
        if not widgets:
            return

        mw._lastChangedWidgest  = widgets
        mw._lastReader          = reader_name
        mw._lastReaderSegInd.append([new_coords, None, [], 0])
        mw._lastReaderSegCol.append(colorInd)
        mw._lastReaderSegPrevCol.append(colorInd)
        mw._undoTimes = 0

        if len(mw._lastReaderSegInd) > mw._lastMax:
            mw._lastReaderSegCol     = mw._lastReaderSegCol[1:]
            mw._lastReaderSegInd     = mw._lastReaderSegInd[1:]
            mw._lastReaderSegPrevCol = mw._lastReaderSegPrevCol[1:]

    def _emit_segmentation(self, gl):
        """
        Write the current target_buffer into npSeg and refresh all views.

        The undo snapshot is pushed BEFORE any write so that MELAGE's Undo()
        sees the correct before-state.  We bypass segChanged / update_last
        because those helpers only record a delta and would find nothing to save
        once the voxels are already written.
        """
        view_name   = self.combo_view.currentText()
        reader      = self.data_context.get(view_name)
        if reader is None:
            return

        colorInd    = _COLORINDS[self._label_idx % _MAX_LABELS]
        buf         = np.asarray(self._target_buffer, dtype=bool)
        reader_name = "readView1" if view_name == "view 1" else "readView2"
        is_video    = getattr(reader, "isChunkedVideo", False)
        if is_video and buf.ndim == 3:
            # (H, W, 1) -> (H, W), matching npSeg's 2D shape (see _ensure_session)
            buf = buf[:, :, 0]

        self._ensure_label_visible(colorInd)

        # Push undo state BEFORE any write so Undo() has the correct snapshot.
        self._push_nni_undo(reader_name, buf, colorInd, reader)

        # Apply: clear old prediction for this label, write the new one.
        reader.npSeg[reader.npSeg == colorInd] = 0
        if not np.any(buf):
            self._force_refresh_widgets(reader, view_name)
            self._commit_video_frame(reader, is_video)
            return
        reader.npSeg[buf] = colorInd

        # Refresh all views directly.
        self._force_refresh_widgets(reader, view_name)
        self._commit_video_frame(reader, is_video)

    def _commit_video_frame(self, reader, is_video: bool):
        """
        Persist the (in-place edited) current-frame npSeg into the video's
        per-frame label proxy (`reader.seg_ims`) — without this, edits to
        frames whose mask wasn't already cached would be silently lost the
        next time the frame slider moves (see VideoLabelProxy.get_frame).
        """
        if not is_video or not hasattr(reader, "commit_frame_segmentation_changes"):
            return
        reader.commit_frame_segmentation_changes(reader.npSeg, reader.current_frame)

    def _ensure_label_visible(self, colorInd: int):
        """
        Make sure colorInd appears in every relevant GL widget's colorInds list
        so makeObject() renders it immediately — no manual Color Settings toggle needed.

        Also ticks the matching row in the tree_colors panel (signals blocked so
        changeColorPen's side-effects — changing the active drawing color — don't fire).
        """
        if self._main_window is None:
            return
        view_name = self.combo_view.currentText()
        candidates = [1, 2, 3, 11] if view_name == "view 1" else [4, 5, 6, 12]

        # 1. Add colorInd to every relevant GL widget that doesn't already show it
        for i in candidates:
            w = getattr(self._main_window, f"openGLWidget_{i}", None)
            if w is None:
                continue
            if 9876 not in w.colorInds and colorInd not in w.colorInds:
                w.colorInds = list(w.colorInds) + [colorInd]

        # 2. Tick the matching row in tree_colors (purely visual, no side-effects)
        try:
            tree = getattr(self._main_window, "tree_colors", None)
            if tree is None:
                return
            source = tree.model().sourceModel()
            root   = source.invisibleRootItem()
            source.blockSignals(True)   # prevent changeColorPen from firing
            for i in range(root.rowCount()):
                item = root.child(i)
                if item is None:
                    continue
                try:
                    if int(float(item.text())) == colorInd:
                        if item.checkState() != QtCore.Qt.Checked:
                            item.setCheckState(QtCore.Qt.Checked)
                        break
                except (ValueError, TypeError):
                    pass
            source.blockSignals(False)
        except Exception as e:
            print(f"[nnInteractive] _ensure_label_visible tree sync: {e}")

    def _force_refresh_widgets(self, reader, view_name: str):
        """Directly refresh GL widgets — used when segChanged can't help (empty mask)."""
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
                print(f"[nnInteractive] refresh error widget {i}: {e}")

    # ------------------------------------------------------------------
    # Manual Segment button (live update off)
    # ------------------------------------------------------------------

    def on_btn_apply_clicked(self):
        if not NNI_AVAILABLE:
            QMessageBox.critical(self, "Missing dependency",
                                 "pip install nnInteractive")
            return
        if not _weights_ready():
            QMessageBox.critical(self, "Weights missing",
                                 "Please download the weights first.")
            return

        # Collect all interactions from every axis overlay
        # Use the canonical log (always populated, even with live update off)
        all_interactions = list(self._interaction_log)

        if not all_interactions:
            QMessageBox.warning(self, "No interactions",
                                "Click on the image to add positive/negative points first.")
            return

        try:
            from melage.utils.utils import permute_axis

            device = ("cuda" if self.check_cuda.isChecked()
                      and torch.cuda.is_available() else "cpu")
            if device == "cuda":
                torch.cuda.empty_cache()
            if not self._ensure_session(device):
                return

            view_name = self.combo_view.currentText()
            reader    = self.data_context.get(view_name)
            is_video  = getattr(reader, "isChunkedVideo", False)

            self.btn_apply.setEnabled(False)

            # Reset and replay all recorded interactions from all axis overlays
            self._session.reset_interactions()
            if device == "cuda":
                torch.cuda.empty_cache()
            self._target_buffer[:] = False

            total = len(all_interactions)
            for step, (col, row, sn, win, is_pos) in enumerate(all_interactions):
                pt_2d    = np.array([[int(col), int(row), sn]])
                pt_3d, _ = permute_axis(pt_2d, None, win)
                coords   = pt_3d[0].tolist()
                if is_video:
                    coords[2] = 0
                self._session.add_point_interaction(
                    coordinates=tuple(int(c) for c in coords),
                    include_interaction=is_pos,
                )
                self.progress_bar.setValue(int((step + 1) / total * 90))
                QtWidgets.QApplication.processEvents()

            # Use any overlay's gl widget for the emit (all share the same view)
            ref_gl = self._overlays[0].gl if self._overlays else None
            if ref_gl:
                self._emit_segmentation(ref_gl)
            self.progress_bar.setValue(100)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            self.btn_apply.setEnabled(True)

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------

    def update_data_context(self, data_context):
        """
        Called by the main window whenever a new image is loaded.
        Updates the image reference and reattaches overlays to the new GL widgets.
        """
        self.data_context  = data_context or {}
        self._image_hash   = None   # force set_image on next interaction
        self._session      = None   # release cached session (new image → new session)

        # Refresh the view-selector combo without triggering reattach twice
        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        if self.data_context:
            self.combo_view.addItems(list(self.data_context.keys()))
            self._select_loaded_view()
        self.combo_view.blockSignals(False)

        self._reattach_overlay()

    def _refresh_data_context(self):
        """
        Pull the current image/video readers directly from the main window.

        Used as a fallback when ``self.data_context`` is stale or empty
        (e.g. ``update_data_context`` was never pushed to this widget
        instance) — see ``_ensure_session``.
        """
        main_window = getattr(self, "_main_window", None)
        if main_window is None or not hasattr(main_window, "get_current_image_data"):
            return
        self.update_data_context(main_window.get_current_image_data())

    def _select_loaded_view(self):
        """
        Point combo_view at a view that actually has an image loaded.

        `combo_view` always lists "view 1"/"view 2" regardless of which one
        (if either) has data — defaulting to index 0 ("view 1") meant that
        loading data only into "view 2" (e.g. a video opened in the second
        viewer) left the combo on "view 1", and `_ensure_session` then
        reported "No image loaded" even though "view 2" was ready. Prefer
        keeping the current selection if it is still valid; otherwise pick
        the first view with a loaded reader.
        """
        current = self.combo_view.currentText()
        if self.data_context.get(current) is not None:
            return
        for i in range(self.combo_view.count()):
            view_name = self.combo_view.itemText(i)
            if self.data_context.get(view_name) is not None:
                self.combo_view.setCurrentIndex(i)
                return

    def _on_dock_visibility(self, visible: bool):
        """Called by the dock's visibilityChanged signal. Show/hide overlays accordingly."""
        for ov in self._overlays:
            if visible:
                ov.show()
            else:
                ov.hide()

    def closeEvent(self, event):
        if self._main_window is not None:
            try:
                self._main_window.actionUndo.triggered.disconnect(
                    self._nni_undo_side_effect)
            except (AttributeError, RuntimeError):
                pass
        for ov in self._overlays:
            ov.close()
        self._overlays = []
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

class NNInteractivePlugin(MelagePlugin):
    @property
    def name(self) -> str:
        return "nnInteractive"

    @property
    def description(self) -> str:
        return (
            "3D interactive segmentation with click/box prompts. "
            "One click → full 3D label. Multi-label. "
            "(Isensee et al., 2024)"
        )

    @property
    def reference(self) -> str:
        return (
            'Isensee et al. '
            '<a href="https://arxiv.org/abs/2411.19414">'
            'nnInteractive: Redefining Annotation Interfaces for '
            'Medical Image Segmentation.</a> arXiv, 2024.'
        )

    @property
    def category(self) -> str:
        return "Deep Learning"

    def get_widget(self, data_context=None, parent=None):
        return NNILogic(data_context, parent)
