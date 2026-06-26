import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal

from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .medsam_schema import get_schema

try:
    from segment_anything import sam_model_registry
    from skimage import transform as sk_transform
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

_DOWNLOAD_URLS = [
    "https://zenodo.org/records/10689643/files/medsam_vit_b.pth",
]
_WEIGHTS_FILENAME = "medsam_vit_b.pth"

_MODEL_CACHE: dict = {}

_AXIS_MAP = {"Axial (Z)": 2, "Coronal (Y)": 1, "Sagittal (X)": 0}
_MODE_MAP  = ["bbox", "point_pos", "point_neg"]


# ---------------------------------------------------------------------------
# Platform-appropriate user-writable weights directory
# ---------------------------------------------------------------------------

def _weights_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA", home), "melage", "weights")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "melage", "weights")
    xdg = os.environ.get("XDG_DATA_HOME", os.path.join(home, ".local", "share"))
    return os.path.join(xdg, "melage", "weights")


def _weights_path() -> str:
    return os.path.join(_weights_dir(), _WEIGHTS_FILENAME)


# ---------------------------------------------------------------------------
# Interactive slice canvas
# ---------------------------------------------------------------------------

class SliceCanvas(QtWidgets.QWidget):
    """
    Displays one 2D image slice.  The user can:
      - drag to draw a bounding box  (yellow)
      - click to place positive points (green dot)
      - click to place negative points (red dot)

    All coordinates stored/returned are in *image* pixel space.
    """

    prompt_changed = pyqtSignal()

    _CURSOR_CROSS = QtCore.Qt.CrossCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 380)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.setCursor(self._CURSOR_CROSS)
        self.setStyleSheet("background-color: #111;")

        self._pixmap: QtGui.QPixmap | None = None
        self._img_w = 1
        self._img_h = 1

        self.mode: str = "bbox"
        self.bbox: list | None = None
        self.points: list = []

        self._drag_start: tuple | None = None

    # Keep widget square as the dialog is resized
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return width

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(420, 420)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_slice(self, slc: np.ndarray | None) -> None:
        if slc is None:
            self._pixmap = None
            self.update()
            return
        lo, hi = float(slc.min()), float(slc.max())
        gray = np.ascontiguousarray(
            (slc - lo) / max(hi - lo, 1e-8) * 255, dtype=np.uint8
        )
        h, w = gray.shape
        self._img_h, self._img_w = h, w
        # Stack to RGB888 — more portable across PyQt5 builds than Grayscale8
        rgb = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))
        qi = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qi)
        self.update()

    def clear_prompts(self) -> None:
        self.bbox = None
        self.points = []
        self.prompt_changed.emit()
        self.update()

    def has_prompt(self) -> bool:
        return self.bbox is not None or bool(self.points)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _display_rect(self) -> QtCore.QRect:
        """Image display rect inside the widget (letterboxed, centred)."""
        if self._pixmap is None:
            return QtCore.QRect(0, 0, self.width(), self.height())
        scale = min(self.width() / self._img_w, self.height() / self._img_h)
        dw = int(self._img_w * scale)
        dh = int(self._img_h * scale)
        ox = (self.width()  - dw) // 2
        oy = (self.height() - dh) // 2
        return QtCore.QRect(ox, oy, dw, dh)

    def _to_image(self, sx: int, sy: int) -> tuple[int, int]:
        r = self._display_rect()
        if r.width() == 0 or r.height() == 0:
            return 0, 0
        ix = (sx - r.x()) / r.width()  * self._img_w
        iy = (sy - r.y()) / r.height() * self._img_h
        return (max(0, min(self._img_w - 1, int(ix))),
                max(0, min(self._img_h - 1, int(iy))))

    def _to_screen(self, ix: int, iy: int, r: QtCore.QRect) -> tuple[int, int]:
        sx = r.x() + int(ix / self._img_w * r.width())
        sy = r.y() + int(iy / self._img_h * r.height())
        return sx, sy

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap is None:
            return
        ix, iy = self._to_image(event.x(), event.y())
        if self.mode == "bbox":
            self._drag_start = (ix, iy)
            self.bbox = [ix, iy, ix, iy]
        elif self.mode == "point_pos":
            self.points.append((ix, iy, True))
            self.prompt_changed.emit()
        else:
            self.points.append((ix, iy, False))
            self.prompt_changed.emit()
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode == "bbox" and self._drag_start is not None:
            ix, iy = self._to_image(event.x(), event.y())
            x0, y0 = self._drag_start
            self.bbox = [min(x0, ix), min(y0, iy), max(x0, ix), max(y0, iy)]
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode == "bbox" and self._drag_start is not None:
            self._drag_start = None
            if self.bbox and (self.bbox[2] > self.bbox[0] or self.bbox[3] > self.bbox[1]):
                self.prompt_changed.emit()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#111"))

        if self._pixmap is None:
            p.setPen(QtGui.QColor("#888"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       "Select a view and axis to load slice")
            return

        r = self._display_rect()
        p.drawPixmap(r, self._pixmap)

        # Bounding box — yellow
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            sx1, sy1 = self._to_screen(x1, y1, r)
            sx2, sy2 = self._to_screen(x2, y2, r)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 0), 2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

            # Coordinate label
            p.setPen(QtGui.QColor(255, 220, 0))
            p.drawText(sx1 + 3, sy1 - 4,
                       f"({x1},{y1}) → ({x2},{y2})")

        # Points — green (positive) / red (negative)
        for px, py, is_pos in self.points:
            sx, sy = self._to_screen(px, py, r)
            color = QtGui.QColor(50, 220, 50) if is_pos else QtGui.QColor(220, 50, 50)
            p.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
            p.setBrush(color)
            p.drawEllipse(QtCore.QPoint(sx, sy), 6, 6)

        # Mode hint in corner
        hints = {
            "bbox":      "drag to draw box",
            "point_pos": "click to add  ✚  point",
            "point_neg": "click to add  ✖  point",
        }
        p.setPen(QtGui.QColor("#aaa"))
        p.drawText(r.x() + 4, r.y() + 14, hints.get(self.mode, ""))


# ---------------------------------------------------------------------------
# Background download thread
# ---------------------------------------------------------------------------

class _DownloadThread(QtCore.QThread):
    progress  = pyqtSignal(int)
    succeeded = pyqtSignal(str)
    failed    = pyqtSignal(str)

    def __init__(self, dest: str, parent=None):
        super().__init__(parent)
        self.dest = dest
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        import requests
        os.makedirs(os.path.dirname(self.dest), exist_ok=True)
        tmp = self.dest + ".tmp"
        last_error = "No download URL available."
        for url in _DOWNLOAD_URLS:
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

def _get_model(checkpoint: str, device: str):
    key = (checkpoint, device)
    if key not in _MODEL_CACHE:
        model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        model.to(device)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _extract_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 2:
        return volume[:, :, idx]
    if axis == 1:
        return volume[:, idx, :]
    return volume[idx, :, :]


def _write_slice(label: np.ndarray, axis: int, idx: int, mask: np.ndarray) -> None:
    if axis == 2:
        label[:, :, idx] = mask
    elif axis == 1:
        label[:, idx, :] = mask
    else:
        label[idx, :, :] = mask


def _extract_for_display(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """
    Extract a slice and orient it for anatomical display.

    Raw NIfTI voxel order is (X, Y, Z).  A plain array slice gives shape
    (X, Y) which, when interpreted as (rows, cols), maps X → top-to-bottom
    and Y → left-to-right — a 90° rotation from the expected view.

    Correct standard display (axial example, RAS space):
      - transpose: (X, Y) → (Y, X) so rows = Y, cols = X
      - flipud:    Y = 0 (posterior) ends up at bottom, Y = max (anterior) at top
    The same transform is correct for coronal and sagittal axes too.
    """
    slc = _extract_slice(volume, axis, idx)
    return np.ascontiguousarray(np.flipud(slc.T))


def _display_mask_to_volume(mask: np.ndarray) -> np.ndarray:
    """Inverse of _extract_for_display: map a display-space mask back to volume space."""
    return np.ascontiguousarray(np.flipud(mask).T)


def _preprocess(slc: np.ndarray) -> np.ndarray:
    lo, hi = float(slc.min()), float(slc.max())
    norm = (slc - lo) / max(hi - lo, 1e-8)
    u8 = (norm * 255).astype(np.uint8)
    rgb = np.stack([u8] * 3, axis=-1)
    return sk_transform.resize(
        rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)


def _infer(model, img_1024: np.ndarray,
           bbox: list | None, points: list,
           orig_h: int, orig_w: int, device: str) -> np.ndarray:
    """
    bbox   : [x_min, y_min, x_max, y_max] in image-pixel coords, or None
    points : [(x, y, is_positive), …] in image-pixel coords
    """
    tensor = (
        torch.from_numpy(img_1024.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        embedding = model.image_encoder(tensor)

    # Scale bounding box to 1024 space
    box_torch = None
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        box_1024 = np.array([
            x_min / orig_w * 1024, y_min / orig_h * 1024,
            x_max / orig_w * 1024, y_max / orig_h * 1024,
        ], dtype=np.float32)
        box_torch = (
            torch.as_tensor(box_1024, device=device).unsqueeze(0).unsqueeze(0)
        )

    # Scale point prompts to 1024 space
    pts_torch = lbl_torch = None
    if points:
        coords = np.array(
            [[p[0] / orig_w * 1024, p[1] / orig_h * 1024] for p in points],
            dtype=np.float32,
        )
        labels = np.array([int(p[2]) for p in points], dtype=np.int64)
        pts_torch = torch.as_tensor(coords, device=device).unsqueeze(0)
        lbl_torch = torch.as_tensor(labels, device=device).unsqueeze(0)

    with torch.no_grad():
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(pts_torch, lbl_torch) if pts_torch is not None else None,
            boxes=box_torch,
            masks=None,
        )
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

    pred = F.interpolate(
        torch.sigmoid(logits), size=(orig_h, orig_w),
        mode="bilinear", align_corners=False,
    )
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plugin dialog
# ---------------------------------------------------------------------------

class MedSamLogic(DynamicDialog):
    completed = pyqtSignal(object)

    def __init__(self, data_context, parent=None):
        super().__init__(parent)
        self.data_context = data_context or {}
        self._download_thread = None

        self.create_main_ui(schema=get_schema(), default_items=False)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setSizeGripEnabled(True)

        for widget_id, widget_obj in self.widgets.items():
            setattr(self, widget_id, widget_obj)

        # Insert the interactive canvas after combo_view (index 0 in layout)
        self.slice_canvas = SliceCanvas(self)
        self.main_layout.insertWidget(1, self.slice_canvas)

        if self.data_context:
            self.combo_view.clear()
            self.combo_view.addItems(list(self.data_context.keys()))

        # Signal wiring
        self.combo_view.currentIndexChanged.connect(self._refresh_canvas)
        self.combo_axis.currentIndexChanged.connect(self._on_axis_changed)
        self.spin_slice.valueChanged.connect(self._on_slice_changed)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_download.clicked.connect(self.on_btn_download_clicked)
        self.btn_apply.clicked.connect(self.on_btn_apply_clicked)

        self._refresh_weights_status()
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Canvas management
    # ------------------------------------------------------------------

    def _refresh_canvas(self):
        view_name = self.combo_view.currentText()
        reader = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "im"):
            self.slice_canvas.set_slice(None)
            return

        volume = reader.im.get_fdata()
        if volume.ndim == 4:
            volume = volume[..., 0]

        axis = _AXIS_MAP[self.combo_axis.currentText()]
        n = volume.shape[axis]

        # Update range; default to the middle slice so the first view is not empty
        self.spin_slice.blockSignals(True)
        prev_max = int(self.spin_slice.maximum())
        self.spin_slice.setMaximum(n - 1)
        if prev_max != n - 1 or self.spin_slice.value() == 0:
            self.spin_slice.setValue(n // 2)
        self.spin_slice.blockSignals(False)

        idx = int(self.spin_slice.value())
        self.slice_canvas.set_slice(_extract_for_display(volume, axis, idx))

    def _on_axis_changed(self):
        self.slice_canvas.clear_prompts()
        self._refresh_canvas()

    def _on_slice_changed(self):
        # Only refresh slice image, keep existing prompts so user can compare
        view_name = self.combo_view.currentText()
        reader = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "im"):
            return
        volume = reader.im.get_fdata()
        if volume.ndim == 4:
            volume = volume[..., 0]
        axis = _AXIS_MAP[self.combo_axis.currentText()]
        idx = int(self.spin_slice.value())
        self.slice_canvas.set_slice(_extract_for_display(volume, axis, idx))

    def _on_mode_changed(self):
        self.slice_canvas.mode = _MODE_MAP[self.combo_mode.currentIndex()]

    def _on_clear(self):
        self.slice_canvas.clear_prompts()

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _refresh_weights_status(self):
        path = _weights_path()
        found = os.path.isfile(path)
        if found:
            size_gb = os.path.getsize(path) / 1e9
            self.lbl_weights_status.setText(f"✔ Found ({size_gb:.1f} GB)  —  {path}")
            self.btn_download.setEnabled(False)
            self.btn_apply.setEnabled(True)
        else:
            self.lbl_weights_status.setText(
                f"✘ Not found — will be saved to:\n{path}"
            )
            self.btn_download.setEnabled(True)
            self.btn_apply.setEnabled(False)

    def on_btn_download_clicked(self):
        if not SAM_AVAILABLE:
            QMessageBox.critical(
                self, "Missing dependency",
                "segment-anything is not installed.\n\n"
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

    def _on_download_succeeded(self, path: str):
        self.progress_download.setValue(100)
        self._refresh_weights_status()
        QMessageBox.information(self, "Download complete",
                                f"MedSAM weights saved to:\n{path}")

    def _on_download_failed(self, error: str):
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
                "segment-anything is not installed.\n\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git",
            )
            return

        if not self.slice_canvas.has_prompt():
            QMessageBox.warning(
                self, "No prompt",
                "Draw a bounding box by dragging, or click to place points first.",
            )
            return

        view_name = self.combo_view.currentText()
        reader = self.data_context.get(view_name)
        if reader is None or not hasattr(reader, "im"):
            QMessageBox.warning(self, "No image", "No image loaded in the selected view.")
            return

        checkpoint = _weights_path()
        if not os.path.isfile(checkpoint):
            QMessageBox.critical(self, "Weights missing",
                                 "Please download the weights first.")
            return

        volume = reader.im.get_fdata()
        if volume.ndim == 4:
            volume = volume[..., 0]

        affine = reader.im.affine
        axis = _AXIS_MAP[self.combo_axis.currentText()]
        n_slices = volume.shape[axis]
        all_slices = self.check_all_slices.isChecked()
        indices = list(range(n_slices)) if all_slices else [int(self.spin_slice.value())]

        if all_slices and n_slices > 1:
            reply = QMessageBox.question(
                self, "Apply to all slices?",
                f"The same prompt will be applied to all {n_slices} slices.\n"
                "This works best when the region of interest stays in a\n"
                "similar position across slices. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        bbox   = self.slice_canvas.bbox
        points = self.slice_canvas.points
        device = (
            "cuda" if torch.cuda.is_available() and self.check_cuda.isChecked()
            else "cpu"
        )

        try:
            self.btn_apply.setEnabled(False)
            self.progress_bar.setValue(5)
            QtWidgets.QApplication.processEvents()

            model = _get_model(checkpoint, device)
            self.progress_bar.setValue(15)

            label = np.zeros(volume.shape[:3], dtype=np.uint8)
            total = len(indices)

            for step, idx in enumerate(indices):
                # Use the same display-oriented slice so bbox/point coords match
                slc = _extract_for_display(volume, axis, idx)
                orig_h, orig_w = slc.shape
                img_1024 = _preprocess(slc)
                mask = _infer(model, img_1024, bbox, points, orig_h, orig_w, device)
                # Invert the display transform before writing back to the label volume
                _write_slice(label, axis, idx, _display_mask_to_volume(mask))

                self.progress_bar.setValue(15 + int((step + 1) / total * 80))
                if step % 5 == 0:
                    QtWidgets.QApplication.processEvents()

            self.completed.emit({
                "image": None,
                "affine": affine,
                "label": label,
                "view": view_name,
            })
            self.progress_bar.setValue(100)
            QMessageBox.information(
                self, "Done",
                f"MedSAM segmented {total} slice(s) on {view_name}.",
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            self.btn_apply.setEnabled(True)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

class MedSamPlugin(MelagePlugin):
    @property
    def name(self) -> str:
        return "MedSAM"

    @property
    def description(self) -> str:
        return (
            "Interactive bounding-box / point-prompted segmentation "
            "for medical images (Ma et al., Nature Communications 2024)."
        )

    @property
    def reference(self) -> str:
        return (
            'Ma et al. '
            '<a href="https://doi.org/10.1038/s41467-024-44824-z">'
            'Segment anything in medical images.</a> '
            'Nature Communications, 2024.'
        )

    @property
    def category(self) -> str:
        return "Deep Learning"

    def get_widget(self, data_context=None, parent=None):
        return MedSamLogic(data_context, parent)
