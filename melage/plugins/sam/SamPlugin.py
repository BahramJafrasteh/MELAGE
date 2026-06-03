import os
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal

from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .SamPlugin_schema import get_schema

try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "tiny":      {"config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                  "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt"},
    "small":     {"config": "configs/sam2.1/sam2.1_hiera_s.yaml",
                  "checkpoint": "checkpoints/sam2.1_hiera_small.pt"},
    "base_plus": {"config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                  "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt"},
    "large":     {"config": "configs/sam2.1/sam2.1_hiera_l.yaml",
                  "checkpoint": "checkpoints/sam2.1_hiera_large.pt"},
}

# ---------------------------------------------------------------------------
# Module-level predictor cache
# Avoids reloading weights from disk on every Apply click.
# Key: (checkpoint_path, device_str)  →  predictor object
# ---------------------------------------------------------------------------
_PREDICTOR_CACHE: dict = {}


def _get_predictor(checkpoint_path: str, config_path: str, device: str):
    """Return a cached predictor, or build and cache a new one."""
    key = (checkpoint_path, device)
    if key not in _PREDICTOR_CACHE:
        if torch.cuda.is_available() and device == "cuda":
            torch.backends.cudnn.benchmark = True   # auto-tune kernels
            torch.cuda.empty_cache()
        _PREDICTOR_CACHE[key] = build_sam2_video_predictor(
            config_path, checkpoint_path, device=device)
    return _PREDICTOR_CACHE[key]


def _get_fast_temp_dir() -> tuple[str, bool]:
    """
    Prefer /dev/shm (Linux RAM disk) for frame storage so JPEG I/O never
    hits a spinning or SSD disk.  Falls back to the system temp dir.
    Returns (path, should_cleanup).
    """
    ram_disk = "/dev/shm"
    if os.path.isdir(ram_disk):
        return tempfile.mkdtemp(dir=ram_disk, prefix="melage_sam_"), True
    return tempfile.mkdtemp(prefix="melage_sam_"), True


def _write_jpeg(args: tuple):
    """Write one RGB frame as JPEG — designed for ThreadPoolExecutor."""
    path, frame_rgb = args
    # quality=80: ~4× faster encode than q=95, imperceptible for segmentation.
    # subsampling=2 (4:2:0): fastest chroma encoding; fine for grayscale/US.
    Image.fromarray(frame_rgb).save(path, format="JPEG", quality=80,
                                    subsampling=2, optimize=False)


class Sam2Logic(DynamicDialog):
    completed = pyqtSignal(object)

    # Number of parallel JPEG-writer threads.
    # 4 is a safe default; increase if you have many fast CPU cores.
    _N_WRITE_WORKERS = 4

    def __init__(self, data_context, parent=None, target_temp_dir=None):
        super().__init__(parent)
        self.data_context = data_context
        self.custom_temp_dir = target_temp_dir

        self.create_main_ui(schema=get_schema(), default_items=False)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._current_index = 0
        self.BUFFER = 50

        for widget_id, widget_obj in self.widgets.items():
            setattr(self, widget_id, widget_obj)
            if hasattr(widget_obj, "clicked"):
                getattr(widget_obj, "clicked").connect(
                    lambda _, x=widget_id: getattr(self, f"on_{x}_clicked")()
                    if hasattr(self, f"on_{x}_clicked") else None
                )

        if self.data_context:
            self.combo_view.clear()
            self.combo_view.addItems(list(self.data_context.keys()))

        self.combo_view.currentIndexChanged.connect(self._on_changed_current_index)
        self._on_changed_current_index()
        self.bind_range_logic()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def bind_range_logic(self):
        self.spin_start.valueChanged.connect(self.on_start_changed)
        self.spin_end.valueChanged.connect(self.on_end_changed)
        self.on_start_changed(self.spin_start.value())
        self.on_end_changed(self.spin_end.value())

    def on_start_changed(self, new_start_val):
        if new_start_val >= self.spin_end.value():
            self.spin_end.setValue(new_start_val + 1)
        self.spin_end.setMinimum(new_start_val + 1)

    def on_end_changed(self, new_end_val):
        if new_end_val <= self.spin_start.value():
            self.spin_start.setValue(new_end_val - 1)
        self.spin_start.setMaximum(new_end_val - 1)

    def _on_changed_current_index(self):
        total_frames, current_index = 100, 0
        view_index = self.combo_view.currentIndex()
        data_obj = self.data_context.get('view 1' if view_index == 0 else 'view 2')
        if data_obj and getattr(data_obj, "isChunkedVideo", False):
            proxy = data_obj.video_im
            total_frames  = int(proxy.frames)
            current_index = int(proxy.current_index)
        self.spin_start.setMaximum(total_frames)
        self.spin_start.setValue(max(0, current_index - self.BUFFER))
        self.spin_end.setMaximum(total_frames)
        self.spin_end.setValue(min(total_frames, current_index + self.BUFFER))
        self._current_index = current_index

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_frame_rgb(image_proxy, index: int) -> np.ndarray:
        """Extract one RGB uint8 frame from a proxy or NIfTI object."""
        if hasattr(image_proxy, "get_frame"):
            return image_proxy.get_frame(index)

        raw = image_proxy.get_fdata() if hasattr(image_proxy, "get_fdata") else image_proxy
        if raw.ndim == 4:
            raw = raw[..., 0]
        if 0 <= index < raw.shape[2]:
            sl = raw[:, :, index]
            if sl.dtype != np.uint8:
                lo, hi = sl.min(), sl.max()
                sl = ((sl - lo) / (hi - lo) * 255.0).astype(np.uint8) if hi > lo \
                     else np.zeros_like(sl, dtype=np.uint8)
            return np.stack([sl] * 3, axis=-1)
        return np.zeros((256, 256, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Parallel frame export
    # ------------------------------------------------------------------

    def _export_frames(self, image_proxy, start_z: int, end_z: int,
                       temp_dir: str) -> None:
        """
        Extract frames (serially — proxy may not be thread-safe) then
        encode and write JPEG files in parallel using a thread pool.
        Extracting into memory first avoids holding the GIL during I/O.
        """
        n = end_z - start_z
        tasks = []
        for i, global_i in enumerate(range(start_z, end_z)):
            frame = self._get_frame_rgb(image_proxy, global_i)
            path  = os.path.join(temp_dir, f"{i:05d}.jpg")
            tasks.append((path, frame))

            # Keep UI responsive during the extraction phase
            if i % 30 == 0:
                QtWidgets.QApplication.processEvents()

        workers = min(self._N_WRITE_WORKERS, max(1, n))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_write_jpeg, tasks))   # blocks until all written

    # ------------------------------------------------------------------
    # Mask writing
    # ------------------------------------------------------------------

    def _write_mask_to_proxy(self, label_proxy, out_logits,
                              start_z: int, out_idx: int, sub_len: int) -> None:
        global_z = start_z + out_idx

        if len(out_logits) > 0:
            logit = out_logits[0]          # shape: (1, H, W) on GPU

            # Check on GPU first — avoids a full CPU transfer for empty frames
            if logit.any():
                mask = (logit > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                label_proxy[..., global_z] = mask

        # Progress: base 50% + up to 50% for propagation
        self.progress_bar.setValue(50 + int((out_idx / max(1, sub_len)) * 50))

        # Reduced from every-5 to every-20: fewer interruptions to the GPU pipeline
        if out_idx % 20 == 0:
            QtWidgets.QApplication.processEvents()

    # ------------------------------------------------------------------
    # Main apply handler
    # ------------------------------------------------------------------

    def on_btn_apply_clicked(self):
        if not SAM2_AVAILABLE:
            QMessageBox.critical(self, "Error", "SAM 2 is not installed.")
            return

        view_name = self.combo_view.currentText()
        if view_name not in self.data_context:
            return
        data_obj = self.data_context[view_name]

        # --- Identify proxies ---
        if getattr(data_obj, "isChunkedVideo", False):
            label_proxy = data_obj.seg_ims
            image_proxy = data_obj.video_im
        else:
            QMessageBox.critical(self, "Error", "SAM 2 for MRI/CT not implemented yet.")
            return

        if not hasattr(image_proxy, "frames"):
            QMessageBox.critical(self, "Error", "SAM 2 requires a video/chunked input.")
            return

        total_frames = image_proxy.frames

        # --- Temp directory ---
        if self.custom_temp_dir:
            os.makedirs(self.custom_temp_dir, exist_ok=True)
            temp_dir, should_cleanup = self.custom_temp_dir, False
        else:
            temp_dir, should_cleanup = _get_fast_temp_dir()

        # --- Resolve model paths ---
        model_size = self.combo_model_size.itemText(self.combo_model_size.currentIndex())
        PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
        config_path = MODEL_REGISTRY[model_size]["config"]
        checkpoint_path = os.path.join(PLUGIN_DIR, MODEL_REGISTRY[model_size]["checkpoint"])

        if not os.path.exists(checkpoint_path):
            QMessageBox.critical(self, "Error", f"Missing checkpoint: {checkpoint_path}")
            return

        device = "cuda" if torch.cuda.is_available() and self.check_cuda.isChecked() else "cpu"

        try:
            self.progress_bar.setValue(5)
            self.btn_apply.setEnabled(False)
            QtWidgets.QApplication.processEvents()

            # --- Range validation ---
            start_z, end_z = 0, total_frames
            if getattr(self, "check_limit_range", None) and self.check_limit_range.isChecked():
                start_z = int(max(0, self.spin_start.value()))
                end_z   = int(min(total_frames, self.spin_end.value())) + 1

                if not (start_z <= self._current_index < end_z):
                    QMessageBox.warning(self, "Out of Range",
                        f"Current slice ({self._current_index}) is outside the selected "
                        f"range ({start_z}–{end_z}).\nNavigate to a slice inside the range.")
                    return

                if label_proxy and hasattr(label_proxy, "segmented_indices"):
                    if len(label_proxy.segmented_indices) == 0:
                        QMessageBox.warning(self, "No Segmentation Found",
                            "SAM 2 needs at least one segmented slice as a prompt.\n"
                            "Please manually segment the object on the current slice.")
                        return
                    if np.sum(label_proxy.get_frame(self._current_index)) == 0:
                        QMessageBox.warning(self, "Current Slice Empty",
                            f"Slice {self._current_index} has no segmentation.\n"
                            "SAM 2 needs a seed mask on the visible slice.")
                        return

            sub_len = end_z - start_z
            print(f"Exporting {sub_len} frames to {temp_dir} ...")

            # --- Export frames in parallel ---
            self._export_frames(image_proxy, start_z, end_z, temp_dir)
            self.progress_bar.setValue(20)

            # --- Load (or reuse cached) predictor ---
            print(f"Loading SAM 2 [{model_size}] on {device} ...")
            predictor = _get_predictor(checkpoint_path, config_path, device)
            inference_state = predictor.init_state(video_path=temp_dir)

            # --- Apply prompts ---
            prompts_applied = 0
            if label_proxy and hasattr(label_proxy, "segmented_indices"):
                for global_idx in label_proxy.segmented_indices:
                    if not (start_z <= global_idx < end_z):
                        continue
                    mask_uint8 = label_proxy.get_frame(global_idx)
                    if np.any(mask_uint8):
                        predictor.add_new_mask(
                            inference_state,
                            global_idx - start_z,
                            obj_id=1,
                            mask=(mask_uint8 > 0).astype(np.float32),
                        )
                        prompts_applied += 1

            self.progress_bar.setValue(50)
            print("Propagating ...")

            # Determine seed frame
            start_prop_idx = self._current_index - start_z
            if not (0 <= start_prop_idx < sub_len):
                start_prop_idx = 0

            # --- Propagate with inference_mode + optional fp16 ---
            use_fp16 = (device == "cuda")
            autocast_ctx = (torch.autocast("cuda", dtype=torch.float16)
                            if use_fp16 else torch.inference_mode())

            with torch.inference_mode():
                cm = (torch.autocast("cuda", dtype=torch.float16)
                      if use_fp16 else torch.no_grad())
                with cm:
                    # Backward pass
                    print(f"  <-- Backward (frame {start_prop_idx} → 0)")
                    for out_idx, _, out_logits in predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=start_prop_idx,
                            reverse=True):
                        self._write_mask_to_proxy(
                            label_proxy, out_logits, start_z, out_idx, sub_len)

                    # Forward pass
                    print(f"  --> Forward (frame {start_prop_idx} → {sub_len})")
                    for out_idx, _, out_logits in predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=start_prop_idx,
                            reverse=False):
                        self._write_mask_to_proxy(
                            label_proxy, out_logits, start_z, out_idx, sub_len)

            predictor.reset_state(inference_state)

            if label_proxy and hasattr(data_obj, "seg_ims"):
                data_obj.seg_ims = label_proxy

            self.completed.emit({"view": view_name, "data_obj": data_obj})
            self.progress_bar.setValue(100)
            QMessageBox.information(
                self, "Success",
                f"Processed {sub_len} frames with {prompts_applied} prompt(s).\n"
                f"Model loaded from: {'cache' if (checkpoint_path, device) in _PREDICTOR_CACHE else 'disk'}.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_apply.setEnabled(True)
            if should_cleanup and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


# --- Plugin entry point ---
class Sam2Plugin(MelagePlugin):
    @property
    def name(self) -> str: return "SAM 2 (Video/3D)"

    @property
    def category(self) -> str: return "Deep Learning"

    def get_widget(self, data_context=None, parent=None):
        return Sam2Logic(data_context, parent, target_temp_dir=None)
