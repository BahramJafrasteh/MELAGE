"""
melage.api._volume
==================
Lightweight, GUI-free container for medical image data.

A Volume holds:
  • image data  (numpy float array)
  • affine      (4×4 world-space transform)
  • nibabel header (optional, preserves original metadata)
  • segmentation  (integer numpy array, same spatial shape, optional)

It round-trips cleanly to/from nibabel Nifti1Image and is repr-friendly in
Jupyter notebooks.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from typing import Optional, Tuple


class Volume:
    """
    Immutable-by-convention container for one medical image volume.

    Parameters
    ----------
    image : np.ndarray | nib.Nifti1Image
        3-D (or 4-D) image data, or a nibabel image (affine/header inferred).
    affine : np.ndarray, optional
        4×4 affine matrix (required when *image* is an ndarray).
    header : nib.Nifti1Header, optional
        Preserves spacing, units, etc.
    segmentation : np.ndarray, optional
        Integer label array with the same spatial shape as *image*.
    """

    def __init__(
        self,
        image,
        affine: Optional[np.ndarray] = None,
        header=None,
        segmentation: Optional[np.ndarray] = None,
    ):
        if isinstance(image, nib.Nifti1Image):
            self._nib = image
            self._data = np.asarray(image.dataobj)
            self._affine = image.affine.copy()
            self._header = image.header
        else:
            self._data = np.asarray(image)
            self._affine = np.asarray(affine) if affine is not None else np.eye(4)
            self._header = header
            self._nib = nib.Nifti1Image(self._data, self._affine, self._header)

        self.segmentation: Optional[np.ndarray] = (
            np.asarray(segmentation, dtype=np.int32) if segmentation is not None else None
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_reader(cls, reader) -> "Volume":
        """Build a Volume from a melage.core.io.readData object."""
        vol = cls.__new__(cls)
        vol._nib = reader.im
        vol._data = np.asarray(reader.im.dataobj)
        vol._affine = reader.im.affine.copy()
        vol._header = reader.im.header
        # npSeg may be None or an uninitialized array
        seg = getattr(reader, "_npSeg", None)
        if seg is None:
            seg = getattr(reader, "npSeg", None)
        if seg is not None and np.any(seg):
            vol.segmentation = seg.copy().astype(np.int32)
        else:
            vol.segmentation = None
        return vol

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Raw image voxel array (float, read-only convention)."""
        return self._data

    @property
    def affine(self) -> np.ndarray:
        """4×4 voxel-to-world affine."""
        return self._affine

    @property
    def header(self):
        """nibabel Nifti1Header (may be None if created from bare array)."""
        return self._header

    @property
    def shape(self) -> Tuple:
        """Spatial + optional time dimensions."""
        return self._data.shape

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Voxel size in mm (x, y, z)."""
        if self._header is not None:
            try:
                zooms = self._header.get_zooms()
                return tuple(float(z) for z in zooms[:3])
            except Exception:
                pass
        # fall back to diagonal of affine
        return tuple(float(np.linalg.norm(self._affine[:3, i])) for i in range(3))

    @property
    def dtype(self):
        return self._data.dtype

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_nibabel(self) -> nib.Nifti1Image:
        """Return the underlying Nifti1Image."""
        return self._nib

    def seg_to_nibabel(self) -> nib.Nifti1Image:
        """Return the segmentation as a Nifti1Image (same affine)."""
        if self.segmentation is None:
            raise ValueError("This Volume has no segmentation.")
        return nib.Nifti1Image(self.segmentation, self._affine, self._header)

    def with_segmentation(self, seg: np.ndarray) -> "Volume":
        """Return a new Volume identical to this one but with the given segmentation."""
        v = Volume(self._nib, segmentation=seg)
        return v

    # ------------------------------------------------------------------
    # Jupyter / REPL display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sp = self.spacing
        seg_info = (
            f", seg_labels={int(self.segmentation.max())}"
            if self.segmentation is not None
            else ""
        )
        return (
            f"Volume(shape={self.shape}, "
            f"spacing=({sp[0]:.2f},{sp[1]:.2f},{sp[2]:.2f}) mm, "
            f"dtype={self.dtype}{seg_info})"
        )

    def _repr_html_(self) -> str:
        sp = self.spacing
        rows = [
            ("Shape", str(self.shape)),
            ("Spacing (mm)", f"{sp[0]:.3f} × {sp[1]:.3f} × {sp[2]:.3f}"),
            ("Dtype", str(self.dtype)),
            ("Intensity range", f"[{float(self._data.min()):.1f}, {float(self._data.max()):.1f}]"),
            ("Segmentation", f"labels 0–{int(self.segmentation.max())}" if self.segmentation is not None else "None"),
        ]
        if self._header is not None:
            try:
                units = self._header.get_xyzt_units()
                rows.append(("Units", str(units)))
            except Exception:
                pass
        cells = "".join(
            f"<tr><td style='padding:3px 8px;font-weight:bold'>{k}</td>"
            f"<td style='padding:3px 8px'>{v}</td></tr>"
            for k, v in rows
        )
        return (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:0.9em'>"
            f"<tr><th colspan='2' style='text-align:left;padding:4px 8px;background:#2d6a9f;color:white'>"
            "melage.Volume</th></tr>"
            f"{cells}</table>"
        )
