"""
melage.api._io
==============
load() and save() — the entry/exit points of every MELAGE pipeline.

Examples
--------
    vol = melage.load("brain.nii.gz")
    vol = melage.load("/data/study", fmt="dicom")   # DICOM folder

    melage.save(vol, "output.nii.gz")
    melage.save(vol, "seg.nii.gz", what="seg")
"""

from __future__ import annotations

import os
import numpy as np
import nibabel as nib
from typing import Optional, Union

from melage.api._volume import Volume


def load(
    path: str,
    *,
    target_system: str = "IPL",
    modality: str = "eco",
) -> Volume:
    """
    Load a medical image from disk and return a :class:`Volume`.

    Supported formats: NIfTI (.nii, .nii.gz), NRRD (.nrrd, .nhdr),
    DICOM (.dcm or a folder), GE Kretz (.vol), MP4/AVI video.

    Parameters
    ----------
    path : str
        Path to an image file or DICOM folder.
    target_system : str
        Orientation target for internal resampling (default ``"IPL"``).
    modality : str
        Hint for intensity normalisation (``"eco"``, ``"t1"``, …).

    Returns
    -------
    Volume
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"melage.load: path not found — {path!r}")

    from melage.core.io import load_image_core

    reader, info, fmt = load_image_core(path, target_system=target_system, type=modality)

    if not info[1]:
        raise RuntimeError(
            f"melage.load: failed to read {path!r} — {info[2]}"
        )

    return Volume.from_reader(reader)


def save(
    vol: Volume,
    path: str,
    *,
    what: str = "image",
    dtype=None,
) -> None:
    """
    Save a Volume to disk (NIfTI by default).

    Parameters
    ----------
    vol : Volume
        The volume to persist.
    path : str
        Output file path.  The extension determines the format
        (``.nii``, ``.nii.gz``, ``.nrrd``).
    what : {"image", "seg", "segmentation"}
        Which array to write.  ``"image"`` saves the image data;
        ``"seg"`` saves the integer segmentation.
    dtype : numpy dtype, optional
        Cast the output array before saving.  Defaults to float32
        for images and int16 for segmentations.
    """
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if what in ("seg", "segmentation", "label"):
        if vol.segmentation is None:
            raise ValueError(
                "melage.save: Volume has no segmentation. "
                "Run a segmentation method first (e.g. melage.segment.bet(vol))."
            )
        data = vol.segmentation
        out_dtype = dtype or np.int16
        img = nib.Nifti1Image(data.astype(out_dtype), vol.affine, vol.header)
    else:
        data = vol.data
        out_dtype = dtype or np.float32
        img = nib.Nifti1Image(data.astype(out_dtype), vol.affine, vol.header)

    nib.save(img, path)
    print(f"Saved → {path}")


def info(path: str) -> dict:
    """
    Return a dictionary of metadata for a file *without* fully loading voxels.

    Also prints a human-readable summary.

    Parameters
    ----------
    path : str
        Image file path.

    Returns
    -------
    dict with keys: shape, spacing, dtype, affine, format
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"melage.info: path not found — {path!r}")

    try:
        img = nib.load(path)
        shape = img.shape
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        dtype = img.get_data_dtype()
        affine = img.affine
        fmt = "NIfTI"
    except Exception:
        # fall back to a full load for non-NIfTI
        vol = load(path)
        shape = vol.shape
        spacing = vol.spacing
        dtype = vol.dtype
        affine = vol.affine
        fmt = "unknown"

    result = {
        "path": path,
        "format": fmt,
        "shape": shape,
        "spacing_mm": spacing,
        "dtype": dtype,
        "affine": affine,
    }

    print(
        f"File   : {path}\n"
        f"Format : {fmt}\n"
        f"Shape  : {shape}\n"
        f"Spacing: {spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f} mm\n"
        f"Dtype  : {dtype}"
    )
    return result
