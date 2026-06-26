"""
melage.api._preprocess
======================
Headless preprocessing operations.

All functions accept a :class:`~melage.api._volume.Volume` and return a new
Volume (the input is never mutated).

Examples
--------
    import melage

    vol = melage.load("brain.nii.gz")
    vol = melage.preprocess.n4_bias(vol)
    vol = melage.preprocess.resize(vol, spacing=1.0)
    vol = melage.preprocess.normalize(vol)
    melage.save(vol, "brain_corrected.nii.gz")
"""

from __future__ import annotations

from typing import Union, Sequence
import numpy as np
import nibabel as nib

from melage.api._volume import Volume
from melage.api._progress import make_progress


def n4_bias(
    vol: Volume,
    *,
    iterations: int = 50,
    shrink_factor: int = 1,
    fitting_levels: int = 4,
    use_otsu: bool = True,
    progress=None,
) -> Volume:
    """
    N4 bias-field correction (SimpleITK).

    Parameters
    ----------
    vol : Volume
    iterations : int
        Maximum number of fitting iterations per level (default 50).
    shrink_factor : int
        Down-sample factor before fitting; 1 = full resolution (default 1).
    fitting_levels : int
        Number of multi-resolution fitting levels (default 4).
    use_otsu : bool
        Build the binary brain mask with multi-Otsu thresholding (default True).
        Set to False to use a simple non-zero mask.
    progress : bool | callable | None
        Progress reporting. ``True``/``None`` → print to stdout,
        ``False`` → silent, callable → receives ``(pct, msg)``.

    Returns
    -------
    Volume
        Bias-corrected image; segmentation is carried over unchanged.
    """
    bar = make_progress(progress, "N4")
    bar.setValue(0)
    bar.setText("Running N4 bias correction …")

    from melage.plugins.N4_bias.main.utils import N4_bias_correction

    corrected_nib = N4_bias_correction(
        vol.to_nibabel(),
        use_otsu=use_otsu,
        shrinkFactor=shrink_factor,
        numberFittingLevels=fitting_levels,
        max_iter=iterations,
    )
    bar.setValue(100)
    return Volume(corrected_nib, segmentation=vol.segmentation)


def resize(
    vol: Volume,
    spacing: Union[float, Sequence[float]],
    *,
    method: str = "spline",
    progress=None,
) -> Volume:
    """
    Resample a volume to a new voxel spacing.

    Parameters
    ----------
    vol : Volume
    spacing : float | sequence of 3 floats
        Target voxel size in mm.  A single float sets isotropic spacing.
    method : {"spline", "linear", "nearest"}
        Interpolation method (default ``"spline"``).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Resampled image.  Segmentation is *not* transferred automatically
        because it would require nearest-neighbour resampling; resample it
        explicitly with ``method="nearest"`` if needed.
    """
    bar = make_progress(progress, "resize")
    bar.setValue(0)

    from melage.utils.utils import resample_to_spacing

    if isinstance(spacing, (int, float)):
        spacing = [float(spacing)] * 3
    else:
        spacing = [float(s) for s in spacing]

    bar.setText(f"Resampling → {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm …")
    resampled = resample_to_spacing(vol.to_nibabel(), spacing, method)
    bar.setValue(100)
    return Volume(resampled)


def reorient(
    vol: Volume,
    target: str = "RAS",
    *,
    progress=None,
) -> Volume:
    """
    Reorient a volume to a target anatomical coordinate system.

    Parameters
    ----------
    vol : Volume
    target : str
        Three-letter axis code (e.g. ``"RAS"``, ``"LPS"``), default ``"RAS"``.
    progress : bool | callable | None

    Returns
    -------
    Volume
        Reoriented image. The segmentation (if present) is reoriented to match.
    """
    bar = make_progress(progress, "reorient")
    bar.setValue(0)

    from melage.plugins.change_coord.main.utils import changeCoordSystem

    reoriented = changeCoordSystem(vol.to_nibabel(), target=target)
    bar.setValue(50)

    seg = None
    if vol.segmentation is not None:
        seg_img = nib.Nifti1Image(vol.segmentation, vol.affine, vol.header)
        seg = changeCoordSystem(seg_img, target=target).get_fdata().astype(np.int32)

    bar.setValue(100)
    return Volume(reoriented, segmentation=seg)


def normalize(
    vol: Volume,
    *,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
    out_range: tuple = (0.0, 255.0),
    progress=None,
) -> Volume:
    """
    Intensity normalization by percentile clipping and rescaling.

    Parameters
    ----------
    vol : Volume
    percentile_low : float
        Lower percentile for clipping (default 0.5).
    percentile_high : float
        Upper percentile for clipping (default 99.5).
    out_range : tuple
        Output intensity range after rescaling (default ``(0, 255)``).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Normalised image.
    """
    bar = make_progress(progress, "normalize")
    bar.setValue(0)
    bar.setText("Normalising intensities …")

    data = vol.data.astype(np.float32).copy()
    lo = float(np.percentile(data, percentile_low))
    hi = float(np.percentile(data, percentile_high))
    if hi == lo:
        raise ValueError("normalize: image has zero dynamic range.")
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo) * (out_range[1] - out_range[0]) + out_range[0]

    bar.setValue(100)
    img = nib.Nifti1Image(data, vol.affine, vol.header)
    return Volume(img, segmentation=vol.segmentation)


def threshold(
    vol: Volume,
    low: float,
    high: float,
    *,
    fill_below: float = 0.0,
    fill_above: float = 0.0,
    progress=None,
) -> Volume:
    """
    Simple intensity threshold — voxels outside [low, high] are set to *fill*.

    Parameters
    ----------
    vol : Volume
    low, high : float
        Intensity window.
    fill_below, fill_above : float
        Replacement values outside the window (default 0).
    progress : bool | callable | None

    Returns
    -------
    Volume
    """
    bar = make_progress(progress, "threshold")
    bar.setValue(0)

    data = vol.data.astype(np.float32).copy()
    data[data < low] = fill_below
    data[data > high] = fill_above

    bar.setValue(100)
    img = nib.Nifti1Image(data, vol.affine, vol.header)
    return Volume(img, segmentation=vol.segmentation)


def largest_component(
    vol: Volume,
    *,
    label: int = 1,
    connectivity: int = 3,
    progress=None,
) -> Volume:
    """
    Keep only the largest connected component of a segmentation label.

    Parameters
    ----------
    vol : Volume
        Must have a segmentation array.
    label : int
        Which label to clean (default 1).
    connectivity : int
        Voxel connectivity: 1 (face), 2 (edge), 3 (corner, default).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Same image, segmentation with isolated components removed.
    """
    if vol.segmentation is None:
        raise ValueError("largest_component: Volume has no segmentation.")

    bar = make_progress(progress, "LargestCC")
    bar.setValue(0)

    from melage.utils.utils import LargestCC

    seg = vol.segmentation.copy()
    binary = (seg == label).astype(np.uint8)
    cleaned = LargestCC(binary, connectivity=connectivity)
    seg[seg == label] = 0
    seg[cleaned == 1] = label

    bar.setValue(100)
    return vol.with_segmentation(seg)
