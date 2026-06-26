"""
melage.api._segment
===================
Headless segmentation operations.

Each function returns a new :class:`~melage.api._volume.Volume` whose
``segmentation`` attribute contains the integer label array.

Examples
--------
    import melage

    vol = melage.load("brain.nii.gz")

    # Brain extraction
    vol = melage.segment.bet(vol)
    melage.save(vol, "brain_mask.nii.gz", what="seg")

    # Tissue segmentation
    vol = melage.segment.fcm(vol, n_classes=3)
    melage.save(vol, "tissues.nii.gz", what="seg")
"""

from __future__ import annotations

import os
from typing import Union
import numpy as np

from melage.api._volume import Volume
from melage.api._progress import make_progress, PrintProgress


# ──────────────────────────────────────────────────────────────────────
# BET — Brain Extraction Tool
# ──────────────────────────────────────────────────────────────────────

def bet(
    vol: Volume,
    *,
    thresholding: bool = True,
    fractional_threshold: float = 0.50,
    t02: float = 0.02,
    t98: float = 0.98,
    d1: float = 7.0,
    d2: float = 2.0,
    r_min: float = 3.3,
    r_max: float = 10.0,
    iterations: int = 1000,
    progress=None,
) -> Volume:
    """
    Brain Extraction Tool (BET).

    Identifies the brain boundary and returns a binary brain mask.

    Parameters
    ----------
    vol : Volume
    thresholding : bool
        Use multi-Otsu thresholding to derive t02/t98 automatically
        (default True).  When False, ``t02`` and ``t98`` are used as
        raw intensity percentile fractions (0–1).
    fractional_threshold : float
        Controls aggressiveness; lower → more liberal extraction (default 0.5).
    t02, t98 : float
        Intensity fraction thresholds (used only when *thresholding* is False).
    d1, d2 : float
        Search distance parameters (mm, defaults 7 and 2).
    r_min, r_max : float
        Curvature radius limits (mm, defaults 3.3 and 10).
    iterations : int
        Maximum mesh evolution iterations (default 1000).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Input image with ``segmentation`` set to the binary brain mask.
    """
    bar = make_progress(progress, "BET")
    bar.setValue(0)

    from melage.plugins.bet.main.BET import BET as _BET

    data = vol.data.copy().astype(np.float32)
    spacing = max(vol.spacing)

    bet_obj = _BET(data, spacing)
    bet_obj._progress = 0  # internal progress counter

    if thresholding:
        # Threshold_MultiOtsu(data, n) returns n-1 cut points; n=1 or n=2 both
        # collapse to a single value (t02 == t98), which makes BET's deformation
        # step divide by zero. n=3 gives two distinct low/high cut points.
        from melage.utils.utils import Threshold_MultiOtsu
        thresholds = Threshold_MultiOtsu(data, 3)
        _t02 = float(thresholds[0])
        _t98 = float(thresholds[-1])
    else:
        _t02 = t02
        _t98 = t98

    params = [_t02, _t98, fractional_threshold, d1, d2, r_min, r_max, np.int64(iterations)]
    bet_obj.update_params(params, thresholding)

    bet_obj.initialization(bar, bar)
    bet_obj.run(bar, bar)
    mask = bet_obj.compute_mask()

    bar.setValue(100)
    return vol.with_segmentation(mask.astype(np.int32))


# ──────────────────────────────────────────────────────────────────────
# FCM — Fuzzy C-Means tissue segmentation
# ──────────────────────────────────────────────────────────────────────

def fcm(
    vol: Volume,
    *,
    n_classes: int = 3,
    method: str = "FCM",
    max_iter: int = 100,
    post_correction: bool = True,
    progress=None,
) -> Volume:
    """
    Tissue segmentation via Fuzzy C-Means (esFCM).

    Parameters
    ----------
    vol : Volume
    n_classes : int
        Number of tissue classes (default 3: background / GM / WM).
    method : str
        Clustering variant: ``"FCM"``, ``"PFCM"``, ``"eFCM"`` (default ``"FCM"``).
    max_iter : int
        Maximum iterations (default 100).
    post_correction : bool
        Apply spatial post-processing to reduce noise (default True).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Input image with ``segmentation`` set to integer tissue labels.
    """
    bar = make_progress(progress, "FCM")
    bar.setValue(0)

    from melage.plugins.esfcm.main.test import get_inference

    data = vol.data.copy()
    seg = get_inference(
        method, bar, data,
        affine=None,
        num_tissues=n_classes,
        post_correction=post_correction,
        max_iter=max_iter,
    )
    bar.setValue(100)
    return vol.with_segmentation(seg.astype(np.int32))


# ──────────────────────────────────────────────────────────────────────
# MGA-Net — brain extraction + image reconstruction
# ──────────────────────────────────────────────────────────────────────

def mga_net(
    vol: Volume,
    *,
    modality: str = "mri",
    threshold: float = 0.0,
    high_quality: bool = True,
    model_path: str = None,
    use_cuda: bool = True,
    progress=None,
) -> Volume:
    """
    MGA-Net: deep-learning brain extraction with simultaneous image
    reconstruction (denoising/super-resolution of the brain region).

    Parameters
    ----------
    vol : Volume
    modality : {"mri", "us"}
        Imaging modality (default ``"mri"``); use ``"us"`` for ultrasound.
    threshold : float
        Mask probability threshold (default 0.0).
    high_quality : bool
        Run a second, higher-resolution pass for the reconstructed image
        (default True; slower but sharper reconstruction).
    model_path : str, optional
        Custom checkpoint path; defaults to the bundled ``MGA_NET.pth``.
    use_cuda : bool
        Use CUDA if available (default True).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Reconstructed image with ``segmentation`` set to the binary brain mask.
    """
    bar = make_progress(progress, "MGA-Net")
    bar.setValue(0)

    import torch
    from melage.plugins.mga_net.main.test_mgaNet import build_model, get_inference

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    bar.setValue(10)

    model = build_model(model_path=model_path, device=device)
    bar.setValue(30)

    data = vol.data.copy()
    eco_mri = 1 if modality.lower() == "mri" else -1
    rec_nifti, mask = get_inference(
        model, data, vol.affine, device,
        eco_mri=eco_mri, threshold=threshold, high_quality_rec=high_quality,
    )
    bar.setValue(100)
    return Volume(rec_nifti, segmentation=mask.astype(np.int32))


# ──────────────────────────────────────────────────────────────────────
# WarpSeg — deep-learning structural / tissue segmentation
# ──────────────────────────────────────────────────────────────────────

def warpseg(
    vol: Volume,
    *,
    output: str = "whole",
    model_path: str = None,
    use_cuda: bool = True,
    progress=None,
) -> Volume:
    """
    WarpSeg: deep-learning whole-brain structural / tissue segmentation.

    Parameters
    ----------
    vol : Volume
    output : {"whole", "tissue"}
        Which map to return: structural ROI labels (``"whole"``, default)
        or coarse tissue classes (``"tissue"``).
    model_path : str, optional
        Custom checkpoint path; defaults to
        ``<DEFAULT_MODELS_DIR>/WarpSeg_Adult.pth``.
    use_cuda : bool
        Use CUDA if available (default True).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Input image with ``segmentation`` set to the predicted labels.
    """
    bar = make_progress(progress, "WarpSeg")
    bar.setValue(0)

    import torch
    from melage.config import settings
    from melage.plugins.warpseg.warpseg_main.test import build_model, get_inference

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    path = model_path or os.path.join(settings.DEFAULT_MODELS_DIR, "WarpSeg_Adult.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"WarpSeg weights not found at {path}. Pass model_path= explicitly "
            "or place WarpSeg_Adult.pth in settings.DEFAULT_MODELS_DIR."
        )
    bar.setValue(10)

    model = build_model(path, device)
    bar.setValue(30)

    data = vol.data.copy()
    seg_tissue, seg_whole = get_inference(model, data, device, post_processing=False)
    bar.setValue(100)

    seg = seg_whole if output == "whole" else seg_tissue
    return vol.with_segmentation(seg.astype(np.int32))


# ──────────────────────────────────────────────────────────────────────
# Masking Operation — boolean combination of two existing labels
# ──────────────────────────────────────────────────────────────────────

def combine_labels(
    vol: Volume,
    label1: int,
    op: str,
    label2: int,
    *,
    output_label: int = None,
    progress=None,
) -> Volume:
    """
    Combine two existing segmentation labels with a boolean set operation.

    Parameters
    ----------
    vol : Volume
        Must have a segmentation array.
    label1, label2 : int
        Label indices to combine.
    op : {"+", "-", "*", "/"}
        ``"+"`` = union, ``"-"`` = difference (label1 minus label2),
        ``"*"`` = intersection, ``"/"`` = symmetric difference.
    output_label : int, optional
        Label index to write the result into (default: ``label1``).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Same image, segmentation with ``output_label`` replaced by the
        combined mask.
    """
    if vol.segmentation is None:
        raise ValueError("combine_labels: Volume has no segmentation.")

    bar = make_progress(progress, "combine_labels")
    bar.setValue(0)

    from melage.plugins.masking_operation.mo import combine_label_masks

    seg = combine_label_masks(vol.segmentation, label1, op, label2, output_label)
    bar.setValue(100)
    return vol.with_segmentation(seg.astype(np.int32))


# ──────────────────────────────────────────────────────────────────────
# N4 + BET convenience pipeline
# ──────────────────────────────────────────────────────────────────────

def preprocess_and_bet(
    vol: Volume,
    *,
    n4_first: bool = True,
    progress=None,
) -> Volume:
    """
    Convenience pipeline: optional N4 bias correction → BET.

    Parameters
    ----------
    vol : Volume
    n4_first : bool
        Run N4 bias correction before BET (default True).
    progress : bool | callable | None

    Returns
    -------
    Volume
        Bias-corrected (if requested) image with brain mask attached.
    """
    from melage.api._preprocess import n4_bias

    bar = make_progress(progress, "pipeline")

    if n4_first:
        bar.setText("Step 1/2: N4 bias correction …")
        vol = n4_bias(vol, progress=False)
        bar.setValue(50)

    bar.setText("Step 2/2: Brain extraction …")
    vol = bet(vol, progress=False)
    bar.setValue(100)
    return vol
