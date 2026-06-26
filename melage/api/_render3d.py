"""
melage.api._render3d
====================
Headless, GPU-accelerated 3-D rendering that reuses MELAGE's real OpenGL
volume viewer (:class:`melage.rendering.glScientific.glScientific`) — driven
offscreen via Qt's ``offscreen`` platform plugin, so it works without a
visible display (servers, CI, notebooks) while still using hardware-
accelerated OpenGL.

This is the ``engine="gl"`` backend for :mod:`melage.api._visualize`. The
default ``engine="mesh"`` backend remains the dependency-light fallback used
when PyQt5 / OpenGL / a GPU isn't available — see :func:`melage.api._visualize.render`.
"""

from __future__ import annotations

import os

import numpy as np

from melage.api._volume import Volume

# Process-wide hidden QApplication (and a throwaway "primer" GL widget, see
# _ensure_qapplication). Created lazily on first use and never torn down —
# mirrors how melage.main leaves a live QApplication running for the rest of
# the session, and avoids a Qt/Mesa offscreen teardown crash observed when
# repeatedly creating *and destroying* QOpenGLWidget-based instances.
_qapp = None
_primer = None


def _ensure_qapplication():
    """Return a (possibly hidden, offscreen) QApplication, creating one if needed.

    Also "primes" a throwaway QOpenGLWidget the first time around: creating
    and showing one GL-backed widget initialises GL extension function
    pointers / the default framebuffer in a way that `glScientific`'s first
    instance otherwise misses under the offscreen QPA platform — without it,
    `GLVolumeItem`'s `GL_PROXY_TEXTURE_3D` check spuriously reports "too
    large for this hardware" and volumes silently fail to upload (verified
    empirically: identical volume + GL context, only difference is whether a
    QOpenGLWidget was shown/made-current beforehand).
    """
    global _qapp, _primer
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is not None:
        _qapp = app
    else:
        # Only force the offscreen platform plugin when no display is configured —
        # if the caller already has a real Qt session (e.g. ran from the GUI), reuse it.
        if "QT_QPA_PLATFORM" not in os.environ and not os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["melage-headless"])

    if _primer is None:
        _primer = QtWidgets.QOpenGLWidget()
        _primer.resize(64, 64)
        _primer.show()
        _primer.makeCurrent()
        _qapp.processEvents()

    return _qapp


def _qimage_to_array(qimage) -> np.ndarray:
    """Convert a QImage to an HxWx3 uint8 numpy array (RGB)."""
    from PyQt5.QtGui import QImage

    qimage = qimage.convertToFormat(QImage.Format_RGB888)
    width, height, stride = qimage.width(), qimage.height(), qimage.bytesPerLine()
    ptr = qimage.constBits()
    ptr.setsize(height * stride)
    # Qt pads each scanline to a 4-byte boundary, so `stride` (bytes/row) can
    # exceed `width * 3` (e.g. width=350 -> 1050 bytes of pixels, but a 1052-
    # byte stride) — reshaping straight to (height, stride // 3, 3) then
    # breaks (`stride // 3 != width`). Reshape on the *byte* stride first,
    # then drop the row-padding bytes before reshaping to pixels.
    # `np.array(...)` (not `np.ascontiguousarray`) forces an owned copy: this
    # array would otherwise alias QImage's internal buffer, which the
    # singleton viewer reuses/overwrites on every subsequent capture() —
    # silently corrupting previously-returned arrays in place.
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, stride)
    return np.array(arr[:, :width * 3].reshape(height, width, 3))


def _select_data(vol: Volume, label):
    """Pick the array to render: one/several segmentation labels, the full
    segmentation, or — if there is none — the raw image data."""
    if label is not None:
        if vol.segmentation is None:
            raise ValueError("Volume has no segmentation — run melage.segment.* first")
        labels = [int(label)] if np.isscalar(label) else [int(l) for l in label]
        data = np.where(np.isin(vol.segmentation, labels), vol.segmentation, 0)
        if not np.any(data):
            raise ValueError(f"Label(s) {labels} are empty in this segmentation")
        return data.astype(np.float64), labels
    if vol.segmentation is not None:
        return vol.segmentation.astype(np.float64), None
    return np.asarray(vol.data, dtype=np.float64), None


# Process-wide hidden glScientific viewers, created lazily and reused across
# calls, one per (width, height). glScientific is a QOpenGLWidget —
# repeatedly creating *and tearing down* GL contexts inside one offscreen
# QApplication is what crashes Qt's offscreen/Mesa software-rasterizer combo
# (segfault on the second instance's teardown, observed empirically), so
# instances are cached for the process lifetime — the same way `_qapp` above
# is cached and never destroyed — and never destroyed.
#
# Each size gets its own instance rather than `resize()`-ing a shared one:
# under the offscreen QPA platform, resizing an existing QOpenGLWidget to a
# *larger* size does not reallocate its framebuffer object, so
# `grabFramebuffer()` returns an image padded with stale/uninitialized GPU
# memory outside the old (smaller) framebuffer's bounds (verified
# empirically: a fresh viewer created directly at 500x400 renders correctly,
# but resizing a 350x260 viewer up to 500x400 leaves garbage outside the
# original 350x260 region).
_viewers = {}


def _get_viewer(width, height):
    """Return the process-wide offscreen glScientific viewer for (width, height),
    creating it on first use."""
    _ensure_qapplication()
    from melage.rendering.glScientific import glScientific

    key = (int(width), int(height))
    viewer = _viewers.get(key)
    if viewer is None:
        viewer = glScientific(colorsCombinations={})
        viewer.resize(*key)
        # Cold-start quirk: the very first paint of a freshly created
        # glScientific under the offscreen QPA platform can render a blank/
        # near-black frame (observed mean pixel value ~0.3 vs. ~50-60 for
        # subsequent, otherwise-identical frames) regardless of what data is
        # loaded — a one-time GL/shader warm-up effect, not a texture-upload
        # bug. A single throwaway capture() before any real rendering absorbs
        # it, verified empirically (works even before any volume is loaded).
        viewer.capture(*key)
        _viewers[key] = viewer
    return viewer


_VIEW_PRESETS = {
    "axial": (90, 90),
    "sagittal": (0, 0),
    "coronal": (0, 90),
}

# Mirrors glScientific.cut_actions_list (glScientific.py:1419-1428) — the GUI's
# "Clipping (Cut)" context-menu actions, which set `_artistic` to one of these
# names. The QAction objects themselves are only built lazily by
# ShowContextMenu (never called headlessly), but `_artistic` alone is what
# `cmap_image`/`paint`/`GLVolumeItem.setData` actually consume.
_CUT_ACTIONS = {
    "cut_remove_half_action",
    "cut_remove_left_half_action",
    "cut_remove_top_half_action",
    "cut_remove_bottom_half_action",
    "cut_remove_front_half_action",
    "cut_remove_back_half_action",
    "cut_remove_quarter_action",
    "cut_remove_eighth_action",
}


def render_gl(vol: Volume, *, label=None, cmap="gray", overlay: bool = False,
              alpha: float = 1.0, image_intensity: float = None,
              bgcolor=None, axis: bool = False, grid: bool = False,
              threshold: float = 0, cut: str = None,
              distance=None, elevation: float = 30, azimuth: float = 30,
              view: str = None,
              width: int = 800, height: int = 600) -> np.ndarray:
    """
    Render ``vol`` headlessly with MELAGE's real OpenGL volume ray-caster
    (the same renderer behind the GUI's 3-D viewer) and return the result
    as an HxWx3 uint8 RGB array.

    Parameters
    ----------
    vol : Volume
    label : int | sequence[int], optional
        Restrict rendering/colouring to one or more segmentation labels.
        If omitted, every non-zero label is shown (``overlay=True``) or the
        full segmentation / raw image is shown as-is (``overlay=False``).
    cmap : str
        Matplotlib colormap name used for non-overlay (image / full-segmentation)
        rendering — passed straight through to ``glScientific.cmap_image``.
        Ignored when ``overlay=True``.
    overlay : bool
        If True, render the grayscale image with the (coloured) segmentation
        blended on top — MELAGE's "Show Img+Seg" mode (``glScientific.paint`` /
        ``updateSegVolItem``), the same alpha-composited overlay the GUI's 3-D
        view shows. Requires ``vol.segmentation``. If False (default), render
        either the selected label(s)/segmentation or the raw image alone via
        ``cmap_image``.
    alpha : float
        ``overlay=True`` only. Opacity of the segmentation overlay (0 =
        invisible / image only, 1 = fully opaque, the default) — scales the
        alpha channel of each label's colour before blending.
    image_intensity : float, optional
        ``overlay=True`` only. Per-voxel alpha weight for background image
        voxels in the 3-D volumetric composite (``glScientific.intensityImg``,
        0-1, default 0.1). Lower values keep outer/non-segmented voxels nearly
        transparent so the inner segmentation overlay dominates; higher values
        make those voxels more opaque, which occludes the inner structure and
        darkens the result overall. The default (0.1) is MELAGE's tuned balance
        — most users should leave this alone.
    bgcolor : sequence of 4 floats, optional
        Background colour as ``(r, g, b, a)`` in 0-1, e.g. ``(1, 1, 1, 1)``
        for white — same values as the GUI's right-click "Background Color"
        menu (``glScientific.changeBG``). Defaults to MELAGE's standard
        near-black ``(0.05, 0.05, 0.05, 1)``.
    axis : bool
        Show the 3-D axis triad (``glScientific.axis_status`` / the GUI's
        "Axis" toggle). Default off.
    grid : bool
        Show the background grid (``glScientific.grid_status`` / the GUI's
        "Grid" toggle). Default off.
    threshold : float
        Voxel-intensity cutoff in 0-100 (percent of the data's max), below
        which voxels are masked out — same as the 3-D toolbar's threshold
        slider (``glScientific._threshold``). Default 0 (no masking).
    cut : str, optional
        Apply one of the GUI's "Clipping (Cut)" actions to remove part of
        the volume: ``"cut_remove_half_action"`` (right half),
        ``"cut_remove_left_half_action"``, ``"cut_remove_top_half_action"``,
        ``"cut_remove_bottom_half_action"``, ``"cut_remove_front_half_action"``,
        ``"cut_remove_back_half_action"``, ``"cut_remove_quarter_action"``
        (top-right quarter), or ``"cut_remove_eighth_action"`` (top-right-front
        eighth). Default ``None`` (no clipping).
    distance, elevation, azimuth : float, optional
        Camera controls forwarded to ``glScientific.setCameraPosition``.
        ``distance`` is auto-computed to fit the whole volume in frame
        (from its bounding-box diagonal and field of view) when omitted —
        pass an explicit value to zoom in/out from that framing.
    view : str, optional
        Standard anatomical view, overriding ``elevation``/``azimuth``:
        ``"axial"`` (elevation=90, azimuth=90), ``"sagittal"``
        (elevation=0, azimuth=0), or ``"coronal"`` (elevation=0,
        azimuth=90).
    width, height : int
        Output image size in pixels.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 RGB image.
    """
    viewer = _get_viewer(width, height)

    # Reset all toolbar-equivalent state explicitly on every call — the
    # singleton viewer otherwise carries state over from a previous render.
    viewer.opts['bgcolor'] = list(bgcolor) if bgcolor is not None else [0.05, 0.05, 0.05, 1]
    viewer.axis_action.setChecked(bool(axis))
    viewer.grid_action.setChecked(bool(grid))
    viewer._threshold = float(np.clip(threshold, 0, 100))
    if cut is not None and cut not in _CUT_ACTIONS:
        raise ValueError(f"cut must be one of {sorted(_CUT_ACTIONS)} or None, got {cut!r}")
    viewer._artistic = cut if cut is not None else False

    if view is not None:
        if view not in _VIEW_PRESETS:
            raise ValueError(f"view must be one of {sorted(_VIEW_PRESETS)} or None, got {view!r}")
        elevation, azimuth = _VIEW_PRESETS[view]

    if overlay:
        if vol.segmentation is None:
            raise ValueError("overlay=True requires a segmentation — run melage.segment.* first")

        from melage.api._visualize import _color_for
        from melage.utils.utils import detect_modality_and_window

        seg = np.asarray(vol.segmentation).astype(np.int32)
        if label is None:
            labels = sorted(int(l) for l in np.unique(seg) if l != 0)
            viewer.colorInds = [9876]  # sentinel meaning "show every label" (see updateSegVolItem)
        else:
            labels = [int(label)] if np.isscalar(label) else [int(l) for l in label]
            viewer.colorInds = labels
        a = float(np.clip(alpha, 0.0, 1.0))
        viewer.colorsCombinations = {l: (*_color_for(l)[:3], a) for l in labels}
        # Always reset intensityImg so a prior call with a non-default value
        # doesn't bleed into this call via the singleton viewer's state.
        viewer.intensityImg = float(image_intensity) if image_intensity is not None else 0.1

        # `updateSegVolItem` blends a *uint8* grayscale image with the coloured
        # segmentation (it assumes 0-255 input, see glScientific.py:1651/1669) —
        # `detect_modality_and_window` is the same modality-aware normalisation
        # MELAGE uses to build `reader.npImage` for display.
        img_u8 = detect_modality_and_window(np.asarray(vol.data, dtype=np.float64))

        viewer.createGridAxis(list(seg.shape))
        viewer._renderMode = 'Seg'
        viewer.im_seg_action.setChecked(True)
        viewer.seg_action.setChecked(False)
        viewer.paint(seg, img_u8)
        shape = seg.shape
    else:
        data, labels = _select_data(vol, label)
        viewer.colorsCombinations = {l: (0.85, 0.20, 0.20, 1.0) for l in (labels or [])}
        viewer.createGridAxis(list(data.shape))
        # NOTE: "original" passes the data through untouched (RGBA = the raw
        # values), which only makes sense for already-intensity-scaled images —
        # binary/label masks (values like 0/1) end up at alpha ~= 1/255 and are
        # invisible. A real colormap scales by `_image / _image.max()`, which is
        # what we want for both label masks and raw image data.
        # `cmap_image(reset=True)` always resets `_threshold` to 0 as a side
        # effect (mirroring the GUI's "new colormap" behaviour) — re-apply
        # ours and recompute with `reset=False`, like the GUI's threshold
        # slider does (`threshold_change` -> `update_cmap_image(..., reset=False)`).
        viewer.cmap_image(data, cmap, reset=True)
        thresh_val = float(np.clip(threshold, 0, 100))
        if thresh_val:
            viewer._threshold = thresh_val
            viewer.cmap_image(data, cmap, reset=False)
        shape = data.shape

    if distance is None:
        # `createGridAxis` seeds opts['distance'] with max(shape) — tuned for
        # the GUI, where the user can zoom/pan interactively. For a single
        # static capture that's too close: the volume overflows the view
        # frustum and gets clipped at the frame edges (looks off-centre, with
        # chunks "missing" at top/bottom). Auto-frame instead from the
        # bounding-box diagonal — the worst-case extent across all rotations —
        # and the viewer's field of view, with a small margin so the volume
        # comfortably fits regardless of elevation/azimuth.
        diagonal = float(np.linalg.norm(shape))
        fov = viewer.opts['fov']
        distance = (diagonal / 2) / np.tan(np.radians(fov / 2)) * 1.15

    viewer.setCameraPosition(elevation=elevation, azimuth=azimuth, distance=distance)
    return _qimage_to_array(viewer.capture(width, height))


def screenshot_gl(vol: Volume, path: str, **render_kwargs) -> None:
    """
    Render ``vol`` headlessly via OpenGL and save the result directly to
    ``path`` (PNG, JPG, …).

    Examples
    --------
        melage.visualize.screenshot(vol, "brain_3d.png", label=1, engine="gl")
    """
    from matplotlib.image import imsave

    imsave(path, render_gl(vol, **render_kwargs))
