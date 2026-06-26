"""
melage.api._visualize
=====================
Lightweight, headless 3-D visualisation helpers.

MELAGE's interactive 3-D viewer (``melage.rendering.glScientific``) is built
on PyQt5 / OpenGL and needs a live display. These helpers extract surface
meshes from a :class:`~melage.api.Volume`'s segmentation with scikit-image's
marching cubes (the same family of techniques used elsewhere in MELAGE, e.g.
the BET plugin's mesh export) and plot them with matplotlib — so a quick 3-D
view works in scripts, servers, and Jupyter notebooks without Qt/OpenGL.

Quick start
-----------
    import melage

    vol = melage.load("brain.nii.gz")
    vol = melage.segment.bet(vol)

    fig = melage.visualize.render(vol, label=1, title="Brain surface")
    melage.visualize.screenshot(vol, "brain_3d.png", label=1)
    melage.visualize.export_mesh(vol, "brain.stl", label=1)
"""

from __future__ import annotations

import warnings

import numpy as np

from melage.api._volume import Volume

# Default label -> RGBA palette, mirroring MELAGE's standard label colours.
_DEFAULT_COLORS = {
    1: (0.85, 0.20, 0.20, 1.0),
    2: (0.20, 0.65, 0.85, 1.0),
    3: (0.30, 0.80, 0.30, 1.0),
    4: (0.95, 0.75, 0.10, 1.0),
    5: (0.65, 0.35, 0.85, 1.0),
}


def _color_for(label: int):
    return _DEFAULT_COLORS.get(int(label), (0.7, 0.7, 0.7, 1.0))


def mesh(vol: Volume, label: int = 1, *, level: float = 0.5,
         step_size: int = 1, smooth: bool = False, smooth_iterations: int = 10):
    """
    Extract a surface mesh for one segmentation label via marching cubes.

    Parameters
    ----------
    vol : Volume
        Must have ``vol.segmentation`` set (e.g. from ``melage.segment.bet``).
    label : int
        Segmentation label to extract a surface for (default 1).
    level : float
        Iso-surface threshold for marching cubes (default 0.5 = mask boundary).
    step_size : int
        Marching-cubes step size; raise it for a coarser/faster mesh.
    smooth : bool
        Apply Taubin smoothing to the extracted surface.
    smooth_iterations : int
        Number of smoothing iterations when ``smooth=True``.

    Returns
    -------
    trimesh.Trimesh
        Surface mesh with vertices in world (mm) coordinates.
    """
    import trimesh
    from skimage import measure

    if vol.segmentation is None:
        raise ValueError("Volume has no segmentation — run melage.segment.* first")

    binary = (vol.segmentation == int(label))
    if not np.any(binary):
        raise ValueError(f"Label {label} is empty in this segmentation")

    verts, faces, normals, _ = measure.marching_cubes(
        binary.astype(np.float32), level=level, step_size=step_size,
    )

    # voxel -> world (mm) coordinates via the image affine
    verts_h = np.column_stack([verts, np.ones(len(verts))])
    verts_world = (verts_h @ vol.affine.T)[:, :3]

    m = trimesh.Trimesh(vertices=verts_world, faces=faces,
                        vertex_normals=normals, process=False)
    if smooth:
        trimesh.smoothing.filter_taubin(m, iterations=smooth_iterations)
    return m


def export_mesh(vol: Volume, path: str, label: int = 1, **mesh_kwargs) -> None:
    """
    Extract a surface mesh for ``label`` and write it to disk.

    The format is inferred from the file extension (``.stl``, ``.obj``,
    ``.ply``, ``.glb`` … — anything trimesh supports).

    Examples
    --------
        melage.visualize.export_mesh(vol, "brain.stl", label=1, smooth=True)
    """
    mesh(vol, label=label, **mesh_kwargs).export(path)


def _render_gl_figure(vol, *, label, cmap, overlay, alpha, image_intensity,
                      bgcolor, axis, grid, threshold, cut,
                      distance, elevation, azimuth, view,
                      width, height, figsize, title):
    """Render via the real OpenGL viewer (offscreen) and wrap the result in
    a matplotlib Figure, so engine="gl" is a drop-in for engine="mesh"."""
    import matplotlib.pyplot as plt
    from melage.api._render3d import render_gl

    img = render_gl(vol, label=label, cmap=cmap, overlay=overlay, alpha=alpha,
                    image_intensity=image_intensity, bgcolor=bgcolor, axis=axis,
                    grid=grid, threshold=threshold, cut=cut, distance=distance,
                    elevation=elevation, azimuth=azimuth, view=view,
                    width=width, height=height)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig


def render(vol: Volume, *, label=1, color=None, alpha: float = 1.0,
           ax=None, figsize=(6, 6), elev: float = 20, azim: float = -60,
           title: str = None, engine: str = "mesh", cmap: str = "gray",
           overlay: bool = False, image_intensity: float = None,
           bgcolor=None, axis: bool = False, grid: bool = False,
           threshold: float = 0, cut: str = None, view: str = None,
           distance: float = None, width: int = 800, height: int = 600,
           **mesh_kwargs):
    """
    Render a quick static 3-D view of one or more segmentation labels — a
    lightweight way to look at a volume in scripts, servers, and notebooks
    without the GUI.

    Two engines are available:

    - ``engine="mesh"`` (default): extracts a surface mesh via marching
      cubes and plots it with matplotlib. No Qt/OpenGL/GPU required —
      always available, the safe default for servers and CI.
    - ``engine="gl"``: renders with MELAGE's real OpenGL volume ray-caster
      (:class:`~melage.rendering.glScientific.glScientific`, the same
      renderer behind the GUI's 3-D viewer), driven offscreen via Qt's
      ``offscreen`` platform plugin. Requires PyQt5 + a working OpenGL
      driver; falls back to ``engine="mesh"`` (with a warning) if that
      isn't available.

    Parameters
    ----------
    vol : Volume
    label : int | sequence[int]
        One label, or a list of labels to render together.
    color : RGBA tuple | sequence[RGBA tuple], optional
        Colour(s) in 0–1 range (``engine="mesh"`` only). Defaults to
        MELAGE's standard label palette.
    alpha : float
        Surface opacity (``engine="mesh"``), or — when ``engine="gl"`` and
        ``overlay=True`` — the segmentation overlay's opacity (0 = invisible/
        image-only, 1 = fully opaque, the default).
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Existing 3-D axes to draw into (``engine="mesh"`` only); a new
        figure is created if omitted.
    figsize : tuple
    elev, azim : float
        Camera elevation / azimuth in degrees (both engines).
    title : str, optional
    engine : "mesh" | "gl"
    cmap : str
        Matplotlib colormap name used by ``engine="gl"`` for non-overlay
        (image / full-segmentation) rendering. Ignored when ``overlay=True``.
    overlay : bool
        ``engine="gl"`` only. Render the grayscale image with the (coloured)
        segmentation alpha-blended on top — MELAGE's "Show Img+Seg" mode,
        the same overlay the GUI's 3-D view uses. Requires a segmentation.
    image_intensity : float, optional
        ``engine="gl"`` + ``overlay=True`` only. Per-voxel alpha weight for
        background image voxels in the 3-D volumetric composite
        (``glScientific.intensityImg``, 0-1, default 0.1). Higher values make
        outer/non-segmented voxels more opaque, which occludes inner structure
        and darkens the result — counter-intuitively the default (0.1) gives the
        most vivid overlay. Leave this at its default unless you intentionally
        want to dim/suppress the background volume.
    bgcolor : sequence of 4 floats, optional
        ``engine="gl"`` only. Background colour as ``(r, g, b, a)`` in 0-1 —
        same as the GUI's right-click "Background Color" menu. Defaults to
        MELAGE's standard near-black ``(0.05, 0.05, 0.05, 1)``.
    axis : bool
        ``engine="gl"`` only. Show the 3-D axis triad (the GUI's "Axis"
        toggle). Default off.
    grid : bool
        ``engine="gl"`` only. Show the background grid (the GUI's "Grid"
        toggle). Default off.
    threshold : float
        ``engine="gl"`` only. Voxel-intensity cutoff in 0-100 (percent of
        the data's max), below which voxels are masked out — same as the
        3-D toolbar's threshold slider. Default 0 (no masking).
    cut : str, optional
        ``engine="gl"`` only. Apply one of the GUI's "Clipping (Cut)"
        actions, e.g. ``"cut_remove_half_action"`` (right half),
        ``"cut_remove_left_half_action"``, ``"cut_remove_top_half_action"``,
        ``"cut_remove_bottom_half_action"``, ``"cut_remove_front_half_action"``,
        ``"cut_remove_back_half_action"``, ``"cut_remove_quarter_action"``,
        ``"cut_remove_eighth_action"``. Default ``None`` (no clipping).
    view : str, optional
        ``engine="gl"`` only. Standard anatomical view, overriding
        ``elev``/``azim``: ``"axial"``, ``"sagittal"``, or ``"coronal"``.
    distance : float, optional
        Camera distance for ``engine="gl"``. Auto-computed to fit the whole
        volume in frame when omitted — pass a value to zoom in/out from there.
    width, height : int
        Output pixel size for ``engine="gl"``.
    **mesh_kwargs
        Forwarded to :func:`mesh` (``level``, ``step_size``, ``smooth`` …),
        ``engine="mesh"`` only.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if engine == "gl":
        try:
            return _render_gl_figure(vol, label=label, cmap=cmap, overlay=overlay, alpha=alpha,
                                     image_intensity=image_intensity, bgcolor=bgcolor, axis=axis,
                                     grid=grid, threshold=threshold, cut=cut, distance=distance,
                                     elevation=elev, azimuth=azim, view=view, width=width, height=height,
                                     figsize=figsize, title=title)
        except Exception as exc:
            warnings.warn(
                f"melage.visualize: engine='gl' unavailable ({exc!r}); "
                "falling back to engine='mesh' (marching-cubes + matplotlib).",
                stacklevel=2,
            )

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    labels = [label] if np.isscalar(label) else list(label)
    if color is None:
        colors = [_color_for(lab) for lab in labels]
    elif np.isscalar(color[0]):
        colors = [color] * len(labels)
    else:
        colors = list(color)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    all_verts = []
    for lab, col in zip(labels, colors):
        m = mesh(vol, label=lab, **mesh_kwargs)
        coll = Poly3DCollection(m.vertices[m.faces], alpha=alpha)
        coll.set_facecolor(col)
        ax.add_collection3d(coll)
        all_verts.append(m.vertices)

    pts = np.vstack(all_verts)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    if title:
        ax.set_title(title)

    return fig


def screenshot(vol: Volume, path: str, **render_kwargs) -> None:
    """
    Render a 3-D view and save it straight to an image file (PNG, JPG …).

    Examples
    --------
        melage.visualize.screenshot(vol, "brain_3d.png", label=1, elev=10)
    """
    import matplotlib.pyplot as plt

    fig = render(vol, **render_kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
