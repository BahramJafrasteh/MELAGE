"""
MELAGE Segmentation Report Generator
Produces a self-contained HTML file with embedded images.
"""
from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _png_to_b64(fig) -> str:
    """Convert a matplotlib Figure to a base-64 PNG data-URI."""
    import matplotlib
    matplotlib.use('Agg')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _array_to_b64(arr_uint8: np.ndarray) -> str:
    """Convert an HxW or HxWx3 uint8 array to a base-64 PNG data-URI."""
    from PIL import Image
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _label_names_from_color_name(color_name_list: list) -> dict:
    """
    Parse self.color_name list (format: '<index>_<name>') into {index: name}.
    9876 → 'Combined' is the catch-all; everything else is a structure name.
    """
    out = {}
    for cn in color_name_list:
        parts = cn.split('_', 1)
        try:
            idx = int(float(parts[0]))
            name = parts[1].replace('_', ' ') if len(parts) > 1 else str(idx)
            out[idx] = name
        except (ValueError, IndexError):
            pass
    return out


def _color_for_label(label_idx: int, colors_combinations: dict) -> str:
    """Return a CSS rgb() string for a label index."""
    rgb = colors_combinations.get(label_idx)
    if rgb and len(rgb) >= 3:
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        return f"rgb({r},{g},{b})"
    return "rgb(200,200,200)"


# ──────────────────────────────────────────────────────────────────────────────
# 3-D / neuroimaging stats
# ──────────────────────────────────────────────────────────────────────────────

def _seg_stats_3d(npSeg: np.ndarray,
                  npImage: np.ndarray,
                  spacing: np.ndarray,
                  label_ids: list) -> list[dict]:
    """
    Compute per-label statistics for a 3-D segmentation.
    spacing: [sx, sy, sz] in mm.
    Returns a list of stat dicts.
    """
    voxel_vol_mm3 = float(np.prod(spacing))
    rows = []
    for lid in label_ids:
        mask = npSeg == lid
        n = int(mask.sum())
        if n == 0:
            continue
        vol_mm3 = n * voxel_vol_mm3
        vol_cm3 = vol_mm3 / 1000.0

        # Centre of mass in voxel space, then convert to mm
        coords = np.argwhere(mask)
        com_vox = coords.mean(axis=0)
        com_mm = com_vox * spacing

        # Bounding box (voxel indices)
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        bbox_str = (f"[{mn[0]}–{mx[0]}, {mn[1]}–{mx[1]}, {mn[2]}–{mx[2]}]")
        bbox_size = (mx - mn + 1) * spacing

        # Mean intensity in the ROI
        mean_int = float(npImage[mask].mean()) if npImage is not None else float('nan')

        rows.append({
            'label_id':  lid,
            'n_voxels':  n,
            'vol_mm3':   vol_mm3,
            'vol_cm3':   vol_cm3,
            'com_mm':    com_mm,
            'bbox':      bbox_str,
            'bbox_size': bbox_size,
            'mean_int':  mean_int,
        })
    return rows


_CANVAS = 320  # pixels — all three slices are padded/resized to this square


def _to_canvas(arr2d: np.ndarray,
               phys_h: float, phys_w: float,
               target: int = _CANVAS) -> np.ndarray:
    """
    Resize a 2-D uint8 array so it fits inside a target×target square while
    preserving the physical aspect ratio (phys_h × phys_w in mm), then
    centre-pad with black to make it exactly target×target.
    Returns a uint8 array of shape (target, target).
    """
    from PIL import Image as _PILImage

    phys_h = max(phys_h, 1e-6)
    phys_w = max(phys_w, 1e-6)
    aspect = phys_h / phys_w          # > 1 means taller than wide

    if aspect >= 1:
        new_h = target
        new_w = max(1, round(target / aspect))
    else:
        new_w = target
        new_h = max(1, round(target * aspect))

    resized = np.array(
        _PILImage.fromarray(arr2d.astype(np.uint8)).resize(
            (new_w, new_h), _PILImage.BILINEAR))

    canvas = np.zeros((target, target), dtype=np.uint8)
    r0 = (target - new_h) // 2
    c0 = (target - new_w) // 2
    canvas[r0:r0 + new_h, c0:c0 + new_w] = resized
    return canvas


def _seg_to_canvas(seg2d: np.ndarray,
                   phys_h: float, phys_w: float,
                   target: int = _CANVAS) -> np.ndarray:
    """Same resize/pad logic for an integer segmentation mask (nearest-neighbour)."""
    from PIL import Image as _PILImage

    phys_h = max(phys_h, 1e-6)
    phys_w = max(phys_w, 1e-6)
    aspect = phys_h / phys_w

    if aspect >= 1:
        new_h = target
        new_w = max(1, round(target / aspect))
    else:
        new_w = target
        new_h = max(1, round(target * aspect))

    resized = np.array(
        _PILImage.fromarray(seg2d.astype(np.int16)).resize(
            (new_w, new_h), _PILImage.NEAREST))

    canvas = np.zeros((target, target), dtype=np.int16)
    r0 = (target - new_h) // 2
    c0 = (target - new_w) // 2
    canvas[r0:r0 + new_h, c0:c0 + new_w] = resized
    return canvas


def _slice_preview_3d(npImage: np.ndarray,
                      npSeg: np.ndarray,
                      colors_combinations: dict,
                      spacing: Optional[np.ndarray] = None) -> str:
    """
    Render three orthogonal slices with segmentation overlay.

    Axis convention (from MELAGE's getCurrentSlice / read_pars):
      axis 0 → Coronal  (npImage[i, :, :])
      axis 1 → Sagittal (npImage[:, i, :])
      axis 2 → Axial    (npImage[:, :, i])

    All three panels are rendered at the same canvas size (_CANVAS × _CANVAS px)
    so they appear identical in the report regardless of the original voxel grid.
    Physical aspect ratio is preserved inside each canvas; unused area is black.

    origin='upper' — standard clinical/radiological report orientation.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    D, H, W = npImage.shape

    # Voxel spacing along each axis (sx, sy, sz)
    if spacing is not None and len(spacing) >= 3:
        sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    else:
        sx = sy = sz = 1.0

    # Centre-of-mass slice selection
    #   axis 0 → Axial    (npImage[i, :, :])
    #   axis 1 → Coronal  (npImage[:, i, :])
    #   axis 2 → Sagittal (npImage[:, :, i])
    seg_coords = np.argwhere(npSeg > 0)
    if seg_coords.size:
        com = seg_coords.mean(axis=0).astype(int)
        idx_ax  = int(np.clip(com[0], 0, D - 1))
        idx_cor = int(np.clip(com[1], 0, H - 1))
        idx_sag = int(np.clip(com[2], 0, W - 1))
    else:
        idx_ax, idx_cor, idx_sag = D // 2, H // 2, W // 2

    # Physical sizes in mm:
    #   Axial    slice (H rows, W cols) → physical (H*sy) × (W*sz)
    #   Coronal  slice (D rows, W cols) → physical (D*sx) × (W*sz)
    #   Sagittal slice (D rows, H cols) → physical (D*sx) × (H*sy)
    planes = [
        (npImage[idx_ax,  :, :], npSeg[idx_ax,  :, :],
         f'Axial    [{idx_ax}]',  H * sy, W * sz,  ('A', 'A', 'P', 'P')),
        (npImage[:, idx_cor, :], npSeg[:, idx_cor, :],
         f'Coronal  [{idx_cor}]', D * sx, W * sz,  ('S', 'S', 'I', 'I')),
        (npImage[:, :, idx_sag], npSeg[:, :, idx_sag],
         f'Sagittal [{idx_sag}]', D * sx, H * sy,  ('S', 'S', 'I', 'I')),
    ]

    # Resize every slice to the same _CANVAS × _CANVAS square
    canvases = []
    for sl, seg_sl, title, ph, pw, corners in planes:
        img_c = _to_canvas(sl,     ph, pw)
        seg_c = _seg_to_canvas(seg_sl, ph, pw)
        canvases.append((img_c, seg_c, title, corners))

    # All three subplots now receive identical-sized arrays → equal display size
    px = _CANVAS / 80          # figure size in inches for this resolution
    fig, axes = plt.subplots(1, 3, figsize=(px * 3 + 1.2, px + 0.8))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, (img_c, seg_c, title, corners) in zip(axes, canvases):
        ax.imshow(img_c, cmap='gray', origin='upper',
                  vmin=0, vmax=255, aspect='equal')

        for lid in np.unique(seg_c):
            if lid == 0:
                continue
            rgb = colors_combinations.get(int(lid))
            r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2])) \
                      if (rgb and len(rgb) >= 3) else (1.0, 0.2, 0.2)
            overlay = np.zeros((*seg_c.shape, 4), dtype=float)
            overlay[seg_c == lid] = [r, g, b, 0.45]
            ax.imshow(overlay, origin='upper', aspect='equal')

        ax.set_title(title, color='#b0c4de', fontsize=10, pad=4)
        ax.axis('off')

        kw = dict(color='#ffd700', fontsize=8, fontweight='bold',
                  transform=ax.transAxes)
        ax.text(0.02, 0.97, corners[0], va='top',    ha='left',  **kw)
        ax.text(0.98, 0.97, corners[1], va='top',    ha='right', **kw)
        ax.text(0.02, 0.03, corners[2], va='bottom', ha='left',  **kw)
        ax.text(0.98, 0.03, corners[3], va='bottom', ha='right', **kw)

    fig.tight_layout(pad=0.5)
    uri = _png_to_b64(fig)
    plt.close(fig)
    return uri


def _intensity_histogram(npImage: np.ndarray, npSeg: np.ndarray) -> str:
    """
    Plot the intensity histogram of the full image and per-label distributions.
    Returns a base-64 PNG data-URI.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d0d1a')

    ax.hist(npImage.ravel(), bins=128, color='#4fc3f7', alpha=0.7,
            label='Full image', density=True)

    unique_ids = [l for l in np.unique(npSeg) if l != 0]
    palette = ['#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#74c0fc']
    for i, lid in enumerate(unique_ids[:5]):
        vals = npImage[npSeg == lid]
        if vals.size > 10:
            ax.hist(vals, bins=64,
                    color=palette[i % len(palette)], alpha=0.6,
                    label=f'Label {lid}', density=True)

    ax.set_xlabel('Intensity', color='white')
    ax.set_ylabel('Density', color='white')
    ax.set_title('Intensity distribution', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    uri = _png_to_b64(fig)
    plt.close(fig)
    return uri


# ──────────────────────────────────────────────────────────────────────────────
# Video / photo stats
# ──────────────────────────────────────────────────────────────────────────────

def _seg_stats_video(seg_proxy,
                     total_frames: int,
                     pixel_area_mm2: float = 1.0) -> dict:
    """
    Compute per-label area (in pixels²) for each frame.
    Returns {label_id: np.ndarray of length total_frames}.
    """
    areas: dict[int, list[float]] = {}
    for f in range(total_frames):
        try:
            frame_seg = seg_proxy.get_frame(f)
        except Exception:
            continue
        for lid in np.unique(frame_seg):
            if lid == 0:
                continue
            areas.setdefault(lid, [0.0] * total_frames)
            areas[lid][f] = float((frame_seg == lid).sum()) * pixel_area_mm2
    return {lid: np.array(v) for lid, v in areas.items()}


def _area_timeseries_plot(area_dict: dict,
                           label_names: dict,
                           colors_combinations: dict) -> str:
    """
    Plot area vs. frame for each label.
    Returns a base-64 PNG data-URI.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d0d1a')

    for lid, areas in area_dict.items():
        rgb = colors_combinations.get(lid)
        if rgb and len(rgb) >= 3:
            color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        else:
            color = None
        name = label_names.get(lid, f'Label {lid}')
        ax.plot(areas, label=name, color=color, linewidth=1.5)

    ax.set_xlabel('Frame', color='white')
    ax.set_ylabel('Area (px²)', color='white')
    ax.set_title('Segmented area per frame', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    uri = _png_to_b64(fig)
    plt.close(fig)
    return uri


def _video_frame_previews(image_proxy,
                            seg_proxy,
                            frame_indices: list,
                            colors_combinations: dict) -> list[str]:
    """
    Capture a set of frames with segmentation overlay.
    Returns a list of base-64 PNG data-URIs.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    uris = []
    for f in frame_indices:
        try:
            frame_img = image_proxy.get_frame(f)
            frame_seg = seg_proxy.get_frame(f)
        except Exception:
            continue

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.imshow(frame_img, cmap='gray' if frame_img.ndim == 2 else None,
                  origin='upper')

        for lid in np.unique(frame_seg):
            if lid == 0:
                continue
            rgb = colors_combinations.get(lid)
            if rgb and len(rgb) >= 3:
                r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            else:
                r, g, b = 1.0, 0.2, 0.2
            h, w = frame_seg.shape[:2]
            overlay = np.zeros((h, w, 4), dtype=float)
            overlay[frame_seg == lid] = [r, g, b, 0.45]
            ax.imshow(overlay, origin='upper')

        ax.set_title(f'Frame {f}', color='white', fontsize=9)
        ax.axis('off')
        fig.tight_layout(pad=0.1)
        uris.append(_png_to_b64(fig))
        plt.close(fig)
    return uris


# ──────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
  :root { --bg: #1a1a2e; --card: #16213e; --accent: #4fc3f7;
          --text: #e0e0e0; --muted: #888; --border: #2a3a5e; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
         padding: 32px; line-height: 1.6; }
  h1 { color: var(--accent); font-size: 24px; margin-bottom: 4px; }
  h2 { color: var(--accent); font-size: 17px; margin: 28px 0 10px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  h3 { color: #90caf9; font-size: 14px; margin: 16px 0 6px; }
  .subtitle { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; }
  .meta-item { font-size: 13px; }
  .meta-item span { color: var(--muted); display: inline-block; min-width: 140px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th { background: #0d1b3e; color: var(--accent); padding: 8px 10px; text-align: left; font-weight: 600; }
  td { padding: 7px 10px; border-bottom: 1px solid var(--border); }
  tr:nth-child(even) td { background: rgba(255,255,255,0.03); }
  .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  .img-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
  .img-row img { border-radius: 6px; border: 1px solid var(--border); max-width: 100%; }
  .full-img { width: 100%; border-radius: 6px; border: 1px solid var(--border); }
  footer { margin-top: 40px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--border); padding-top: 10px; }
</style>
"""


def _measurements_table_html(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return '<p style="color:#666">No measurements recorded.</p>'
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"


def _meta_item(label: str, value: str) -> str:
    return f'<div class="meta-item"><span>{label}</span> {value}</div>'


# ──────────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────────

def generate_neuroimaging_report(reader,
                                  filename: str,
                                  color_name_list: list,
                                  colors_combinations: dict,
                                  measurements: list[list[str]],
                                  output_path: str) -> str:
    """
    Generate a report for a 3-D neuroimaging dataset (MRI/CT/NIfTI/DICOM).

    Parameters
    ----------
    reader : readData instance with .npImage, .npSeg, .ImSpacing, .im
    filename : source filename
    color_name_list : self.color_name (list of '<idx>_<name>' strings)
    colors_combinations : self.colorsCombinations
    measurements : list of row lists from the measurements table
    output_path : where to write the HTML file
    """
    label_names = _label_names_from_color_name(color_name_list)
    spacing = np.array(reader.ImSpacing[:3], dtype=float)
    npSeg   = reader.npSeg
    npImage = reader.npImage

    label_ids = sorted(int(l) for l in np.unique(npSeg) if l != 0)

    # ── Compute stats ──
    stats = _seg_stats_3d(npSeg, npImage, spacing, label_ids)

    # ── Render visuals ──
    slice_uri  = _slice_preview_3d(npImage, npSeg, colors_combinations, spacing)
    hist_uri   = _intensity_histogram(npImage, npSeg)

    # ── Image metadata ──
    shape = npImage.shape
    try:
        modality = str(reader.im.header.get('intent_name', 'MRI')).strip() or 'MRI'
    except Exception:
        modality = 'MRI'
    try:
        orient = ' '.join(reader.ImDirection)
    except Exception:
        orient = 'Unknown'

    # ── Build HTML ──
    meas_headers = ['Description', 'Type', 'Measure 1', 'Measure 2',
                    'Slice', 'Plane', 'CenterXY', 'File']
    meas_html = _measurements_table_html(measurements, meas_headers)

    stat_rows = ""
    for s in stats:
        lid   = s['label_id']
        name  = label_names.get(lid, f'Label {lid}')
        dot   = f'<span class="dot" style="background:{_color_for_label(lid, colors_combinations)}"></span>'
        com   = f"{s['com_mm'][0]:.1f}, {s['com_mm'][1]:.1f}, {s['com_mm'][2]:.1f}"
        bbox_sz = f"{s['bbox_size'][0]:.1f} × {s['bbox_size'][1]:.1f} × {s['bbox_size'][2]:.1f} mm"
        stat_rows += (
            f"<tr>"
            f"<td>{dot}{name}</td>"
            f"<td>{s['n_voxels']:,}</td>"
            f"<td>{s['vol_mm3']:.1f}</td>"
            f"<td>{s['vol_cm3']:.3f}</td>"
            f"<td>{com}</td>"
            f"<td>{s['bbox']}</td>"
            f"<td>{bbox_sz}</td>"
            f"<td>{s['mean_int']:.1f}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>MELAGE Report — {os.path.basename(filename)}</title>
{_CSS}</head>
<body>
<h1>MELAGE Segmentation Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; Neuroimaging</p>

<h2>Image Information</h2>
<div class="card">
  <div class="meta-grid">
    {_meta_item('File', os.path.basename(filename))}
    {_meta_item('Modality', modality)}
    {_meta_item('Dimensions (D×H×W)', f'{shape[0]} × {shape[1]} × {shape[2]}')}
    {_meta_item('Voxel spacing (mm)', f'{spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f}')}
    {_meta_item('Orientation', orient)}
    {_meta_item('Labels found', str(len(label_ids)))}
  </div>
</div>

<h2>Slice Previews</h2>
<img class="full-img" src="{slice_uri}" alt="Slice previews"/>

<h2>Segmentation Statistics</h2>
<div class="card">
<table>
  <thead><tr>
    <th>Structure</th><th>Voxels</th><th>Volume (mm³)</th><th>Volume (cm³)</th>
    <th>Centre of mass (mm)</th><th>Bounding box</th><th>Extent</th><th>Mean intensity</th>
  </tr></thead>
  <tbody>{stat_rows}</tbody>
</table>
</div>

<h2>Intensity Histogram</h2>
<img class="full-img" src="{hist_uri}" alt="Intensity histogram"/>

<h2>Measurements Table</h2>
<div class="card">{meas_html}</div>

<footer>MELAGE v2.1.1 &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d')}</footer>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


def generate_video_report(reader,
                           filename: str,
                           color_name_list: list,
                           colors_combinations: dict,
                           measurements: list[list[str]],
                           output_path: str) -> str:
    """
    Generate a report for a video / photo segmentation dataset.

    Parameters
    ----------
    reader : readData instance with .isChunkedVideo, .video_im, .seg_ims
    filename : source filename
    color_name_list : self.color_name
    colors_combinations : self.colorsCombinations
    measurements : list of row lists
    output_path : where to write the HTML file
    """
    label_names = _label_names_from_color_name(color_name_list)

    image_proxy = reader.video_im
    seg_proxy   = reader.seg_ims

    total_frames = int(image_proxy.frames)
    height = getattr(image_proxy, 'height', '?')
    width  = getattr(image_proxy, 'width',  '?')

    # ── Area stats per frame ──
    area_dict = _seg_stats_video(seg_proxy, total_frames)

    # ── Summary stats ──
    summary = {}
    for lid, areas in area_dict.items():
        nonzero = areas[areas > 0]
        summary[lid] = {
            'frames_present': int((areas > 0).sum()),
            'mean_area':      float(nonzero.mean()) if nonzero.size else 0.0,
            'max_area':       float(areas.max()),
            'frame_of_max':   int(areas.argmax()),
        }

    # ── Visuals ──
    plot_uri = _area_timeseries_plot(area_dict, label_names, colors_combinations) \
               if area_dict else None

    # Sample frames: first, middle, last with a label present + current frame
    sample_indices = sorted({
        0,
        total_frames // 2,
        max(0, total_frames - 1),
        *[int(area_dict[lid].argmax()) for lid in list(area_dict.keys())[:2]],
    })[:6]
    frame_uris = _video_frame_previews(image_proxy, seg_proxy,
                                        sample_indices, colors_combinations)

    # ── HTML ──
    meas_headers = ['Description', 'Type', 'Measure 1', 'Measure 2',
                    'Slice', 'Plane', 'CenterXY', 'File']
    meas_html = _measurements_table_html(measurements, meas_headers)

    sum_rows = ""
    for lid in sorted(summary.keys()):
        s    = summary[lid]
        name = label_names.get(lid, f'Label {lid}')
        dot  = f'<span class="dot" style="background:{_color_for_label(lid, colors_combinations)}"></span>'
        sum_rows += (
            f"<tr>"
            f"<td>{dot}{name}</td>"
            f"<td>{s['frames_present']:,} / {total_frames}</td>"
            f"<td>{s['mean_area']:.1f}</td>"
            f"<td>{s['max_area']:.1f}</td>"
            f"<td>{s['frame_of_max']}</td>"
            f"</tr>"
        )

    frame_imgs_html = ""
    for i, (fi, uri) in enumerate(zip(sample_indices, frame_uris)):
        frame_imgs_html += f'<img src="{uri}" style="height:200px" alt="Frame {fi}"/>'

    plot_html = f'<img class="full-img" src="{plot_uri}" alt="Area time series"/>' \
                if plot_uri else '<p style="color:#666">No segmented frames found.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>MELAGE Report — {os.path.basename(filename)}</title>
{_CSS}</head>
<body>
<h1>MELAGE Segmentation Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; Video / Photo</p>

<h2>File Information</h2>
<div class="card">
  <div class="meta-grid">
    {_meta_item('File', os.path.basename(filename))}
    {_meta_item('Total frames', str(total_frames))}
    {_meta_item('Frame size (H×W)', f'{height} × {width}')}
    {_meta_item('Labels found', str(len(summary)))}
  </div>
</div>

<h2>Segmentation Summary</h2>
<div class="card">
<table>
  <thead><tr>
    <th>Label</th><th>Frames present</th>
    <th>Mean area (px²)</th><th>Peak area (px²)</th><th>Peak frame</th>
  </tr></thead>
  <tbody>{sum_rows}</tbody>
</table>
</div>

<h2>Area over Time</h2>
{plot_html}

<h2>Sample Frames</h2>
<div class="img-row">{frame_imgs_html}</div>

<h2>Measurements Table</h2>
<div class="card">{meas_html}</div>

<footer>MELAGE v2.1.1 &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d')}</footer>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path
