"""
melage.api
==========
Public Python API for headless / scripted use of MELAGE.

Designed for Jupyter notebooks, shell scripts, and automated pipelines.
No PyQt5 or display required.

Quick Start
-----------
    import melage

    vol = melage.load("brain.nii.gz")
    print(vol)                              # shape, spacing, dtype

    # Preprocessing
    vol = melage.preprocess.n4_bias(vol)
    vol = melage.preprocess.resize(vol, spacing=1.0)
    vol = melage.preprocess.reorient(vol, target="RAS")

    # Segmentation
    vol = melage.segment.bet(vol)
    vol = melage.segment.fcm(vol, n_classes=3)
    vol = melage.segment.mga_net(vol)               # deep-learning brain extraction
    vol = melage.segment.warpseg(vol)                # deep-learning structural segmentation
    vol = melage.segment.combine_labels(vol, 1, "+", 2)  # boolean label combination

    # Save
    melage.save(vol, "output.nii.gz")       # saves image
    melage.save(vol, "seg.nii.gz", what="seg")  # saves segmentation

    # 3-D visualisation (no GUI / OpenGL required)
    melage.visualize.render(vol, label=1)            # matplotlib 3-D plot
    melage.visualize.screenshot(vol, "brain_3d.png", label=1)
    melage.visualize.export_mesh(vol, "brain.stl", label=1)

    # Image info (no full load)
    melage.info("brain.nii.gz")

    # String-based dispatcher (for loops / config-driven pipelines)
    vol = melage.run("n4_bias", vol)
    vol = melage.run("bet",     vol)
    vol = melage.run("resize",  vol, spacing=1.0)
"""

from melage.api._volume import Volume
from melage.api._io import load, save, info
from melage.api import _preprocess as preprocess
from melage.api import _segment as segment
from melage.api import _visualize as visualize

# ──────────────────────────────────────────────────────────────────────
# String-dispatch runner — useful in config-driven pipelines
# ──────────────────────────────────────────────────────────────────────

_REGISTRY: dict = {}  # populated lazily on first call to run()


def _build_registry():
    global _REGISTRY
    if _REGISTRY:
        return
    _REGISTRY = {
        # preprocessing
        "n4_bias":    preprocess.n4_bias,
        "n4":         preprocess.n4_bias,
        "resize":     preprocess.resize,
        "resample":   preprocess.resize,
        "normalize":  preprocess.normalize,
        "threshold":  preprocess.threshold,
        "largest_cc": preprocess.largest_component,
        "reorient":   preprocess.reorient,
        # segmentation
        "bet":        segment.bet,
        "brain_extraction": segment.bet,
        "fcm":        segment.fcm,
        "tissue_seg": segment.fcm,
        "mga_net":    segment.mga_net,
        "warpseg":    segment.warpseg,
        "combine_labels": segment.combine_labels,
        "mask_op":    segment.combine_labels,
        "preprocess_and_bet": segment.preprocess_and_bet,
    }


def run(tool: str, vol: Volume, **kwargs) -> Volume:
    """
    Dispatch a processing step by name.

    Parameters
    ----------
    tool : str
        One of the registered tool names (see ``melage.list_tools()``).
    vol : Volume
    **kwargs
        Forwarded to the underlying function.

    Returns
    -------
    Volume

    Examples
    --------
        vol = melage.run("n4_bias", vol, iterations=50)
        vol = melage.run("bet",     vol, fractional_threshold=0.4)
    """
    _build_registry()
    tool_key = tool.lower().replace("-", "_")
    fn = _REGISTRY.get(tool_key)
    if fn is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"melage.run: unknown tool {tool!r}.\n"
            f"Available: {available}"
        )
    return fn(vol, **kwargs)


def list_tools() -> list:
    """
    Return a sorted list of all registered tool names.

    Examples
    --------
        for name in melage.list_tools():
            print(name)
    """
    _build_registry()
    return sorted(set(_REGISTRY))


__all__ = [
    "Volume",
    "load",
    "save",
    "info",
    "run",
    "list_tools",
    "preprocess",
    "segment",
    "visualize",
]
