# -*- coding: utf-8 -*-
__author__ = 'Bahram Jafraste'
__version__ = '2.2.0'

# ── Non-GUI utilities (always safe to import) ──────────────────────────
from . import utils

# ── GUI sub-packages: imported only when PyQt5 / OpenGL are available ──
# Scripts and notebooks that only use melage.api never trigger this import.
#
# Importing them has a side effect: melage/dialogs/HistogramDialog.py and
# ThresholdingDialog.py call `matplotlib.use('Agg')` at module load time.
# That's a *global* backend switch — and merely saving/restoring the backend
# name around the import isn't enough: the round trip (real -> Agg -> real)
# still corrupts IPython/Jupyter's inline-display wiring (the Figure -> PNG
# formatter is wired up by IPython's `enable_matplotlib`, not restored by a
# plain `matplotlib.use()` call), so `plt.show()` ends up printing
# "<Figure size ... with N Axes>" instead of rendering inline.
# Instead, neutralise `matplotlib.use` for the duration of the GUI import so
# the switch never happens at all. This is safe: those dialogs build their
# plots via explicit `FigureCanvasQTAgg`/`Figure` objects, not `pyplot`, so
# they don't actually depend on the global backend being "Agg" to work.
# (Must wrap the *eager* import, not a lazy one — `melage.utils.utils` and
# `melage.widgets` resolve a circular import between themselves that depends
# on this import order.)
try:
    import matplotlib as _mpl
    _real_mpl_use = _mpl.use
    _mpl.use = lambda *args, **kwargs: None
except Exception:
    _mpl = None
    _real_mpl_use = None

try:
    from . import widgets
    from .melage import main
except Exception:  # OpenGL, PyQt5, or display not available
    pass
finally:
    if _mpl is not None and _real_mpl_use is not None:
        _mpl.use = _real_mpl_use
    del _mpl, _real_mpl_use

# ── Public scripting / pipeline API (no GUI required) ─────────────────
# Heavy plugin code is loaded on demand inside each function.
from melage.api import (
    Volume,
    load,
    save,
    info,
    run,
    list_tools,
    preprocess,
    segment,
    visualize,
)

__all__ = [
    "main",
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
