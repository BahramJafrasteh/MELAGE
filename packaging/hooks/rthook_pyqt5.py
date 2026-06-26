# Runtime hook: tell Qt where its plugins live inside the frozen bundle.
import os
import sys

if getattr(sys, 'frozen', False):
    _base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    _qt_plugins = os.path.join(_base, 'PyQt5', 'Qt5', 'plugins')
    if os.path.isdir(_qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = _qt_plugins
    # Prevent Qt from searching system-wide plugin paths that may conflict
    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')
