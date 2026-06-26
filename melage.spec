# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for MELAGE
#
# Build commands (run from project root):
#   macOS / Linux : pyinstaller melage.spec
#   Windows       : pyinstaller melage.spec
#
# Output:
#   macOS   → dist/MELAGE.app  (wrap in DMG for distribution)
#   Windows → dist/MELAGE/     (wrap with Inno Setup for distribution)

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# ---------------------------------------------------------------------------
# Collect data files and hidden imports from scientific packages
# ---------------------------------------------------------------------------
datas    = []
binaries = []
hiddenimports = []

_auto_collect = [
    'sklearn', 'skimage', 'scipy', 'shapely',
    'nibabel', 'pydicom', 'cv2', 'matplotlib',
]
for pkg in _auto_collect:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

# SimpleITK ships its own libITK* shared libs — collect them explicitly
try:
    d, b, h = collect_all('SimpleITK')
    datas += d; binaries += b; hiddenimports += h
except Exception:
    pass

# antspyx: collect the whole package directory (native libs included)
try:
    import ants as _ants
    _ants_dir = os.path.dirname(os.path.abspath(_ants.__file__))
    datas += [(_ants_dir, 'ants')]
    hiddenimports += collect_submodules('ants')
except ImportError:
    pass

# numba + llvmlite (BET plugin)
try:
    d, b, h = collect_all('numba')
    datas += d; binaries += b; hiddenimports += h
    d, b, h = collect_all('llvmlite')
    datas += d; binaries += b; hiddenimports += h
except Exception:
    pass

# trimesh (BET plugin + API mesh export)
try:
    d, b, h = collect_all('trimesh')
    datas += d; binaries += b; hiddenimports += h
except Exception:
    pass

# packaging (used by update checker)
try:
    d, b, h = collect_all('packaging')
    datas += d; binaries += b; hiddenimports += h
except Exception:
    pass

# ---------------------------------------------------------------------------
# MELAGE application data files
# ---------------------------------------------------------------------------
datas += [
    ('assets',          'assets'),           # icons, color LUTs, manual images
    ('melage/plugins',  'melage/plugins'),   # plugin scripts + configs
    ('data',            'data'),             # MNI templates etc.
]

# ---------------------------------------------------------------------------
# Explicit hidden imports
# ---------------------------------------------------------------------------
hiddenimports += [
    # PyQt5
    'PyQt5.QtOpenGL',
    'PyQt5.QtPrintSupport',
    'PyQt5.QtSvg',
    'PyQt5.sip',
    # OpenGL
    'OpenGL.platform.ctypes_standalone',
    'OpenGL.arrays.numpymodule',
    'OpenGL.arrays.ctypesarrays',
    'OpenGL.arrays.lists',
    'OpenGL.arrays.strings',
    'OpenGL.GL.exceptional',
    # melage itself (plugin system uses dynamic imports)
    *collect_submodules('melage'),
    # stdlib / misc
    'pkg_resources.py2_compat',
    'packaging.version',
    'importlib.metadata',
]

# ---------------------------------------------------------------------------
# Packages to exclude (AI extras — downloaded on first use, not bundled)
# ---------------------------------------------------------------------------
excludes = [
    'torch', 'torchvision', 'torchaudio',
    'sam2', 'segment_anything', 'nnInteractive',
    'tensorflow', 'keras',
    'IPython', 'jupyter', 'notebook',
    'pytest', 'sphinx',
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ['melage/melage.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['packaging/hooks'],
    runtime_hooks=['packaging/hooks/rthook_pyqt5.py'],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MELAGE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    icon='assets/resource/main.ico',
)

# ---------------------------------------------------------------------------
# Collect (one-dir bundle)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MELAGE',
)

# ---------------------------------------------------------------------------
# macOS app bundle
# ---------------------------------------------------------------------------
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MELAGE.app',
        # .icns generated by build/macos/make_icns.py in CI
        icon='assets/resource/main.icns',
        bundle_identifier='com.melage.neuroimaging',
        info_plist={
            'CFBundleDisplayName': 'MELAGE',
            'CFBundleShortVersionString': '2.2.0',
            'CFBundleVersion': '2.2.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'NSPrincipalClass': 'NSApplication',
        },
    )
