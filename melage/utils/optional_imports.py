# melage/utils/optional_imports.py
import warnings

# --- 1. TORCH (AI) ---
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # Placeholder

# --- 2. ANTSPY (Registration) ---
try:
    import ants
    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False
    ants = None
# --- 3. NIBABEL (Neuroimaging) ---
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    nib = None
# --- 4. Pydicom (DICOM handling) ---
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    pydicom = None
# --- 5. SimpleITK (Medical Image Processing) ---
try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False
    sitk = None
# --- 6. OpenCV (Computer Vision) ---
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
# --- 7. Matplotlib (Plotting) ---
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    matplotlib = None
# --- 8. Scikit-image (Image Processing) ---
try:
    import skimage
    HAS_SCIKIT_IMAGE = True
except ImportError:
    HAS_SCIKIT_IMAGE = False
    skimage = None
# --- 9. Pandas (Data Analysis) ---
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
# --- 10. 
# --- Helper to check functionality ---
def require_torch(parent_widget=None):
    if not HAS_TORCH:
        if parent_widget:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(parent_widget, "Feature Missing",
                                "This feature requires PyTorch, which is not installed.")
        print("Error: PyTorch not found.")
        return False
    return True