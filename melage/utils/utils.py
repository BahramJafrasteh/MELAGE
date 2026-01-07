

__AUTHOR__ = 'Bahram Jafrasteh'

# --- 1. Python Standard Library ---
import sys
import os
import re
import json
import math
import struct
import shutil
import csv
import functools
from pathlib import Path
from operator import index
from dataclasses import dataclass
from collections import defaultdict

# --- 2. Core Math & Data ---
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm

# --- 3. SciPy & Sklearn (Math/ML) ---
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn, splprep, splev
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# --- 4. Geometry & Computer Vision ---
from shapely.geometry import LineString, Polygon, Point
from shapely.validation import make_valid
import cv2

# Scikit-Image
from skimage.draw import polygon as polygon_skimage
from skimage.transform import resize as image_resize_skimage
from skimage.transform import rotate as image_rotate_skimage
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import label as label_connector
# NOTE: 'skimage.external.tifffile' is deprecated in newer versions.
# Use 'import tifffile' if possible, otherwise keep this line.
try:
    from skimage.external import tifffile as tif
except ImportError:
    import tifffile as tif

# --- 5. Medical Imaging (I/O) ---
import SimpleITK as sitk
try:
    import nrrd
except ImportError:
    print("Warning: nrrd not found")
try:
    from pydicom.filereader import dcmread
except ImportError:
    print("Warning: pydicom not found")

# Nibabel (Consolidated)
try:
    import nibabel as nib
    from nibabel.orientations import (
        aff2axcodes, axcodes2ornt, apply_orientation, ornt_transform
    )
    from nibabel.processing import resample_to_output, resample_from_to
    from nibabel.funcs import four_to_three
except ImportError:
    print("Warning: nibabel not found")

try:
    import vtk
except ImportError:
    print("Warning: vtk not found")

# --- 6. GUI & Visualization (PyQt / VTK) ---
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtWidgets import QFileDialog

from melage.config import settings
from melage.widgets.toggle import AnimatedToggle
# Direction of medical image Left, Right, Posterior Anterior, Inferior, Superior
code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))


###################### Item class for read kretz ######################
@dataclass
class Item:
    tagcl: bytes
    tagel: bytes
    size: bytes





def get_filter_for_file(filename, all_filters):
    """
    Returns the specific filter string (e.g., 'Nifti (*.nii)') that matches
    the given filename.
    """
    # 1. Split the massive filter string into individual categories
    # "Nifti(...);;Vol(...)" -> ["Nifti(...)", "Vol(...)"]
    categories = all_filters.split(';;')

    # We compare in lowercase to handle extensions robustly (e.g., .NII vs .nii)
    filename_lower = filename.lower()

    for category in categories:
        # 2. Extract the part inside parentheses: "*.nii *.nii.gz"
        start = category.find('(')
        end = category.find(')')

        if start != -1 and end != -1:
            # Get the content inside ()
            pattern_str = category[start + 1:end]

            # Split into individual extensions: ["*.nii", "*.nii.gz"]
            patterns = pattern_str.split()

            for pattern in patterns:
                # Clean the pattern to get the suffix
                # "*.nii.gz" -> ".nii.gz"
                # "**" -> skip
                clean_ext = pattern.replace('*', '').lower()

                if not clean_ext:
                    continue  # Skip empty or pure wildcards

                # 3. Check if the filename ends with this extension
                # "mri.nii.gz" ends with ".nii.gz" -> Match!
                if filename_lower.endswith(clean_ext):
                    return category

    # Fallback: If no specific match found, return "All Files" or the first one
    return "All Files (*)"

def guess_num_image_index(nifti_input):
    """
    Guess which axis of the image shape represents the 'num_images' (volume/time/channel) dimension.

    Parameters:
        nifti_input: str or nib.Nifti1Image or nib.Nifti1Header

    Returns:
        index (int or None): The axis index that likely corresponds to num_images, or None if 3D.
    """
    if isinstance(nifti_input, nib.Nifti1Image):
        hdr = nifti_input.header
    else:
        return np.argmin(nifti_input.shape), 1

    ndim = hdr['dim'][0]
    shape = hdr['dim'][1:4]
    if ndim == 4:
        ind_num_image = 3
    else:
        ind_num_image = 3
    num_image = hdr['dim'][4]
    return ind_num_image, num_image



def get_size_minDim(img):
    """
    Efficiently split a 4D nibabel image along the smallest dimension.
    Mimics nibabel's speed by using dataobj (memmap) without copying.

    Parameters
    ----------
    img : nibabel image
        4D nibabel image with `.dataobj`, `.header`, and `.affine`.

    Returns
    -------
    imgs : list of nibabel images
        3D images sliced along the smallest dimension.
    """
    arr = img.dataobj  # usually a memmap, avoids full load
    shape = arr.shape
    if len(shape) != 4:
        raise ValueError("Image must be 4D.")

    min_dim = np.argmin(shape)
    return min_dim, shape[min_dim]



def fast_split_min_dim(img, min_dim, desired_index=0):
    """
    Efficiently split a 4D nibabel image along the smallest dimension.
    Mimics nibabel's speed by using dataobj (memmap) without copying.

    Parameters
    ----------
    img : nibabel image
        4D nibabel image with `.dataobj`, `.header`, and `.affine`.

    Returns
    -------
    imgs : list of nibabel images
        3D images sliced along the smallest dimension.
    """
    arr = img.dataobj  # usually a memmap, avoids full load
    shape = arr.shape
    if len(shape) != 4:
        raise ValueError("Image must be 4D.")

    image_maker = img.__class__
    header = img.header.copy()
    affine = img.affine

    num_dims = shape[min_dim]
    for i in range(shape[min_dim]):
        if i!=desired_index:
            continue
        slicer = [slice(None)] * 4
        slicer[min_dim] = i
        sliced_data = arr[tuple(slicer)]
        return image_maker(sliced_data, affine, header)

###################### Compare the to data elements ######################
def Item_equal(Item, tag):
    """
    Args:
        Item:
        tag: to tags in the item

    Returns: boolean true or false

    """
    a = struct.pack('H', tag[0])
    b = struct.pack('H', tag[1])
    if a == Item.tagcl and b == Item.tagel:
        return True
    return False


###################### convert to string with a precission ######################
def str_conv(a):
    return f"{a:.1f}"

###################### save VTK image to nifti ######################
def save_as_nifti(VtkImage, meta_data, pathfile):
    """
    Args:
        VtkImage: vtk image data
        meta_data: meta data of the image
        pathfile: output path
    Returns: write the file with info into the path specified
    """

    nifit_writer = vtk.vtkNIFTIImageWriter()
    nifit_writer.SetInputData(VtkImage)
    nifit_writer.SetFileName(pathfile+'.nii.gz')
    nifit_writer.Write()
    with open(pathfile+'.json', 'w') as fp:
        json.dump(meta_data, fp)

###################### Reading files with desired coordinate system ######################
def read_file_with_cs(atlas_file, expected_source_system='RAS'):
    # Read NIFTI images with desired coordinate system

    im = nib.load(atlas_file)
    orig_orient = nib.io_orientation(im.affine)
    code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))
    source_system = ''.join(list(aff2axcodes(im.affine, code_direction)))
    if source_system != expected_source_system:
        print('converted to RAS')
        target_orient = axcodes2ornt('RAS', code_direction)
        transform = ornt_transform(orig_orient, target_orient)
        im = im.as_reoriented(transform)

    return im

###################### Read SITK image as nib ######################
def read_sitk_as_nib(sitk_im):
    return nib.Nifti1Image(sitk.GetArrayFromImage(sitk_im).transpose(),
                         make_affine(sitk_im), None)

###################### Convert NIBABEL image to SITK ######################
def read_nib_as_sitk(image_nib, dtype=None):
    # From https://github.com/gift-surg/PySiTK/blob/master/pysitk/simple_itk_helper.py
    if dtype is None:
        dtype = np.float32#image_nib.header["bitpix"].dtype
    nda_nib = image_nib.get_fdata().astype(dtype)
    nda_nib_shape = nda_nib.shape
    nda = np.zeros((nda_nib_shape[2],
                    nda_nib_shape[1],
                    nda_nib_shape[0]),
                   dtype=dtype)

    # Convert to (Simple)ITK data array format, i.e. reorder to
    # z-y-x-components shape
    for i in range(0, nda_nib_shape[2]):
        for k in range(0, nda_nib_shape[0]):
            nda[i, :, k] = nda_nib[k, :, i]
    # Get SimpleITK image
    vector_image_sitk = sitk.GetImageFromArray(nda)
    # Update header from nibabel information
    # (may introduce some header inaccuracies?)
    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    affine_nib = image_nib.affine.astype(np.float64)
    R_nib = affine_nib[0:-1, 0:-1]

    spacing_sitk = np.array(image_nib.header.get_zooms(), dtype=np.float64)
    spacing_sitk = spacing_sitk[0:R_nib.shape[0]]
    S_nib_inv = np.diag(1. / spacing_sitk)

    direction_sitk = R.dot(R_nib).dot(S_nib_inv).flatten()

    t_nib = affine_nib[0:-1, 3]
    origin_sitk = R.dot(t_nib)

    vector_image_sitk.SetSpacing(np.array(spacing_sitk).astype('double'))
    vector_image_sitk.SetDirection(direction_sitk)
    vector_image_sitk.SetOrigin(origin_sitk)
    return vector_image_sitk

###################### Save modified file to NIFTI ######################
def save_modified_nifti(reader, source, filename):
    """
    Args:
        reader: a reader with image information
        source: folder to write new image
        filename: new file name

    Returns:

    """
    print ('not implemented yet! save_modified')


    _imRotate = sitk.GetImageFromArray(reader.npImage)
    for key in reader.im.GetMetaDataKeys():
        _imRotate.SetMetaData(key, reader.im.GetMetaData(key))
    _imRotate.SetSpacing(reader.ImSpacing)
    _imRotate.SetOrigin(reader.ImOrigin)
    _imRotate.SetDirection(reader.im.GetDirection())
    im = sitk.Flip(_imRotate, [False, True, False])
    sitk.WriteImage(im, '_tmp.nii.gz')
    try:
        fl, ext = os.path.splitext(filename)
        if ext == '.gz':
            fl, ext = os.path.splitext(fl)

        eco_im = nib.load('_tmp.nii.gz')
        d = eco_im.get_fdata()
        newd = d.astype(np.uint8)
        new_eco_im = nib.Nifti1Image(newd, eco_im.affine, header=eco_im.header)
        path_file = os.path.join(source, fl+ '_modified_.nii.gz')
        shutil.copy(os.path.join(source, fl+'.json'), os.path.join(source, fl+'_modified_.json'))
        new_eco_im.to_filename(path_file)
    except Exception as e:
        print('Error Saving File')
        print(e)



###################### Write VTK image as dicom ######################
def save_as_dicom(VtkImage, meta_data, pathfile):
    """
    Write VTK Image to dicom image
    Args:
        VtkImage: vtk image data
        meta_data: meta data information
        pathfile: output path

    Returns:

    """
    save_as_nifti(VtkImage, meta_data, '.temp/tmp_1')
    nifti_itk = sitk.ReadImage('.temp/tmp_1.nii.gz')
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkUInt16)

    # Convert floating type image (imgSmooth) to int type (imgFiltered)
    nifti_itk = castFilter.Execute(nifti_itk)
    sitk.WriteImage(nifti_itk, pathfile+'.dcm')
###################### save vtk as nrrd ######################
def save_as_nrrd(VtkImage, meta_data, pathfile):
    """
    Write vtk image to nrrd file
    Args:
        VtkImage:
        meta_data:
        pathfile:

    Returns:

    """
    save_as_nifti(VtkImage, meta_data, '.temp/tmp_1')
    nifti_itk = sitk.ReadImage('.temp/tmp_1.nii.gz')
    sitk.WriteImage(nifti_itk, pathfile+'.nrrd')



def resample_to_size(im, new_size, scale_factor=None,method='linear'):
    original_image = read_nib_as_sitk(im)
    # Get the current size of the image
    size = original_image.GetSize()

    # Calculate the scale factor for resizing
    if scale_factor is None:
        scale_factor = [(float(sz)/new_sz)*spc for sz, new_sz, spc in zip(size, new_size, original_image.GetSpacing())]

    # Resample the image using the scale factor
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)
    resampler.SetOutputSpacing(scale_factor)
    resampler.SetSize([int(el) for el in new_size])

    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetOutputDirection(original_image.GetDirection())

    if method.lower() == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)  # You can choose different interpolators
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)  # You can choose different interpolators
    # Perform resampling
    resized_image = resampler.Execute(original_image)
    #inverse_scale_factor = [s1/float(sz) for sz, s1 in zip(*[resized_image.GetSize(), size])]
    return read_sitk_as_nib(resized_image)#, inverse_scale_factor, list(size)

def extract_patches(image, patch_size, overlap):

    patches = []
    strides = [p-o for p, o in zip(patch_size, overlap)]


    range_x = image.shape[0] + 1
    range_y = image.shape[1]  + 1
    range_z = image.shape[2]  + 1

    for x in range(0, range_x, strides[0]):
        for y in range(0, range_y, strides[1]):
            for z in range(0, range_z, strides[2]):
                patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                patches.append(patch)
    return patches


def reconstruct_image(patches, original_size, overlap, patch_size=None):
    reconstructed_image = np.zeros(original_size)
    if patch_size is None:
        patch_size = patches[0].shape
    strides = [p-o for p, o in zip(patch_size, overlap)]
    counts = np.zeros(original_size, dtype=np.int32)

    idx = 0
    for x in range(0, original_size[0] + 1, strides[0]):
        for y in range(0, original_size[1]  + 1, strides[1]):
            for z in range(0, original_size[2] + 1, strides[2]):
                reconstructed_image[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] += patches[idx]
                counts[x:x + patch_size[0], y:y + patch_size[1], z:z + patch_size[2]] += 1
                idx += 1

    # Average the overlapping regions
    reconstructed_image /= counts.astype(float)

    return reconstructed_image


###################### Resample images to desired spacing ######################
def resample_to_spacing(im, newSpacing, method='spline'):
    try:
        original_image = read_nib_as_sitk(im)


        # Define the new spacing (voxel size) for resampling
        if type(newSpacing)==list or type(newSpacing)==tuple:
            new_spacing = newSpacing
        else:
            new_spacing = (newSpacing, newSpacing, newSpacing)

        # Calculate the new size based on the original size and spacing
        new_size = [int(sz * spc / new_spc + 0.5) for sz, spc, new_spc in
                    zip(original_image.GetSize(), original_image.GetSpacing(), new_spacing)]

        # Set up the resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(original_image.GetOrigin())
        resampler.SetOutputDirection(original_image.GetDirection())
        if method.lower()=='linear':
            resampler.SetInterpolator(sitk.sitkLinear)  # You can choose different interpolators
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)  # You can choose different interpolators

        # Perform resampling
        resampled_image = resampler.Execute(original_image)

        return read_sitk_as_nib(resampled_image)
    except:

        return resample_to_output(im, new_spacing)


###################### Help dialogue to open new image ######################
def help_dialogue_open_image(path):
    try:
        im = nib.load(path).dataobj
    except:
        im = dcmread(path).pixel_array
    if len(im.shape)==4:
        d = im[...,0]
        s1 = d[im.shape[0] // 2, :, :]
        s2 = d[:, im.shape[1] // 2, :]
        s3 = d[:, :, im.shape[2] // 2]
    elif len(im.shape)==3:
        d = im
        s1 = d[im.shape[0] // 2, :, :]
        s2 = d[:, im.shape[1] // 2, :]
        s3 = d[:, :, im.shape[2] // 2]
    elif len(im.shape)==2:
        d = im
        s1 = d
        s2 = s1
        s3 = s1
    else:
        return

    size = 200
    s1 = image_resize_skimage(s1, [size, size])[::-1]
    s2 = image_resize_skimage(s2, [size, size])
    s3 = image_resize_skimage(s3, [size, size])
    s0 = np.zeros((size * 2, size * 2))
    s0[:size, :size] = s1
    s0[:size, size:] = s2
    s0[size:, size - size // 2:size + size // 2] = s3
    s = image_rotate_skimage(s0, 90)
    s = s / s.max()
    # pixmap = QPixmap(fileName+'.jpg')
    s *= 255
    s = np.expand_dims(s, -1)
    s = s.astype(np.uint8)
    return s



def calculate_snr(image_array):

    # Calculate mean signal intensity
    mean_signal = np.mean(image_array)

    # Calculate standard deviation of the noise
    # For simplicity, consider the noise as the difference between the original image and a smoothed version

    smoothed_image = gaussian_filter(image_array, sigma=2.0)
    noise_array = np.abs(image_array - (smoothed_image))
    std_noise = np.std(noise_array)

    # Calculate SNR
    snr = mean_signal / std_noise

    return snr

###################### A class to  resize image######################

########### COMBO BOX To read 4D images #####################
class ComboBox_Dialog(QtWidgets.QDialog):
    """
    A dialog for combo box created for reading 4d images
    """
    selectedInd = QtCore.pyqtSignal(object)
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        Dialog = self.window()
        self.setupUi(Dialog)

    def accepted_emit(self):
        ind = self.comboBox.currentIndex()
        self.selectedInd = ind
        self.accept()
    def reject_emit(self):
        self.selectedInd = None
        self.reject()
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(500, 112)
        self.splitter = QtWidgets.QSplitter(Dialog)
        self.splitter.setGeometry(QtCore.QRect(10, 10, 480, 91))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.comboBox = QtWidgets.QComboBox(self.splitter)
        self.comboBox.setObjectName("comboBox")
        cbstyle = """
            QComboBox QAbstractItemView {border: 1px solid grey;
            background: #03211c; 
            selection-background-color: #03211c;} 
            QComboBox {background: #03211c;margin-right: 1px;}
            QComboBox::drop-down {
        subcontrol-origin: margin;}
            """
        self.comboBox.setStyleSheet(cbstyle)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.splitter)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(self.accepted_emit)
        self.buttonBox.rejected.connect(self.reject_emit)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

########### normalizing DWI images #####################
def norm_dti(vec):
    normc = np.linalg.norm(vec)
    if normc>1e-4:
        vec/= normc
    return vec


########### combo box to read images #####################
def create_combo_box_new(seridsc_total, sizes):
    """

    :param seridsc_total:
    :param sizes:
    :return:
    """
    combo = ComboBox_Dialog()
    r = 0
    for seridsc, size in zip(seridsc_total, sizes):
        combo.comboBox.addItem("{} {} shape: {}".format(seridsc, r, size))
        r += 1
    return combo

def show_message_box(text = 'There is no file to read'):
    """
    Display a message box
    Args:
        text: the text content of a message box
    Returns:

    """
    MessageBox = QtWidgets.QMessageBox()
    MessageBox.setText(text)
    MessageBox.setWindowTitle('Warning')
    MessageBox.show()

###################### change image system ####################
def convert_to_ras(affine, target = "RAS"):
    """
    Args:
        affine: affine matrix
        target: target system

    Returns:

    """

    orig_orient = nib.io_orientation(affine)
    source_system = ''.join(list(aff2axcodes(affine, code_direction)))# get direction
    target_orient = axcodes2ornt(target, code_direction)
    transform = ornt_transform(orig_orient, target_orient)

    return transform, source_system

###################### identify current coordinate system ####################
def getCurrentCoordSystem(affine):

    orig_orient = nib.io_orientation(affine)
    source_system = ''.join(list(aff2axcodes(affine, code_direction)))# get direction
    return source_system


def is_valid_format(file, type_f='nifti'):
    #if type_f == 'nifti':
    valid_exts = [".nia", ".nii", ".nii.gz", ".hdr", ".img", ".img.gz", ".mgz"]
    status = any(file.endswith(ext) for ext in valid_exts)
    if status:
        return "nifti", True
    valid_exts = [".nhdr", ".nrrd"]
    status = any(file.endswith(ext) for ext in valid_exts)
    if status:
        return "nrrd", True
    valid_exts = [".dcm"]
    status = any(file.endswith(ext) for ext in valid_exts)
    if status:
        return "dicom", True
    else:
        return "none", False




def dicom_series_to_nib(dicom_file):
    # Read the DICOM series
    dicom_dir = os.path.dirname(dicom_file)
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_IDs:
        return None

    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)
    sitk_image = reader.Execute()

    # Convert SimpleITK image to numpy array and affine
    affine = make_affine(sitk_image) # shape: (slices, height, width)

    nib_im = nib.Nifti1Image(sitk.GetArrayFromImage(sitk_image).transpose(), affine)
    return nib_im

###################### Read Segmentation file ##################
def read_segmentation_file(self, file, reader, update_color_s=True):
    """
    Read the segmentation files
    Args:
        self:
        file:
        reader:
        type:

    Returns: image data and state

    """
    #from nibabel.orientations import apply_orientation
    type_found, val_stat = is_valid_format(file, type_f='nifti')
    if not val_stat:
        return 0, False, True
    if type_found=='nifti':
        im = nib.load(file)  # read imag
        affine = im.affine
    elif type_found=='nrrd':

        data, header = nrrd.read(file)

        img = sitk.ReadImage(file)
        if 'space directions' in header and 'space origin' in header:
            directions = np.array(header['space directions'])
            origin = np.array(header['space origin'])
            affine = np.eye(4)
            affine[:3, :3] = directions
            affine[:3, 3] = origin
        else:
            affine = np.eye(4)  # fallback
        im = nib.Nifti1Image(data, affine)
    elif type_found=='dicom':
        im = dicom_series_to_nib(file)

        affine = im.affine
        if im is None:
            return 0, False, True


    im.get_data_dtype()
    if im.ndim == 4:

        im = four_to_three(im)[0]  # select the first image

    transform, _ = convert_to_ras(im.affine, target=reader.target_system)
    im = im.as_reoriented(transform)

    data = im.get_fdata()
    data = data.transpose(2, 1, 0)[::-1, ::-1, ::-1] # new march 17, 2025 #BJ
    if not np.issubdtype(data.dtype, np.integer):
        data = np.round(data).astype('int')
    if not all([i == j for i, j in zip(reader.npImage.shape, data.shape)]):
        return data, True, False
    data_add = None
    if rhasattr(self,'readView1.npSeg'):
        if reader != self.readView1:
            data_add = self.readView1.npSeg
    if rhasattr(self, 'readView2.npSeg'):
        if reader != self.readView2:
            data_add = self.readView2.npSeg
    uq = np.unique(data)
    if uq.shape[0]>255:
        if data.max()>80:
            ind_g = data>50
            data[ind_g] = 1
            data[~ind_g]=0
        elif data.max()<1:
            ind_g = data ==0
            data[~ind_g] = 1
    #if len(uq)==2:
    #    ind = data==0
    #    data[~ind]=1
    if update_color_s:
        data, state = update_color_scheme(self, data, data_add=data_add)

        return data, state, True
    else:
        return data,True, True

###################### Manually check items in color tree ##################
def manually_check_tree_item(self, txt='9876'):
    """
        Put items checked manual according to the input
    Args:
        self: self
        txt: item id

    Returns:

    """
    root = self.tree_colors.model().sourceModel().invisibleRootItem()
    num_rows = root.rowCount()
    ls = [i for i in range(num_rows) if
          root.child(i).text() == txt]
    for l in ls:
        root.child(l).setCheckState(Qt.Checked)
    return ls
####################### Discover all scheme files #####################
def discover_color_files(base_folder):
    """
    Dynamically finds all .txt files in the 'color' subdirectory.

    Args:
        base_folder (str): The root folder containing the 'color' directory.

    Returns:
        dict: A dictionary where keys are filenames (without extension)
              and values are the relative paths to the .txt files.
              Returns an empty dictionary if the folder doesn't exist.
    """
    color_dir = Path(base_folder) / 'color'
    if not color_dir.is_dir():
        print(f"Warning: Color directory not found at '{color_dir}'")
        return {}

    # Create a dictionary mapping the filename (e.g., 'Simple')
    # to its relative path (e.g., 'color/Simple.txt')
    # The .lower() makes the key consistent for lookups.
    return {
        path.stem: f"/color/{path.name}"
        for path in color_dir.glob('*.txt')
    }
###################### Updating current color scheme ##################

MAX_COLORS = 255
COMBINED_COLOR_ID = 9876
DEFAULT_COLOR_FILES = {
    'simple': '/color/Simple.txt',
    'tissue': '/color/Tissue.txt',
    'albert': '/color/albert_LUT.txt',
    'mcrib': '/color/mcrib_LUT.txt',
}

# --- Helper Functions (These perform the specific, isolated tasks) ---

def _get_required_indices(data, data_add=None):
    """Extracts unique positive integer indices from data arrays."""

    indices = set([l for l in np.unique(data).astype('int') if l > 0])
    if data_add is not None:
        indices.update([l for l in np.unique(data_add).astype('int') if l > 0])
    return {idx for idx in indices if idx > 0}


def _get_color_filepath(app_instance, show_dialog, unique_count):

    """Gets color file path via dialog or automatic selection."""
    if show_dialog:
        filters = "Text files (*.txt *.lut)"
        opts = QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(app_instance, "Open Color File", settings.DEFAULT_USE_DIR, filters,
                                                   options=opts)
        if file_path!='':
            return file_path
    available_files = discover_color_files(settings.RESOURCE_DIR)
    if not available_files:
        print("Error: No color files found.")
        return None # No files to choose from
    # Automatic selection logic
    # This example selects a file based on name, similar to before.
    if unique_count <= 2 and 'Simple' in available_files:
        filepath = available_files['Simple']
    elif unique_count <= 9 and 'Tissue' in available_files:
        filepath = available_files['Tissue']
    elif unique_count < 90 and 'albert_LUT' in available_files:
        filepath = available_files['albert_LUT']
    elif 'mcrib_LUT' in available_files: # Fallback for larger counts
        filepath = available_files['mcrib_LUT']
    else:
        # If no specific logic matches, just pick the first available file
        filepath = next(iter(available_files.values()))
    filepath = available_files["FreeSurferColorLUT"]
    return settings.RESOURCE_DIR + filepath


def _load_color_definitions(file_path):
    """Loads color definitions from a text file, trying multiple formats."""
    try:
        names, rgb_data, _ = read_txt_color(file_path, from_one=False, mode='albert')
        return {'names': names, 'rgb': rgb_data}
    except Exception as e1:
        print(f"Could not read '{file_path}' in primary format. Trying fallback. Error: {e1}")
        try:
            names, rgb_data, _ = read_txt_color(file_path, from_one=False)
            return {'names': names, 'rgb': rgb_data}
        except Exception as e2:
            print(f"Error: Failed to read color file '{file_path}'. Error: {e2}")
            return None


def _update_ui_widgets(app_instance):
    """Safely disconnects, updates, and reconnects UI signals and widgets."""
    try:
        #self.dw2_cb.currentTextChanged.disconnect(self.changeColorPen)
        app_instance.tree_colors.itemChanged.disconnect(app_instance.changeColorPen)
    except:
        pass

    set_new_color_scheme(app_instance)
    try:
        #self.dw2_cb.currentTextChanged.connect(self.changeColorPen)
        app_instance.tree_colors.itemChanged.connect(app_instance.changeColorPen)
    except:
        pass


# --- Main Function ---

def update_color_scheme(app_instance, data, data_add=None, dialog=True, update_widget=True):
    """
    Updates the application's color scheme based on indices present in the data.

    Args:
        app_instance: The instance of the application class (what was 'self').
        data (np.ndarray): The primary data array with color indices.
        data_add (np.ndarray, optional): Additional data array with color indices.
        dialog (bool): If True, shows a file dialog to select a color file.
        update_widget (bool): If True, updates the UI widgets after the scheme is changed.

    Returns:
        tuple[np.ndarray, bool]: The original data array and a success flag.
    """
    if data is None:
        data = np.array([0, 1])

    # 1. Determine which color indices are required
    required_indices = _get_required_indices(data, data_add)
    if len(required_indices) > MAX_COLORS:
        print(f"Error: Number of unique colors exceeds the limit of {MAX_COLORS}.")
        return data, False

    current_indices = set(app_instance.color_index_rgb[:, 0].astype(int))
    new_indices_needed = required_indices - current_indices

    if not new_indices_needed:
        return data, True  # Success, no update was needed

    # 2. Get the color file and load the color definitions
    color_file_path = _get_color_filepath(app_instance, dialog, len(required_indices))
    #if not color_file_path:
    #    return data, False  # User cancelled or no file found

    loaded_colors = _load_color_definitions(color_file_path)
    #if loaded_colors is None:
    #    return data, False  # Failed to load file

    # 3. Merge new colors with the existing scheme and update the app's state
    loaded_rgb_map = {int(row[0]): row for row in loaded_colors['rgb']}
    loaded_name_map = {int(name.split('_')[0]): name for name in loaded_colors['names']}

    final_rgb = list(app_instance.color_index_rgb)
    final_names = list(app_instance.color_name)

    for ii, index in enumerate(new_indices_needed):
        if index in loaded_rgb_map:
            final_rgb.append(loaded_rgb_map[index])
            final_names.append(loaded_name_map.get(index, f"{index}_unknown"))
        else:
            new_rgb_row = [index, np.random.rand(), np.random.rand(), np.random.rand(), 1]
            final_rgb.append(new_rgb_row)
            final_names.append(f"{index}_structure_unknown")

    # Ensure the special "Combined" color exists
    if COMBINED_COLOR_ID not in {int(row[0]) for row in final_rgb}:
        final_rgb.append([COMBINED_COLOR_ID, 1, 0, 0, 1])
        final_names.append(f"{COMBINED_COLOR_ID}_Combined")


    # Update the application state directly on the passed instance
    app_instance.color_index_rgb, app_instance.color_name, app_instance.colorsCombinations = \
        generate_color_scheme_info(app_instance,np.array(final_rgb), final_names)

    widgets_num = [0, 1, 2, 3, 4, 5, 10, 11, 13, 23]
    for num in widgets_num:
        name = 'openGLWidget_' + str(num + 1)
        widget = getattr(app_instance, name)
        if hasattr(widget, 'colorsCombinations'):
            widget.colorsCombinations = app_instance.colorsCombinations
        if hasattr(widget, 'color_name'):
            widget.color_name = app_instance.color_name

    # 4. Update the UI components
    if update_widget:
        _update_ui_widgets(app_instance)

    return data, True


def update_color_scheme0(self, data, data_add=None, dialog=True, update_widget=True):
    """
    Updating current color scheme
    Args:
        self:
        data:
        data_add:
        dialog:
        update_widget:
    Returns:

    """
    if data_add is not None:
        uq1 = [l for l in np.unique(data_add).astype('int') if l > 0]
    else:
        uq1 = []
    if data is None:
        data = np.array([0,1])
    uq = [l for l in np.unique(data).astype('int') if l > 0]
    if len(uq) > 255:
        return data, False
    uq = list(set(uq) | set(uq1))
    list_dif = list(set(uq)- set(self.color_index_rgb[:, 0]))
    if len(list_dif)<1:
        return data, True
    else:
        if dialog:

            filters = "TXT(*.txt)"
            opts = QFileDialog.DontUseNativeDialog
            fileObj = QFileDialog.getOpenFileName(self, "Open COLOR File", self.source_dir, filters, options=opts)
            filen = fileObj[0]
        else:
            filen = ''
        from_one = False


        if filen == '':
            print('automatic files')
            len_u = len(np.unique(data))
            if len_u<=2:
                filen = settings.RESOURCE_DIR + '/color/Simple.txt'
            elif len_u<=9:
                filen = settings.RESOURCE_DIR + '/color/Tissue.txt'
            elif len_u <90:
                filen = settings.RESOURCE_DIR + '/color/albert_LUT.txt'
            else:
                filen = settings.RESOURCE_DIR + '/color/mcrib_LUT.txt'
            from_one = False
        try:
            possible_color_name, possible_color_index_rgb, _ = read_txt_color(filen, from_one=from_one, mode='albert')
        except:
            try:
                possible_color_name, possible_color_index_rgb, _ = read_txt_color(filen, from_one=from_one)
            except:
                return data, False
        #uq = np.unique(data)

        set_not_in_new_list = set(uq) - (set(possible_color_index_rgb[:, 0].astype('int')))
        set_kept_new_list = set_not_in_new_list - (set_not_in_new_list - set(self.color_index_rgb[:, 0].astype('int')))
        set_create_new_list = set_not_in_new_list - set_kept_new_list
        for element in list(set_kept_new_list):
            new_color_rgb = self.color_index_rgb[self.color_index_rgb[:,0]==element,:]
            possible_color_index_rgb = np.vstack((possible_color_index_rgb, new_color_rgb))
            try:
                new_colr_name = [l for l in self.color_name if l.split('_')[0]==str(element)][0]
            except:
                r, l = [[r, l] for r, l in enumerate(self.color_name) if l.split('_')[0] == str(float(element))][0]
                l2 =str(int(float(l.split('_fre')[0]))) + '_' + '_'.join(l.split('_')[1:])
                self.color_name[r] = l2
                new_colr_name = [l for l in self.color_name if l.split('_')[0]==str(element)][0]
            possible_color_name.append(new_colr_name)

        for element in set_create_new_list:
            new_colr_name = '{}_structure_unknown'.format(element)
            possible_color_name.append(new_colr_name)
            new_color_rgb = [element, np.random.rand(), np.random.rand(), np.random.rand(), 1]
            possible_color_index_rgb = np.vstack((possible_color_index_rgb, np.array(new_color_rgb)))
        if 9876 not in possible_color_index_rgb[:,0]:
            new_colr_name = '9876_Combined'
            new_color_rgb = [9876, 1, 0, 0, 1]
            possible_color_name.append(new_colr_name)
            possible_color_index_rgb = np.vstack((possible_color_index_rgb, np.array(new_color_rgb)))

        #self.color_index_rgb, self.color_name, self.colorsCombinations = combinedIndex(self.colorsCombinations, possible_color_index_rgb, possible_color_name, np.unique(data), uq1)
        self.color_index_rgb, self.color_name, self.colorsCombinations = generate_color_scheme_info(self, possible_color_index_rgb, possible_color_name)
        try:
            #self.dw2_cb.currentTextChanged.disconnect(self.changeColorPen)
            self.tree_colors.itemChanged.disconnect(self.changeColorPen)
        except:
            pass

        set_new_color_scheme(self)
        try:
            #self.dw2_cb.currentTextChanged.connect(self.changeColorPen)
            self.tree_colors.itemChanged.connect(self.changeColorPen)
        except:
            pass

        if update_widget:
            update_widget_color_scheme(self)
        return data, True


###################### Add ultimate color ##################
def addLastColor(self, last_color):
    """
    Add ultimate color if it does not exist
    Args:
        self:
        last_color:

    Returns:

    """
    if last_color not in self.color_name:
        rm =int(float(last_color.split('_')[0]))
        self.color_name.append(last_color)
        self.colorsCombinations[rm] = [1, 0, 0, 1]
        if rm == 9876:
            clr = [rm, 1, 0, 0, 1]
        else:
            clr = [rm, np.random.rand(), np.random.rand(), np.random.rand(), 1]
        self.color_index_rgb = np.vstack((self.color_index_rgb, np.array(clr)))

###################### Add new tree widget ##################
def add_new_tree_widget(self, newindex, newText, color_rgb):
    """
    Adding new tree widget
    Args:
        self:
        newindex:
        newText:
        color_rgb:

    Returns:

    """
    int_index = int(float(newindex))
    self.colorsCombinations[int_index] = color_rgb
    parent = self.tree_colors.model().sourceModel().invisibleRootItem()
    addTreeRoot(parent, newindex, newText, color_rgb)
    new_color_name = newindex+'_'+newText
    if new_color_name not in self.color_name:
        self.color_name.append(new_color_name)
    if int_index not in self.color_index_rgb[:,0]:
        clr = color_rgb.copy()
        clr.insert(0, int_index)
        self.color_index_rgb = np.vstack((self.color_index_rgb, np.array(clr)))

    manually_check_tree_item(self, newindex)


###################### Adapt to previous version ##################
def adapt_previous_versions(self):
    """
    Adapt to the previous version of MELAGE
    Args:
        self:

    Returns:

    """
    rm = 9876
    if rm not in self.colorsCombinations:
        last_color = '9876_Combined'
        self.colorsCombinations[rm] = [1, 0, 0, 1]
        if last_color not in self.color_name:
            self.color_name.append(last_color)
        if rm not in self.color_index_rgb[:, 0]:
            clr = [rm, 1, 0, 0, 1]
            self.color_index_rgb = np.vstack((self.color_index_rgb, np.array(clr)))

###################### Updating widget colors ##################
def update_widget_color_scheme(self):
    widgets_num = [0, 1, 2, 3, 4, 5, 10, 11, 13, 23]
    for num in widgets_num:
        name = 'openGLWidget_' + str(num+1)
        widget = getattr(self, name)
        if hasattr(widget, 'colorsCombinations'):
            widget.colorsCombinations = self.colorsCombinations
        if hasattr(widget, 'color_name'):
            widget.color_name = self.color_name

###################### Make segmentation visible ##################
def make_all_seg_visibl(self):
    """
    This function makes all segmentations visible
    Args:
        self:

    Returns:

    """
    widgets = find_avail_widgets(self)
    prefix = 'openGLWidget_'
    ind = 9876#self.dw2Text.index('X_Combined')+1#len(self.colorsCombinations)
    #self.dw2_cb.setCurrentIndex(ind)
    #self.dw2_cb.setCurrentText('9876_Combined')
    manually_check_tree_item(self, '9876')
    colorPen = [1,0,0,1]#self.colorsCombinations[ind]
    for k in widgets:
        name = prefix + str(k)
        widget = getattr(self, name)
        widget.colorInd = ind
        if k in [14]:
            widget.paint(self.readView1.npSeg,
                         self.readView1.npImage, None)
        elif k in [24]:
            widget.paint(self.readView2.npSeg,
                         self.readView2.npImage, None)
        else:
            widget.colorObject = colorPen
            #widget.colorInd = len(self.colorsCombinations)
            widget.makeObject()
            widget.update()

###################### Compute volume according to the selected region ##################
def compute_volume(reader, filename, inds, in_txt=None, ind_screen=0):
    """
    Compute total volume of visible structures
    Args:
        reader:
        filename:
        inds:

    Returns:

    """
    if 9876 in inds:
        vol = (reader.npSeg > 0).sum()*reader.ImSpacing[0] ** 3 / 1000
    else:
        vol = 0
        for ind in inds:
            vol += (reader.npSeg == ind).sum()
        vol *= reader.ImSpacing[0] ** 3 / 1000
    #txt = 'File: {}, '.format(filename)
    division_ind = in_txt.find(' ; ')
    if in_txt is None or division_ind==-1:
        in_txt = ';'
    if ind_screen == 1:#'MRI'
        if division_ind!=0:
            kept_part = in_txt[:division_ind]
        else:
            kept_part = ''
    else:
        kept_part = in_txt[division_ind+2:]
    if len(filename)>10:
        txt = '{}..., '.format(filename[:10])
    else:
        txt = '{}, '.format(filename)

    if ind_screen==1:#'MRI'
        txt += 'Vol : {0:0.2f} cm\u00b3'.format((vol))
        #if len(kept_part)>0:
        out_txt = kept_part.replace('  ', '') + ' ; ' + txt
        #else:
        #    out_txt = txt
    else:
        txt += 'Vol : {0:0.2f} cm\u00b3'.format((vol))
        #if len(kept_part)>0:
        out_txt = txt + ' ; ' + kept_part.replace('  ', '')
        #else:
        #    out_txt = txt
    return out_txt

###################### Unique of an array ##################
def getUnique(mat):
    return np.unique(mat)

###################### Generate info for color schemes ##################
def generate_color_scheme_info(reader, color_index_rgb, color_name):
    """
    Generate color scheme information
    Args:
        color_index_rgb:
        color_name:

    Returns:

    """
    new_colorsCombinations = defaultdict(list)
    for color_index in color_index_rgb[:,0]:
        ind_l = color_index_rgb[:, 0] == color_index
        new_colorsCombinations[color_index] = [color_index_rgb[ind_l, 1][0], color_index_rgb[ind_l, 2][0],
                                     color_index_rgb[ind_l, 3][0], 1]

    return color_index_rgb, color_name, new_colorsCombinations



################# SET COLOR SCHEME ###########################
def set_new_color_scheme(self):
    """
    SET new color scheme
    Args:
        self:

    Returns:

    """
    #from widgets.tree_widget import TreeWidgetItem
    if self.color_index_rgb is None:

        normalized = matplotlib.colors.Normalize(vmin=0,vmax=len(self.color_name))
        colors = []
        for i in range(len(self.color_name)):
            color = list(matplotlib.cm.tab20c(normalized(i)))
            color.insert(0, i + 1)
            colors.append(color)
        self.color_index_rgb = np.array(colors)


    ######################################
    if hasattr(self, 'tree_colors'):
        self.tree_colors.model().sourceModel().clear()
        self.tree_colors.model().sourceModel().setColumnCount(2)
        self.tree_colors.model().sourceModel().setHorizontalHeaderLabels(['Index', 'Name'])
        parent = self.tree_colors.model().sourceModel().invisibleRootItem()
        for i in range(len(self.color_name)):
            cln = self.color_name[i]
            indc, descp = cln.split('_')[0], '_'.join(cln.split('_')[1:])
            try:
                clrvalue = self.color_index_rgb[self.color_index_rgb[:,0]==int(float(indc)),:][0]
            except:
                continue
            addTreeRoot(parent, indc, descp, clrvalue[1:-1])

    check_nul = [l for l in self.color_name if '9876' in l.split('_')]
    colr = [1, 0, 0, 1]
    if len(check_nul)==0:
        parent = self.tree_colors.model().sourceModel().invisibleRootItem()
        addTreeRoot(parent, '9876', 'Combined', colr)

################# SET IMAGE SCHEME ###########################

# Place this helper method inside your main class
def _is_image_already_added(self, file_path, view_index):
    """Checks if an image has already been added to a specific view."""
    for item_data in self.imported_images:
        existing_file = item_data[0][0][0]
        existing_view_index = item_data[0][0][2]
        if file_path == existing_file and view_index == existing_view_index:
            return True
    return False

# Your updated function
def update_image_sch(self, info=None, color=[1, 1, 0], loaded=False):
    """
    Sets a new color scheme and adds items for multiple images.
    """
    # SUGGESTION: Add a guard clause for safety
    if info is None:
        return

    if hasattr(self, 'tree_images'):
        parent = self.tree_images.model().sourceModel().invisibleRootItem()
        fileObj, index, index_view = info

        # Determine the description string for the view
        if index >= 3:
            indc = f'View {index_view + 1} (seg)'
        else:
            indc = f'View {index_view + 1}'

        color_255 = [int(c * 255) for c in color]

        for file in fileObj[0]:
            if '*' in file:
                continue

            # SUGGESTION: Use the clearer helper function
            if not _is_image_already_added(self, file, index_view):
                self.imported_images.append([[[file, fileObj[1], index_view], index], color, loaded, indc])
            else:
                # If it already exists, skip adding it to the tree
                continue

            # --- Create Tree Items ---
            node1 = QtGui.QStandardItem(indc)
            node1.setForeground(QtGui.QBrush(QtGui.QColor(*color_255)))
            node1.setData(index_view)

            # SUGGESTION: Set flags correctly to prevent the double-toggle bug
            # DO NOT include ItemIsUserCheckable
            node1.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsTristate)
            node1.setCheckable(True)
            node1.setFlags(node1.flags() & QtCore.Qt.ItemIsUserCheckable)

            # Set initial check state
            node1.setCheckState(QtCore.Qt.Checked if loaded else QtCore.Qt.Unchecked)

            descp = os.path.basename(file)
            node2 = QtGui.QStandardItem(descp)
            node2.setForeground(QtGui.QBrush(QtGui.QColor(*color_255)))
            # SUGGESTION: No need for flags on a non-checkable item
            # node2.setFlags(...)
            #node2.setFlags(node2.flags() | QtCore.Qt.ItemIsTristate)

            parent.appendRow([node1, node2])


def update_image_sch_to_remove(self, info=None, color = [1,1,0],loaded = False):
    """
    SET new color scheme to read multiple images
    Args:
        self:

    Returns:

    """

    ######################################
    if hasattr(self, 'tree_images'):

        parent = self.tree_images.model().sourceModel().invisibleRootItem()


        [fileObj, index, index_view] = info
        if index_view==0:
            indc = 'View 1'
        elif index_view==1:
            indc = 'View 2'
        if index>=3:
            if index_view==0:
                indc = 'View 1 (seg)'
            elif index_view==1:
                indc = 'View 2 (seg)'
        """
        
        indc='Unknow'
        if index==0:
            indc='View 1'
        elif index==1:
            indc='View 1 (fetal)'
        elif index==2:
            indc='View 2'
        elif index==3:
            indc='View 1 (Seg)'
        elif index==4:
            indc='View 1 (fetal, seg)'
        elif index == 5:
            indc = 'View 2 (seg)'
        """
        color = [int(c * 255) for c in color]
        for file in fileObj[0]:
            if '*' in file:
                continue
            descp = os.path.basename(file)
            if info is not None:
                existence_indices_view = [[file in f[0][0][0], f[0][0][2]] for f in self.imported_images]
                #if np.sum([el[0] for el in existence_indices_view])==0:
                if np.sum([f[0][0][2]==index_view for f in self.imported_images if file in f[0][0][0]])==0:
                    self.imported_images.append([[[file, fileObj[1], index_view], index], color, loaded, indc])
                else:
                    return

            node1 = QtGui.QStandardItem(indc)

            node1.setForeground(QtGui.QBrush(QtGui.QColor(color[0], color[1], color[2], 255)))


            node1.setData(index_view)

            node1.setFlags(
                node1.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable )
            if loaded:
                node1.setCheckState(Qt.Checked)
            else:
                node1.setCheckState(Qt.Unchecked)
            node2 = QtGui.QStandardItem(descp)
            node2.setForeground(QtGui.QBrush(QtGui.QColor(color[0], color[1], color[2], 255)))
            node2.setFlags(node2.flags() | QtCore.Qt.ItemIsTristate)
            # node2.setCheckState(0)
            parent.appendRow([node1, node2])


def get_back_data(im, shape_initial, pad_zero, border_value):
    im_fill = np.ones(shape_initial) * border_value
    im_fill[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
    pad_zero[2][0]:pad_zero[2][1] + 1] = im
    return im_fill





def magic_selection(im, initial_point, connectivity = 4, tol = 60):
    #(int(realy), int(realx))
    h, w = im.shape[:2]

    tolerance = (tol,) * 3

    segmented_area = np.zeros((h + 2, w + 2), dtype=np.uint8)

    segmented_area[:] = 0
    try:
        cv2.floodFill(im.astype(np.float32), segmented_area, initial_point, 0,
                      (tol,), (tol,), (
                connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 >> 8
        ))
        magic_mask = segmented_area[1:-1, 1:-1].copy()
        magic_mask[im == 0] = 0
        segmented_area = magic_mask>0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        segmented_area = cv2.morphologyEx(segmented_area.astype(np.float32), cv2.MORPH_CLOSE, kernel)
        
        return segmented_area
    except:
        return None

################# Second way to clean parent image ###########################
def clean_parent_image2(self, filename, indc,index_view):

    parent = self.tree_images.model().sourceModel().invisibleRootItem()
    fn = os.path.basename(filename)
    index_row = None
    for i in range(parent.rowCount()):
        signal1 = parent.child(i, 0).text()
        signal2 = parent.child(i, 1).text()
        if signal2==fn and signal1 in indc:
            [info, _, _, _] = self.imported_images[i]
            if indc in info[0][1]:
                continue
            else:
                index_row = i
                signal = parent.child(i)
                signal.setCheckState(Qt.Checked)
                break
    if index_row is None:
        return
    indices = []
    for i in range(parent.rowCount()):
        signal = parent.child(i)
        if i==index_row or signal.data()!=index_view:
            continue
        if signal.checkState() == Qt.Checked:
            if signal.text() in indc:
                try:
                    #self.tree_images.model().sourceModel().itemChanged.disconnect(self.changeImage)
                    self.tree_images.clicked.disconnect(self.on_row_clicked_image)
                except:
                    pass
                signal.setCheckState(Qt.Unchecked)
                [info, _, _, _] = self.imported_images[i]
                #if indc in info[0][1]:
                indices.append(i)
                try:
                    #self.tree_images.model().sourceModel().itemChanged.connect(self.changeImage)
                    self.tree_images.clicked.connect(self.on_row_clicked_image)
                except:
                    pass
    for ind in indices:
        try:
            self.imported_images.pop(ind)
            parent = self.tree_images.model().sourceModel().invisibleRootItem()
            parent.removeRow(ind)
        except Exception as e:
            print(e)


################# first way to clean parent image ###########################
def clean_parent_image(self, index_row, indc,index_view):
    parent = self.tree_images.model().sourceModel().invisibleRootItem()
    for i in range(parent.rowCount()):
        signal = parent.child(i)
        if signal.data()!=index_view or i==index_row:
            continue

        if signal.checkState() == Qt.Checked:
            if signal.text() in indc:
                try:
                    #self.tree_images.model().sourceModel().itemChanged.disconnect(self.changeImage)
                    self.tree_images.clicked.disconnect(self.on_row_clicked_image)
                except:
                    pass
                signal.setCheckState(Qt.Unchecked)
                try:
                    #self.tree_images.model().sourceModel().itemChanged.connect(self.changeImage)
                    self.tree_images.clicked.connect(self.on_row_clicked_image)
                except:
                    pass


################# read color information from text files ###########################


def _is_number(s):
    """Helper to check if a string represents a number."""
    return bool(re.match(r'^-?\d+(\.\d+)?$', str(s).strip()))
def _read_standard_color(file):
    inital_col = []
    color_name = []
    color_index_rgb = np.array([])

    names = []
    data = []
    with open(file, 'r') as fp:
        for line in fp:
            # Remove comments (#) and strip whitespace
            clean_line = line.split('#')[0].strip()
            if not clean_line:
                continue

            parts = re.split(r'[,\t\s]+', clean_line)

            # Standard Format: ID Name R G B (A)
            # We need at least ID, Name, R, G, B (5 parts)
            if len(parts) >= 5 and is_number(parts[0]):
                try:
                    idx = int(parts[0])
                    name = parts[1]

                    if _is_number(name):
                        raise ValueError(f"Standard format expects text name, found number: {name}")
                    name = f"{idx}_{name}"
                    r = float(parts[2])
                    g = float(parts[3])
                    b = float(parts[4])

                    # Alpha is optional. If missing, assume 255 (opaque).
                    # Note: FreeSurfer files often list 0 for alpha.
                    # If you want to force opacity, change 'float(parts[5])' to 255.0
                    a = float(parts[5]) if len(parts) > 5 else 255.0

                    names.append(name)
                    data.append([idx, r, g, b, a])
                except ValueError:
                    if _is_number(name):
                        raise ValueError(f"Standard format expects text name, found number: {name}")
                    continue  # Skip malformed lines
    if len(data) > 0:
        color_index_rgb = np.array(data)
        # Normalize R, G, B, A columns (indices 1, 2, 3, 4) to 0-1
        color_index_rgb[:, 1:] = color_index_rgb[:, 1:] / 255.0

        color_name = names
        inital_col = color_index_rgb[:, 0].copy()
    return color_name, color_index_rgb, inital_col


def _read_heuristic_color(file):
    inital_col = []
    color_name = []
    color_index_rgb = np.array([])

    with open(file, 'r') as fp:
        lines = fp.readlines()

        # Helper logic to find columns (kept mostly same as your original)
        index_colr_start = 0
        index_colr_end = 0
        indices_non_numeric = 0

        for id_line, l in enumerate(lines):
            if l.strip().startswith('#'):
                continue
            try:
                # Clean line
                spl = list(filter(None, re.split(r'[,\t\s"]+', l.strip())))
                spl_1 = spl[1:]  # Skip first element (usually ID)

                if len(spl_1) > 3:
                    # Heuristic: Find sequences of 3, 4, or 6 numbers
                    indices_3 = [i for i in range(len(spl_1) - 2) if all(is_number(spl_1[i + j]) for j in range(3))]
                    indices_4 = [i for i in range(len(spl_1) - 3) if all(is_number(spl_1[i + j]) for j in range(4))]
                    indices_6 = [i for i in range(len(spl_1) - 5) if all(is_number(spl_1[i + j]) for j in range(6))]

                    if len(indices_3) == 1:  # RGB
                        index_colr_start = indices_3[0] + 1
                        index_colr_end = index_colr_start + 3
                    elif len(indices_4) == 1:  # RGBA
                        index_colr_start = indices_4[0] + 1
                        index_colr_end = index_colr_start + 4
                    elif len(indices_6) == 1:  # RGB (CCC)
                        index_colr_start = indices_6[0] + 1
                        index_colr_end = index_colr_start + 4

                    # Find name index
                    indices_non_numeric_list = [r for r, s in enumerate(spl_1) if not is_number(s)]
                    if indices_non_numeric_list:
                        indices_non_numeric = indices_non_numeric_list[np.argmax(
                            [len(spl_1[el]) for el in indices_non_numeric_list])] + 1
                    break
            except:
                continue

        # Parsing based on found indices
        clean_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

        # Extract names
        color_name = []
        for l in clean_lines:
            #tokens = [r.replace('"', '') for r in re.sub(r'\s+', ' ', l.strip()).split()]
            tokens = list(filter(None, re.split(r'[,\t\s"]+', l.strip())))
            if len(tokens) > indices_non_numeric:
                prefix = tokens[0]
                suffix = tokens[indices_non_numeric]
                cur_name = f"{prefix}_{suffix}"
                if _is_number(cur_name):
                    raise ValueError(f"Standard format expects text name, found number: {cur_name}")

                color_name.append(cur_name)

        # Extract IDs
        indices_el = [int(l.replace(',', ' ').replace('"', '').split()[0]) for l in clean_lines if l.strip()]

        # Extract RGB/RGBA
        rgb_data = []
        for l in clean_lines:
            tokens = list(filter(None, re.split(r'[,\t\s"]+', l.strip())))
            vals = [float(s) for s in tokens[index_colr_start:index_colr_end]]
            rgb_data.append(vals)

        color_index_rgb = np.array(rgb_data)
        value_use = 1
        if color_index_rgb.max()>1:
            value_use = 255
        # Handle 3-column (RGB) vs 4-column (RGBA)
        if color_index_rgb.shape[1] == 3:
            # Add Alpha = 1.0 column (pre-normalized) or 255 (if pre-division)
            # Assuming raw values are 0-1 based on logic below
            color_index_rgb = np.hstack((color_index_rgb, np.full((color_index_rgb.shape[0], 1), value_use)))
        #elif color_index_rgb.shape[1] == 4:
        #if np.max(color_index_rgb[:, -1])<= 1:
        #    color_index_rgb[:, -1] *= 255

        # Normalize colors to 0-1
        color_index_rgb = color_index_rgb / value_use

        # Prepend ID column
        ids_col = np.array(indices_el).reshape(-1, 1)
        color_index_rgb = np.hstack((ids_col, color_index_rgb))

        # Filter zero colors? (Your original logic: inds = color_index_rgb[:, [1, 2, 3]].sum(1) != 0)
        # This removes black labels. Keep or remove based on preference.
        inds = color_index_rgb[:, 1:4].sum(1) != 0
        color_index_rgb = color_index_rgb[inds, :]
        color_name = list(np.array(color_name)[inds])

        # Fix: Populate inital_col for ITK mode
        inital_col = color_index_rgb[:, 0].copy()
    return color_name, color_index_rgb, inital_col

def _read_custom_color(file, from_one):
    inital_col = []
    color_name = []
    color_index_rgb = np.array([])
    with open(file, 'r') as fp:
        lines = fp.readlines()
        color_info = []
        r = 0
        for n, l in enumerate(lines):
            raw_name = l.rstrip('\n')
            if _is_number(raw_name.split()[0]):
                raise ValueError(f"Line {n} expected name but found number: {raw_name}")
            if n % 2 == 0:
                if from_one:
                    color_name.append('{}_'.format(r + 1) + raw_name)
                    r += 1
                else:
                    color_name.append(lines[n + 1].split(' ')[0] + '_' + raw_name)
            else:
                color_info.append([int(i) for i in l.rstrip("\n").split(' ')])

        color_index_rgb = np.array(color_info).astype('float')
        # Normalize RGBA (Cols 1-4)
        color_index_rgb[:, [1, 2, 3, 4]] = color_index_rgb[:, [1, 2, 3, 4]] / 255.0
        inital_col = color_index_rgb[:, 0].copy()
        if from_one:
            color_index_rgb[:, 0] = np.arange(color_index_rgb.shape[0]) + 1
    return color_name, color_index_rgb, inital_col

def is_number(vl):
    return bool(re.match(r'^-?\d+(\.\d+)?$', vl))



def read_txt_color(file, mode='lut', from_one=False):
    """
    Read color information from text files (LUTs).
    Supports: 'lut' (legacy), 'freeview'/'fsleyes' (FreeSurfer/FSL), and generic 'itk'.
    """
    inital_col = []
    color_name = []
    color_index_rgb = np.array([])

    # Define the pool of strategies
    # We wrap them in lambdas or partials if arguments differ
    strategies = {
        'custom': lambda: _read_custom_color(file, from_one),
        'standard': lambda: _read_standard_color(file),
        'heuristic': lambda: _read_heuristic_color(file)
    }

    # 1. Add Primary Strategy based on input mode
    # 2. Add remaining strategies as fallback
    # The order here determines preference if the primary fails.
    # Standard is usually safest, then Heuristic, then Custom (which is very brittle)
    execution_order = [strategies['standard'],strategies['heuristic'], strategies['custom']]

    last_error = None

    for strategy in execution_order:
        try:
            return strategy()
        except Exception as e:
            last_error = e
            continue  # Try next strategy


    return color_name, color_index_rgb, inital_col

###################### Load Tractography file ######################
def load_trk(file):
    try:
        stk = nib.streamlines.load(file)
        success = True
    except:
        stk = None
        success = False
    return stk, success

###################### related to Tractography file ######################
def divide_track_to_prinicipals(trk):
    """

    Args:
        trk:

    Returns:

    """
    rng = np.linspace(np.floor(np.min((trk[:, 0]))), np.ceil(np.max((trk[:, 0]))), int(np.ceil(np.max((trk[:, 0])))-np.floor(np.min((trk[:, 0]))))*2)
    trks = []
    for r, rn in enumerate(rng):
        if r < len(rng)-1:
            ind_rng = (trk[:,0]>= rng[r])*(trk[:,0] <= rng[r+1])
            a = trk[ind_rng, :]

            if a.shape[0]>0:
                trks.append(a.mean(0))
    return np.array(trks)



###################### Return voxel coordinates for reference x, y, z and vice versa ######################
def apply_affine(coord, affine):
    """ Return voxel coordinates for reference x, y, z and vice versa"""
    if coord.shape[1] != 4:
           c = np.zeros((coord.shape[0], 4))
           c[:, :-1] = coord
           c[:, -1] = np.ones(coord.shape[0])
           coord = c.T
    return np.matmul(affine,coord).T

###################### Generate colors for tractography files ######################
def generate_colors_track(streamls):
    """
    Generate colors for tractography according to stream lines
    Args:
        streamls:

    Returns:

    """
    def rgbcolor(line):
        if line.ndim == 1:
            norml = np.linalg.norm(line)
            color = np.abs(np.divide(line, norml, where=norml != 0))

        return color

    colors = [rgbcolor(strl[-1] - strl[0])
                for strl in streamls]
    return np.vstack(colors)


###################### Get real world coordinates from tractography information ######################
def get_world_from_trk(streamlines, affine, inverse=False, color_based_on_length=False):
    """
    Get real world coordinates from tractography information
    Args:
        streamlines: stream lines
        affine: affine matrix
        inverse: inverse of affine matrix
        color_based_on_length: boolean to select color according to the length of segments

    Returns:

    """
    length = streamlines._lengths
    ind_large = length > 1

    if color_based_on_length:

        normalized = matplotlib.colors.Normalize(vmin=np.quantile(length[ind_large], 0.2), vmax=np.quantile(length[ind_large], 0.7))
        colors = [matplotlib.cm.jet(normalized(len(strl)))
         for strl in streamlines]

    else:

        colors = generate_colors_track(streamlines)

    if inverse:
        affine0 = np.linalg.inv(affine)
    else:
        affine0 = affine.copy()
    str_world = []
    #colors = []
    r = 0
    for ln, clr in zip(streamlines[ind_large], colors[ind_large]):
        a = apply_affine(ln, affine0)
        #a = np.round(a).astype('int')
        #a[:,-1] = r
        a = np.unique(a, axis=0)
        color = clr

        b = np.zeros((a.shape[0], 8))
        b[:, :3] = a[:,:-1]
        b[:, 3:6] = color
        b[:,-1] = r
        b[:, -2] = a.shape[0]
        r += 1
        str_world.append(b)

    return np.concatenate(str_world)

def vox2ref(affine, ref):
   """ Return X, Y, Z coordinates for i, j, k """
   return np.matmul(affine, ref)[:3]

###################### Cursors ######################

def cursorOpenHand():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/Hand.png")
    return QtGui.QCursor(bitmap)

def cursorClosedHand():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/Handsqueezed.png")
    return QtGui.QCursor(bitmap)
def cursorZoomIn():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/zoom_in.png")
    return QtGui.QCursor(bitmap)
def cursorZoomOut():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/zoom_out.png")
    return QtGui.QCursor(bitmap)
def cursorRotate():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/rotate.png")
    return QtGui.QCursor(bitmap)
def cursorArrow():
    #bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/arrow.png")
    return QtGui.QCursor(Qt.ArrowCursor)


def cursorPaint(size=24):
    """
    Creates a cursor from an image with a corrected 'Hotspot'.
    """
    path = settings.RESOURCE_DIR + "/paint-brush-2.png"
    pixmap = QtGui.QPixmap(path)

    if pixmap.isNull():
        print(f"Warning: Could not load cursor from {path}")
        return QtCore.Qt.ArrowCursor

    # 1. Resize: Ensure it's not too big (Standard is 24x24 or 32x32)
    pixmap = pixmap.scaled(size, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    # 2. Define Hotspot: The pixel where the click actually happens.
    # For a typical brush icon (handle top-right, tip bottom-left):
    # X = 0 (Left), Y = size-1 (Bottom)
    hot_x = 0
    hot_y = size//2

    # NOTE: Adjust hot_x/hot_y if your brush tip is in a different spot!

    return QtGui.QCursor(pixmap, hot_x, hot_y)
def cursorPaintX():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/paint-brush-2.png")
    return QtGui.QCursor(bitmap)



def cursorErase():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/HandwritingMinus.png")
    return QtGui.QCursor(bitmap)
def cursorEraseX():
    bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/HandwritingMinusX.png")
    return QtGui.QCursor(bitmap)

###################### locate proper widgets ######################
def find_avail_widgets(self):
    """
    Find available active widgets
    Args:
        self:

    Returns:

    """
    prefix = 'openGLWidget_'
    widgets = [1, 2, 3, 4, 5, 6, 11, 12]
    widgets_view2 = [4, 5, 6, 12, 24]
    widgets_view1 = [11, 1, 2, 3, 14]
    _eco = False
    _mri = False

    for k in widgets:
        name = prefix + str(k)
        widget = getattr(self, name)
        if widget.isVisible():
            if k in widgets_view1:
                _eco = True
            elif k in widgets_view2:
                _mri = True
    if _eco and _mri:
        widgets = widgets
    elif _eco:
        widgets = widgets_view1
    elif _mri:
        widgets = widgets_view2
    return widgets

###################### set cursors ######################
def setCursorWidget(widget, code, reptime, rad_circle=50):
    """
    Set Cursor Widgets
    Args:
        widget: widget
        code: integer code of the widget
        reptime: repetition time
        rad_circle: raidus of circle in case of circles

    Returns:

    """
    try_disconnect(widget)
    widget.enabledPan = False  # Panning
    widget.enabledRotate = False  # Rotating
    widget.enabledPen = False  # Polygon Drawing
    widget.enabledMagicTool = False  # FreeHand Drawing
    widget.enabledErase = False  # Erasing the points
    widget.enabledZoom = False  # ZOOM DISABLED
    widget.enabledPointSelection = False  # Point Selection
    widget.enabledRuler = False
    widget.enabledCircle = False
    #widget.enabledGoTo = False
    widget.enabledLine = False
    widget._magic_slice =  None # for magic coloring
    widget.setMouseTracking(False)
    widget.makeObject()
    widget.update()
    if reptime <=1:
        if code == 0:
            widget.updateEvents()
            widget.setCursor(Qt.ArrowCursor)

        elif code == 1:  # ImFreeHand
            widget.updateEvents()
            #widget.enabledMagicTool = True
            #widget.setMouseTracking(True)
            widget.enabledPen = True
            widget.setCursor(cursorPaint())


            #from segment_anything import SamPredictor, sam_model_registry

            # Choose the model variant to download
            #model_type = "vit_h"  # Options: "vit_h", "vit_l", "vit_b"
            #checkp = os.path.join(os.path.dirname(os.path.dirname(__file__)),
            #                      'widgets/DeepLModels/sam_vit_h_4b8939.pth')
            # Download and load the model
            #sam = sam_model_registry[model_type](checkpoint=checkp)
            #widget._sam_predictor = SamPredictor(sam)

            try:
                widget.customContextMenuRequested.connect(widget.ShowContextMenu)
            except Exception as e:
                print('Cursor Widget Error')
                print(e)
        elif code == 2:  # Panning
            widget.updateEvents()
            widget.enabledPan = True
            widget.setCursor(cursorOpenHand())

        elif code == 3:  # Erasing
            #widget.setCursor(cursorOpenHand())
            widget.updateEvents()
            widget.enabledErase = True
            widget.setCursor(cursorErase())

        elif code == 4:  # ImPaint Contour
            widget.updateEvents()
            widget.enabledPen = True

            #widget.setCursor(cursorPaint())
            widget.setCursor(Qt.CrossCursor)
            try:
                widget.customContextMenuRequested.connect(widget.ShowContextMenu_contour)
            except Exception as e:
                print('Cursor Widget Error')
                print(e)
        elif code == 5: # point locator
            widget.updateEvents()
            widget.enabledPointSelection = True
            widget.setCursor(Qt.CrossCursor)
        elif code == 6: # ruler
            widget.updateEvents()
            widget.enabledRuler=True
            widget.setCursor(Qt.CrossCursor)
            try:
                widget.customContextMenuRequested.connect(widget.ShowContextMenu_ruler)
            except Exception as e:
                print('Cursor Widget Error')
                print(e)
        elif code == 7: # goto
            widget.updateEvents()
            widget.enabledGoTo = True
            widget.setCursor(Qt.CrossCursor)
        elif code == 8: # goto
            widget.updateEvents()
            widget.enabledLine = True
            widget.setCursor(cursorPaint())
            try:
                widget.customContextMenuRequested.connect(widget.ShowContextMenu_gen)
            except Exception as e:
                print('Cursor Widget Error')
                print(e)
        elif code == 9: #circle
            #widget.updateEvents()
            widget.setMouseTracking(True)
            widget.enabledCircle = True
            #widget.setCursor(cursorCircle(rad_circle))
            widget.setCursor(QtCore.Qt.ArrowCursor)
            widget.is_cursor_on_screen = False

    else:
        if code == 4:  # ImPaint
            widget.updateEvents()
            widget.enabledPen = True
            widget.setCursor(cursorPaintX())
        elif code == 3:  # Erasing
            widget.updateEvents()
            widget.enabledErase = True
            widget.setCursor(cursorEraseX())

###################### try disconnect widgets ######################
def try_disconnect(widget):
    """
    Try to disconnect connected widgets
    Args:
        widget:

    Returns:

    """
    funcs = [widget.ShowContextMenu, widget.ShowContextMenu_ruler, widget.ShowContextMenu_contour, widget.ShowContextMenu_gen]
    for f in funcs:
        try:
            while True:
                widget.customContextMenuRequested.disconnect(f)
        except Exception as e:
            pass

###################### try disconnect widgets ######################
def zonePoint(x1, y1, xc, yc):
    """
    Find zone of a point regarding to another point
    Args:
        x1:
        y1:
        xc:
        yc:

    Returns:

    """
    difx = x1 - xc
    dify = y1 - yc
    zone = 0
    if difx>0 and dify>0:
        zone = 4
    elif difx> 0 and dify<0:
        zone = 1
    elif difx < 0 and dify<0:
        zone = 2
    elif difx < 0 and dify>0:
        zone = 3
    return zone

###################### Convert points to polygons ######################


def ConvertPToPolygons(points, ignoredInd=0):
    """
    Optimized: Uses C++ based make_valid instead of slow python logic.
    """
    if len(points) < 3:
        return []

    # 2. Create Polygon directly
    # Shapely automatically closes the ring, no need to append points[0] manually
    poly = Polygon(points)

    # 3. Fast Path: If it's already valid, return immediately
    if poly.is_valid:
        return [poly]

    # 4. Slow Path: Fix self-intersections (Figure-8, Bowties)
    # make_valid splits a self-intersecting polygon into a MultiPolygon
    cleaned = make_valid(poly)

    # 5. Handle standard output types
    if cleaned.geom_type == 'Polygon':
        return [cleaned]
    elif cleaned.geom_type == 'MultiPolygon':
        return list(cleaned.geoms)
    elif cleaned.geom_type == 'GeometryCollection':
        # Sometimes it returns lines/points too, filter for Polygons only
        return [p for p in cleaned.geoms if p.geom_type == 'Polygon']

    return []

###################### Convert points to polygons with a defined buffer size ######################
def ConvertPointsToPolygons(points, width = 0):
    """
    Convert Points to polygons
    Args:
        points:
        width:

    Returns:

    """
    if width <= 0:
        return Polygon(points)
    ls = LineString(points)
    d = list(ls.buffer(width).exterior.coords)
    return Polygon(np.hstack((np.array(d), np.ones((len(d), 1)) * points[0][-1])))





###################### fill inside the polygon (To fill pixels) ######################
def fillInsidePol(poly):
    """
    Fill inside polygons - Optimized using Scanline Algorithm
    Args:
        poly: Shapely Polygon object

    Returns:
        total_points: (N, 3) array of [x, y, sliceNo]
        coords: (M, 3) array of exterior coordinates
    """
    try:
        # 1. Extract coordinates
        coords = np.array(poly.exterior.coords)

        # 2. Get Slice Number (Z-axis)
        # Assuming all points on this polygon are on the same slice
        sliceNo = coords[0, 2]

        # 3. separate x and y
        x_poly = coords[:, 0]
        y_poly = coords[:, 1]

        # 4. FAST RASTERIZATION (The Optimization)
        # skimage returns the indices (row, col) of pixels inside the polygon
        # Note: skimage uses (row, col) which maps to (y, x)
        rr, cc = polygon_skimage(y_poly, x_poly)

        # 5. Stack into (x, y, z) format
        # rr = y coordinates, cc = x coordinates
        # We assume integer coordinates for pixels
        selected_points = np.column_stack((cc, rr))

        # 6. Add Slice Number column
        # Create a column of the slice number matching the length of points
        z_col = np.full((len(selected_points), 1), sliceNo)

        total_points = np.hstack([selected_points, z_col])

        return total_points, coords

    except Exception as e:
        print(f'Error in fillInsidePol: {e}')
        return np.array([]), np.array([])
###################### fill index of white voxels ######################
def findIndexWhiteVoxels(poly, segmentShowWindowName, is_pixel=False, bool_permute_axis=True):
    """
    Find index of white voxels in the segmentation.
    Refactored for performance, safety, and readability.
    """
    try:
        if is_pixel:
            # CASE 1: 'poly' is already a list of pixel coordinates (e.g., Pen/Brush tool)
            # Optimization: Specify dtype=int immediately to avoid double-copying
            whiteVoxels = np.array(poly, dtype=int)

            # For raw pixels, the "edge" is just the pixels themselves
            edges = whiteVoxels.copy()

        else:
            # CASE 2: 'poly' is a geometric shape (e.g., Lasso/Polygon tool)
            # Use the OPTIMIZED fillInsidePol (from previous step)
            pixels, edges = fillInsidePol(poly)

            # Safety Check: Handle empty polygons gracefully
            if pixels is None or len(pixels) == 0:
                # Warning: Returning None might break tuple unpacking (a, b = func())
                # Consider returning (np.array([]), np.array([])) if your caller expects tuples.
                return None

                # Optimization: Convert to int only once
            whiteVoxels = pixels.astype(int)
            edges = edges.astype(int)

        # --- Shared Logic (Refactored to follow DRY principle) ---
        # We only apply permutation if we actually found voxels
        if bool_permute_axis and len(whiteVoxels) > 0:
            whiteVoxels, edges = permute_axis(whiteVoxels, edges, segmentShowWindowName)

        return whiteVoxels, edges

    except Exception as e:
        # Optimization: Print the ACTUAL error instead of a mysterious line number
        print(f'Error in findIndexWhiteVoxels: {e}')
        return None

###################### permute axis of white voxels for painting ######################
def permute_axis(whiteVoxels, edges, segmentShowWindowName):
    """
    Permute axis according to plane
    Args:
        whiteVoxels:
        edges:
        segmentShowWindowName:

    Returns:

    """
    if segmentShowWindowName == 'sagittal':
        whiteVoxels = whiteVoxels[:, [1, 0, 2]]
        if edges is not None:
            edges = edges[:, [1, 0, 2]]
    elif segmentShowWindowName == 'coronal':
        whiteVoxels = whiteVoxels[:, [1, 2, 0]]
        if edges is not None:
            edges = edges[:, [1, 2, 0]]
    elif segmentShowWindowName == 'axial':
        whiteVoxels = whiteVoxels[:, [2, 1, 0]]
        if edges is not None:
            edges = edges[:, [2, 1, 0]]
    elif segmentShowWindowName=='video':
        whiteVoxels = whiteVoxels[:, [1, 0, 2]]
        if edges is not None:
            edges = edges[:, [1, 0, 2]]
    else:
        raise exit("unknown window name")
    return whiteVoxels, edges

###################### find painted voxels ######################
def findWhiteVoxels(totalPs, segmentShowWindowName,seg = None):
    """
    Find white voxels
    Args:
        totalPs: total points
        segmentShowWindowName: texts
        seg: segmetations

    Returns:

    """
    pixels = []
    for sliceN in totalPs.keys():
        for key in totalPs[sliceN].keys():
            poly, _ = totalPs[sliceN][key]
            ones, edges = fillInsidePol(poly)
            pixels += ones
    whiteVoxels = np.array(pixels).astype("int")

    if segmentShowWindowName == 'sagittal':
        whiteVoxels = whiteVoxels[:, [1, 0, 2]]
    elif segmentShowWindowName == 'coronal':
        whiteVoxels = whiteVoxels[:, [1, 2, 0]]
    elif segmentShowWindowName == 'axial':
        whiteVoxels = whiteVoxels[:, [2, 1, 0]]
    newSeg = np.zeros_like(seg)
    newSeg[tuple(zip(*whiteVoxels))] = 255.0
    return newSeg

###################### Permute to proper axis #################
def PermuteProperAxis(whiteVoxels, segmentShowWindowName, axis = None):
    """
    Permute to proper axis
    Args:
        whiteVoxels:
        segmentShowWindowName:
        axis:

    Returns:

    """
    if axis is None:
        if segmentShowWindowName == 'sagittal':
            axis = [1, 0, 2]
        elif segmentShowWindowName == 'coronal':
            axis = [1, 2, 0]
        elif segmentShowWindowName == 'axial':
            axis = [2, 1, 0]
        elif segmentShowWindowName=='video':
            axis = [1, 0, 2]
        whiteVoxels = whiteVoxels[:, axis]
    else:
        whiteVoxels = whiteVoxels[:, axis]
    return whiteVoxels

###################### 3D rotation of images #################
def rotation3d(image,  theta_axial, theta_coronal, theta_sagittal, remove_zeros=False):

    """
    
    [0,0,1] : axial
    [0,1,0]: coronal
    rotate an image around proper axis with angle theta
    :param image: A nibabel image
    :param axis: rotation axis [0,0,1] is around z
    :param theta: Rotation angle
    :return: The rotated image
    """
    def get_offset(f_data):
        xs, ys, zs = np.where(f_data > 2) #find zero values
        return np.array([np.mean(xs), np.mean(ys), np.mean(zs)])

    def x_affine(theta):
        #https://nipy.org/nibabel/coordinate_systems.html#rotate-axis-0
        """ Rotation aroud x axis
        """
        cosine = np.cos(theta)
        sinus = np.sin(theta)
        return np.array([[1, 0, 0, 0],
                         [0, cosine, -sinus, 0],
                         [0, sinus, cosine,0],
                         [0, 0, 0, 1]])
    def y_affine(theta):
        #https://nipy.org/nibabel/coordinate_systems.html#rotate-axis-0
        """ Rotation aroud y axis

        """
        cosine = np.cos(theta)
        sinus = np.sin(theta)
        return np.array([[cosine, 0, sinus, 0],
                         [0, 1, 0, 0],
                         [-sinus, 0, cosine, 0],
                         [0, 0, 0, 1]])

    def z_affine(theta):
        #https://nipy.org/nibabel/coordinate_systems.html#rotate-axis-0
        """ Rotation aroud z axis
        """
        cosine = np.cos(theta)
        sinus = np.sin(theta)
        return np.array([[cosine, -sinus, 0, 0],
                         [sinus, cosine, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    if theta_axial==0 and theta_sagittal==0 and theta_coronal==0:
        return image.get_fdata(), image.affine


    theta_axial *= np.pi/180
    theta_sagittal *= np.pi / 180
    theta_coronal *= np.pi / 180

    M = x_affine(theta_sagittal).dot(y_affine(theta_coronal)).dot(z_affine(theta_axial))

    offset = (np.array(image.shape)-M[:-1, :-1].dot(np.array(image.shape))/2.0)

    f_data = resample_itk(image, M[:-1, :-1])#np.array(image.shape).astype('float')/2.0

    if remove_zeros:
        xs, ys, zs = np.where(f_data != 0) #find zero values
        tol = 4

        min_max = []
        for x in [xs, ys, zs]:
            minx = min(x)-tol if min(x)-tol>1 else min(x)
            maxx = max(x) + tol if max(x) + tol < f_data.shape[0]-1 else max(x)
            min_max.append([minx, maxx])
        f_data = f_data[min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1, min_max[2][0]:min_max[2][1] + 1]

    return f_data, M
###################### MultiOtsu thresholding #################
def Threshold_MultiOtsu(a, numc):

    if numc > 1 and numc <= 5:
        thresholds = threshold_multiotsu(a, classes=numc)
    elif numc == 1:
        thresholds = [threshold_otsu(a)]
    else:

        thresholds = list(threshold_multiotsu(a, classes=5))
        b = np.digitize(a, thresholds)

        if numc == 6:
            vls = [4]
        elif numc == 7:
            vls = [4, 3]
        elif numc == 8:
            vls = [4, 3, 2]
        elif numc == 9:
            vls = [4, 3, 2, 1]
        elif numc == 10:
            vls = [4, 3, 2, 1]
        else:
            return
        for j in vls:
            c = a.copy()
            c[b != j] = 0
            if numc == 10 and j == 4:
                new_t = list(threshold_multiotsu(c, classes=4))
            else:
                new_t = list(threshold_multiotsu(c, classes=3))
            [thresholds.append(i) for i in new_t]
        thresholds = sorted(thresholds)

    thresholds = [el for el in thresholds if el > 5]
    return thresholds

###################### apply thresholding #################
def apply_thresholding(image, _currentThresholds):
    if not len(_currentThresholds)>0:
        return np.zeros_like(image)
    regions = np.digitize(image,_currentThresholds)
    return regions+1

###################### find affine of simpleITK image #################
def make_affine(simpleITKImage):
    # https://niftynet.readthedocs.io/en/v0.2.1/_modules/niftynet/io/simple_itk_as_nibabel.html
    # get affine transform in LPS
    if simpleITKImage.GetDimension() == 4:
        c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
             for p in ((1, 0, 0, 0),
                       (0, 1, 0, 0),
                       (0, 0, 1, 0),
                       (0, 0, 0, 0))]
        c = np.array(c)
        c = c[:, :-1]
    elif simpleITKImage.GetDimension() == 3:
        c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
             for p in ((1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1),
                       (0, 0, 0))]
        c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine


###################### resample image after rotation #################
def resample_itk(image, M, offset=None):
    """
    Resample iamge after rotation
    Args:
        image:
        M:
        offset:

    Returns:

    """
    a = sitk.GetImageFromArray(image.get_fdata())
    width, height, depth = a.GetSize()
    center = a.TransformIndexToPhysicalPoint((int(np.ceil(width / 2)),
                                     int(np.ceil(height / 2)),
                                     int(np.ceil(depth / 2))))
    tr = sitk.Euler3DTransform()
    tr.SetCenter(center)
    tr.SetMatrix(np.asarray(M).flatten().tolist())
    return sitk.GetArrayFromImage(sitk.Resample(a, a, tr, sitk.sitkLinear, 0))

###################### Recursive search for an attribute of a class #################
def rhasattr(obj, path):
    """
    Recursive search for an attribute of a class
    Args:
        obj:
        path:

    Returns:

    """

    try:
        functools.reduce(getattr, path.split("."), obj)
        return True
    except AttributeError:
        return False

###################### Compute Anisotropy Elipse ######################
def computeAnisotropyElipse(kspacedata):
    """
    This function computes anisotropy elipse based on the momentum inertia of elipse
    :param image: K space 2D data
    :return: a binary function to determine a point inside the elipse
    """
    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    image = np.absolute(kspacedata)
    if image.max() > 0:
        scale = -3
        scaling_c = np.power(10., scale)
        np.log1p(image * scaling_c, out=image)
        # normalize between zero and 255
        fmin = float(image.min())
        fmax = float(image.max())
        if fmax != fmin:
            coeff = fmax - fmin
            image[:] = np.floor((image[:] - fmin) / coeff * 255.)

    pixelX, pixelY = np.where(image >= 0)
    value = image.flatten()

    Ixy = -np.sum(value * pixelX * pixelY)
    Iyy = np.sum(value * pixelX ** 2)
    Ixx = np.sum(value * pixelY ** 2)
    A = np.array([[Ixx, Ixy], [Ixy, Iyy]])

    eigVal, eigVec = np.linalg.eig(A)
    BOverA = np.sqrt(eigVal[0] / eigVal[1])
    #if BOverA<1:
     #   BOverA = 1/BOverA
    rotationAngle = math.degrees(
        math.atan(dotproduct(eigVec[0], eigVec[1]) / (length(eigVec[0]) * length(eigVec[1]))))
    x0 = 0
    y0 = 0
    convertX = lambda x, y: (x - x0) * np.cos(np.deg2rad(rotationAngle)) + (y - y0) * np.sin(
        np.deg2rad(rotationAngle))
    convertY = lambda x, y: -(x - x0) * np.sin(np.deg2rad(rotationAngle)) + (y - y0) * np.cos(
        np.deg2rad(rotationAngle))

    return lambda x, y, bb: (convertX(x, y) ** 2 / (BOverA * bb+1e-5) ** 2 + convertY(x, y) ** 2 / (
                bb ** 2+1e-5) - 1) < 0


##########################Search for a point in a contours################################
def point_in_contour(segSlice, point, color):
    """
    Search for a point in a contours
    Args:
        segSlice:
        point:
        color:

    Returns:

    """
    segSlice[segSlice != color] = 0
    segSlice[segSlice==color]=1
    contours, hierarchy = cv2.findContours(image=segSlice.astype('uint8'), mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)
    for contour1 in contours:
        if cv2.pointPolygonTest(contour1,point,True)>=0: #point in contour
            M = cv2.moments(contour1)
            xy = [M['m10'] / M['m00'], M['m01']/M['m00']] #centroid
            perimeter = cv2.arcLength(contour1, True) # perimeter
            area = cv2.contourArea(contour1) # area
            return area, perimeter, xy, contour1.squeeze()
    return 0.0, 0.0, [0.0, 0.0], [0.0, 0.0]



######extract attributes from widget and assign it to the dictionary######################


def getAttributeWidget(widget, nameWidget, dic):
    """
    Extract attributes from widget and assign it to the dictionary.
    Safe for lists containing Widgets.
    """

    def IsSafe(val):
        """
        Recursively checks if a value (or collection) is safe to save.
        Returns False if it contains QObjects, QWidgets, or QActions.
        """
        # 1. Explicitly reject Qt Objects
        if isinstance(val, (QtCore.QObject, QtWidgets.QWidget, QtWidgets.QAction, QtWidgets.QLabel,
                            QtWidgets.QFrame)):
            return False

        # 2. Allow Primitives
        if val is None:
            return True
        if isinstance(val, (int, float, str, bool, np.ndarray, tuple, Qt.GlobalColor)):
            return True

        # 3. Check Lists (Recursive)
        if isinstance(val, list):
            # If empty, it's safe. If not, check every item.
            if len(val) == 0: return True
            return all(IsSafe(x) for x in val)

        # 4. Check Dictionaries (Recursive) - THIS FIXES YOUR ISSUE
        if isinstance(val, (dict, defaultdict)):
            # Check every value in the dictionary
            if len(val) == 0: return True
            return all(IsSafe(v) for v in val.values())

        # Reject anything else
        return False

    def updateDic(val, attr, at):
        for el in attr:
            try:
                if hasattr(val, el):
                    vl = getattr(val, el)()
                    if IsSafe(vl):
                        dic[nameWidget][at][el] = vl
            except Exception as e:
                print(f'Update Dictionary Error for {at}: {e}')

    dic[nameWidget] = defaultdict(list)

    # Iterate over all attributes
    for at in dir(widget):
        if at.startswith('_') or at == 'program':
            continue
        success = False
        val = getattr(widget, at)

        # --- 1. HANDLE WIDGET TYPES ---
        if isinstance(val, QtWidgets.QSlider):
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'QSlider'
            updateDic(val, ['minimum', 'maximum', 'value', 'isHidden'], at)
            success = 1

        elif isinstance(val, QtWidgets.QLabel):
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'QLabel'
            updateDic(val, ['text', 'isHidden'], at)
            success = 2

        elif isinstance(val, QtWidgets.QRadioButton):
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'QRadioButton'
            updateDic(val, ['isHidden', 'isChecked'], at)
            success = 3

        elif type(val).__name__ == 'AnimatedToggle':
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'AnimatedToggle'
            updateDic(val, ['isHidden', 'isChecked'], at)
            success = 4

        elif type(val).__name__ == 'CollapsibleBox':
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'CollapsibleBox'
            is_expanded = val.toggle_button.isChecked() if hasattr(val, 'toggle_button') else False
            dic[nameWidget][at]['isExpanded'] = is_expanded
            dic[nameWidget][at]['isHidden'] = val.isHidden()
            success = 5

        # --- 2. HANDLE VARIABLES (Ints, Lists, Arrays) ---
        # The IsSafe check now correctly rejects dicts containing QObjects
        elif IsSafe(val):
            # Special handling for 'items' which might be a dictionary
            if at == 'items' and isinstance(val, defaultdict):
                val = list(val.keys())
                at = 'items_names'

            dic[nameWidget][at] = val
            success = 6


    return dic


######extract attributes from a dictionary and assign it to a widget######################
def loadAttributeWidget(widget, nameWidget, dic, progressbar):
    """
    extract attributes from dictionary and assign it to a widget
    Set attribute of a widget
    :param widget:
    :param nameWidget:
    :param dic:
    :return:
    """
    if type(dic[nameWidget])==list:
        return
    lenkeys = len(dic[nameWidget].keys())
    for key in dic[nameWidget].keys():
        progressbar.setValue(progressbar.value())
        if type(dic[nameWidget][key]) == defaultdict and nameWidget=='main':
            if 'type' in dic[nameWidget][key]:
                tpe = dic[nameWidget][key]['type']
                try:
                    Qel = getattr(widget, key)
                except Exception as e:
                    print(e)
                    continue
                subDic = dic[nameWidget][key]
                if tpe == 'QLabel':
                    attr = ['text', 'isHidden']
                    Qel.setVisible(not subDic['isHidden'])
                    #Qel.setText(subDic['text'])
                elif tpe == 'QRadioButton' or tpe == 'AnimatedToggle':
                    attr = ['isHidden', 'isChecked']
                    Qel.setVisible(not subDic['isHidden'])
                    #Qel.setChecked(subDic['isChecked'])
                elif tpe == 'QSlider':
                    attr = ['minimum', 'maximum', 'value', 'isHidden']
                    Qel.setVisible(not subDic['isHidden'])
                    #Qel.setRange(subDic['minimum'], subDic['maximum'])
                    #Qel.setValue(subDic['value'])
            else:
                setattr(widget, key, dic[nameWidget][key])
        else:
            #if key == 'ImCenter':
            #    print(key)
            if key == "segSlice":
                try:
                    if dic[nameWidget][key].ndim==3:
                        dic[nameWidget][key] = dic[nameWidget][key][:,:,0]
                except:
                    pass
            setattr(widget, key, dic[nameWidget][key])



    if hasattr(widget, 'update'):
        if hasattr(widget, 'resetInit'):
            widget.resetInit()
        widget.setDisabled(False)
        widget.update()

########Extract info from current slice##############
def getCurrentSlice(widget, npImage, npSeg, sliceNum, tract=None, tol_slice=3):
    """
    Extract current slice information from image
    Args:
        widget:
        npImage:
        npSeg:
        sliceNum:
        tract:
        tol_slice:

    Returns:

    """
    imSlice = None
    segSlice = None
    trk = None
    if widget.currentWidnowName=='video':
        return npImage, npSeg, trk

    if widget.activeDim == 0:
        imSlice = npImage[sliceNum, :, :]
        segSlice = npSeg[sliceNum, :, :]
        if tract is not None:
            trk = tract[(tract[:,2]>sliceNum-tol_slice)*(tract[:,2]<sliceNum+tol_slice),:]
            trk = trk[:,[0, 1, 2, 3,4,5,6,7]]
    elif widget.activeDim == 1:
        imSlice = npImage[:, sliceNum, :]
        segSlice = npSeg[:, sliceNum, :]
        if tract is not None:
            trk = tract[(tract[:,1]>sliceNum-tol_slice)*(tract[:,1]<sliceNum+tol_slice),:]
            trk = trk[:,[0, 2, 1, 3,4,5,6,7]]
    elif widget.activeDim == 2:
        imSlice = npImage[:, :, sliceNum]
        segSlice = npSeg[:, :, sliceNum]
        if tract is not None:
            trk = tract[(tract[:,0]>sliceNum-tol_slice)*(tract[:,0]<sliceNum+tol_slice), :]
            trk = trk[:, [1,2,0,3,4,5,6,7]]

    return imSlice, segSlice, trk

########assign segmentation to a widget##############
def setSliceSeg(widget, npSeg):
    """
    assign segmentation to a widget
    Args:
        widget:
        npSeg:

    Returns:

    """
    sliceNum = widget.sliceNum
    if widget.currentWidnowName=='video':
        widget.segSlice = npSeg
        return
    if widget.activeDim == 0:
        segSlice = npSeg[sliceNum, :, :]
    elif widget.activeDim == 1:
        segSlice = npSeg[:, sliceNum, :]
    elif widget.activeDim == 2:
        segSlice = npSeg[:, :, sliceNum]
    widget.segSlice = segSlice

########Get current slider value##############
def getCurrentSlider(slider, widget, value):
    """
    Get current slider value
    Args:
        slider:
        widget:
        value:

    Returns:

    """
    rng = slider.maximum() - slider.minimum()
    rngnew = widget.imDepth
    sliceNum = (value - slider.minimum()) * rngnew / rng
    if sliceNum >= widget.imDepth:
        sliceNum = widget.imDepth - 1
    return int(sliceNum)

########Updating image view##############



def updateSight(slider, widget, reader, value, tol_slice=3):
    """
    Update slider and widget
    Args:
        slider:
        widget:
        reader:
        value:
        tol_slice:

    Returns:

    """


    try:
        sliceNum = getCurrentSlider(slider,
                                    widget, value)
        widget.points = []
        widget.selectedPoints = []

        widget.updateInfo(*getCurrentSlice(widget,
                                                        reader.npImage, reader.npSeg,
                                                        sliceNum, reader.tract, tol_slice=tol_slice), sliceNum, reader.npImage.shape,
                          imSpacing = reader.ImSpacing)

        widget.update()
    except Exception as e:
        print(e)
        print('Impossible')

########Change from sagital to coronal and axial##############
def changeCoronalSagittalAxial(slider, widget, reader, windowName, indWind, label, initialState = False, tol_slice=3):
    try:
        widget.changeView(windowName, widget.zRot)
        widget.updateCurrentImageInfo(reader.npImage.shape)
        slider.blockSignals(True)
        slider.setRange(0, reader.ImExtent[indWind])
        slider.blockSignals(False)
        slider.setValue(reader.ImExtent[indWind] // 2)
        label.setText(str_conv(reader.ImExtent[indWind] // 2))

        sliceNum = slider.value()
        widget.points = []
        widget.selectedPoints = []

        widget.updateInfo(*getCurrentSlice(widget,reader.npImage, reader.npSeg,
                                                         sliceNum, reader.tract, tol_slice=tol_slice), sliceNum, reader.npImage.shape, initialState=initialState,
                          imSpacing = reader.ImSpacing)

        widget.update()
    except Exception as e:
        print(e)
        print('Impossible')

###################### standardize between 0 and 255 ######################

def standardize( imdata, value=255.0):
    """
    Standardize image between 0 and 255.0
    Args:
        imdata:

    Returns:

    """
    imdata = (imdata - imdata.min()) * value / np.ptp(imdata) #range
    return imdata

###################### find non zero segmentation values ######################
def getNoneZeroSeg(seg, whiteInd, colorInd, ind_color):
    if colorInd != ind_color:
        return whiteInd[np.where(seg[tuple(zip(*whiteInd))]==colorInd)[0],:]
    else:
        return whiteInd[np.where(seg[tuple(zip(*whiteInd))]>0)[0],:]

###################### find pixel that are not segmented ######################
def getZeroSeg(seg, whiteInd, colrInd):
    return whiteInd[np.where(seg[tuple(zip(*whiteInd))] != colrInd)[0], :]
    return whiteInd[np.hstack((np.where(seg[tuple(zip(*whiteInd))]==9)[0], np.where(seg[tuple(zip(*whiteInd))]==48)[0],
                               np.where(seg[tuple(zip(*whiteInd))]==1235)[0], np.where(seg[tuple(zip(*whiteInd))]==1234)[0])),:]
    return whiteInd[np.hstack((np.where(seg[tuple(zip(*whiteInd))]==9)[0], np.where(seg[tuple(zip(*whiteInd))]==48)[0])),:]

###################### repeating the segmentation ######################
def repetition(shp, coords, numRep, windowName):
    """
    repeat coordinate based on the number of repetitions
    Args:
        shp:
        coords:
        numRep:
        windowName:

    Returns:

    """
    if numRep <= 1 and numRep>=-1:
        return coords

    def updateCoord(ind, coords_final, numRep):
        abs_numRep = abs(numRep)
        for i in range(abs_numRep-1):
            #self.progressBarSaving.setValue((i+1)/(numRep-1))
            tmp = coords.copy()
            if numRep>0:
                tmp[:,ind]+=(i+1)
            elif numRep <0:
                tmp[:,ind]-=(i+1)
            coords_final = np.vstack((coords_final, tmp))

        return coords_final

    coords_final = coords.copy()
    if windowName.lower() == "coronal":
        if numRep>0:
            max_rep = shp[1]-coords_final[0][1]-1
            numRep = max_rep if numRep > max_rep else numRep
        else:
            numRep = max(numRep, -coords_final[0][1]-1)
        coords_final = updateCoord(1, coords_final, numRep)
    elif windowName.lower() == "sagittal":
        if numRep>0:
            max_rep = shp[2]-coords_final[0][2]-1
            numRep = max_rep if numRep > max_rep else numRep
        else:
            numRep = max(numRep, -coords_final[0][2]-1)
        coords_final = updateCoord(2, coords_final,numRep)
    elif windowName.lower() == "axial":
        if numRep>0:
            max_rep = shp[0]-coords_final[0][0]-1
            numRep = max_rep if numRep > max_rep else numRep
        else:
            numRep = max(numRep, -coords_final[0][0]-1)
        coords_final = updateCoord(0, coords_final, numRep)
    return coords_final


###################### Linking MRI and Ultrasound images ######################
def LinkMRI_ECO(pointsMRI, pointsECO, degree = 1):
    """
    Link MRI image to US image
    Args:
        pointsMRI:
        pointsECO:
        degree:

    Returns:

    """


    pointsMRI = np.asarray(pointsMRI)
    pointsECO = np.asarray(pointsECO)
    models = []
    for i in range(3):
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                          ('linear', LinearRegression(fit_intercept=True))])
        model = model.fit(pointsMRI, pointsECO[:,i])
        models.append(model)

    for i in range(3):
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                          ('linear', LinearRegression(fit_intercept=True))])
        model = model.fit(pointsECO, pointsMRI[:,i])
        models.append(model)


    #poly_reg = PolynomialFeatures(degree=degree)
    #X_poly = poly_reg.fit_transform(pointsMRI)
    #pol_reg = LinearRegression()
    #pol_reg.fit(X_poly, pointsECO)
    return models


######################  ######################
def destacked(x,y,z):
    return np.vstack((x, y, z)).transpose(1,0)


###################### generate extrapoints on a line ######################
def generate_extrapoint_on_line(l1, l2, sliceNum):
    """
    :param l1: start line
    :param l2: end line
    :param sliceNum: slice number
    :return: point on the line
    """
    angleline = math.atan2((l2[0] - l1[0]), (l2[1] - l1[1])) * 180 / np.pi
    #print(angleline)

    if (abs(angleline) > 25 and abs(angleline) < 165):
        m = (l2[1] - l1[1]) / (l2[0] - l1[0])
        c = l2[1] - (m * l2[0])
        pts = np.sort([l1[0], l2[0]])
        #argm = np.argmin([l1[1], l2[1]])
        xs = np.linspace(pts[0], pts[1], int(pts[1] - pts[0]) + 1)
        ys = (m * xs + c)
    else:
        m = (l2[0] - l1[0]) / (l2[1] - l1[1])
        c = l2[0] - (m * l2[1])
        pts = np.sort([l1[1], l2[1]])
        #argm = np.argmin([l1[1], l2[1]])
        ys = np.linspace(pts[0], pts[1], int(pts[1] - pts[0]) + 1)
        xs = (m * ys + c)
#xs = np.round(xs)
#ys = np.round(ys)
    d = [[x0, y0, z0] for x0, y0, z0 in zip(xs, ys, [sliceNum] * len(xs))]
    if sum([abs(x-y) for x, y in zip(d[-1],l1)])<5:
        d = d[::-1]
    return d

###################### Add tree root items ######################
def addTreeRoot(treeItem, name, description, color):
    """

    Args:
        treeItem:
        name:
        description:
        color:

    Returns:

    """

    #for i in [0,1]:
    #    treeItem.setForeground(i,QtGui.QBrush(QtGui.QColor(color[0]*255, color[1]*255, color[2]*255, 255)))
    color = [int(c*255) for c in color]
    node1 = QtGui.QStandardItem(name)
    node1.setForeground(QtGui.QBrush(QtGui.QColor(color[0], color[1], color[2], 255)))
    node1.setFlags(node1.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEditable)
    node1.setCheckState(0)
    node2 = QtGui.QStandardItem(description)
    node2.setForeground(QtGui.QBrush(QtGui.QColor(color[0], color[1], color[2], 255)))
    node2.setFlags(node2.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsEditable)
    #node2.setCheckState(0)
    treeItem.appendRow([node1, node2])

###################### find larget connected components ######################
def LargestCC(segmentation, connectivity=3):
    """
    Get largets connected components
    """
    ndim = 3

    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(-1)
        ndim = 4
    labels = label_connector(segmentation, connectivity=connectivity)
    frequency = np.bincount(labels.flat)
    # frequency = -np.sort(-frequency)
    return labels, frequency


def getscale_safe(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Robustly scales image intensities to range dst_min..dst_max.
    Optimized for standard 0-255 medical video frames (excludes black background from stats).
    """
    # 1. Input Validation
    # If data is all zeros or flat, return identity
    if np.min(data) == np.max(data):
        return 0.0, 1.0

    # 2. Get Source Min/Max (Robustly)
    # We only care about NON-ZERO pixels for robust statistics
    # (ignoring the black background mask we just created)
    valid_pixels = data[data > 0]

    if len(valid_pixels) == 0:
        # Fallback if image is somehow entirely black
        src_min, src_max = dst_min, dst_max
    else:
        # Sort pixels to find percentiles accurately without histogram binning errors
        # This is more accurate than the binning method in your original code
        valid_pixels = np.sort(valid_pixels)
        n_pixels = len(valid_pixels)

        # Find index for lower and upper bounds
        idx_min = int(f_low * n_pixels)
        idx_max = int(f_high * n_pixels) - 1

        # Clamp indices to be safe
        idx_min = max(0, min(idx_min, n_pixels - 1))
        idx_max = max(0, min(idx_max, n_pixels - 1))

        src_min = valid_pixels[idx_min]
        src_max = valid_pixels[idx_max]

    # 3. Calculate Scale
    # Avoid division by zero
    if src_max <= src_min:
        scale = 1.0
    else:
        scale = (dst_max - dst_min) / (src_max - src_min)


    return src_min, scale


def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.
    :param np.ndarray data: image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: float src_min: (adjusted) offset
    :return: float scale: scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    #print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        idx = 0
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0
    elif src_max< src_min:
        scale = (dst_max - dst_min) / (src_max - src_min)
    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    #print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale





def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values
    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: np.ndarray data_new: scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    #print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new


def normalize_mri(img):
    src_min, scale = getscale(img, 0, 255)
    new_data = scalecrop(img, 0, 255, src_min, scale)
    return new_data


###################### find convex hul segmentation ######################
def convexhull_spline(total_points, currentWidnowName, sliceNum, npSeg):
    """
    :param total_points:
    :return:
    """


    distance_point_line = lambda p1, p2, p3: np.linalg.norm(np.cross(p3 - p1, p1 - p2)) / np.linalg.norm(p3 - p1)
    angle_ps = lambda p1, p2: math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
    angle_cor = lambda a1: (a1 - 360) if a1 > 180 else a1 + 360 if a1 < -180 else abs(a1) if (a1 < 0, a1 > -180) else a1

    # angle_three = lambda p1,p2, p3: [abs(angle_ps(p1,p3)-angle_ps(p1,p2)), abs(angle_ps(p1,p3)-angle_ps(p2,p3))]
    # angle_correct = lambda a1, a2: [(180-a1) if a1>=180 else a1, (180-a2) if a2>=180 else a2]
    criterion_met = lambda p1, p2, p3: abs(angle_cor(angle_ps(p1, p3)) - angle_cor(angle_ps(p1, p2))) < 75 and abs(
        angle_cor(angle_ps(p1, p3)) - angle_cor(angle_ps(p2, p3))) < 75 and abs(
        angle_ps(p2, p3) - angle_ps(p2, p1)) > 45
    def find_best(outrS, outrind):
        dists = []
        indc = []
        bls = []
        for indices in outrS:
            p1, p2, p3 = d[int(outrind[indices, 3]), :], d[int(outrind[indices, 2]), :], d[int(outrind[indices, 4]), :]
            dists.append(distance_point_line(p1, p2, p3))
            bls.append(criterion_met(p1,p2,p3))
            indc.append(indices)
            if len(indc) > 5:
                break
        ind_m = np.argsort(dists)
        succes = False
        indices = -1
        if dists[ind_m[0]]>=1.2:#1.2 pixel
            for ind in ind_m:
                if bls[ind]:
                    indices = indc[ind]
                    succes = True
                    break

        return indices, succes

    x, y, z = np.where(npSeg == 1500)
    additional_point = np.vstack((x,y,z)).transpose()
    #total_points = np.loadtxt('totalpoints.txt')
    total_points = np.unique(total_points, axis=0)
    remps = []
    if currentWidnowName == 'coronal':
        main_axis = 1
        newp = total_points[total_points[:, 1] == sliceNum, :]
        other_axis = [2,0]
        remps = total_points[total_points[:, 1] != sliceNum, :]
    elif currentWidnowName == 'sagittal':
        main_axis = 2
        newp = total_points[total_points[:,2]==sliceNum,:]
        remps = total_points[total_points[:, 2] != sliceNum, :]
        other_axis = [1, 0]
    elif currentWidnowName == 'axial':
        main_axis = 0
        newp = total_points[total_points[:,0]==sliceNum,:]
        remps = total_points[total_points[:, 0] != sliceNum, :]
        other_axis = [2, 1]

    additional_point = additional_point[:, [other_axis[0], other_axis[1], main_axis]]

    d = np.array([newp[:, other_axis[0]], newp[:, other_axis[1]]]).T

    convex_hull = np.array(LineString(d).convex_hull.exterior.xy).T
    routes = [np.argmin(np.sum(np.abs(con - d), 1)) for con in convex_hull]
    out_route = list(set(np.arange(d.shape[0])) - set(routes))

    if len(out_route)>0:
        sorted_dist = list(cdist(d[routes, :], d[out_route, :]).min(0).argsort(0))
        out_route = np.array(out_route)[sorted_dist]
        dist = lambda x, y: np.linalg.norm(x - y)

        path_dist = lambda ind, routes: np.array(
            [[dist(d[routes[r], :], d[ind, :]), dist(d[routes[r + 1], :], d[ind, :]), ind, routes[r], routes[r + 1]] for r
             in range(len(routes) - 1)])
        for ind in out_route:
            outrind = path_dist(ind, routes)

            outrS = outrind[:, [1]].argsort(0)
            indices, succes = find_best(outrS, outrind)
            if succes:
                indsel = np.where(routes == outrind[indices,3])[0][0]
                routes.insert(indsel + 1, ind)
    tt = newp[routes, :][:,[other_axis[0], other_axis[1]]].T

    tck, u = splprep(tt, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    debug = False
    if debug:

        plt.scatter(convex_hull[:, 0], convex_hull[:, 1]);
        plt.plot(d[routes, 0], d[routes, 1]);
        plt.scatter(d[:, 0], d[:, 1])
        plt.scatter(x_new, y_new)
        plt.show()

    newp = np.array([x_new, y_new, np.repeat(sliceNum, x_new.shape[0])]).transpose()


    pl = Polygon(newp)
    if not pl.is_valid:
        pls = ConvertPToPolygons(newp)
        for ij, pl in enumerate(pls):
            if ij == 0:
                selected_points, edges = fillInsidePol(pl)
            else:
                n_points, edges= fillInsidePol(pl)
                selected_points = np.vstack([selected_points, n_points])
    else:
        selected_points, edges = fillInsidePol(pl)


    selected_points = PermuteProperAxis(np.vstack((additional_point, selected_points)), currentWidnowName)
    return selected_points.astype('int'), []#list(remps)

###################### find selected widget ######################
def locateWidgets(sender, mainw):
    """
    locate widget
    Args:
        sender:
        mainw:

    Returns:

    """
    if sender == mainw.openGLWidget_4 or sender == mainw.openGLWidget_5 or sender == mainw.openGLWidget_6:
        # mri Image
        readerName = 'readView2'
        reader = mainw.readView2
        widgets = []
        if mainw.tabWidget.currentIndex() == 0:
            widgets = [mainw.openGLWidget_4, mainw.openGLWidget_5, mainw.openGLWidget_6]
        return readerName, reader, widgets
    elif sender == mainw.openGLWidget_1 or sender == mainw.openGLWidget_2 or sender == mainw.openGLWidget_3 or sender == mainw.openGLWidget_11:
        # eco
        readerName = 'readView1'
        reader = mainw.readView1
        widgets = []
        #if mainw.tabWidget.currentIndex() == 0:
        widgets = [mainw.openGLWidget_1, mainw.openGLWidget_2, mainw.openGLWidget_3,mainw.openGLWidget_11]
        #elif mainw.tabWidget.currentIndex() == 2:
            #widgets = [mainw.openGLWidget_11]
        return readerName, reader, widgets
    else:
        return None, None, None

###################### searching for additional points ######################
def SearchForAdditionalPoints(nseg, sliceNum, windowname, max_threshold_to_be_line=30, max_lines=2,
                              threshold_jump = 4, line_info = None, active_color_ind=1):
    """
    Search for additional points ...
    Args:
        nseg:
        sliceNum:
        windowname:
        max_threshold_to_be_line:
        max_lines:
        threshold_jump:
        line_info:

    Returns:

    """
    seg = nseg.copy()
    len_lines = dict()
    if line_info is not None:
        if windowname == 'coronal':
            point_sel = np.vstack(line_info)[:, [0, 2]]
            seg[tuple(zip(*point_sel))] = 0
        elif windowname == 'sagittal':
            point_sel = np.vstack(line_info)[:, [0, 1]]
            seg[tuple(zip(*point_sel))] = 0
        elif windowname == 'axial':
            point_sel = np.vstack(line_info)[:, [1, 2]]
            seg[tuple(zip(*point_sel))] = 0
    [x_ind, y_ind] = np.where(seg == np.inf)
    points = np.array([[x, y] for x, y in zip(x_ind, y_ind)])
    if points.shape[0]==0:
        len_lines['h'] = 0
        len_lines['v'] = 0
        return [], False, len_lines
    mean_x, mean_y = points.mean(0)
    def divide_to_lines(points, axis, threshold_jump):
        line_breaks = np.where(np.diff(np.sort(points[:, axis])) > threshold_jump)[0]
        breaks = [0]
        for br in line_breaks:
            if (br-breaks[-1])>5 and (points.shape[0]-br)>5:
                breaks.append(br+1)
            else:
                breaks[-1] = br+1
        breaks.append(points.shape[0])

        return breaks
    success_h = False
    success_v = False

    total_points = []
    for ax in ['h', 'v']:
        len_lines[ax] = 0
        if ax.lower()=='h':
            axis_used = 0
            axis_second=1
            mean_axis = mean_y

        else:
            axis_used = 1
            axis_second = 0
            mean_axis = mean_x
        unique, counts = np.unique(points[:,axis_used], return_counts=True)
        if counts.max() > max_threshold_to_be_line:
            lines = np.sort(unique[counts > max_threshold_to_be_line])
            len_lines[ax] = len(lines)
            l_prev = np.inf
            if len(lines) >= max_lines:
                success = True
                if ax.lower() == 'h':
                    success_h = True
                else:
                    success_v = True
                for l in lines:
                    point_in_line = points[points[:, axis_used] == l, :]
                    breaks = divide_to_lines(point_in_line, axis_second, threshold_jump)
                    if len(breaks)<5:

                        if abs(l_prev - l) < 2 and len(total_points) > 0 and len(breaks)==2:
                            xy_st, xy_end = total_points[-2], total_points[-1]
                            if ax.lower()=='h':
                                xy2_st, xy2_end = [l, point_in_line[:, axis_second].min()], [l, point_in_line[:, axis_second].max()]
                            else:
                                xy2_st, xy2_end = [point_in_line[:, axis_second].min(), l], [point_in_line[:, axis_second].max(), l]
                            if (xy2_st[axis_second] + xy_st[axis_second]) / 2 < mean_axis:
                                ind_st = np.argmax([xy_st[axis_second], xy2_st[axis_second]])
                                xy_s = xy_st if ind_st == axis_used else xy2_st
                                ind_end = np.argmin([xy_end[axis_second], xy2_end[axis_second]])
                                xy_e = xy_end if ind_end == axis_used else xy2_end
                            else:
                                ind_st = np.argmin([xy_st[axis_second], xy2_st[axis_second]])
                                xy_s = xy_st if ind_st == axis_used else xy2_st
                                ind_end = np.argmax([xy_end[axis_second], xy2_end[axis_second]])
                                xy_e = xy_end if ind_end == axis_used else xy2_end
                            total_points = total_points[:-2]
                            total_points.append(xy_s)
                            total_points.append(xy_e)
                        else:
                            if ax.lower() == 'h':
                                for b in range(len(breaks)-1):
                                    total_points.append([l, point_in_line[breaks[b]:breaks[b+1], axis_second].min()])
                                    total_points.append([l, point_in_line[breaks[b]:breaks[b+1], axis_second].max()])
                                #total_points.append([l, point_in_line[:, axis_second].min()])
                                #total_points.append([l, point_in_line[:, axis_second].max()])
                            else:
                                for b in range(len(breaks)-1):
                                    total_points.append([point_in_line[breaks[b]:breaks[b+1], axis_second].min(), l])
                                    total_points.append([point_in_line[breaks[b]:breaks[b+1], axis_second].max(), l])
                        l_prev = l

    if len(total_points)>0:
        total_points = np.unique(total_points, axis=0)
        total_points = total_points[:, [1, 0]]
        total_points = np.hstack((total_points,np.repeat(sliceNum, total_points.shape[0]).reshape(-1,1)))
        total_points = PermuteProperAxis(total_points, windowname)
    return total_points, success_h*success_v, len_lines

###################### find index of selected colors ######################
def _get_color_index2(npSeg, WI):
    uq = np.unique(npSeg[tuple(zip(*WI))])
    inds, us = [], []
    for u in uq:
        ind = npSeg[tuple(zip(*WI))] == u
        inds.append(ind)
        us.append(u)
    return inds, us


def _get_color_index(npSeg, WI):
    if WI is None or len(WI) == 0:
        return [], []

    idx = tuple(zip(*WI))
    extracted_vals = npSeg[idx]  # Shape: (N_pixels,)
    uq = np.unique(extracted_vals)  # Shape: (N_labels,)

    # Broadcasting Magic
    # Compare (1, N_pixels) vs (N_labels, 1)
    # Result is a boolean matrix of shape (N_labels, N_pixels)
    masks = (extracted_vals[None, :] == uq[:, None])

    # Convert rows to list
    return list(masks), list(uq)

###################### updating segmentation ######################

def update_last_video(self, reader, colorInd, whiteInd_all, colorInd2, guide_lines=False):
    """
    Optimized update_last_video:
    - Removes dangerous full-volume calculations.
    - Uses safe Numpy indexing instead of zip(*WI).
    - Prevents crashes during fast painting.
    """
    # 1. Get Frames involved in this update
    # (Assuming whiteInd_all is N x 3: [x, y, frame_idx])
    num_update = np.unique(whiteInd_all[:, 2])

    for iw in num_update:
        # Get the specific frame (2D array)
        current_npseg = reader.seg_ims.get_frame(iw)

        # Extract pixels for this specific frame 'iw'
        # Mask: Check column 2 for frame index
        mask_frame = (whiteInd_all[:, 2] == iw)

        # Get (X, Y) coordinates for this frame
        # Make sure to copy to avoid view issues
        WI = whiteInd_all[mask_frame][:, [0, 1]]

        # --- SAFETY CHECK 1: BOUNDS ---
        # Ensure we don't try to write pixels outside the image (Causes SEGFAULT)
        h, w = current_npseg.shape
        valid_mask = (WI[:, 0] >= 0) & (WI[:, 0] < h) & \
                     (WI[:, 1] >= 0) & (WI[:, 1] < w)
        WI = WI[valid_mask]

        if len(WI) < 1:
            continue

        # --- LOGIC SELECTION ---
        if colorInd != 0:
            # OPTIMIZATION: Use Numpy Indexing (Fast & Safe)
            # Old: tuple(zip(*WI)) -> Slow & Crashy on large arrays
            # New: (WI[:, 0], WI[:, 1])

            # Find pixels that are NOT already the target color
            current_vals = current_npseg[WI[:, 0], WI[:, 1]]
            diff_mask = (current_vals != colorInd)
            WI = WI[diff_mask]

            if len(WI) < 1:
                return

            # History / Undo Logic
            # Note: Ensure _get_color_index uses the new safe version we wrote
            inds, us = _get_color_index(current_npseg, WI)
            self._lastReaderSegCol.append(colorInd)
            self._lastReaderSegInd.append([WI, inds, us, iw])

        else:
            # Erase Mode
            WI = getNoneZeroSeg(current_npseg, WI, colorInd2, 9876)
            if len(WI) < 1:
                return

            self._lastReaderSegCol.append(colorInd)
            inds, us = _get_color_index(current_npseg, WI)
            self._lastReaderSegInd.append([WI, inds, us, iw])

        # Update History Buffers
        self._lastReaderSegPrevCol.append(colorInd2)
        self._undoTimes = 0
        if len(self._lastReaderSegInd) > self._lastMax:
            self._lastReaderSegCol = self._lastReaderSegCol[1:]
            self._lastReaderSegInd = self._lastReaderSegInd[1:]
            self._lastReaderSegPrevCol = self._lastReaderSegPrevCol[1:]

        if guide_lines:
            self._lastlines.append(WI)

        # --- APPLY CHANGES ---
        # Fast Numpy Write
        current_npseg[WI[:, 0], WI[:, 1]] = colorInd

        # Commit to Proxy (Ensure this doesn't call makeObject!)
        reader.commit_frame_segmentation_changes(current_npseg, iw)

        # --- TEXT UPDATE (SAFE VERSION) ---
        # Calculating total volume of a video ((npSeg>0).sum()) is too heavy.
        # It reads 10GB of data for every pixel you draw.
        # SOLUTION: Just calculate the CURRENT FRAME volume for display.

        spacing_vol = self.readView1.ImSpacing[0] ** 3 / 1000

        # Only sum the current frame (Fast)
        if colorInd == 9876:
            vol = (current_npseg > 0).sum() * spacing_vol
            label_type = "Total"
        else:
            colsel = colorInd if colorInd != 0 else colorInd2
            vol = (current_npseg == colsel).sum() * spacing_vol
            label_type = f"Label {colsel}"

        # Update Text
        txt = f'File: {self.filenameView1} | Frame {int(iw)} {label_type} Vol: {vol:0.2f} cm\u00b3'
        self.openedFileName.setText(txt)

        if colorInd == 1500:
            self._lineinfo.append(WI)

def update_last_video2(self, reader, colorInd, whiteInd_all, colorInd2, guide_lines = False):
    """
    update last
    Args:
        self:
        reader:
        colorInd:
        whiteInd:
        colorInd2:
        guide_lines:

    Returns:

    """

    num_update = np.unique(whiteInd_all[:, 2])
    for iw in num_update:
        current_npseg = reader.seg_ims.get_frame(iw)
        whiteInd = whiteInd_all[whiteInd_all[:, 2] == iw, :]
        whiteInd = whiteInd[:, [0,1]]

        if colorInd != 0:
            WI = whiteInd[np.where(current_npseg[tuple(zip(*whiteInd))] != colorInd)[0], :]
            #WI = whiteInd
            if WI.shape[0]< 1:
                return
            inds, us = _get_color_index(current_npseg, WI)
            self._lastReaderSegCol.append(colorInd)
            self._lastReaderSegInd.append([WI, inds, us, iw])

            WI = whiteInd # to be commented to not check

        else:
            WI = getNoneZeroSeg(current_npseg, whiteInd, colorInd2, 9876)
            if WI.shape[0]< 1:
                return
            self._lastReaderSegCol.append(colorInd)
            inds, us = _get_color_index(current_npseg, WI)
            self._lastReaderSegInd.append([WI, inds, us, iw])

        self._lastReaderSegPrevCol.append(colorInd2)
        self._undoTimes = 0
        if len(self._lastReaderSegInd) > self._lastMax:
            self._lastReaderSegCol = self._lastReaderSegCol[1:]
            self._lastReaderSegInd = self._lastReaderSegInd[1:]
            self._lastReaderSegPrevCol = self._lastReaderSegPrevCol[1:]
        if guide_lines:
            self._lastlines.append(WI)
        current_npseg[tuple(zip(*WI))] = colorInd
        reader.commit_frame_segmentation_changes(current_npseg, iw)
        if colorInd == 0:
            colsel = colorInd2
        else:
            colsel = colorInd
        if self._sender in [getattr(self, 'openGLWidget_{}'.format(f)) for f in self.widgets_view1]:

            txt = 'File: {}'.format(self.filenameView1)
            if colorInd == 9876:
                txt += ' TV (US) : {0:0.2f} cm\u00b3'.format((self.readView1.npSeg > 0).sum() * self.readView1.ImSpacing[0] ** 3 / 1000)
            else:
                txt += ' TV (US) : {0:0.2f} cm\u00b3'.format((self.readView1.npSeg == colsel).sum() * self.readView1.ImSpacing[0] ** 3 / 1000)
            self.openedFileName.setText(txt)
        else:
            txt = 'File: {}'.format(self.filenameView2)
            if colorInd==9876:
                txt += ' TV (View2) : {0:0.2f} cm\u00b3'.format((self.readView2.npSeg > 0).sum() * self.readView2.ImSpacing[0] ** 3 / 1000)
            else:
                txt += ' TV (View2) : {0:0.2f} cm\u00b3'.format((self.readView2.npSeg == colsel).sum() * self.readView2.ImSpacing[0] ** 3 / 1000)
            self.openedFileName.setText(txt)
        if colorInd == 1500:
            self._lineinfo.append(WI)


def update_last(self, npSeg, colorInd, whiteInd, colorInd2, guide_lines = False):
    """
    update last
    Args:
        self:
        npSeg:
        colorInd:
        whiteInd:
        colorInd2:
        guide_lines:

    Returns:

    """

    if colorInd != 0:
        WI = getZeroSeg(npSeg, whiteInd, colorInd)
        #WI = whiteInd
        if WI.shape[0]< 1:
            return
        inds, us = _get_color_index(npSeg, WI)
        self._lastReaderSegCol.append(colorInd)
        self._lastReaderSegInd.append([WI, inds, us, 0])
        WI = whiteInd # to be commented to not check

    else:
        WI = getNoneZeroSeg(npSeg, whiteInd, colorInd2, 9876)
        if WI.shape[0]< 1:
            return
        self._lastReaderSegCol.append(colorInd)
        inds, us = _get_color_index(npSeg, WI)
        self._lastReaderSegInd.append([WI, inds, us, 0])

    self._lastReaderSegPrevCol.append(colorInd2)
    self._undoTimes = 0
    if len(self._lastReaderSegInd) > self._lastMax:
        self._lastReaderSegCol = self._lastReaderSegCol[1:]
        self._lastReaderSegInd = self._lastReaderSegInd[1:]
        self._lastReaderSegPrevCol = self._lastReaderSegPrevCol[1:]
    if guide_lines:
        self._lastlines.append(WI)
    npSeg[tuple(zip(*WI))] = colorInd
    if colorInd == 0:
        colsel = colorInd2
    else:
        colsel = colorInd
    if self._sender in [getattr(self, 'openGLWidget_{}'.format(f)) for f in self.widgets_view1]:

        txt = 'File: {}'.format(self.filenameView1)
        if colorInd == 9876:
            txt += ' TV (US) : {0:0.2f} cm\u00b3'.format((self.readView1.npSeg > 0).sum() * self.readView1.ImSpacing[0] ** 3 / 1000)
        else:
            txt += ' TV (US) : {0:0.2f} cm\u00b3'.format((self.readView1.npSeg == colsel).sum() * self.readView1.ImSpacing[0] ** 3 / 1000)
        self.openedFileName.setText(txt)
    else:
        txt = 'File: {}'.format(self.filenameView2)
        if colorInd==9876:
            txt += ' TV (View2) : {0:0.2f} cm\u00b3'.format((self.readView2.npSeg > 0).sum() * self.readView2.ImSpacing[0] ** 3 / 1000)
        else:
            txt += ' TV (View2) : {0:0.2f} cm\u00b3'.format((self.readView2.npSeg == colsel).sum() * self.readView2.ImSpacing[0] ** 3 / 1000)
        self.openedFileName.setText(txt)
    if colorInd == 1500:
        self._lineinfo.append(WI)

###################### select proper widgets ######################
def select_proper_widgets(self):

    widgets = []
    sender = QObject.sender(self)
    if self.tabWidget.currentIndex() == 0:
        widgets = []
        if sender in [self.openGLWidget_4, self.openGLWidget_6,self.openGLWidget_5]:
            widgets.append(self.openGLWidget_4)
            widgets.append(self.openGLWidget_5)
            widgets.append(self.openGLWidget_6)
        elif sender in [self.openGLWidget_1, self.openGLWidget_2,self.openGLWidget_3]:
            widgets.append(self.openGLWidget_1)
            widgets.append(self.openGLWidget_2)
            widgets.append(self.openGLWidget_3)
    elif self.tabWidget.currentIndex() == 1:
        wndnm = self.openGLWidget_11.currentWidnowName
        if wndnm.lower() == 'sagittal':
            widgets.append(self.openGLWidget_2)
        elif wndnm.lower() == 'coronal':
            widgets.append(self.openGLWidget_1)
        elif wndnm.lower() == 'axial':
            widgets.append(self.openGLWidget_3)
        widgets.append(self.openGLWidget_11)
    elif self.tabWidget.currentIndex() == 2:
        wndnm = self.openGLWidget_12.currentWidnowName
        if wndnm.lower() == 'sagittal':
            widgets.append(self.openGLWidget_5)
        elif wndnm.lower() == 'coronal':
            widgets.append(self.openGLWidget_4)
        elif wndnm.lower() == 'axial':
            widgets.append(self.openGLWidget_6)
    return widgets



###################### save numpy array to image ######################
def save_snapshot(q_image, filename):
    # Qt detects format from extension (.png, .jpg)
    # For JPG: quality=100 (0-100)
    # For PNG: quality is compression level (0-9, usually ignored or -1 default)
    if filename.endswith(".jpg"):
        q_image.save(filename, "JPG", 100)
    else:
        q_image.save(filename, "PNG")

    print(f"Saved to {filename}")


def get_global_crop_box(image_list, threshold=10):
    """
    Finds the smallest rectangle that contains all non-black pixels
    across the entire set of images.
    """
    # 1. Create a "Maximum Projection"
    # We stack all images effectively on top of each other to see the full valid area
    max_proj = np.zeros_like(image_list[0])
    for img in image_list:
        max_proj = np.maximum(max_proj, img)
    max_proj = cv2.cvtColor(max_proj, cv2.COLOR_BGR2GRAY)
    # 2. Threshold to separate image content from pure black background
    # Any pixel value > threshold is considered "content"
    _, thresh = cv2.threshold(max_proj, threshold, 255, cv2.THRESH_BINARY)

    # 3. Find the bounding box of the white area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: If image is all black, return full size
        return image_list
    main_contour = max(contours, key=cv2.contourArea)

    # --- 3. Filter: Keep ONLY the largest connected object ---
    # This ignores the "small not connected object" you mentioned
    main_contour = max(contours, key=cv2.contourArea)

    # Create a "Cookie Cutter" mask (White on main object, Black everywhere else)
    mask = np.zeros_like(max_proj)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    # Calculate crop box from this clean mask
    x, y, w, h = cv2.boundingRect(main_contour)
    print(f"Detected Main Area: x={x}, y={y}, w={w}, h={h}")

    # --- 4. Apply Mask & Crop to every frame ---
    cleaned_stack = []
    for img in image_list:
        # Ensure img matches mask channels
        if len(img.shape) == 3:
            # If image is color, we need a 3-channel mask
            mask_3ch = cv2.merge([mask, mask, mask])
            cleaned_img = cv2.bitwise_and(img, mask_3ch)
        else:
            cleaned_img = cv2.bitwise_and(img, mask)

        # Crop to the bounding box
        cropped_img = cleaned_img[y:y + h, x:x + w]
        cleaned_stack.append(cropped_img)

    return cleaned_stack


def detect_modality_and_window(data):
    """
    Analyzes the numpy array to guess the modality and returns
    the prepared uint8 image.
    """
    # 1. Check Dimensions
    # 4D data is almost always MRI (fMRI, DTI, or Multi-phase)
    if data.ndim == 4:
        print("Detected: 4D MRI/DTI")
        # For visualization, we usually take the first volume or mean
        # But per your logic, we might visualize one channel
        return (255.0*(data - data.min())/(data.max()-data.min())).astype(np.uint8)

    min_val = data.min()
    max_val = data.max()

    # 2. Check for CT Signature (Negative Values)
    # CT Air is -1000. If we see values significantly below 0, it's CT.
    # We use -200 as a safe threshold to distinguish from noisy MRI background.
    if min_val < -200:
        print(f"Detected: CT (Range: {min_val} to {max_val})")
        # Apply Soft Tissue Window (modify as needed)
        return apply_ct_window(data, window_center=40, window_width=400)

    # 3. Check for MRI Signature (Positive only, start at 0)
    else:
        print(f"Detected: MRI (Range: {min_val} to {max_val})")
        return normalize_mri(data)



def apply_ct_window(data, window_center=40, window_width=400):
    min_visible = window_center - (window_width / 2.0)
    max_visible = window_center + (window_width / 2.0)
    windowed = np.clip(data, min_visible, max_visible)
    # Normalize 0-255
    normalized = (windowed - min_visible) / (max_visible - min_visible)
    return (normalized * 255.0).astype(np.uint8)


def normalize_mri(data):
    # Robust normalization using percentiles (ignores outliers)
    p1 = np.percentile(data, 1)  # Ignore bottom 1% (noise)
    p99 = np.percentile(data, 99)  # Ignore top 1% (artifacts)

    normalized = np.clip(data, p1, p99)
    normalized = (normalized - p1) / (p99 - p1)
    return (normalized * 255.0).astype(np.uint8)


###################### saving 3D images ######################
def save_3d_img(reader, file, img, format='tif', type_im = 'mri', cs=['RAS', 'AS', True]):

    cors, asto, save_csv = cs

    if format == 'tif':

        tif.imsave(file+'.tif', img, bigtiff=True)
    elif format == 'nifti':
        if file[-7:] != '.nii.gz':
            file = file +'.nii.gz'
        if type_im == 'mri':
            transpose_axis = [2, 1, 0]
            flip_axis = None
        elif type_im == 'eco':
            transpose_axis = [2, 1, 0]
            flip_axis = 1
        #img = np.transpose(img,transpose_axis)
        #img = np.flip(img, axis=flip_axis)

        if hasattr(reader, 'affine'):
            affine = reader.im.affine
            #if reader.s2c:
            #    try:
            #        affine = reader._imChanged_affine
            #    except Exception as e:
            #        print(e)
        else:
            affine = np.eye(4)
            affine[:-1, -1] = np.array(reader.ImOrigin)
            np.fill_diagonal(affine[:-1, :-1], reader.ImSpacing)
        try:
            transform, _ = convert_to_ras(affine, target=cors)
        except:
            if hasattr(reader, 'source_system'):
                transform , _ = convert_to_ras(affine, target=reader.source_system)
            else:
                transform, _ = convert_to_ras(affine, target=reader.target_system)
        if hasattr(reader, 'header'):
            hdr = reader.header
        else:
            hdr = nib.Nifti1Header()
            hdr['dim'] = np.array([3, img.shape[0], img.shape[1], img.shape[2], 1, 1, 1, 1])
        new_im = nib.Nifti1Image(img.transpose(2, 1, 0)[::-1, ::-1, ::-1], affine) #get back to original

        if asto.lower()=='as':
            new_im =  new_im.as_reoriented(transform) # reorient to the right transformation system
        elif asto.lower()=='to':
            img2 = apply_orientation(img, transform)
            #new_affine = affine @ nib.orientations.inv_ornt_aff(transform, img.shape)
            new_im = nib.Nifti1Image(img2, affine)
            #new_affine = new_im.as_reoriented(transform).affine
            #new_im = nib.Nifti1Image(img, new_affine, header=hdr)
        new_im.header['pixdim'] = hdr['pixdim']
        nib.save(new_im, file)

        if save_csv:
            with open(file+'.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
                w = csv.DictWriter(f, reader.metadata.keys())
                w.writeheader()
                w.writerow(reader.metadata)

        #with open(file + '.json', 'w') as fp:
        #    json.dump(reader.metadata, fp)

###################### compute volume of segmented area ######################
def compute_vol_seg(npSeg_orig, segwnd):
    """
    Compute segmentation volume
    Args:
        npSeg_orig:
        segwnd:

    Returns:

    """
    npSeg = npSeg_orig.copy()
    selected_points_total = np.zeros((0,0,0))
    try:
        segwnd = segwnd.lower()
        if segwnd == 'coronal':
            searchwnd = npSeg.shape[1]
        elif segwnd == 'sagittal':
            searchwnd = npSeg.shape[2]
        elif segwnd == 'axial':
            searchwnd = npSeg.shape[0]
        nd = 0
        for slc in range(searchwnd):

            if segwnd=='coronal':
                seg = npSeg[:, slc, :]
            elif segwnd == 'sagittal':
                seg = npSeg[:,:,slc]
            elif segwnd == 'axial':
                seg = npSeg[slc,:,:]
            [x_ind, y_ind] = np.where(seg > 0)
            pointxy = np.array([[x, y] for x, y in zip(x_ind, y_ind)])
            if pointxy.shape[0]>1:
                total_points, success, len_lines = SearchForAdditionalPoints(seg, slc, segwnd)
                if len(total_points) > 0 and success:

                    selected_points,_ = convexhull_spline(total_points, segwnd, slc, npSeg)
                    tmp_seg = seg.copy()

                    tmp_seg[selected_points[:, 0], selected_points[:, 1]] = 1
                    if (selected_points.shape[0] - sum(npSeg[tuple(zip(*selected_points))]>0))>0 and sum([len_lines[key] for key in len_lines.keys()])<40:
                        if nd==0:
                            selected_points_total = selected_points
                            nd=1
                        else:
                            selected_points_total = np.vstack((selected_points_total, selected_points))
                        #seg[seg != 0] = 0
                        #seg[selected_points[:, 0], selected_points[:, 1]] = 1
    except Exception as e:
        print('Compute Vol Seg Error')
        print(e)
    return selected_points_total


###################### export information from a table ######################
def export_tables(self, file):
    """
    Export data to tables
    Args:
        self:
        file:

    Returns:

    """
    if file[0]=='':
        return
    num_header = self.table_widget_measure.columnCount()
    headers = []
    dicts = defaultdict(list)
    rows = self.table_widget_measure.rowCount()
    cols = self.table_widget_measure.columnCount()
    for i in range(num_header):
        itm = self.table_widget_measure.horizontalHeaderItem(i)
        if itm is not None:
            txt = itm.text()
        else:
            txt = 'unknown'
        headers.append(txt)

    dicts_0 = defaultdict(list)
    for r in range(rows):
        dicts_0[r] = []
        for c in range(cols):
             itm = self.table_widget_measure.item(r, c)
             if itm is not None:
                txt = itm.text()
                dicts_0[r].append(txt)

    with open(file + '.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        f.write(','.join(headers)+'\n')

        for key in dicts_0.keys():
            f.write(','.join(dicts_0[key])+'\n')

    #with open(file+'.json', 'w') as fp:
    #    json.dump(dicts, fp)

###################### n4 Bias filed correction ######################
def N4_bias_correction(image_nib, use_otsu=True, shrinkFactor=1,
                       numberFittingLevels=6, max_iter=5):
    inputImage = read_nib_as_sitk(image_nib)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    if use_otsu:
        #maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        threshold_val = Threshold_MultiOtsu(image_nib.get_fdata(), 1)[0]
        a = image_nib.get_fdata().copy()
        a[a <= threshold_val] = 0
        a[a > threshold_val] = 1
        mask_image = make_image_using_affine(a, image_nib.affine)
        maskImage = read_nib_as_sitk(mask_image)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)

    else:

        mask_image = nib.Nifti1Image((image_nib.get_fdata()>0).astype(np.int8), image_nib.affine, header=image_nib.header)
        #maskImage = sitk.Cast(sitk.GetImageFromArray((image.get_fdata()>0).astype('int'), sitk.sitkInt8), sitk.sitkUInt8)
        maskImage = read_nib_as_sitk(mask_image)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)


    if shrinkFactor > 1:
        inputImage = sitk.Shrink(
            inputImage, [shrinkFactor] * inputImage.GetDimension()
        )
        maskImage = sitk.Shrink(
            maskImage, [shrinkFactor] * inputImage.GetDimension()
        )


    corrector = sitk.N4BiasFieldCorrectionImageFilter()



    if max_iter > 5:
        corrector.SetMaximumNumberOfIterations(
            [max_iter] * numberFittingLevels
        )

    corrected_image = corrector.Execute(inputImage, maskImage)
    affine = make_affine(corrected_image)
    nib_im = nib.Nifti1Image(sitk.GetArrayFromImage(corrected_image).transpose(), affine)
    return nib_im



###################### creat an image given affine matrix ######################
def make_image_using_affine(data, affine, header=None):
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data, affine, header)

###################### creat an image based on another one ######################
def make_image(data, target):
    return nib.Nifti1Image(data, target.affine, target.header)


###################### unique value of an image ######################
def len_unique(im):
    uq = np.unique(im)
    return uq, uq.shape[0]


###################### function used in slice interpolation ######################
def bwperim(bw, n=4):
    #https://github.com/lforet/CoinVision/blob/master/build/mahotas/mahotas/bwperim.py
    # with some modifications
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

###################### function used in slice interpolation ######################
def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

###################### Find distance map of image ######################
def bwdist(im):
    '''
    Find distance map of image
    '''

    dist_im = distance_transform_edt(1-im)
    return dist_im
###################### Slice interpolation ######################
def slice_intepolation(reader, slices, currentWidnowName, colorInd, WI):

    '''
    Interpolate between two slices in the image
    '''

    slices = np.unique(slices)
    slices.sort()
    interpolated_slices = []
    if currentWidnowName== 'coronal':
        selected_slices = reader.npSeg[:, slices, :]
        selected_slices = np.transpose(selected_slices, [0, 2, 1])
    elif currentWidnowName == 'sagittal':
        selected_slices = reader.npSeg[:, :, slices]
    elif currentWidnowName == 'axial':
        selected_slices = reader.npSeg[slices, :, :]
        selected_slices = np.transpose(selected_slices, [1, 2, 0])
    elif currentWidnowName=='video':
        # 1. Initialize Array: (Height, Width, N_Slices)
        # Use 'len(slices)' instead of shape if slices is a list
        num_slices = len(slices)
        selected_slices = np.zeros((reader.npSeg.shape[0], reader.npSeg.shape[1], num_slices), dtype=np.uint8)
        # 2. Loop with Enumerate to keep track of index 'i'
        for i, slice_idx in enumerate(slices):
            # READ FROM SEG PROXY (Not Video Image)
            # We want the Label (2D), not the MRI/Video (3D RGB)
            mask = reader.seg_ims.get_frame(slice_idx)
            # 3. Assign to the specific layer 'i'
            selected_slices[:, :, i] = mask
    ind_zero = selected_slices != colorInd
    selected_slices[ind_zero]=0
    for j in range(slices.shape[0]-1):
        top = (selected_slices[..., j]>0).astype('int')
        bottom = (selected_slices[..., j+1]>0).astype('int')
        max_slice =slices[j+1]
        min_slice = slices[j]
        precisions = np.linspace(0, 2, (max_slice - min_slice+1))[1:-1]#np.arange(2 / (max_slice - min_slice-1), 2, 2 / (max_slice - min_slice-1))
        top = signed_bwdist(top)
        bottom = signed_bwdist(bottom)

        # row,cols definition
        r, c = top.shape

        # rejoin top, bottom into a single array of shape (2, r, c)
        top_and_bottom = np.stack((top, bottom))

        # create ndgrids
        points = (np.r_[0, 2], np.arange(r), np.arange(c))
        for k, precision in enumerate(precisions):
            xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
            xi = np.c_[np.full((r*c),precision), xi]
            # Interpolate for new plane
            out = interpn(points, top_and_bottom, xi)
            out = out.reshape((r, c))
            sliceNo = min_slice + k + 1
            # Threshold distmap to values above 0
            #out = out > 0
            out = np.argwhere(out > 0)
            out = np.hstack([out[:,[1,0]], np.ones([out.shape[0], 1]) * sliceNo])
            interpolated_slices.append(out)

    return PermuteProperAxis(np.concatenate(interpolated_slices), currentWidnowName).astype('int')

###################### function used in semgentation based on circle ######################
def seperate_lcc(whiteInd, center):
    minw = whiteInd.min(0)
    maxw = whiteInd.max(0)
    df = maxw - minw
    img = np.zeros([df[0] + 1, df[1] + 1])
    img[whiteInd[:, 0] - minw[0], whiteInd[:, 1] - minw[1]] = 1
    img_l, img_f = LargestCC(img, connectivity=1)
    index_sel = img_l[center[1]-minw[0], center[0]-minw[1] ]
    nw = np.argwhere(img_l == index_sel)
    nw[:, 0] += minw[0]
    nw[:, 1] += minw[1]
    whiteInd = np.hstack([nw, np.ones([nw.shape[0], 1]) * (whiteInd[0, 2])]).astype('int')
    return whiteInd

###################### find sequence name from DICOM ######################
def get_SequenceName(SequenceName):
    if 'ep_b' in SequenceName:
        return 'dwi'
    elif 'epfid2d' in SequenceName:
        return 'perf'
    elif 'epfid3d1_15' in SequenceName or 'fl3d1r' in SequenceName:
        return 'swi'
    elif 'epse2d' in SequenceName:
        return 'dwi' #(when b-vals specified)
    elif 'fl2d' in SequenceName:
        return 'localizer'
    elif 'fl3d1r_t' in SequenceName:
        return 'angio'
    elif 'spc3d' in SequenceName:
        return 'T2'
    elif 'spcir' in SequenceName or 'tir2d' in SequenceName:
        return 'flair'
    elif 'spcR' in SequenceName:
        return 'PD'
    elif 'tfl3d' in SequenceName:
        return 'T1'
    elif 'tfl_me3d5_16ns' in SequenceName:
        return 'T1' #T1 (ME-MPRAGE)
    elif 'tse2d' in SequenceName or 'tse3d' in SequenceName:
        return 'T2'


def adapt_to_size(imAzero, NewSpacing, Spacing, borderp):

    def signdf(df):
        if df <= 0:
            return -1
        else:
            return 1
    # minus_sign = True
    maxdim = np.array([el * sp / NewSpacing for el in imAzero.shape for sp in Spacing]).max()
    df = (maxdim - 192)
    minus_sign = False
    prev_sign = signdf(df)
    while True:
        maxdim = np.array([el * sp / NewSpacing for el in imAzero.shape for sp in Spacing]).max()
        df = (maxdim - 192)
        if signdf(df) != prev_sign:
            break
        if abs(abs(df) - borderp) < 10:  # and df <= 0:
            break
        if minus_sign:
            NewSpacing -= 0.1
        else:
            NewSpacing += 0.1
        prev_sign = signdf(df)
    while True:
        imAa = resample_to_output(imAzero, [NewSpacing, NewSpacing, NewSpacing])
        df = (np.max(imAa.shape) - 192)
        if signdf(df) != prev_sign and df < 0:
            break
        prev_sign = signdf(df)
        if abs(df) <= borderp and df <= 0:
            break
        else:
            if signdf(df) == -1:
                minus_sign = True
            else:
                minus_sign = False
        if minus_sign:
            NewSpacing -= 0.1
        else:
            NewSpacing += 0.1
    return imAa


def histogram_equalization(source):
    def histogram_equalization_3d(image, method='ehist'):

        # Reshape the 3D image into a 2D array with shape (num_slices, height * width)
        num_slices, height, width = image.shape
        image = normalize_mri(image)
        flattened_image = image.reshape((num_slices, height * width))
        #flattened_image = normalize_mri(flattened_image)
        # Apply histogram equalization to the flattened image
        if method=='ehist':
            alg = cv2.equalizeHist
        elif method=='clahe':
            clip_limit = 2; tile_grid_size=(8,8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            alg = clahe.apply

        equalized_flattened = np.apply_along_axis(alg, axis=1,
                                                  arr=(flattened_image).astype(np.uint8))

        # Reshape the equalized 2D array back to 3D
        equalized_image = equalized_flattened.reshape((num_slices, height, width))

        return equalized_image
    # Compute histograms
    reference = histogram_equalization_3d(source, method='clahe')
    #source = rescale_between_a_b(sourceo, 0, 255)
    return normalize_mri(reference)
