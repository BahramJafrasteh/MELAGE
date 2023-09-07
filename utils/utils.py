

__AUTHOR__ = 'Bahram Jafrasteh'

import sys
sys.path.append("../../")
import numpy as np
import struct
from dataclasses import dataclass
from PyQt5 import QtWidgets, QtCore, QtGui
import json
import math
import os
import SimpleITK as sitk

from PyQt5.QtCore import Qt
import cv2
import numpy as np
from skimage.transform import resize as image_resize_skimage
from skimage.transform import rotate as image_rotate_skimage
from collections import defaultdict
from qtwidgets import AnimatedToggle
try:
    import nibabel as nib
except:
    None
try:
    from utils.source_folder import source_folder
except:
    pass
# Direction of medical image Left, Right, Posterior Anterior, Inferior, Superior
code_direction = (('L', 'R'), ('P', 'A'), ('I', 'S'))


###################### Item class for read kretz ######################
@dataclass
class Item:
    tagcl: bytes
    tagel: bytes
    size: bytes



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


###################### Reading files with desired coordinate system ######################
def read_file_with_cs(atlas_file, expected_source_system='RAS'):
    # Read NIFTI images with desired coordinate system
    from nibabel.orientations import aff2axcodes, axcodes2ornt, apply_orientation, ornt_transform
    import nibabel as nib
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


###################### Convert NIBABEL image to SITK ######################
def read_nib_as_sitk(image_nib, dtype=None):
    # From https://github.com/gift-surg/PySiTK/blob/master/pysitk/simple_itk_helper.py
    if dtype is None:
        dtype = image_nib.header["bitpix"].dtype
    nda_nib = image_nib.get_data().astype(dtype)
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






###################### Resample images to desired spacing ######################
def resample_to_spacing(im, newSpacing):
    from nibabel.processing import resample_to_output
    return resample_to_output(im, [newSpacing, newSpacing, newSpacing])

###################### Help dialogue to open new image ######################
def help_dialogue_open_image(path):
    try:
        im = nib.load(path).dataobj
    except:
        from pydicom.filereader import dcmread
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

###################### A class to  resize image######################

class resize_window(QtWidgets.QDialog):
    from PyQt5.QtCore import pyqtSignal
    closeSig = pyqtSignal()
    resizeim = pyqtSignal(object)
    comboboxCh = pyqtSignal(object, object)
    """
    A dialog for combo box created for reading 4d images
    """
    def __init__(self, parent=None, use_combobox=False):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        Dialog = self.window()
        self.use_combobox = use_combobox
        self.setupUi(Dialog)

        self._status = False

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(500, 112)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(10,10,10,10)
        self.grid_main.setObjectName("gridLayout")
        self.hbox = QtWidgets.QHBoxLayout()

        self.label_warning = QtWidgets.QLabel()
        self.label_warning.setText('The pixels are not isotropic. Do you want to resize image (isotropic)?')

        self.label_current_spc0 = QtWidgets.QLabel()
        self.label_current_spc0.setText('Current Spacing')

        self.label_current_spc = QtWidgets.QLabel()
        #self.label_current_spc.setReadOnly(True)
        self.label_current_spc.setText('0,0,0')

        self.hbox.addWidget(self.label_current_spc0)
        self.hbox.addWidget(self.label_current_spc)

        self.hbox2 = QtWidgets.QHBoxLayout()
        self.label_new_spc0 = QtWidgets.QLabel()
        self.label_new_spc0.setText('New Spacing')
        self.label_new_spc0.setStyleSheet('color: Red')
        self.label_new_spc = QtWidgets.QDoubleSpinBox()
        self.label_new_spc.setMinimum(0.01)
        self.label_new_spc.setMaximum(20)
        self.label_new_spc.setValue(1.0)
        self.label_new_spc.setDecimals(20)

        self.hbox2.addWidget(self.label_new_spc0)
        self.hbox2.addWidget(self.label_new_spc)
        _translate = QtCore.QCoreApplication.translate
        self.pushbutton = QtWidgets.QDialogButtonBox()
        self.pushbutton_cancel = QtWidgets.QPushButton()
        self.pushbutton.setStandardButtons(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.pushbutton.accepted.connect(self.accept_it)
        self.pushbutton.rejected.connect(self.reject_it)
        #self.pushbutton.setText(_translate("Dialog", "OK"))
        self.grid_main.addWidget(self.label_warning, 0,0,1,1)
        self.grid_main.addLayout(self.hbox, 1,0,1,1)
        self.grid_main.addLayout(self.hbox2,2,0,1,1)
        if self.use_combobox:
            self.comboBox_image = QtWidgets.QComboBox()
            self.comboBox_image.setObjectName("comboBox_image")
            self.comboBox_image.addItem("")
            self.comboBox_image.addItem("")
            self.comboBox_image.currentIndexChanged.connect(self.comboBOX_changed)
            self.grid_main.addWidget(self.comboBox_image, 3, 0, 1, 1)

            self.grid_main.addWidget(self.pushbutton, 4, 0, 1, 1)
        else:
            self.grid_main.addWidget(self.pushbutton, 4, 0, 1, 1)



        self.retranslateUi(Dialog)

        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def comboBOX_changed(self):
        ind = self.comboBox_image.currentIndex()
        self.comboboxCh.emit(None, ind)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Resize..."))
        _translate = QtCore.QCoreApplication.translate
        if self.use_combobox:
            self.comboBox_image.setItemText(0, _translate("Form", "        Top Image        "))
            self.comboBox_image.setItemText(1, _translate("Form", "      Bottom Image      "))
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(resize_window, self).closeEvent(a0)
    def accept_it(self):
        self._status = True
        if self.use_combobox:
            index = self.comboBox_image.currentIndex()
            self.resizeim.emit(index)
        self.accept()

    def reject_it(self):
        self._status = False
        self.reject()

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
    from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform
    orig_orient = nib.io_orientation(affine)
    source_system = ''.join(list(aff2axcodes(affine, code_direction)))# get direction
    target_orient = axcodes2ornt(target, code_direction)
    transform = ornt_transform(orig_orient, target_orient)

    return transform, source_system

###################### identify current coordinate system ####################
def getCurrentCoordSystem(affine):
    from nibabel.orientations import aff2axcodes
    orig_orient = nib.io_orientation(affine)
    source_system = ''.join(list(aff2axcodes(affine, code_direction)))# get direction
    return source_system

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
    im = nib.load(file)  # read image

    if im.ndim == 4:
        from nibabel.funcs import four_to_three
        im = four_to_three(im)[0]  # select the first image

    transform, _ = convert_to_ras(im.affine, target=reader.target_system)
    im = im.as_reoriented(transform)

    data = im.get_fdata()
    if not all([i == j for i, j in zip(reader.npImage.shape, data.shape)]):
        return data, True, False
    data_add = None
    if rhasattr(self,'readImECO.npSeg'):
        if reader != self.readImECO:
            data_add = self.readImECO.npSeg
    if rhasattr(self, 'readImMRI.npSeg'):
        if reader != self.readImMRI:
            data_add = self.readImMRI.npSeg
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

###################### Updating current color scheme ##################
def update_color_scheme(self, data, data_add=None, dialog=True, update_widget=True):
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
        uq1 = [l for l in np.unique(data_add) if l > 0]
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
            from PyQt5.QtWidgets import QFileDialog
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
                filen = source_folder + '/color/Simple.txt'
            elif len_u<=9:
                filen = source_folder + '/color/Tissue.txt'
            elif len_u <90:
                filen = source_folder + '/color/albert_LUT.txt'
            else:
                filen = source_folder + '/color/mcrib_LUT.txt'
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
        self.color_index_rgb, self.color_name, self.colorsCombinations = generate_color_scheme_info(possible_color_index_rgb, possible_color_name)
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
            widget.paint(self.readImECO.npSeg,
                         self.readImECO.npImage, None)
        elif k in [24]:
            widget.paint(self.readImMRI.npSeg,
                         self.readImMRI.npImage, None)
        else:
            widget.colorObject = colorPen
            #widget.colorInd = len(self.colorsCombinations)
            widget.makeObject()
            widget.update()

###################### Compute volume according to the selected region ##################
def compute_volume(reader, filename, inds):
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
    txt = 'File: {}, '.format(filename)
    if reader.type=='t1':
        txt += 'TV (MRI) : {0:0.2f} cm\u00b3'.format((vol))
    else:
        txt += 'TV (US) : {0:0.2f} cm\u00b3'.format((vol))
    return txt


###################### Generate info for color schemes ##################
def generate_color_scheme_info(color_index_rgb, color_name):
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
        import matplotlib
        import matplotlib.cm
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
                pass
            addTreeRoot(parent, indc, descp, clrvalue[1:-1])

    check_nul = [l for l in self.color_name if '9876' in l.split('_')]
    colr = [1, 0, 0, 1]
    if len(check_nul)==0:
        parent = self.tree_colors.model().sourceModel().invisibleRootItem()
        addTreeRoot(parent, '9876', 'Combined', colr)

################# SET IMAGE SCHEME ###########################
def update_image_sch(self, info=None, color = [1,1,0],loaded = False):
    """
    SET new color scheme to read multiple images
    Args:
        self:

    Returns:

    """

    ######################################
    if hasattr(self, 'tree_images'):

        parent = self.tree_images.model().sourceModel().invisibleRootItem()


        [fileObj, index] = info

        indc='Unknow'
        if index==0:
            indc='US'
        elif index==1:
            indc='Fetal'
        elif index==2:
            indc='MRI'
        elif index==3:
            indc='US_Seg'
        elif index==4:
            indc='Fetal_Seg'
        elif index == 5:
            indc = 'MRI_Seg'
        color = [int(c * 255) for c in color]
        for file in fileObj[0]:
            if '*' in file:
                continue
            descp = os.path.basename(file)
            if info is not None:
                self.imported_images.append([[[file, fileObj[1]], index], color, loaded, indc])

            node1 = QtGui.QStandardItem(indc)

            node1.setForeground(QtGui.QBrush(QtGui.QColor(color[0], color[1], color[2], 255)))

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

################# Second way to clean parent image ###########################
def clean_parent_image2(self, filename, indc):

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
        if i==index_row:
            continue
        if signal.checkState() == Qt.Checked:
            if signal.text() in indc:
                try:
                    self.tree_images.model().sourceModel().itemChanged.disconnect(self.changeImage)
                except:
                    pass
                signal.setCheckState(Qt.Unchecked)
                [info, _, _, _] = self.imported_images[i]
                if indc in info[0][1]:
                    indices.append(i)
                try:
                    self.tree_images.model().sourceModel().itemChanged.connect(self.changeImage)
                except:
                    pass
    for ind in indices:
        self.imported_images.pop(ind)
        parent = self.tree_images.model().sourceModel().invisibleRootItem()
        parent.removeRow(ind)


################# first way to clean parent image ###########################
def clean_parent_image(self, index_row, indc):
    parent = self.tree_images.model().sourceModel().invisibleRootItem()
    for i in range(parent.rowCount()):
        signal = parent.child(i)
        if i==index_row:
            continue
        if signal.checkState() == Qt.Checked:
            if signal.text() in indc:
                try:
                    self.tree_images.model().sourceModel().itemChanged.disconnect(self.changeImage)
                except:
                    pass
                signal.setCheckState(Qt.Unchecked)
                try:
                    self.tree_images.model().sourceModel().itemChanged.connect(self.changeImage)
                except:
                    pass


################# read color information from text files ###########################
def read_txt_color(file, mode ='lut', from_one= False):
    """
    read color information from text files
    Args:
        file: name of the file
        mode: mode of reading
        from_one:

    Returns:

    """
    import re
    inital_col = []
    if mode=='lut':
        with open(file, 'r') as fp:
            lines = fp.readlines()
            num_colors = len(lines) // 2
            color_name = []
            color_info = []
            r = 0
            for n, l in enumerate(lines):
                if n%2 ==0:
                    if from_one:
                        color_name.append('{}_'.format(r+1)+l.rstrip('\n'))
                        r += 1
                    else:
                        color_name.append(lines[n+1].split(' ')[0]+ '_'+l.rstrip('\n'))
                else:
                    color_info.append([int(i) for i in l.rstrip("\n").split(' ')])

            #color_info = [[int(i) for i in l.rstrip("\n").split(' ')] for l in lines if not l[4].isalpha()]
            color_index_rgb = np.array(color_info).astype('float')
            color_index_rgb[:, [1, 2, 3, 4]] = color_index_rgb[:, [1, 2, 3, 4]] / 255.0
            inital_col = color_index_rgb[:, 0].copy()
            if from_one:
                color_index_rgb[:, 0] = np.arange(color_index_rgb.shape[0])+1
            #color_name = [l.rstrip('\n') for l in lines if l[0].isalpha()]
    else:
        # itk
        with open(file, 'r') as fp:
            lines = fp.readlines()
            #color_name = [l.split('\n')[0].split(',')[0]+'_'+l.split('\n')[0].split(',')[-1] for l in lines]
            for id, l in enumerate(lines):
                if l[0] == '#':
                    continue
                try:
                    #spl = [r for r in re.split(r'[ ,|;"]+', l[:-1]) if r != '']
                    spl = [r.replace('"', '') for r in re.sub('\s+', ' ', l[:-1]).split() if r != '']
                    int(spl[0])
                    if [r for r, s in enumerate(spl) if not s.isnumeric()][0]==7:
                        #itk mode
                        last_before_name = 7
                        index_colr = 4
                    elif [r for r, s in enumerate(spl) if not s.isnumeric()][0]==4:
                        last_before_name=4
                        index_colr=4
                    else:
                        last_before_name = 7
                        index_colr = 4
                    break
                except:
                    continue

            #color_name = [l.split('\t')[0]+"_"+l.split('\t')[-1][1:-2] for l in lines[id:]]
            color_name = [[r for r in re.sub('\s+', ' ', l).split() if r != '' and r!='\n'][0] + "_"+' '.join([r.replace('"', '') for r in re.sub('\s+', ' ', l[:-1]).split() if r != '' and r!='\n'][last_before_name:]) for l in lines[id:] if l[0]!='#']
            color_index_rgb = np.array([[int(float(s)) for s in [r for r in re.sub('\s+', ' ', l).split() if r != '' and r != '\n'][:index_colr]] for l in
             lines[id:] if l[0] != '#']).astype('float')
            #color_index_rgb = np.array([[int(l.split('\t')[0]), int(l.split('\t')[1]), int(l.split('\t')[2]), int(l.split('\t')[3]), 1] for l in
            #          lines[id:]]).astype('float')
            color_index_rgb = np.hstack( (color_index_rgb, np.ones((color_index_rgb.shape[0],1))))
            color_index_rgb[:, [1, 2, 3]] = color_index_rgb[:, [1, 2, 3]] / 255.0
            #if from_one:
            #    color_index_rgb[:, 0] = np.arange(color_index_rgb.shape[0])+1
            inds = color_index_rgb[:, [1, 2, 3]].sum(1) != 0
            color_index_rgb = color_index_rgb[inds, :]
            color_name = list(np.array(color_name)[inds])

    return color_name, color_index_rgb, inital_col



###################### Return voxel coordinates for reference x, y, z and vice versa ######################
def apply_affine(coord, affine):
    """ Return voxel coordinates for reference x, y, z and vice versa"""
    if coord.shape[1] != 4:
           c = np.zeros((coord.shape[0], 4))
           c[:, :-1] = coord
           c[:, -1] = np.ones(coord.shape[0])
           coord = c.T
    return np.matmul(affine,coord).T



def vox2ref(affine, ref):
   """ Return X, Y, Z coordinates for i, j, k """
   return np.matmul(affine, ref)[:3]

###################### Cursors ######################

def cursorOpenHand():
    bitmap = QtGui.QPixmap(source_folder+"/Hand.png")
    return QtGui.QCursor(bitmap)


def cursorArrow():
    #bitmap = QtGui.QPixmap(source_folder+"/arrow.png")
    return QtGui.QCursor(Qt.ArrowCursor)
def cursorPaint():
    bitmap = QtGui.QPixmap(source_folder+"/HandwritingPlus.png")
    return QtGui.QCursor(bitmap)
def cursorPaintX():
    bitmap = QtGui.QPixmap(source_folder+"/HandwritingPlusX.png")
    return QtGui.QCursor(bitmap)
def cursorCircle(size = 50):
    from PIL import Image, ImageDraw

    size_im = 200
    image = Image.new('RGBA', (size_im, size_im))
    draw = ImageDraw.Draw(image)
    # Size of Bounding Box for ellipse

    x, y = size_im, size_im
    eX, eY = size / 2, size / 2
    bbox = (x / 2 - eX / 2, y / 2 - eY / 2, x / 2 + eX / 2, y / 2 + eY / 2)

    draw.ellipse(bbox, fill=None, outline='blue', width=2)

    eX, eY = size / 12, size / 12
    bbox2 = (x / 2 - eX / 2, y / 2 - eY / 2, x / 2 + eX / 2, y / 2 + eY / 2)
    draw.ellipse(bbox2, fill='blue', outline='blue', width=2)


    return QtGui.QCursor(image.toqpixmap())

def cursorErase():
    bitmap = QtGui.QPixmap(source_folder+"/HandwritingMinus.png")
    return QtGui.QCursor(bitmap)
def cursorEraseX():
    bitmap = QtGui.QPixmap(source_folder+"/HandwritingMinusX.png")
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
    widgets_mri = [4, 5, 6, 12, 24]
    widgets_eco = [11, 1, 2, 3, 14]
    _eco = False
    _mri = False

    for k in widgets:
        name = prefix + str(k)
        widget = getattr(self, name)
        if widget.isVisible():
            if k in widgets_eco:
                _eco = True
            elif k in widgets_mri:
                _mri = True
    if _eco and _mri:
        widgets = widgets
    elif _eco:
        widgets = widgets_eco
    elif _mri:
        widgets = widgets_mri
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
    widget.enabledZoom = False  # ZOOM DISABLED
    widget.enabledRuler = False
    widget.enabledCircle = False



    if reptime <=1:
        if code == 0:
            widget.updateEvents()
            widget.setCursor(Qt.ArrowCursor)

        elif code == 1:  # ImFreeHand
            widget.updateEvents()
            widget.setCursor(cursorPaint())

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
            widget.updateEvents()
            widget.setCursor(cursorErase())

        elif code == 4:  # ImPaint Contour
            widget.updateEvents()
            widget.setCursor(cursorPaint())
            try:
                widget.customContextMenuRequested.connect(widget.ShowContextMenu_contour)
            except Exception as e:
                print('Cursor Widget Error')
                print(e)
        elif code == 5: # point locator
            widget.updateEvents()
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

            widget.setCursor(cursorPaint())
            try:
                widget.customContextMenuRequested.connect(widget.ShowContextMenu_gen)
            except Exception as e:
                print('Cursor Widget Error')
                print(e)
        elif code == 9: #circle
            #widget.updateEvents()
            widget.enabledCircle = True
            widget.setCursor(cursorCircle(rad_circle))

    else:
        if code == 4:  # ImPaint
            widget.updateEvents()
            widget.setCursor(cursorPaintX())
        elif code == 3:  # Erasing
            widget.updateEvents()
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
    from skimage.filters import threshold_multiotsu, threshold_otsu
    if numc > 1 and numc <= 5:
        thresholds = threshold_multiotsu(a, classes=numc)
    elif numc == 1:
        thresholds = [threshold_otsu(a)]
    else:
        import numpy as np
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
    import functools

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
    extract attributes from widget and assign it to the dictionary
    Args:
        widget:
        nameWidget:
        dic:

    Returns:

    """
    def IsKnownType(val):
        return type(val) == int or type(val) == np.ndarray or type(val) == list or val is None or type(
            val) == float or type(val) == defaultdict or \
               type(val) == Qt.GlobalColor or \
               type(val) == str or type(val)==bool or \
               type(val) == tuple
    def updateDic(val, attr, at):
        for el in attr:
            try:
                vl = getattr(val, el)()

                if IsKnownType(vl):
                    dic[nameWidget][at][el] = vl
            except Exception as e:
                print('Update Dictionary Error')
                print(e)

    dic[nameWidget] = defaultdict(list)
    for at in dir(widget):
        if at[0] == '_' or at == 'program':
            continue
        val = getattr(widget, at)
        if type(val) == QtWidgets.QSlider: # slider

            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'QSlider'
            attr = ['minimum', 'maximum', 'value', 'isHidden']
            updateDic(val, attr, at)
        elif type(val)== QtWidgets.QLabel:
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'QLabel'
            attr = ['text', 'isHidden']
            updateDic(val, attr, at)
        elif type(val)== QtWidgets.QRadioButton:
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'QRadioButton'
            attr = ['isHidden', 'isChecked']
            updateDic(val, attr, at)
        elif type(val)== AnimatedToggle:
            dic[nameWidget][at] = defaultdict(list)
            dic[nameWidget][at]['type'] = 'AnimatedToggle'
            attr = ['isHidden', 'isChecked']
            updateDic(val, attr, at)
        elif IsKnownType(val):
            if at=='items':
                if type(val)==defaultdict:
                    val = list(val.keys())
                    at = 'items_names'
            dic[nameWidget][at] = val
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
                    Qel.setText(subDic['text'])
                elif tpe == 'QRadioButton' or tpe == 'AnimatedToggle':
                    attr = ['isHidden', 'isChecked']
                    Qel.setVisible(not subDic['isHidden'])
                    Qel.setChecked(subDic['isChecked'])
                elif tpe == 'QSlider':
                    attr = ['minimum', 'maximum', 'value', 'isHidden']
                    Qel.setVisible(not subDic['isHidden'])
                    Qel.setRange(subDic['minimum'], subDic['maximum'])
                    Qel.setValue(subDic['value'])
            else:
                setattr(widget, key, dic[nameWidget][key])
        else:
            #if key == 'ImCenter':
            #    print(key)
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
        slider.setRange(0, reader.ImExtent[indWind])
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

def standardize( imdata):
    """
    Standardize image between 0 and 255.0
    Args:
        imdata:

    Returns:

    """
    imdata = (imdata - imdata.min()) * 255.0 / np.ptp(imdata) #range
    return imdata





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
    if name.lower()=='mri':
        clr = 55
    else:
        clr = 155
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






###################### select proper widgets ######################
def select_proper_widgets(self):
    from PyQt5.QtCore import QObject
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
    elif self.tabWidget.currentIndex() == 2:
        wndnm = self.openGLWidget_11.currentWidnowName
        if wndnm.lower() == 'sagittal':
            widgets.append(self.openGLWidget_2)
        elif wndnm.lower() == 'coronal':
            widgets.append(self.openGLWidget_1)
        elif wndnm.lower() == 'axial':
            widgets.append(self.openGLWidget_3)
        widgets.append(self.openGLWidget_11)
    elif self.tabWidget.currentIndex() == 3:
        wndnm = self.openGLWidget_11.currentWidnowName
        if wndnm.lower() == 'sagittal':
            widgets.append(self.openGLWidget_5)
        elif wndnm.lower() == 'coronal':
            widgets.append(self.openGLWidget_4)
        elif wndnm.lower() == 'axial':
            widgets.append(self.openGLWidget_6)
    return widgets



###################### save numpy array to image ######################
def save_numpy_to_png(file, img):
    from matplotlib.image import imsave
    imsave(file, img)


###################### saving 3D images ######################
def save_3d_img(reader, file, img, format='tif', type_im = 'mri', cs=['RAS', 'AS', True]):
    import csv
    cors, asto, save_csv = cs

    if format == 'tif':
        from skimage.external import tifffile as tif
        tif.imsave(file+'.tif', img, bigtiff=True)
    elif format == 'nifti':
        if file[-7:] != '.nii.gz':
            file = file +'.nii.gz'

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
        new_im = nib.Nifti1Image(img, affine, header=hdr)
        if asto.lower()=='as':
            new_im =  new_im.as_reoriented(transform) # reorient to the right transformation system
        elif asto.lower()=='to':
            new_affine = new_im.as_reoriented(transform).affine
            new_im = nib.Nifti1Image(img, new_affine, header=hdr)

        nib.save(new_im, file)

        if save_csv:
            with open(file+'.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
                w = csv.DictWriter(f, reader.metadata.keys())
                w.writeheader()
                w.writerow(reader.metadata)

        #with open(file + '.json', 'w') as fp:
        #    json.dump(reader.metadata, fp)



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
    import csv
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




