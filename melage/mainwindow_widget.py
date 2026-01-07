__AUTHOR__ = 'Bahram Jafrasteh'

# --- 1. Python Standard Library ---
import os
import sys
import time
import subprocess
import traceback
from functools import partial
from collections import defaultdict
from os.path import join, basename, dirname

# Avoid modifying sys.path manually if possible (use pip install -e .)
sys.path.append('..')

# --- 2. Third-Party Libraries ---
import numpy as np
import cv2

# Robust Pickle Import (Handles Python 2/3 and Linux specific backports)
try:
    import cPickle as pickle
except ModuleNotFoundError:
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle

# --- 3. GUI (PyQt5) ---
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QSettings, QEvent

# --- 4. Melage: Config & Core ---
from melage.config import settings, __VERSION__
from melage.core.io import readData

# --- 5. Melage: UI Components (Dialogs & Widgets) ---
from melage.dialogs import (
    iminfo_dialog, about_dialog,
    Masking, MaskOperationsDialog, HistImage,
    ThresholdingImage, RegistrationDialog, TransformationDialog
)
from melage.dialogs.helpers import (
    QFileDialogPreview, repeatN, screenshot, NewDialog
)
from melage.widgets import (
    enhanceIm, SettingsDialog, dockWidgets, openglWidgets, PluginManager
)

# --- 6. Melage: Rendering ---
from melage.rendering.DisplayIm import GLWidget
from melage.rendering.glScientific import glScientific

# --- 7. Melage: Utilities (Consolidated) ---
from melage.utils.utils import (
    # General Utilities
    rhasattr, str_conv, getUnique, len_unique,
    repetition, destacked, adapt_previous_versions,

    # UI/Widget Helpers
    select_proper_widgets, setCursorWidget, find_avail_widgets,
    locateWidgets, loadAttributeWidget,
    add_new_tree_widget, addTreeRoot, manually_check_tree_item,

    # Image/View State
    getCurrentSlice, updateSight, changeCoronalSagittalAxial,
    setSliceSeg, update_last, update_last_video, update_image_sch,
    clean_parent_image, clean_parent_image2,
    set_new_color_scheme, update_widget_color_scheme, make_all_seg_visibl,

    # File I/O & Export
    get_filter_for_file, save_snapshot, read_segmentation_file,
    save_3d_img, export_tables, save_as_nifti, save_as_nrrd,
    save_as_dicom, save_modified_nifti,

    # Image Processing & Geometry
    compute_volume, apply_thresholding, rotation3d,
    make_image, make_image_using_affine, LinkMRI_ECO,
    convexhull_spline, slice_intepolation, SearchForAdditionalPoints,
    generate_extrapoint_on_line,

    # Tracking/TRK
    get_world_from_trk, load_trk
)

def time_profile():
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    print(f"[Profile] Update: {(t2 - t1) * 1000:.2f} ms")
class Ui_Main(dockWidgets, openglWidgets):
    """
    Main widgets
    """
    setNewImage_view1 = QtCore.pyqtSignal(object)
    setNewImage_view2 = QtCore.pyqtSignal(object)
    def __init__(self):
        """
        Initializing the main attributes
        """
        self._filters = "Nifti(*.nia *.nii *.nii.gz *.hdr *.img *.img.gz *.mgz);;Vol (*.vol *.V00);;DICOM(*.dcm **);;NRRD(*.nrrd *.nhdr);;DICOMDIR(*DICOMDIR*);;Video (*.mp4 *.avi *.mov *.mkv)"
        formats = [ll.replace(' ', '').replace(')', '') for el in self._filters.split(';;') for ll in el.split('*')[1:]]
        formats = [el for el in formats if el != '' and '.' in el]
        super(Ui_Main, self).__init__()
        pwd = os.path.abspath(__file__)
        self.startTime = time.time()
        self.colorsCombinations = defaultdict(list)
        self._timer_id = -1
        self._last_index_select_image_view1 = 2 # index for selection of image type ('neonatal', 'fetal', 'mri')
        self._last_index_select_image_view2 = 2 # index for selection of image type ('neonatal', 'fetal', 'mri')
        self._last_state_guide_lines = False # guide lines are not activate
        self._last_state_preview = False # preview to show image preview before opening
        self.format_view1 = 'None'
        self.format_view2 = 'None'
        self._loaded = False
        self.readView1 = []
        self.is_view1_video = False
        self.is_view2_video = False
        self.full_path_view1 = None
        self.full_path_view2 = None
        self.readView2 = []
        self.View2_RADIO_NAMES = [
            'radioButton_21', 'radioButton_21_1',
            'radioButton_21_2', 'radioButton_21_3'
        ]
        self.View1_RADIO_NAMES = ['radioButton_1', 'radioButton_2',
                               'radioButton_3', 'radioButton_4']
        self.VIEW2_INDICES = [4, 5, 6, 12]
        self.VIEW1_INDICES = [1, 2, 3, 11]
        self.current_view = 'horizontal'
        self._num_adapted_points = 0
        self._firstSelection = None
        self._Xtimes = 1
        self._rad_circle = 50
        self._rad_circle_dot = 50
        self.num_measure_area = 0
        self.num_measure_length = 0
        #self.source_dir = os.path.dirname(os.path.dirname(pwd))
        self.settingsBN = SettingsDialog(self)
        self.expectedTime = self.settingsBN.auto_save_spinbox.value() * 60
        self.iminfo_dialog = iminfo_dialog(self)
        self._points_adapt_view1 = []
        self._points_adapt_view2 = []
        self.linePoints = []
        self._lastlines = []
        self._lineinfo = []
        self._slice_interp = [[], [], [], []] #(Support Sagittal, Axial, Coronal, Video)
        self.tol_trk = 3
        self.linked_models = None
        self.linked = False
        self.filenameView1 = ''
        self.filenameView2 = ''

        self.create_dialog()
        self.all_dialog_connect()


        if not os.path.exists('.temp'):
            os.mkdir('.temp')
        self.MouseButtonPress = False
        self.MouseButtonRelease = False
        self._translate = QtCore.QCoreApplication.translate
        self._rotationAngleView1_coronal = 0
        self._rotationAngleView1_axial = 0
        self._rotationAngleView1_sagittal = 0
        self._rotationAngleView2_coronal = 0
        self._rotationAngleView2_axial = 0
        self._rotationAngleView2_sagittal = 0
        self._lastReaderSegCol = []
        self._lastReaderSegInd = []

        self._lastReaderSegPrevCol = []
        self._lastMax = 10
        self._undoTimes = 0
        self._lastWindowName = None

        self.allowChangeScn = False

        self._availableFormats = formats
        self.settings = QSettings("./brainNeonatal.ini", QSettings.IniFormat) # setting to save
        self._basefileSave = ''

        ### plugins
        self.plugin_widgets = []  # To keep references to open plugin dialogs


    def create_dialog(self):




        self.repeatTimes = repeatN(self)
        #self.screenShot = screenshot(self)

        self.Masking = Masking(self)
        self.ImageThresholding = ThresholdingImage(self)
        self.HistImage = HistImage(self)

        self.registrationD = RegistrationDialog(self, settings.DEFAULT_USE_DIR)
        self.transformationD = TransformationDialog(self, settings.DEFAULT_USE_DIR)
        self.MaskingOperations = MaskOperationsDialog(self)

        self.enhanceIm = enhanceIm(self)
        # Group all dialogs that need resizing
        dialogs_to_resize = [
            self.repeatTimes,
            self.Masking, self.ImageThresholding, self.HistImage,
            self.registrationD, self.transformationD,
            self.MaskingOperations, self.enhanceIm
        ]
        for dialog in dialogs_to_resize:
            self.adapt_dialog_size(dialog)
    def all_dialog_connect(self):
        self.registrationD.datachange.connect(self.updateDataRegistration)


        self.repeatTimes.numberN.connect(
            lambda value: self.setXTimes(value)
        )
        self.Masking.closeSig.connect(partial(self.maskingClose, 3))

        self.HistImage.closeSig.connect(partial(self.maskingClose, 3))
        self.ImageThresholding.closeSig.connect(partial(self.maskingClose, 3))
        self.MaskingOperations.closeSig.connect(partial(self.maskingClose, 3))
        self.registrationD.closeSig.connect(partial(self.maskingClose, 3))
        self.transformationD.closeSig.connect(partial(self.maskingClose, 3))

        self.iminfo_dialog.closeSig.connect(partial(self.maskingClose, 3))
        self.iminfo_dialog.buttonBox.clicked.connect(partial(self.maskingClose, 3))

        self.ImageThresholding.applySig.connect(partial(self.Thresholding, 'apply'))
        self.ImageThresholding.histeqSig.connect(
                                   lambda value: self.Thresholding('histeq', value))
        self.ImageThresholding.repltSig.connect(partial(self.Thresholding, 'replot'))

        self.Masking.apply_pressed.connect(
            lambda  value: self.applyMaskToImage(value, False)
        )
        self.MaskingOperations.apply_pressed.connect(
            lambda value: self.applyMaskToImage(value, True)
        )

    def adapt_dialog_size(self, dialog):
        parent_size = self.size()
        dialog_width = int(parent_size.width()*1.5 )  # e.g., 60% of parent width
        dialog_height = int(parent_size.height()*1.5 )  # e.g., 50% of parent height

        # Don't let it be smaller than its own minimum size
        min_size = dialog.minimumSize()
        if dialog_width < min_size.width():
            dialog_width = min_size.width()
        if dialog_height < min_size.height():
            dialog_height = min_size.height()

        dialog.resize(dialog_width, dialog_height)

    def setConf(self, list_vals):
        """
        Configuration of MELAGE
        :param list_vals:
        :return:
        """
        self.expectedTime = list_vals[0]*60

    def setXTimes(self, val):
        """
        Set number of repetition
        :param val:
        :return:
        """
        self._Xtimes = val

    def showScreenShotWindow(self):
        # --- Helper: Horizontally concatenate a list of widget IDs ---
        def create_row_image(indices):
            images = []
            for k in indices:
                name = f'openGLWidget_{k + 1}'
                widget = getattr(self, name, None)
                if hasattr(self, "imSlice"):
                    if widget and widget.imSlice is not None:
                        widget.makeCurrent()
                        widget.paintGL()
                        q_img = widget.grabFramebuffer()
                        if not q_img.isNull():
                            images.append(q_img)
                else:
                    if widget:
                        widget.makeCurrent()
                        widget.paintGL()
                        q_img = widget.grabFramebuffer()
                        if not q_img.isNull():
                            images.append(q_img)
            if not images:
                return None
                # Calculate Row Dimensions
            total_width = sum(img.width() for img in images)
            max_height = max(img.height() for img in images)

            # Create Row Canvas
            row_img = QtGui.QImage(total_width, max_height, QtGui.QImage.Format_ARGB32_Premultiplied)
            row_img.fill(QtCore.Qt.transparent)

            # Paint Side-by-Side
            painter = QtGui.QPainter(row_img)
            current_x = 0
            for img in images:
                painter.drawImage(current_x, 0, img)
                current_x += img.width()  # No gaps added
            painter.end()

            return row_img
        """
        Take screen shot from window
        :return:
        """

        if self.tabWidget.currentIndex()==0:
            row1_indices = [0, 1, 2]
            row2_indices = [3, 4, 5]
        elif self.tabWidget.currentIndex() == 1:
            row1_indices = [10, 13]
            row2_indices = []
        elif self.tabWidget.currentIndex() == 2:
            row1_indices = [11, 23]
            row2_indices = []

        # 1. Create the two horizontal strips (h1 and h2)
        img_h1 = create_row_image(row1_indices)
        img_h2 = create_row_image(row2_indices)

        # Filter out empty rows (in case one row has no data)
        valid_rows = [img for img in [img_h1, img_h2] if img is not None]
        final_image = None
        if not valid_rows:
            print("No images captured.")
            return
        # 2. Vertically concatenate the rows
        # The final width is the widest row
        final_width = max(row.width() for row in valid_rows)
        # The final height is the sum of all row heights
        final_height = sum(row.height() for row in valid_rows)

        final_image = QtGui.QImage(final_width, final_height, QtGui.QImage.Format_ARGB32_Premultiplied)
        final_image.fill(QtCore.Qt.white)  # Background color (fill gaps if rows have different widths)

        painter = QtGui.QPainter(final_image)
        current_y = 0

        for row in valid_rows:
            # Draw the row at (0, current_y)
            painter.drawImage(0, current_y, row)

            # Move cursor down by the height of the row we just drew
            current_y += row.height()

        painter.end()


        if final_image is None:
            return
        filters = "png(*.png)"
        opts = QtWidgets.QFileDialog.DontUseNativeDialog

        fileObj = QtWidgets.QFileDialog.getSaveFileName(self, "Open File", settings.DEFAULT_USE_DIR, filters,
                                                        options=opts)
        if fileObj[0] == '':
            return
        filename = fileObj[0] + '.png'
        self.save_screenshot(final_image, filename)

    def get_latest_pypi_version(self, package_name="melage"):
        """
        Queries the PyPI JSON API to find the latest version number.
        """
        try:
            import json
            import urllib.request
            from importlib.metadata import version  # To get installed version
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                e
            )
            return

        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            # Set a timeout so the UI doesn't freeze forever if offline
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                return data["info"]["version"]
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                e
            )
            return None

    def update_application(self):
        """
        Checks for updates and installs them if the user confirms.
        """
        try:
            import urllib.request
            from importlib.metadata import version  # To get installed version
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                e
            )
            return
        # 1. Get the CURRENT installed version
        try:
            current_ver = version("melage")
        except Exception:
            current_ver = "Unknown"

        # 2. Get the LATEST version from PyPI
        # Change cursor to indicate loading/network activity
        self.setCursor(Qt.WaitCursor)
        latest_ver = self.get_latest_pypi_version("melage")
        self.unsetCursor()

        # 3. Handle Network Errors
        if latest_ver is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Connection Error",
                "Could not check for updates.\nPlease check your internet connection."
            )
            return

        # 4. Compare Versions
        if current_ver == latest_ver:
            QtWidgets.QMessageBox.information(
                self,
                "Up to Date",
                f"You are using the latest version of MELAGE.\n\nVersion: {current_ver}"
            )
            return

        # 5. If different, prompt the user
        reply = QtWidgets.QMessageBox.question(
            self,
            'Update Available',
            f"A new version is available!\n\n"
            f"🔹 Current Version: {current_ver}\n"
            f"🔸 New Version:     {latest_ver}\n\n"
            "Do you want to install the update and restart?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )

        if reply == QtWidgets.QMessageBox.No:
            return

        # 6. Perform the Update
        try:
            self.setCursor(Qt.WaitCursor)

            # Run pip upgrade using the current interpreter
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "melage"]
            )

            self.unsetCursor()

            QtWidgets.QMessageBox.information(
                self,
                "Update Successful",
                f"Updated to version {latest_ver} successfully.\n\n"
                "Please restart the application to see the changes."
            )

            # Exit to force a clean restart
            #sys.exit(0)

        except subprocess.CalledProcessError as e:
            self.unsetCursor()
            QtWidgets.QMessageBox.critical(self, "Update Failed", f"Error during installation:\n{e}")

    def showRpeatWindow(self):
        """
        Repetition window
        :return:
        """
        if self.repeatTimes.exec_() == self.repeatTimes.Accepted:
            val = self.repeatTimes.doubleSpinBox.value()
            self._Xtimes = int(val)
        else:
            self._Xtimes = 1

    def showInfoWindow(self):
        """
        Show information window
        :return:
        """
        self.settingsBN.show()

    def showIMVARSWindow(self):
        self.enhanceIm.show()

    def showImInfoWindow(self):
        self.iminfo_dialog.show()


    def _setup_main_window(self, Main):
        """Sets up the main QMainWindow properties (size, font, statusbar)."""
        Main.setObjectName("Main")
        Main.setEnabled(True)
        Main.setMinimumSize(QtCore.QSize(500, 400))  # Your reasonable minimum size

        # Set default size relative to user's screen
        try:
            availableGeometry = Main.screen().availableGeometry()  # Use Main.screen()
            default_width = int(availableGeometry.width() * 1)
            default_height = int(availableGeometry.height() * 1)
            Main.resize(default_width, default_height)
        except AttributeError:
            Main.resize(1024, 768)  # Fallback

        # Set font
        font = QtGui.QFont()
        font.setFamily("Ubuntu")  # Note: This font may not exist on all systems
        Main.setFont(font)

        Main.setDockNestingEnabled(True)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(Main)
        self.statusbar.setObjectName("statusbar")
        Main.setStatusBar(self.statusbar)

    def _setup_central_layout(self, Main):
        """Creates the central widget and its main layout."""
        self.centralwidget = QtWidgets.QWidget(Main)
        self.centralwidget.setEnabled(True)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                           QtWidgets.QSizePolicy.MinimumExpanding)
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")

        # **CRITICAL FIX**: Removed the line:
        # self.centralwidget.setMinimumSize(QtCore.QSize(800, 600))
        # This was preventing your UI from shrinking!

        # This single line creates the layout and assigns it.
        #self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        #self.main_layout.setObjectName("main_layout")
        Main.setCentralWidget(self.centralwidget)

        Main.setContextMenuPolicy(Qt.ActionsContextMenu)  # Use Main, not self

        # Create OpenGL Widgets (fits well here as they are in the central layout)
        self.widgets_view2 = [4, 5, 6, 12]
        self.widgets_view1 = [11, 1, 2, 3]
        self.createOpenGLWidgets(self.centralwidget, self.colorsCombinations)

    def _create_actions(self, Main):
        """Creates all QActions and stores them as class attributes."""
        # This function will be long, but it has only ONE job: create actions.

        # --- Color Action ---

        self.actionColor = QtWidgets.QAction(Main)
        self.actionColor.setObjectName("actionColor")
        self._icon_colorXFaded = QtGui.QIcon()
        self.pixmap_box_color = QtGui.QPixmap(settings.RESOURCE_DIR + "/box.png")
        colr = [1, 1, 1]
        self.pixmap_box_color.fill((QtGui.QColor(colr[0] * 255, colr[1] * 255, colr[2] * 255, 1 * 255)))
        self._icon_colorXFaded.addPixmap(self.pixmap_box_color, QtGui.QIcon.Normal, QtGui.QIcon.On)
        self._icon_colorX = QtGui.QIcon()
        colr = [1, 0, 0]
        self.pixmap_box_color.fill((QtGui.QColor(colr[0] * 255, colr[1] * 255, colr[2] * 255, 1 * 255)))
        self._icon_colorX.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/box.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionColor.setIcon(self._icon_colorXFaded)

        # --- File Actions ---
        self.actionOpenView1 = QtWidgets.QAction(Main)
        self.actionOpenView1.setObjectName("actionOpenView1")
        icon_view1 = QtGui.QIcon()
        icon_view1.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/view1.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionOpenView1.setIcon(icon_view1)
        self.actionOpenView1.setIconText('open')

        self.actionOpenView1.setDisabled(True)

        self.actionOpenView2 = QtWidgets.QAction(Main)
        self.actionOpenView2.setObjectName("actionOpenView2")
        icon_view2 = QtGui.QIcon()
        icon_view2.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/view2.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionOpenView2.setIcon(icon_view2)
        self.actionOpenView2.setIconText('open')
        self.actionOpenView2.setDisabled(True)

        self.actionComboBox = QtWidgets.QComboBox(Main)
        self.actionComboBox.setObjectName("actionComboBox")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setRetainSizeWhenHidden(True)
        sizePolicy.setHeightForWidth(self.actionComboBox.sizePolicy().hasHeightForWidth())
        self.actionComboBox.setMinimumSize(QtCore.QSize(100, 0))
        self.actionComboBox.setSizePolicy(sizePolicy)
        self.actionComboBox.setObjectName("View1")

        self.actionComboBox.setDisabled(True)
        self.actionComboBox.setObjectName("comboBox")
        cbstyle = """
                        QComboBox QAbstractItemView {border: 1px solid grey;
                        background: #03211c; 
                        selection-background-color: #03211c;
                        "text-align: left;"} 
                        QComboBox {background: #03211c;margin-right: 1px;}
                        QComboBox::drop-down {
                    subcontrol-origin: margin;}
                    padding-left
                        """
        self.actionComboBox.setStyleSheet(cbstyle)
        for r in range(2):
            self.actionComboBox.addItem("{}".format(r))
        self.actionComboBox.setVisible(False)

        self.actionOpenFA = QtWidgets.QAction(Main)
        self.actionOpenFA.setObjectName("actionOpenFA")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/dti.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionOpenFA.setIcon(icon)
        self.actionOpenFA.setIconText('open')
        self.actionOpenFA.setDisabled(True)

        self.actionOpenTract = QtWidgets.QAction(Main)
        self.actionOpenTract.setObjectName("actionOpenTract")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/tract.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionOpenTract.setIcon(icon)
        self.actionOpenTract.setIconText('open')
        self.actionOpenTract.setDisabled(True)

        self.actionNew = QtWidgets.QAction(Main)
        self.actionNew.setObjectName("actionNew")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionNew.setIcon(icon)
        self.actionNew.setIconText('New')

        # ...

        self.actionCloseView1 = QtWidgets.QAction(Main)
        self.actionCloseView1.setObjectName("actionNew")
        self.actionCloseView1.setIconText('Close View 1')

        # ...
        ######################### Close View2 ################################
        self.actionCloseView2 = QtWidgets.QAction(Main)
        self.actionCloseView2.setObjectName("actionNew")
        self.actionCloseView2.setIconText('Close View 2')

        self.actionImportSegView2 = QtWidgets.QAction(Main)
        self.actionImportSegView2.setObjectName("action SegView2")
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionCloseView1.setIcon(icon)
        icon_view2S = QtGui.QIcon()
        icon_view2S.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/view2_seg.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionImportSegView2.setIcon(icon_view2S)
        self.actionImportSegView2.setIconText('Segmented View 2')
        self.actionImportSegView2.setDisabled(True)

        self.actionImportSegView1 = QtWidgets.QAction(Main)
        self.actionImportSegView1.setObjectName("action SegView1")
        icon_view1S = QtGui.QIcon()
        icon_view1S.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/view1_seg.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionImportSegView1.setIcon(icon_view1S)
        self.actionImportSegView1.setIconText('Segmented view 1')
        self.actionImportSegView1.setDisabled(True)

        ######################### Export ################################
        self.actionExportImView1 = QtWidgets.QAction(Main)
        self.actionExportImView1.setObjectName("action ImView1")
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionCloseView1.setIcon(icon)
        self.actionExportImView1.setIcon(icon_view1)
        self.actionExportImView1.setIconText('Image View 1')

        self.actionExportSegView1 = QtWidgets.QAction(Main)
        self.actionExportSegView1.setObjectName("action SegView1")
        self.actionExportSegView1.setIcon(icon_view1S)
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionCloseView1.setIcon(icon)
        self.actionExportSegView1.setIconText('Segmented View 1')

        self.actionExportImView2 = QtWidgets.QAction(Main)
        self.actionExportImView2.setObjectName("action ImView2")
        self.actionExportImView2.setIcon(icon_view2)
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionCloseView1.setIcon(icon)
        self.actionExportImView2.setIconText('Image view 2')

        self.actionExportSegView2 = QtWidgets.QAction(Main)
        self.actionExportSegView2.setObjectName("action SegView1")
        self.actionExportSegView2.setIcon(icon_view2S)
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionCloseView1.setIcon(icon)
        self.actionExportSegView2.setIconText('Segmented View 2')

        self.actionExportImView2.setDisabled(True)
        self.actionExportSegView2.setDisabled(True)
        self.actionExportImView1.setDisabled(True)
        self.actionExportSegView1.setDisabled(True)

        self.actionScreenS = QtWidgets.QAction(Main)
        self.actionScreenS.setObjectName("actionNew")
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionCloseView1.setIcon(icon)
        self.actionScreenS.setIconText('Screen Shot')

        self.actionLoad = QtWidgets.QAction(Main)
        self.actionLoad.setObjectName("actionLoad")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/load.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionLoad.setIcon(icon)
        self.actionLoad.setIconText('load')

        self.actionMain_Toolbar = QtWidgets.QAction(Main)
        icon1 = QtGui.QIcon()
        # icon1.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"e/action_check.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # icon1.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/action_check_OFF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        # self.actionMain_Toolbar.setIcon(icon1)
        self.actionMain_Toolbar.setObjectName("actionMain_Toolbar")
        self.actionMain_Toolbar.setCheckable(True)

        self.action_guideLines = QtWidgets.QAction(Main)
        icon1 = QtGui.QIcon()
        self.action_guideLines.setObjectName("Guide lines")
        self.action_guideLines.setCheckable(True)

        ######################### View -> AXIS ################################
        self.action_axisLines = QtWidgets.QAction(Main)
        icon1 = QtGui.QIcon()
        self.action_axisLines.setObjectName("Axis lines")
        self.action_axisLines.setCheckable(True)
        ######################### View -> ZOOM PAN ROTATE TOOLBAR ################################
        self.action_interaction_Toolbar = QtWidgets.QAction(Main)
        icon1 = QtGui.QIcon()
        self.action_interaction_Toolbar.setObjectName("Interaction_Toolbar")
        self.action_interaction_Toolbar.setCheckable(True)

        ######################### File -> INFO ################################
        self.actionFile_info = QtWidgets.QAction(Main)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/settings.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionFile_info.setIcon(icon2)
        self.actionFile_info.setObjectName("actionFile_info")
        ######################### Tools -> Undo ################################
        self.actionUndo = QtWidgets.QAction(Main)
        # icon2 = QtGui.QIcon()
        # icon2.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/settings.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionUndo.setIcon(icon2)
        self.actionUndo.setObjectName("action Undo")
        ######################### Tools -> Redo ################################
        self.actionRedo = QtWidgets.QAction(Main)
        # icon2 = QtGui.QIcon()
        # icon2.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/settings.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionUndo.setIcon(icon2)
        self.actionRedo.setObjectName("action Redo")

        ######################### File -> ChangeImage ################################
        self.actionFile_changeIM = QtWidgets.QAction(Main)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionFile_changeIM.setIcon(icon2)
        self.actionFile_changeIM.setObjectName("actionFile_changeIM")

        ######################### File -> Info ################################
        self.actionfile_iminfo = QtWidgets.QAction(Main)
        self.actionfile_iminfo.setObjectName("actionFile_info")
        ######################### File -> convert ################################
        self.actionconvert = QtWidgets.QAction(Main)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionconvert.setIcon(icon)
        self.actionconvert.setObjectName("actionconvert")



        self.actionexit = QtWidgets.QAction(Main)
        self.actionexit.setObjectName("actionexit")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/close.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionexit.setIcon(icon)

        ######################### Logo ################################
        self.logo = QtWidgets.QLabel(Main)
        self.logo.setPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/melage_top.png"))
        self.logo.resize(100, 50)

        ######################### File -> save ################################
        self.actionsave = QtWidgets.QAction(Main)
        self.actionsave.setObjectName("actionsave")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionsave.setIcon(icon)

        self.actionsave.setDisabled(True)

        ######################### File -> saveas ################################
        self.actionsaveas = QtWidgets.QAction(Main)
        self.actionsaveas.setObjectName("actionsaveas")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/saveas.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionsaveas.setIcon(icon)

        self.actionsaveas.setDisabled(True)

        ######################### Pan Zoom ################################
        self.actionPan = QtWidgets.QAction(Main)
        self.actionPan.setObjectName("actionPan")
        self._icon_Hand_IXFaded = QtGui.QIcon()
        self._icon_Hand_IXFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/Hand_IXFaded.png"), QtGui.QIcon.Normal,
                                          QtGui.QIcon.On)
        self._icon_Hand_IX = QtGui.QIcon()
        self._icon_Hand_IX.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/Hand_IX.png"), QtGui.QIcon.Normal,
                                     QtGui.QIcon.On)

        self.actionPan.setIcon(self._icon_Hand_IXFaded)

        self.actionContour = QtWidgets.QAction(Main)
        self.actionContour.setObjectName("actionContour")
        self._icon_contourFaded = QtGui.QIcon()
        self._icon_contourFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/contourFaded.png"), QtGui.QIcon.Normal,
                                          QtGui.QIcon.On)
        self._icon_contour = QtGui.QIcon()
        self._icon_contour.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/contour.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionContour.setIcon(self._icon_contourFaded)

        self.actionPoints = QtWidgets.QAction(Main)
        self.actionPoints.setObjectName("actionPoints")
        self._icon_pointsFaded = QtGui.QIcon()
        self._icon_pointsFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/pointsFaded.png"), QtGui.QIcon.Normal,
                                         QtGui.QIcon.On)
        self._icon_points = QtGui.QIcon()
        self._icon_points.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/points.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionPoints.setIcon(self._icon_pointsFaded)

        self.actionCircles = QtWidgets.QAction(Main)
        self.actionCircles.setObjectName("action Circles")
        self._icon_CircleFaded = QtGui.QIcon()
        self._icon_CircleFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/circle_faded.png"), QtGui.QIcon.Normal,
                                         QtGui.QIcon.On)
        self._icon_circles = QtGui.QIcon()
        self._icon_circles.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/circle.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionCircles.setIcon(self._icon_CircleFaded)

        self.actionGoTo = QtWidgets.QAction(Main)
        self.actionGoTo.setCheckable(True)
        self.actionGoTo.setObjectName("goto")
        self._icon_gotoFaded = QtGui.QIcon()
        self._icon_gotoFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/synchFaded.png"), QtGui.QIcon.Normal,
                                       QtGui.QIcon.On)
        self._icon_goto = QtGui.QIcon()
        self._icon_goto.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/synch.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionGoTo.setIcon(self._icon_gotoFaded)

        self.action3D = QtWidgets.QAction(Main)
        self.action3D.setCheckable(True)
        self.action3D.setChecked(True)
        self.action3D.setObjectName("goto")
        self._icon_3dFaded = QtGui.QIcon()
        self._icon_3dFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/3dFaded.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self._icon_3d = QtGui.QIcon()
        self._icon_3d.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/3d.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.action3D.setIcon(self._icon_3d)

        self.actionZoomIn = QtWidgets.QAction(Main)
        self.actionZoomIn.setCheckable(True)
        self.actionZoomIn.setChecked(True)
        self.actionZoomIn.setObjectName("goto")
        self._icon_zoomIn = QtGui.QIcon()
        self._icon_zoomIn.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/zoom_in.png"), QtGui.QIcon.Normal,
                                    QtGui.QIcon.On)
        self._icon_zoomIn = QtGui.QIcon()
        self._icon_zoomIn.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/zoom_in.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionZoomIn.setIcon(self._icon_zoomIn)

        self.actionZoomOut = QtWidgets.QAction(Main)
        self.actionZoomOut.setCheckable(True)
        self.actionZoomOut.setChecked(True)
        self.actionZoomOut.setObjectName("goto")
        self._icon_zoomOut = QtGui.QIcon()
        self._icon_zoomOut.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/zoom_out.png"), QtGui.QIcon.Normal,
                                     QtGui.QIcon.On)
        self._icon_zoomOut = QtGui.QIcon()
        self._icon_zoomOut.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/zoom_out.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionZoomOut.setIcon(self._icon_zoomOut)


        self.actionZoomNeutral = QtWidgets.QAction(Main)
        self.actionZoomNeutral.setCheckable(True)
        self.actionZoomNeutral.setChecked(True)
        self.actionZoomNeutral.setObjectName("goto")
        _icon_zoomInN = QtGui.QIcon()
        _icon_zoomInN.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/zoom_neutral.png"), QtGui.QIcon.Normal,
                                    QtGui.QIcon.On)
        self.actionZoomNeutral.setIcon(_icon_zoomInN)



        self.actionContourX = QtWidgets.QAction(Main)
        self.actionContourX.setObjectName("actionContourX")
        self._icon_contourXFaded = QtGui.QIcon()
        self._icon_contourXFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/contourXFaded.png"), QtGui.QIcon.Normal,
                                           QtGui.QIcon.On)
        self._icon_contourX = QtGui.QIcon()
        self._icon_contourX.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/contourX.png"), QtGui.QIcon.Normal,
                                      QtGui.QIcon.On)
        self.actionContourX.setIcon(self._icon_contourXFaded)

        self.actionEraseX = QtWidgets.QAction(Main)
        self.actionEraseX.setObjectName("actionEraseX")
        self._icon_eraseXFaded = QtGui.QIcon()
        self._icon_eraseXFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/EraserXFaded.png"), QtGui.QIcon.Normal,
                                         QtGui.QIcon.On)
        self._icon_eraseX = QtGui.QIcon()
        self._icon_eraseX.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/EraserX.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionEraseX.setIcon(self._icon_eraseXFaded)

        self.actionRuler = QtWidgets.QAction(Main)
        self.actionRuler.setObjectName("actionMeasure")
        self._icon_rulerFaded = QtGui.QIcon()
        self._icon_rulerFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/rulerFaded.png"), QtGui.QIcon.Normal,
                                        QtGui.QIcon.On)
        self._icon_ruler = QtGui.QIcon()
        self._icon_ruler.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/ruler.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRuler.setIcon(self._icon_ruler)

        ##########################################################

        # --- Horizontal (Axial) View Action ---
        self.actionHorizontalView = QtWidgets.QAction(Main)
        self.actionHorizontalView.setObjectName("actionHorizontalView")

        # Icons for horizontal view
        self._icon_HVFaded = QtGui.QIcon()
        self._icon_HVFaded.addPixmap(
            QtGui.QPixmap(settings.RESOURCE_DIR + "/horizontalview.png"),
            QtGui.QIcon.Normal, QtGui.QIcon.On
        )
        self._icon_HV = QtGui.QIcon()
        self._icon_HV.addPixmap(
            QtGui.QPixmap(settings.RESOURCE_DIR + "/horizontalview.png"),
            QtGui.QIcon.Normal, QtGui.QIcon.On
        )

        self.actionHorizontalView.setIcon(self._icon_HV)
        self.actionHorizontalView.setCheckable(True)
        self.actionHorizontalView.setToolTip("Switch to Horizontal (Axial) View")
        self.actionHorizontalView.triggered.connect(lambda: self.setView("horizontal"))

        # --- Vertical (Coronal/Sagittal) View Action ---
        self.actionVerticalView = QtWidgets.QAction(Main)
        self.actionVerticalView.setObjectName("actionVerticalView")

        # Icons for vertical view
        self._icon_VVFaded = QtGui.QIcon()
        self._icon_VVFaded.addPixmap(
            QtGui.QPixmap(settings.RESOURCE_DIR + "/verticalview.png"),
            QtGui.QIcon.Normal, QtGui.QIcon.On
        )
        self._icon_VV = QtGui.QIcon()
        self._icon_VV.addPixmap(
            QtGui.QPixmap(settings.RESOURCE_DIR + "/verticalview.png"),
            QtGui.QIcon.Normal, QtGui.QIcon.On
        )

        self.actionVerticalView.setIcon(self._icon_VV)
        self.actionVerticalView.setCheckable(True)
        self.actionVerticalView.setToolTip("Switch to Vertical (Coronal) View")
        self.actionVerticalView.triggered.connect(lambda: self.setView("vertical"))

        #############################################################
        self.actionLine = QtWidgets.QAction(Main)
        self.actionLine.setObjectName("actionLine")
        self._icon_lineFaded = QtGui.QIcon()
        self._icon_lineFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/linefaded.png"), QtGui.QIcon.Normal,
                                       QtGui.QIcon.On)
        self._icon_line = QtGui.QIcon()
        self._icon_line.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/line.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionLine.setIcon(self._icon_line)

        self.actionPaint = QtWidgets.QAction(Main)
        self.actionPaint.setObjectName("actionPaint")
        self._icon_pencilFaded = QtGui.QIcon()
        self._icon_pencilFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/paint-brush-2.png"), QtGui.QIcon.Normal,
                                         QtGui.QIcon.On)
        self._icon_pencil = QtGui.QIcon()
        self._icon_pencil.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/paint-brush-2.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        self.actionPaint.setIcon(self._icon_pencilFaded)

        self.actionErase = QtWidgets.QAction(Main)
        self.actionErase.setObjectName("actionErase")
        self._icon_EraserFaded = QtGui.QIcon()
        self._icon_EraserFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/EraserFaded.png"), QtGui.QIcon.Normal,
                                         QtGui.QIcon.On)
        self._icon_Eraser = QtGui.QIcon()
        self._icon_Eraser.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/Eraser.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionErase.setIcon(self._icon_EraserFaded)

        self.actionLazyContour = QtWidgets.QAction(Main)
        self.actionLazyContour.setObjectName("actionLazyContour")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/zoom_out.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionLazyContour.setIcon(icon)

        self.actionArrow = QtWidgets.QAction(Main)
        self.actionArrow.setObjectName("actionArrow")
        self._icon_arrowFaded = QtGui.QIcon()
        self._icon_arrowFaded.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/arrowFaded.png"), QtGui.QIcon.Normal,
                                        QtGui.QIcon.On)
        self._icon_arrow = QtGui.QIcon()
        self._icon_arrow.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionArrow.setIcon(self._icon_arrowFaded)

        ######################### Rotate ################################

        self.actionrotate = QtWidgets.QAction(Main)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/action_check.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon1.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/action_check_OFF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionrotate.setIcon(icon1)
        self.actionrotate.setObjectName("actionrotate")
        self.actionrotate.setCheckable(True)

        # self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton.setGeometry(QtCore.QRect(50, 330, 89, 25))
        # self.pushButton.setObjectName("pushButton")



        self.actionHistImage = QtWidgets.QAction(Main)
        self.actionHistImage.setObjectName('Histogram Image')





        self.actionImageThresholding = QtWidgets.QAction(Main)
        self.actionImageThresholding.setObjectName('Image Thresholding')

        self.actionImageRegistration = QtWidgets.QAction(Main)
        self.actionImageRegistration.setObjectName('Image Registration')

        self.actionImageTransformation = QtWidgets.QAction(Main)
        self.actionImageTransformation.setObjectName('Image Transformation')

        self.actionMasking = QtWidgets.QAction(Main)
        self.actionMasking.setObjectName('Image Masking')



        self.actionOperationMask = QtWidgets.QAction(Main)
        self.actionOperationMask.setObjectName('Masking Operations')



        ######################## CALC ########################################

        self.actionTVCor = QtWidgets.QAction(Main)
        self.actionTVCor.setObjectName('3DVolume Coronal')

        self.actionTVSag = QtWidgets.QAction(Main)
        self.actionTVSag.setObjectName('3DVolume Sagital')

        self.actionTVAx = QtWidgets.QAction(Main)
        self.actionTVAx.setObjectName('3DVolume Axial')

        ######################### Help->Aobut ################################
        self.actionabout = QtWidgets.QAction(Main)
        self.actionabout.setObjectName("actionabout")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/about.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionabout.setIcon(icon)

        self.actionVersion = QtWidgets.QWidgetAction(self)
        label = QtWidgets.QLabel("Version {}".format(__VERSION__))
        label.setStyleSheet("color: white;")
        self.actionVersion.setDefaultWidget(label)

        ######################### Help->Manual ################################
        self.actionmanual = QtWidgets.QAction(Main)
        self.actionmanual.setObjectName("actionabout")
        self.update_action = QtWidgets.QAction("Check for Updates...", self)
        self.update_action.triggered.connect(self.update_application)
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/about.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionmanual.setIcon(icon)

    def _setup_menus(self, Main):
        """Creates the menubar and menus, then populates them with actions."""
        self.menubar = QtWidgets.QMenuBar(Main)
        self.menubar.setNativeMenuBar(False)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1050, 22))
        self.menubar.setObjectName("menubar")
        Main.setMenuBar(self.menubar)

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")

        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")

        self.menuToolbar = QtWidgets.QMenu(self.menuView)
        self.menuToolbar.setObjectName("menuToolbar")

        self.menuWidgets = QtWidgets.QMenu(self.menuView)
        self.menuWidgets.setObjectName("menuWidgets")

        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")

        self.menuPlugins = QtWidgets.QMenu(self.menubar)
        self.menuPlugins.setObjectName("MenuPlugins")

        self.menuSeg = QtWidgets.QMenu(self.menuTools)
        self.menuSeg.setObjectName("menuSeg")

        self.menuCalc = QtWidgets.QMenu(self.menubar)
        self.menuCalc.setObjectName("menuCalc")

        self.menuPreprocess = QtWidgets.QMenu(self.menuTools)
        self.menuPreprocess.setObjectName("menuPrep")


        self.menuRegistration = QtWidgets.QMenu(self.menuTools)
        self.menuRegistration.setObjectName("menuRegistration")

        self.menuBasicInfo = QtWidgets.QMenu(self.menuTools)
        self.menuBasicInfo.setObjectName("MenuBasicInfo")

        self.menuImport = QtWidgets.QMenu(self.menubar)
        self.menuImport.setObjectName("menuImport")

        self.menuExport = QtWidgets.QMenu(self.menubar)
        self.menuExport.setObjectName("menuExport")

        self.menuTV = QtWidgets.QMenu(self.menuCalc)
        self.menuTV.setObjectName("menuTV")

        self.menuImport.addAction(self.actionOpenView1)
        self.menuImport.addAction(self.actionOpenView2)

        self.menuImport.addSeparator()
        self.menuImport.addAction(self.actionImportSegView1)
        self.menuImport.addAction(self.actionImportSegView2)

        self.menuExport.addAction(self.actionExportImView2)
        self.menuExport.addAction(self.actionExportImView1)

        self.menuExport.addSeparator()
        self.menuExport.addAction(self.actionExportSegView1)
        self.menuExport.addAction(self.actionExportSegView2)

        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionsave)
        self.menuFile.addAction(self.actionsaveas)
        # self.menuFile.addAction(self.actionsaveModified)
        self.menuFile.addSeparator()
        # self.menuFile.addAction(self.actionconvert)
        self.menuFile.addMenu(self.menuImport)
        self.menuFile.addMenu(self.menuExport)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionScreenS)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionCloseView1)
        self.menuFile.addAction(self.actionCloseView2)

        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionFile_info)
        self.menuFile.addAction(self.actionexit)

        self.menuAbout.addAction(self.actionmanual)
        self.menuAbout.addAction(self.update_action)
        self.menuAbout.addAction(self.actionabout)
        self.menuAbout.addAction(self.actionVersion)


        self.menuView.addAction(self.actionMain_Toolbar)
        self.menuView.addAction(self.action_interaction_Toolbar)
        self.menuView.addAction(self.action_guideLines)
        self.menuView.addAction(self.action_axisLines)
        self.menuView.addMenu(self.menuToolbar)
        self.menuToolbar.addAction(self.actionMain_Toolbar)
        self.menuToolbar.addAction(self.action_interaction_Toolbar)
        self.menuView.addMenu(self.menuWidgets)

        # ... (all menuView adds) ...
        """
        
        actions_widgets = self.createPopupMenu().actions()
        for action in actions_widgets:
            self.menuWidgets.addAction(action)
        """
        self.menuSeg.addSeparator()



        # self.menuSeg.addAction(self.actionNNventricleGatherIm)

        self.menuTV.addSeparator()
        self.menuTV.addAction(self.actionTVCor)
        self.menuTV.addAction(self.actionTVSag)
        self.menuTV.addAction(self.actionTVAx)

        self.menuBasicInfo.addAction(self.actionHistImage)

        self.menuBasicInfo.addAction(self.actionfile_iminfo)

        self.menuRegistration.addAction(self.actionImageRegistration)
        self.menuRegistration.addAction(self.actionImageTransformation)

        self.menuPreprocess.addAction(self.actionMasking)
        self.menuPreprocess.addAction(self.actionOperationMask)
        self.menuPreprocess.addSeparator()
        self.menuPreprocess.addAction(self.actionImageThresholding)






        self.menuCalc.addAction(self.menuTV.menuAction())
        standard_style = """
                QMenu::separator {
                    height: 1px;         /* Set the height of the separator line */
                    background: lightgray; /* Give the line a color */
                    margin-top: 5px;     /* Add 5 pixels of empty space above the line */
                    margin-bottom: 5px;  /* Add 5 pixels of empty space below the line */
                }
            """
        self.menuTools.setStyleSheet(standard_style)
        self.menuPlugins.setStyleSheet(standard_style)



        # Your existing code to add actions and separators
        self.menuTools.addAction(self.actionUndo)
        self.menuTools.addAction(self.actionRedo)
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.menuPreprocess.menuAction())
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.menuRegistration.menuAction())
        self.menuTools.addSeparator()
        self.menuTools.addAction(self.menuBasicInfo.menuAction())

        # self.menuTools.addAction(self.actionNNventricleGatherIm)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuPlugins.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        # self.menubar.addAction(self.actionVersion)

    def _setup_other_widgets(self, Main):
        """Creates widgets not in menus/toolbars (Docks, ComboBox, Logo)."""
        self.pixmap_box_label = QtWidgets.QLabel(Main)
        self.pixmap_box_label.setText('Combined')

        self.createDockWidget(Main)

        # Logo for toolbar
        self.logo = QtWidgets.QLabel(Main)
        self.logo.setPixmap(QtGui.QPixmap(settings.DEFAULT_USE_DIR + "/melage_top.png"))
        self.logo.resize(100, 50)  # This hardcoded size is not ideal

        # NewDialog
        self.newdialog = NewDialog(Main)  # Assumes NewDialog is imported

        # Label Size Policies (This assumes self.label_1, etc. exist from a .ui file)

    def _setup_toolbars(self, Main):
        """
        Creates and populates main application toolbars.
        Optimized for clarity, logical grouping, and styling.
        """
        # --- 1. Shared Styling & Configuration ---
        # Define style once for consistency
        TOOLBAR_STYLE = """
            QToolBar {
                background-color: #000000;
                border-bottom: 0px solid #19232D;
                padding: 2px;
                font-weight: bold;
                spacing: 2px;
            }
            QToolBar::separator:horizontal {
                width: 10px;
                margin: 0 10px;
                border-left: 1px solid #333; /* Optional visual divider */
            }
            QToolButton { margin: 2px; }
        """

        # Dynamic Icon Sizing (High-DPI aware)
        pixel_ratio = self.devicePixelRatio()
        base_size = 36
        icon_size = QtCore.QSize(int(base_size * pixel_ratio), int(base_size * pixel_ratio))

        # Helper to create spacers
        def create_spacer():
            spacer = QtWidgets.QWidget()
            spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            return spacer

        # --- 2. Main File Toolbar (Top) ---
        self.toolBar = QtWidgets.QToolBar(Main)
        self.toolBar.setObjectName("FileToolbar")
        self.toolBar.setStyleSheet(TOOLBAR_STYLE)
        self.toolBar.setIconSize(icon_size)
        self.toolBar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        Main.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        Main.insertToolBarBreak(self.toolBar)  # Ensures it's on its own row if needed

        # Group: Project Management
        self.toolBar.addAction(self.actionNew)
        self.toolBar.addAction(self.actionLoad)
        self.toolBar.addAction(self.actionsave)
        self.toolBar.addSeparator()

        # Group: View 1 (Ultrasound/ECO)
        self.toolBar.addAction(self.actionOpenView1)
        self.toolBar.addAction(self.actionImportSegView1)
        self.toolBar.addSeparator()

        # Group: View 2 (MRI)
        self.toolBar.addAction(self.actionOpenView2)
        self.toolBar.addAction(self.actionImportSegView2)
        self.toolBar.addSeparator()

        # Group: Dynamic Controls (Hidden by default)
        self.actionComboBox_visible = self.toolBar.addWidget(self.actionComboBox)
        self.actionComboBox_visible.setVisible(False)

        # Spacer to push Logo/Exit to the right
        self.toolBar.addWidget(create_spacer())

        # Group: App Controls
        self.toolBar.addWidget(self.logo)
        self.toolBar.addAction(self.actionexit)

        # --- 3. Interaction Toolbar (Bottom/Secondary) ---
        self.toolBar2 = QtWidgets.QToolBar(Main)
        self.toolBar2.setObjectName("InteractionToolbar")
        self.toolBar2.setStyleSheet(TOOLBAR_STYLE)
        self.toolBar2.setIconSize(icon_size)
        self.toolBar2.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        Main.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar2)
        Main.insertToolBarBreak(self.toolBar2)  # Force new line below File toolbar

        # Group: Navigation (Move & Zoom)
        self.toolBar2.addAction(self.actionArrow)  # Select
        self.toolBar2.addAction(self.actionPan)  # Pan
        self.toolBar2.addAction(self.actionZoomNeutral)  # Reset Zoom
        self.toolBar2.addAction(self.actionZoomIn)
        self.toolBar2.addAction(self.actionZoomOut)
        self.toolBar2.addSeparator()

        # Group: Creation (Draw/Mask) - Primary Actions
        self.toolBar2.addAction(self.actionPaint)
        self.toolBar2.addAction(self.actionContour)
        self.toolBar2.addAction(self.actionContourX)
        self.toolBar2.addAction(self.actionCircles)
        self.toolBar2.addSeparator()

        # Group: Modification (Erase)
        self.toolBar2.addAction(self.actionErase)
        self.toolBar2.addAction(self.actionEraseX)
        self.toolBar2.addSeparator()

        # Group: Measurement & Analysis
        self.toolBar2.addAction(self.actionRuler)
        self.toolBar2.addAction(self.actionLine)
        self.toolBar2.addAction(self.actionPoints)
        self.toolBar2.addSeparator()

        # Group: Properties (Color)
        self.toolBar2.addAction(self.actionColor)
        self.toolBar2.addWidget(self.pixmap_box_label)

        # Spacer to push View controls to right
        self.toolBar2.addWidget(create_spacer())

        # Group: View Layouts & 3D
        self.toolBar2.addAction(self.actionVerticalView)
        self.toolBar2.addAction(self.actionHorizontalView)
        self.toolBar2.addSeparator()
        self.toolBar2.addAction(self.actionGoTo)
        self.toolBar2.addAction(self.action3D)

        # Disable Interaction toolbar until data is loaded
        self.toolBar2.setDisabled(True)
    def _setup_toolbars2(self, Main):
        """Creates and populates all toolbars."""

        self.toolBar = QtWidgets.QToolBar(Main)
        self.toolBar.setObjectName("toolBar")
        Main.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        Main.insertToolBarBreak(self.toolBar)
        self.toolBar.addAction(self.actionNew)
        self.toolBar.addAction(self.actionLoad)
        self.toolBar.addAction(self.actionsave)

        self.toolBar.addSeparator()
        # self.toolBar.addAction(self.actionsaveas)
        # spacerItem = QtWidgets.QWidget()
        # spacerItem.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        # self.toolBar.addWidget(spacerItem)
        cts = """
                QToolBar {
          background-color: #000000;
          border-bottom: 0px solid #19232D;
          padding: 0px;
          font-weight: bold;
          spacing: 0px;
        }
        QToolBar::separator:horizontal
        {
        	width: 10px;
        	margin-left: 10px;
        		margin-right: 10px;
        }
        QToolButton{margin: 2px 2px;}
                """
        self.toolBar.setStyleSheet(cts)

        self.toolBar.addAction(self.actionOpenView1)
        self.toolBar.addAction(self.actionImportSegView1)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionOpenView2)
        self.toolBar.addAction(self.actionImportSegView2)

        # self.toolBar.addAction(self.actionOpenFA)
        # self.toolBar.addAction(self.actionOpenTract)
        self.toolBar.addSeparator()
        self.actionComboBox_visible = self.toolBar.addWidget(self.actionComboBox)
        self.actionComboBox_visible.setVisible(False)
        # self.toolBar.addSeparator()
        spacerItem = QtWidgets.QWidget()
        spacerItem.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.toolBar.addWidget(spacerItem)

        self.toolBar.addWidget(self.logo)
        self.toolBar.addAction(self.actionexit)

        self.toolBar2 = QtWidgets.QToolBar(Main)
        self.toolBar2.setObjectName("Interaction")
        Main.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar2)
        Main.insertToolBarBreak(self.toolBar2)
        self.toolBar2.addAction(self.actionArrow)
        self.toolBar2.addAction(self.actionPan)
        self.toolBar2.addSeparator()
        self.toolBar2.setStyleSheet(cts)
        self.toolBar2.addAction(self.actionErase)
        self.toolBar2.addAction(self.actionEraseX)
        self.toolBar2.addSeparator()
        self.toolBar2.addAction(self.actionPaint)
        self.toolBar2.addAction(self.actionContour)
        self.toolBar2.addAction(self.actionContourX)
        self.toolBar2.addAction(self.actionCircles)
        self.toolBar2.addSeparator()
        self.toolBar2.addAction(self.actionColor)
        self.toolBar2.addWidget(self.pixmap_box_label)
        # self.toolBar2.addSeparator()

        spacerItem = QtWidgets.QWidget()
        spacerItem.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.toolBar2.addWidget(spacerItem)

        self.toolBar2.addAction(self.actionLine)

        self.toolBar2.addAction(self.actionPoints)
        self.toolBar2.addSeparator()
        self.toolBar2.addAction(self.actionZoomNeutral)
        self.toolBar2.addAction(self.actionZoomIn)
        self.toolBar2.addAction(self.actionZoomOut)
        self.toolBar2.addSeparator()
        self.toolBar2.addAction(self.actionVerticalView)
        self.toolBar2.addAction(self.actionHorizontalView)
        self.toolBar2.addAction(self.actionRuler)
        self.toolBar2.addAction(self.actionGoTo)
        self.toolBar2.addAction(self.action3D)
        self.toolBar2.setDisabled(True)
        pixel_ratio = self.devicePixelRatio()
        base_size = 36
        dynamic_size = int(base_size * pixel_ratio)
        self.toolBar2.setIconSize(QtCore.QSize(dynamic_size, dynamic_size))
        self.toolBar.setIconSize(QtCore.QSize(dynamic_size, dynamic_size))
        # Set fixed size policy
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.toolBar.setSizePolicy(sizePolicy)
        self.toolBar2.setSizePolicy(sizePolicy)

    def _connect_signals(self):
        """Connects all signals to slots."""

        # --- Action Connections ---
        self.actionColor.triggered.connect(self.color_picker)
        self.horizontalSlider_1.valueChanged.connect(self.changeSight1)
        self.horizontalSlider_2.valueChanged.connect(self.changeSight2)
        self.horizontalSlider_3.valueChanged.connect(self.changeSight3)
        self.horizontalSlider_4.valueChanged.connect(self.changeSight4)
        self.horizontalSlider_5.valueChanged.connect(self.changeSight5)
        self.horizontalSlider_6.valueChanged.connect(self.changeSight6)
        self.horizontalSlider_11.valueChanged.connect(self.changeSightTab3)
        self.horizontalSlider_12.valueChanged.connect(self.changeSightTab4)


        self.radioButton_1.clicked.connect(partial(self.changeToCoronal, 'eco'))
        self.radioButton_2.clicked.connect(partial(self.changeToSagittal, 'eco'))
        self.radioButton_3.clicked.connect(partial(self.changeToAxial, 'eco'))
        self.radioButton_4.clicked.connect(self.showSegOnWindow)

        self.radioButton_21_1.clicked.connect(partial(self.changeToCoronal, 'mri'))
        self.radioButton_21_2.clicked.connect(partial(self.changeToSagittal, 'mri'))
        self.radioButton_21_3.clicked.connect(partial(self.changeToAxial, 'mri'))
        self.radioButton_21.clicked.connect(self.showSegOnWindow)

        self.actionOpenView1.triggered.connect(self.browse_view1)
        self.actionOpenView2.triggered.connect(self.browse_view2)
        self.actionComboBox.currentTextChanged.connect(self.changeVolume)
        self.actionOpenFA.triggered.connect(self.browseFA)
        self.actionOpenTract.triggered.connect(self.browseTractoGraphy)
        self.actionNew.triggered.connect(self.newProject)
        self.actionCloseView1.triggered.connect(self.CloseView1)
        self.actionCloseView2.triggered.connect(self.CloseView2)
        self.actionImportSegView2.triggered.connect(partial(self.importData, 'View2_SEG'))
        self.actionImportSegView1.triggered.connect(partial(self.importData, 'View1_SEG'))
        self.actionExportImView1.triggered.connect(partial(self.exportData, 'View1_IM'))
        self.actionExportSegView1.triggered.connect(partial(self.exportData, 'View1_SEG'))
        self.actionExportImView2.triggered.connect(partial(self.exportData, 'View2_IM'))
        self.actionExportSegView2.triggered.connect(partial(self.exportData, 'View2_SEG'))
        self.actionScreenS.triggered.connect(self.showScreenShotWindow)
        self.actionLoad.triggered.connect(self.loadProject)
        self.actionFile_info.triggered.connect(self.showInfoWindow)
        self.actionUndo.triggered.connect(self.Undo)
        self.actionRedo.triggered.connect(self.Redo)
        self.actionFile_changeIM.triggered.connect(self.showIMVARSWindow)
        self.actionfile_iminfo.triggered.connect(partial(self.maskingShow, 5))
        self.actionconvert.triggered.connect(self.convert)
        self.actionexit.triggered.connect(self.close)
        self.actionsave.triggered.connect(self.save)
        self.actionsaveas.triggered.connect(self.saveas)
        self.actionabout.triggered.connect(self.about)
        self.actionmanual.triggered.connect(self.manual)
        self.actionArrow.triggered.connect(partial(self.setCursors, 0))
        self.actionPan.triggered.connect(partial(self.setCursors, 2))
        self.actionPaint.triggered.connect(partial(self.setCursors, 1))
        self.actionErase.triggered.connect(partial(self.setCursors, 3))
        self.actionRuler.triggered.connect(partial(self.setCursors, 6))
        self.actionLine.triggered.connect(partial(self.setCursors, 8))
        self.actionContour.triggered.connect(partial(self.setCursors, 4))
        self.actionPoints.triggered.connect(partial(self.setCursors, 5))
        self.actionCircles.triggered.connect(partial(self.setCursors, 9, None))
        # self.actionGoTo.triggered.connect(partial(self.setCursors, 7))
        self.actionGoTo.triggered.connect(self.activateGuidelines)
        self.action3D.triggered.connect(self.activate3d)
        self.actionZoomIn.triggered.connect(partial(self.Zoom, 'In'))
        self.actionZoomOut.triggered.connect(partial(self.Zoom, 'Out'))
        self.actionZoomNeutral.triggered.connect(partial(self.Zoom, 'Neutral'))

        self.actionContourX.triggered.connect(self.showRpeatWindow)
        self.actionContourX.triggered.connect(partial(self.setCursorsX, 4))
        self.actionEraseX.triggered.connect(self.showRpeatWindow)
        self.actionEraseX.triggered.connect(partial(self.setCursorsX, 3))
        # self.actionSegExportEco.triggered.connect(self.showScreenShotWindow)
        self.actionMain_Toolbar.triggered.connect(self.toolBar.setVisible)
        self.action_guideLines.triggered.connect(self.activateGuidelines)
        self.action_axisLines.triggered.connect(self.activateAxisLines)
        self.toolBar.visibilityChanged.connect(self.actionMain_Toolbar.setChecked)
        self.t1_1.valueChanged.connect(self.changeBrightness)
        self.t1_2.valueChanged.connect(self.changeContrast)
        self.t1_3.valueChanged.connect(self.changeBandPass)
        self.t1_4.valueChanged.connect(self.changeSobel)
        self.t1_5.valueChanged.connect(self.Rotate)
        self.t1_7.valueChanged.connect(self.changeBandPass)
        self.toggle1_1.clicked.connect(self.changeenable_endo_enhance)
        self.t2_1.valueChanged.connect(self.changeBrightness)
        self.t2_2.valueChanged.connect(self.changeContrast)
        self.t2_3.valueChanged.connect(self.changeBandPass)
        self.t2_7.valueChanged.connect(self.changeBandPass)
        self.t2_4.valueChanged.connect(self.changeSobel)
        self.t2_5.valueChanged.connect(self.Rotate)
        self.toggle2_1.clicked.connect(self.changeenable_endo_enhance)

        self.scrol_tol_rad_circle.valueChanged.connect(self.changeSizePen)
        self.scrol_rad_circle.valueChanged.connect(lambda value: self.changeRadiusCircle(value, True))

        self.scroll_intensity.valueChanged.connect(lambda thrsh: self.ColorIntensityChange(thrsh, 'seg'))
        self.scroll_image_intensity.valueChanged.connect(lambda thrsh: self.ColorIntensityChange(thrsh, 'image'))
        self.page1_rot_cor.currentTextChanged.connect(self.changeRotAx)
        self.page2_rot_cor.currentTextChanged.connect(self.changeRotAx)
        #self.dw5_s1.valueChanged.connect(self.trackDistance)
        #self.dw5_s2.valueChanged.connect(self.trackThickness)


        self.actionHistImage.triggered.connect(partial(self.maskingShow, 4))


        # FCM
        self.actionImageThresholding.triggered.connect(partial(self.maskingShow, 6))
        self.actionImageRegistration.triggered.connect(partial(self.registerShow, 0))
        self.actionImageTransformation.triggered.connect(partial(self.registerShow, 1))
        self.actionMasking.triggered.connect(partial(self.maskingShow, 0))
        self.actionOperationMask.triggered.connect(partial(self.maskingShow, 1))
        self.scroll_intensity.valueChanged.connect(self.lb_scroll_intensity.setNum)

        name = 'openGLWidget_'
        for i in range(12):
            nameWidget = name + str(i + 1)
            if hasattr(self, nameWidget):
                widget = getattr(self, name + str(i + 1))
                widget.segChanged.connect(
                    lambda whiteInd, currentWidnowName, colorInd, sliceNum: self.updateSegmentation(whiteInd,
                                                                                                    currentWidnowName,
                                                                                                    colorInd, sliceNum))

                widget.LineChanged.connect(
                    lambda params: self.updateLP(params))
                widget.interpolate.connect(
                    lambda params: self.Interpolate(params))
                widget.zoomchanged.connect(lambda value, slider: self.changeRadiusCircle(value, slider))

                widget.rulerInfo.connect(
                    lambda distance, colorind: self.update_table_measure(distance, colorind))

                widget.sliceNChanged.connect(
                    lambda sliceNumber: self.updateSliceNumber(sliceNumber)
                )
                widget.intensity_change.connect(
                    lambda thrsh: self.ColorIntensityChange(thrsh, 'seg')
                )
                widget.intensity_change.connect(self.scroll_intensity.setValue)
                widget.goto.connect(
                    lambda slices, currentWidnowName: self.updateAllSlices(slices, currentWidnowName)
                )

        self.setNewImage_view1.connect(
            lambda shapeImage: self.openGLWidget_14.createGridAxis(shapeImage)
        )

        self.setNewImage_view2.connect(
            lambda shapeImage: self.openGLWidget_24.createGridAxis(shapeImage)
        )

        self.openGLWidget_14.point3dpos.connect(
            lambda pose3d, windowName: self.updateLabelPs(pose3d, windowName, 'eco')
        )
        self.openGLWidget_24.point3dpos.connect(
            lambda pose3d, windowName: self.updateLabelPs(pose3d, windowName, 'mri')
        )

        self.openGLWidget_14.update_3dview.connect(
            lambda map_type, reset: self.update3Dview(map_type, None, 'eco')
        )
        self.openGLWidget_24.update_3dview.connect(
            lambda map_type, reset: self.update3Dview(map_type, None, 'mri')
        )

        self.openGLWidget_14.update_cmap.connect(
            lambda map_type, reset: self.update3Dview(map_type, reset, 'eco')
        )
        self.openGLWidget_24.update_cmap.connect(
            lambda map_type, reset: self.update3Dview(map_type, reset, 'mri')
        )

        self.tabWidget.currentChanged.connect(self.changedTab)
        #self.openGLWidget_11.resized.connect(self.changedTab)
        #self.openGLWidget_12.resized.connect(self.changedTab)



    def _set_initial_state(self):
        """Sets the initial visibility and enabled state of widgets."""

        # Actions
        self.actionOpenView1.setDisabled(True)
        self.actionOpenView2.setDisabled(True)
        self.actionComboBox.setDisabled(True)
        self.actionOpenFA.setDisabled(True)
        self.actionOpenTract.setDisabled(True)
        self.actionImportSegView2.setDisabled(True)
        self.actionImportSegView1.setDisabled(True)
        self.actionExportImView2.setDisabled(True)
        self.actionExportSegView2.setDisabled(True)
        self.actionExportImView1.setDisabled(True)
        self.actionExportSegView1.setDisabled(True)
        self.actionsave.setDisabled(True)
        self.actionsaveas.setDisabled(True)

        # Toolbars
        self.toolBar2.setDisabled(True)
        self.actionComboBox_visible.setVisible(False)
        self.actionComboBox.setVisible(False)  # Also hide the widget itself

        # Sliders
        self.horizontalSlider_1.setVisible(False)
        self.horizontalSlider_2.setVisible(False)
        self.horizontalSlider_3.setVisible(False)
        self.horizontalSlider_4.setVisible(False)
        self.horizontalSlider_5.setVisible(False)
        self.horizontalSlider_6.setVisible(False)
        #self.horizontalSlider_7.setVisible(False)
        #self.horizontalSlider_8.setVisible(False)
        #self.horizontalSlider_9.setVisible(False)
        #self.horizontalSlider_10.setVisible(False)
        self.horizontalSlider_11.setVisible(False)
        self.horizontalSlider_12.setVisible(False)

        # Radio Buttons
        self.radioButton_1.setVisible(False)
        self.radioButton_2.setVisible(False)
        self.radioButton_3.setVisible(False)
        self.radioButton_4.setVisible(False)
        self.radioButton_21_1.setVisible(False)
        self.radioButton_21_2.setVisible(False)
        self.radioButton_21_3.setVisible(False)
        self.radioButton_21.setVisible(False)



    def setupUi(self, Main):
        """

        :param Main:
        :return:
        """
        self._setup_main_window(Main)
        self._setup_other_widgets(Main)
        self._setup_central_layout(Main)

        self._create_actions(Main)

        self._setup_menus(Main)
        self.load_plugins()


        self._setup_toolbars(Main)

        # Group all signal/slot connections together
        self._connect_signals()

        # Group all .setVisible(False), .setDisabled(True), etc.
        self._set_initial_state()
        self.retranslateUi(Main)
        QtCore.QMetaObject.connectSlotsByName(Main)

        self.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self)
        self.init_state()
        self.create_cursors()

        self.Main = Main


    def updateDataRegistration(self):
        """
        Image to Image registration window
        :return:
        """
        val = self.registrationD
        ind_image = val.comboBox_image.currentIndex()
        if ind_image==0:
            if not hasattr(self, 'readView1'):
                return
            if not hasattr(self.readView1, 'im'):
                return
            val.setData(self.readView1.im)
        elif ind_image==1:
            if not hasattr(self, 'readView2'):
                return
            if not hasattr(self.readView2, 'im'):
                return

            val.setData(self.readView2.im)


    def applyMaskToImage(self, values, operation=False):
        """
        Apply masks to image
        :param values:
        :return:
        """
        self.setEnabled(True)
        if len(values)==3:
            ind_image, ind_sel, keep = values
            ind_color = int(float(self.color_name[ind_sel].split('_')[0]))
        else:
            ind_image, ind_sel, ind_sel2, type_operation = values
            ind_color = int(float(self.color_name[ind_sel].split('_')[0]))
            ind_color2 = int(float(self.color_name[ind_sel2].split('_')[0]))

        if ind_image==0:
            if not hasattr(self, 'readView1'):
                return
            if not hasattr(self.readView1, 'im'):
                return

            ind_selected = self.readView1.npSeg==ind_color
            if operation:
                ind_selected2 = self.readView1.npSeg == ind_color2
                if type_operation=='+':
                    ind_selected = (ind_selected.astype('int') + ind_selected2.astype('int'))>0
                    self.readView1.npSeg[ind_selected] = ind_color
                elif type_operation== '-':
                    ind_selected = (ind_selected.astype('int') - ind_selected2.astype('int'))>0
                    self.readView1.npSeg[ind_selected] = ind_color
            else:
                if ind_color!=9876:
                    im = self.readView1.npImage.copy()
                else:
                    im = self.readView1.im.get_fdata().copy()
                if ind_selected.sum() > 1:
                    if keep:
                        im[~ind_selected] = 0
                    else:
                        im[ind_selected] = 0

                im = make_image(im, self.readView1.im)
                self.readView1.changeImData(im, axis=[0, 1, 2])
                self.browse_view1(fileObj=None, use_dialog=False)
            self.changedTab()
        elif ind_image==1:
            if not hasattr(self, 'readView2'):
                return
            if not hasattr(self.readView2, 'im'):
                return
            ind_selected = self.readView2.npSeg == ind_color
            if operation:
                ind_selected2 = self.readView2.npSeg == ind_color2
                if type_operation=='+':
                    ind_selected = (ind_selected.astype('int') + ind_selected2.astype('int'))>0
                    self.readView2.npSeg[ind_selected] = ind_color
                elif type_operation== '-':
                    ind_selected = (ind_selected.astype('int') - ind_selected2.astype('int'))>0
                    self.readView2.npSeg[ind_selected] = ind_color
            else:
                if ind_color != 9876:
                    im = self.readView2.npImage.copy()
                else:
                    im = self.readView2.im.get_fdata().copy()

                if ind_selected.sum() > 1:
                    if keep:
                        im[~ind_selected]=0
                    else:
                        im[ind_selected] = 0

                im = make_image(im, self.readView2.im)
                self.readView2.changeImData(im, axis=[0, 1, 2])
                self.browse_view2(fileObj=None, use_dialog=False)
            self.changedTab()


    def applyNewCoordSys(self, values):
        """
        Apply new coordsystem
        :param values:
        :return:
        """
        ind_image, targ_system = values
        if ind_image==0:
            if not hasattr(self, 'readView1'):
                return
            if not hasattr(self.readView1, 'im'):
                return

            status = self.readView1._changeCoordSystem(targ_system)
            if status:

                self.readView1.source_system = targ_system
                self.browse_view1(fileObj=None, use_dialog=False)
                self.changedTab()
        elif ind_image==1:
            if not hasattr(self, 'readView2'):
                return
            if not hasattr(self.readView2, 'im'):
                return
            status = self.readView2._changeCoordSystem(targ_system)
            if status:

                self.browse_view2(fileObj=None, use_dialog=False)
                self.changedTab()

    def maskingClose(self, val):
        """
        Clear information from masking
        :param val:
        :return:
        """
        self.setEnabled(True)




    def reconstruction(self, reconstruct, ind, reader, alg):
        if reconstruct:
            save_var = '_immri_bedl_{}'.format(ind)
            save_var_seg = '_immri_bedl_seg_{}'.format(ind)
            if not hasattr(self, save_var):#[::-1, ::-1, ::-1].transpose(2, 1, 0)
                setattr(self, save_var,
                        reader.im.__class__(reader.im.dataobj[:], reader.im.affine, reader.im.header))
                setattr(self, save_var_seg, reader.npSeg)
            elif getattr(self, save_var) is None:
                    setattr(self, save_var,
                                    reader.im.__class__(reader.im.dataobj[:], reader.im.affine, reader.im.header))
                    setattr(self, save_var_seg, reader.npSeg)
            reader.im = alg.im_rec
            reader.set_metadata()
            reader.read_pars(reset_seg=False)
        else:
            save_var = '_immri_bedl_{}'.format(ind)
            save_var_seg = '_immri_bedl_seg_{}'.format(ind)
            if not hasattr(self, save_var):
                return
            # reader.im = self._immri_tmp
            if getattr(self, save_var) is None:
                return
            reader.im = getattr(self, save_var)
            reader.set_metadata()
            reader.read_pars(adjust_for_show=True)
            self.setNewImage_view2.emit(reader.npImage.shape[:3])
            reader.npSeg = getattr(self, save_var_seg)#[::-1, ::-1, ::-1].transpose(2, 1, 0)
            delattr(self, save_var_seg)
            delattr(self, save_var)






    def Thresholding(self, val, reconstruct=False):


        """
        Brain extraction related
        :param val:
        :return:
        """

        at_view1 = hasattr(self, 'readView1')
        at_view2 = hasattr(self, 'readView2')

        if val in ('BET', 'Deep BET', 'histeq', 'Segmentation', 'MorphSeg'):

            if val == 'histeq':
                alg = self.ImageThresholding
            ind = alg.comboBox_image.currentIndex()
            if ind == 0 and at_view1:
                if hasattr(self.readView1, 'npImage'):
                    if val != 'histeq':
                        self.readView1.npSeg = alg.mask.transpose(2, 1, 0)[::-1, ::-1, ::-1]
                    if hasattr(alg,'im_rec'):
                        reader = self.readView1
                        if alg.im_rec is not None:
                            if not hasattr(alg.im_rec, 'get_fdata'):

                                alg.im_rec = make_image(alg.im_rec.transpose(2, 1, 0)[::-1, ::-1, ::-1], reader.im)
                            self.reconstruction(reconstruct, ind, reader, alg)
                            if val == 'histeq' and not reconstruct:
                                alg.plot(self.readView1.npImage)
                    if self.readView1.npImage is not None:
                        self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)

            elif ind == 1 and at_view2:
                if hasattr(self.readView2, 'npImage'):
                    if val != 'histeq':
                        self.readView2.npSeg = alg.mask.transpose(2, 1, 0)[::-1, ::-1, ::-1]
                    if hasattr(alg,'im_rec'):
                        reader = self.readView2
                        if alg.im_rec is not None:
                            if not hasattr(alg.im_rec, 'get_fdata'):

                                alg.im_rec = make_image(alg.im_rec, reader.im)
                            self.reconstruction(reconstruct, ind, reader, alg)
                            if val == 'histeq' and not reconstruct:
                                alg.plot(self.readView2.npImage)
                    if self.readView2.npImage is not None:
                        self.updateDispView2(self.readView2.npImage, self.readView2.npSeg, initialState=True)

            return

        ind = self.ImageThresholding.comboBox_image.currentIndex()

        if val == 'apply':
            if hasattr(self.ImageThresholding,'_currentThresholds'):

                ind = self.ImageThresholding.comboBox_image.currentIndex()
                if ind == 0 and at_view1:
                    if hasattr(self.readView1, 'npImage'):
                        self.readView1.npSeg = apply_thresholding(self.readView1.npImage, self.ImageThresholding._currentThresholds)
                        self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)
                elif ind == 1 and at_view2:
                    if hasattr(self.readView2, 'npImage'):
                        self.readView2.npSeg = apply_thresholding(self.readView2.npImage, self.ImageThresholding._currentThresholds)
                        self.updateDispView2(self.readView2.npImage, self.readView2.npSeg, initialState=True)
        elif val=='replot':

            repl = False
            if ind==0:

                if at_view1:
                    at_im_view1 = hasattr(self.readView1, 'npImage')
                    if at_im_view1:
                        self.ImageThresholding.plot(self.readView1.npImage)
                        repl = True
            else:

                if at_view2:
                    at_im_view2 = hasattr(self.readView2, 'npImage')
                    if at_im_view2:
                        self.ImageThresholding.plot(self.readView2.npImage)
                        repl = True
            if not repl:
                self.ImageThresholding.emptyPlot()

    def registerShow(self, val=1):
        """
        Image to Image registration widget
        :param val: the parameter that defines type of operation (transformation or registration)
        :return:
        """
        #self.setEnabled(False)

        if val == 0:
            el = self.registrationD
            el.set_source(settings.DEFAULT_USE_DIR)
            at_view1 = hasattr(self, 'readView1')
            at_view2 = hasattr(self, 'readView2')
            if at_view1 and at_view2:
                at_im_view1 = hasattr(self.readView1, 'npImage')
                at_im_view2 = hasattr(self.readView2, 'npImage')
                if not at_im_view1 and at_im_view2 and val in [0, 1, 2, 3, 6, 7, 8, 9]:
                    el.comboBox_image.setCurrentIndex(1)
        elif val == 1:
            el = self.transformationD
            el.set_source(settings.DEFAULT_USE_DIR)
        #el.setEnabled(True)
        el.show()

    def maskingShow(self, val=1):
        """
        Show different widget according to the define parameter val
        :param val:
        :return:
        """
        self.setEnabled(False)
        if val == 0:
            el = self.Masking
        elif val == 1:
            el = self.MaskingOperations

        elif val == 4:
            el = self.HistImage
        elif val == 5:
            el = self.iminfo_dialog

            try:
                for i, widg in enumerate([self.readView1, self.readView2]):
                    el.set_tag_value(widg, ind=i)
            except:
                pass

        elif val == 6:
            el = self.ImageThresholding
        try:
            at_view1 = hasattr(self, 'readView1')
            at_view2 = hasattr(self, 'readView2')

            if val in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12]:
                if at_view2 and hasattr(self.readView2, 'npImage'):
                    el.comboBox_image.setCurrentIndex(1)
                    if hasattr(el, 'comboBox_image_type'):
                        el.comboBox_image_type.setCurrentIndex(1)
                    elif val in [9, 10, 11]:

                        affine = getattr(self.readView2, 'affine', None)
                        header = getattr(self.readView2, 'header', None)
                        img = make_image_using_affine(self.readView2.npImage, affine, header)
                        el.setData(img, self.readView2.ImSpacing)
                        el.comboBox_image.setCurrentIndex(1)

                elif at_view1 and hasattr(self.readView1, 'npImage'):
                    if val in [9, 10, 11]:
                        affine = getattr(self.readView1, 'affine', None)
                        header = getattr(self.readView1, 'header', None)
                        img = make_image_using_affine(self.readView1.npImage, affine, header)
                        el.setData(img, self.readView1.ImSpacing)
                        el.comboBox_image.setCurrentIndex(0)

            if val == 6:
                index = el.comboBox_image.currentIndex()
                if index == 0 and at_view1:
                    el.plot(self.readView1.npImage)
                elif index == 1 and at_view2:
                    el.plot(self.readView2.npImage)

            elif val in [4, 5]:
                name_view2 = self.filenameView2 if at_view2 else None
                name_view1 = self.filenameView1 if at_view1 else None
                if val == 4:
                    if at_view2:
                        el.plot(self.readView2.npImage, 1)
                    if at_view1:
                        el.plot(self.readView1.npImage, 2)
                el.UpdateName(name_view1,name_view2)

            elif val == 8:
                ind = el.comboBox_image.currentIndex()
                self.resize_image(None, ind)

            if val in [0, 1]:
                el.set_color_options(self.color_name)



            el.setEnabled(True)
            el.show()
        except Exception as e:
            self.setEnabled(True)
            print(e)

    def _cutIM(self, _cutIM):
        """
        Cut image based on a defined limits
        :param _cutIM:
        :return:
        """
        sender = QtCore.QObject.sender(self)
        nameS = 'horizontalSlider_'
        totalvalues = []
        if sender.id in [1,2,3]:
            slider_list = [3, 1, 2]
        elif sender.id in [4,5,6]:
            slider_list = [6,4,5]
        for i in slider_list:
            slider = getattr(self, nameS + str(i))
            totalvalues.append([slider._first, slider._second])

        shp = self.readView1.im.shape
        sortedT = [sorted(i) for i in totalvalues]
        for i , el in enumerate(sortedT):
            if el[0]<=0:
                el[0]=0
            if el[1]>=shp[i] or el[1]==0:
                el[1]=shp[i]
        im = self.readView1.im.get_fdata()
        im = im[sortedT[0][0]:sortedT[0][1], sortedT[0][0]:sortedT[1][1], sortedT[2][0]:sortedT[2][1]]

        im = make_image(im, self.readView1.im)
        self.readView1.im = im

        self.readView1.changeData(type='eco', imchange=True, state=False, axis=[0,1,2])
        self.browse_view1(fileObj=None, use_dialog=False)
        #self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)

    def changeVolume(self):
        """
        Change image volume in 4D images
        :return:
        """
        value = self.actionComboBox.objectName()
        if value=='View2' and hasattr(self, 'readView2'):
            self.browse_view2(use_dialog=False)
        elif value=='View1' and hasattr(self, 'readView1'):
            self.browse_view1(use_dialog=False)



    def updateLabelPs(self, pos3d, windowName, typew='eco'):
        """
        Updating label points
        :param pos3d:
        :param windowName:
        :param typew:
        :return:
        """

        if type(pos3d)!=list:
            if any(pos3d < -1) or pos3d.max()> 2000:
                return


            if typew== 'eco':
                self.updateSegPlanes(pos3d, windowName, 'eco')
            elif typew == 'mri':
                self.updateSegPlanes(pos3d, windowName, 'mri')

    def color_picker(self):
        """
        Pick a color
        :return:
        """

        colordialog = QtWidgets.QColorDialog(self.newdialog)
        color = colordialog.getColor()
        if self.newdialog.exec_() == QFileDialogPreview.Accepted:
            newindex = self.newdialog.lineEdit.text()
            newText = self.newdialog.lineEdit2.text()
            root = self.tree_colors.model().sourceModel().invisibleRootItem()
            similar_items = []
            for l in range(root.rowCount()):
                signal = root.child(l)
                if signal.text()==newindex:
                    similar_items.append(signal)
            #    self.tree_colors.invisibleRootItemi().child(l)
            if len(similar_items)>0:
                qm = QtWidgets.QMessageBox(self.newdialog)
                ret = qm.question(self.newdialog, '', "Do you want to override the color?", qm.Yes | qm.No)
                if ret == qm.Yes:
                    for l in similar_items:
                        root.removeRow(l.row())
                else:
                    return

            color_rgb = [l / 255.0 for l in color.getRgb()]
            add_new_tree_widget(self, newindex, newText, color_rgb)
            #parent.setCheckState(True,QtCore.Qt.CheckState())
        #self.styleChoice.setStyleSheet("QWidget { background-color: %s}" % color.name())




    def activateAxisLines(self, val):
        """
        show axis values
        :param val: boolean
        :return:
        """
        widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6, self.openGLWidget_11, self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3, self.openGLWidget_12]

        for widget in widgets:
            widget.showAxis = val
            widget.update()

    def Zoom(self, val):
        """
        Zoomming operation
        :param val:
        :return:
        """
        if self.tabWidget.currentIndex() == 0:
            widgets_view2 = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6]
            widgets_view1 = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3]
            widgets = widgets_view2 + widgets_view1
        elif self.tabWidget.currentIndex() == 1:
            widgets = [self.openGLWidget_11]
        elif self.tabWidget.currentIndex() == 2:
            widgets = [self.openGLWidget_12]
        else:
            widgets = []

        if val=='In':
            for widget in widgets:
                if widget.isVisible():
                    widget.scaleF = widget.ZOOM_OUT_FACTOR
                    if 0.0 < widget.zoom_level_y * widget.scaleF < 5 or 0.0 < widget.zoom_level_x * widget.scaleF < 5:
                        widget.updateScale(0, 0, widget.scaleF, widget.scaleF)
        elif val == 'Out':
            for widget in widgets:
                if widget.isVisible():
                    widget.scaleF = widget.ZOOM_IN_FACTOR
                    if 0.0 < widget.zoom_level_y * widget.scaleF < 5 or 0.0 < widget.zoom_level_x * widget.scaleF < 5:
                        widget.updateScale(0, 0, widget.scaleF, widget.scaleF)
        else:
            for widget in widgets:
                if widget.isVisible():
                    widget.scaleF = widget.ZOOM_IN_FACTOR
                    widget.UpdatePaintInfo()
                    widget.updateScale(0, 0, 1, 1)
    def activate3d(self, val):
        """
        Activate 3D show
        :param val:
        :return:
        """
        widgets = [self.openGLWidget_14, self.openGLWidget_24]
        if val:
            self.action3D.setIcon(self._icon_3d)
            for widget in widgets:
                widget.setVisible(True)



        else:
            self.action3D.setIcon(self._icon_3dFaded)



            for widget in widgets:
                widget.setVisible(False)
                #widget.setMaximumSize(QtCore.QSize(width_3d, self.height() - self.height() / 3))

    def activateGuidelines(self, val):
        """
        Activate guidelines
        :param val:
        :return:
        """
        #widgets = select_proper_widgets(self)
        widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6, self.openGLWidget_11, self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3, self.openGLWidget_12]
        self._last_state_guide_lines = val
        if val:
            for widget in widgets:
                widget.enabledGoTo = True
            self.actionGoTo.setIcon(self._icon_goto)
        else:
            for widget in widgets:
                widget.guidelines_v = []
                widget.guidelines_h = []
                widget.enabledGoTo = False
                widget.makeObject()
                widget.update()
            self.actionGoTo.setIcon(self._icon_gotoFaded)

    def Interpolate(self, params):
        """
        Interpolation with fixed sorting and dedicated video buffer.
        """
        [sliceNum, currentWindowName, apply_interp, WI_index] = params

        # 1. Map Window Names to Buffer Indices
        # Use a dictionary to avoid massive if/else blocks
        # Mapped: sagittal->0, axial->1, coronal->2, video->3 (New separate buffer!)
        buffer_map = {
            'sagittal': 0,
            'axial': 1,
            'coronal': 2,
            'video': 3
        }

        # Safety check
        if currentWindowName not in buffer_map:
            print(f"Error: Unknown window {currentWindowName}")
            return

        idx = buffer_map[currentWindowName]

        # Ensure buffer exists (if you haven't initialized 4 lists in __init__)
        while len(self._slice_interp) <= idx:
            self._slice_interp.append([])

        # 2. Update the Buffer (Remove duplicate if exists, then append)
        # We work on a reference to the list
        current_buffer = self._slice_interp[idx]

        # Check if sliceNum already exists, remove it if so
        # (Using a list comprehension to filter is cleaner)
        current_buffer = [item for item in current_buffer if item[0] != sliceNum]

        # Add new point
        current_buffer.append([sliceNum, WI_index])

        # Save back to main list
        self._slice_interp[idx] = current_buffer

        # 3. Check if we should Run Interpolation
        if not apply_interp:
            return

        # 4. Prepare for Interpolation
        # CRITICAL FIX: Sort by slice number!
        # Otherwise interpolation between [50, 40] fails.
        current_buffer.sort(key=lambda x: x[0])

        slicesWI = current_buffer
        if len(slicesWI) < 2:
            return

        slices = [sl[0] for sl in slicesWI]
        WI = [sl[1] for sl in slicesWI]

        # 5. Resolve Reader & Color (Avoid 'sender' crash)
        # It is safer to pass 'colorInd' in params, but if we must use sender:
        sender = QtCore.QObject.sender(self)

        # Fallback defaults if sender is None (e.g. manual call)
        reader = self.readView1
        colorInd = 1

        if sender:
            if hasattr(sender, 'id'):
                if sender.id in [1, 2, 3, 11]:
                    reader = self.readView1
                elif sender.id in [4, 5, 6, 12]:
                    reader = self.readView2
            if hasattr(sender, 'colorInd'):
                colorInd = sender.colorInd

        # 6. Run Interpolation

        self.setEnabled(False)
        self.app.processEvents()  # Keep UI responsive

        # Call the algo
        whiteInd = slice_intepolation(reader, slices, currentWindowName, colorInd, WI)
        self.setEnabled(True)
        # 7. Update Segmentation (Display)
        # Note: Ensure updateSegmentation handles the VideoLabelProxy if currentWindowName == 'video'
        self.updateSegmentation(whiteInd, currentWindowName, colorInd, sliceNum)

        # 8. Clear Buffer
        self._slice_interp[idx] = []

    def Interpolate2(self, params):
        """
        Interpolation
        :param params:
        :return:
        """
        [sliceNum, currentWidnowName, apply_interp, WI_index] = params

        if currentWidnowName=='sagittal':
            slices = [sl[0] for sl in self._slice_interp[0]]
            if sliceNum in slices:
                index = slices.index(sliceNum)
                self._slice_interp[0].pop(index)
            self._slice_interp[0].append([sliceNum, WI_index])
        elif currentWidnowName == 'axial':
            slices = [sl[0] for sl in self._slice_interp[1]]
            if sliceNum in slices:
                index = slices.index(sliceNum)
                self._slice_interp[1].pop(index)
            self._slice_interp[1].append([sliceNum, WI_index])
        elif currentWidnowName == 'coronal':
            slices = [sl[0] for sl in self._slice_interp[2]]
            if sliceNum in slices:
                index = slices.index(sliceNum)
                self._slice_interp[2].pop(index)
            self._slice_interp[2].append([sliceNum, WI_index])
        elif currentWidnowName=='video':
            slices = [sl[0] for sl in self._slice_interp[2]]
            if sliceNum in slices:
                index = slices.index(sliceNum)
                self._slice_interp[2].pop(index)
            self._slice_interp[2].append([sliceNum, WI_index])
        if not apply_interp:
            return

        sender = QtCore.QObject.sender(self)
        if sender.id in [1,2,3,11]:
            reader = self.readView1
        elif sender.id in [4,5,6,12]:
            reader = self.readView2
        if currentWidnowName == 'sagittal':
            slicesWI = self._slice_interp[0]
            if len(slicesWI) < 2:
                return
            slices = [sl[0] for sl in slicesWI]
            WI = [sl[1] for sl in slicesWI]
            self.app.processEvents()
            whiteInd=  slice_intepolation(reader, slices, currentWidnowName, sender.colorInd, WI)
            self.updateSegmentation(whiteInd, currentWidnowName, sender.colorInd, sender.sliceNum)
            self._slice_interp[0] = []
        elif currentWidnowName == 'axial':
            slicesWI = self._slice_interp[1]
            if len(slicesWI) < 2:
                return
            slices = [sl[0] for sl in slicesWI]
            WI = [sl[1] for sl in slicesWI]
            self.app.processEvents()
            whiteInd= slice_intepolation(reader, slices, currentWidnowName, sender.colorInd, WI)
            self.updateSegmentation(whiteInd, currentWidnowName, sender.colorInd, sender.sliceNum)
            self._slice_interp[1] = []
        elif currentWidnowName == 'coronal':
            slicesWI = self._slice_interp[2]
            if len(slicesWI) < 2:
                return
            slices = [sl[0] for sl in slicesWI]
            WI = [sl[1] for sl in slicesWI]
            if len(slices) < 2:
                return
            self.app.processEvents()
            whiteInd= slice_intepolation(reader, slices, currentWidnowName, sender.colorInd, WI)
            self.updateSegmentation(whiteInd, currentWidnowName, sender.colorInd, sender.sliceNum)
            self._slice_interp[2] = []
        elif currentWidnowName=='video':
            slicesWI = self._slice_interp[2]
            if len(slicesWI) < 2:
                return
            slices = [sl[0] for sl in slicesWI]
            WI = [sl[1] for sl in slicesWI]
            if len(slices) < 2:
                return
            self.app.processEvents()
            whiteInd= slice_intepolation(reader, slices, currentWidnowName, sender.colorInd, WI)
            self.updateSegmentation(whiteInd, currentWidnowName, sender.colorInd, sender.sliceNum)
            self._slice_interp[2] = []


    def updateLP(self, params):
        """
        lp: line points
        :return:
        """
        lp, colorInd, empty, gen_contour = params
        if empty:

            self.linePoints = []
            self._lineinfo= []
            readerName, reader, widgets = locateWidgets(self._lastChangedWidgest[0], self)
            if hasattr(self, '_colorInd'):
                reader.npSeg[reader.npSeg == 1500] = 0

                for widget in widgets:
                    setSliceSeg(widget, reader.npSeg)
                    if widget == self.openGLWidget_11:
                        self.openGLWidget_14.paint(reader.npSeg, reader.npImage, widget.currentWidnowName,
                                                   widget.sliceNum)
                    elif widget == self.openGLWidget_12:
                        self.openGLWidget_24.paint(reader.npSeg, reader.npImage, widget.currentWidnowName,
                                                   widget.sliceNum)
                    widget.makeObject()
                    widget.update()

                #removeLastLines(self._lastlines, reader.npSeg, self._colorInd)
                self._lastlines = []
            return
        elif gen_contour:
            self.GenerateContour()
            return
        self.linePoints.append(lp[0])
        self.linePoints.append(lp[1])
        self._colorInd = colorInd


    def removeTableMeasureItem(self, index):
        """
        remove items from measurement table
        :param index:
        :return:
        """
        def find_current_line_number():
            _num_measure = max(self.num_measure_length, self.num_measure_area)
            for i in range(_num_measure):
                if self.table_widget_measure.item(i,0).text() == '' or self.table_widget_measure.item(i,1).text()== '':
                    self.num_measure_length = i
                    self.num_measure_area = i
                    break

            for i in range(self.table_widget_measure.rowCount()):
                if i >= self.num_measure_length:
                    self.table_widget_measure.removeRow(i)

        try:
            selectedRows = [i.row() for i in self.table_widget_measure.selectedIndexes()]
            selectedColumns = [i.column() for i in self.table_widget_measure.selectedIndexes()]
            for row, column in zip(selectedRows, selectedColumns):
                newitem = QtWidgets.QTableWidgetItem()
                self.table_widget_measure.setItem(row, column, newitem)
            #for item in self.table_widget_measure.selectedItems():
            #    newitem = QtWidgets.QTableWidgetItem()
            #    self.table_widget_measure.setItem(item.row(), item.column(), newitem)
            find_current_line_number()
        except Exception as e:
            print('Measure Table Item')
            pass




    def removeTableItem(self, index):
        """
        remove item from table
        :param index:
        :return:
        """
        if self._points_adapt_view2:
            selectedRows = []

            selectedRows = [i.row() for i in self.table_widget.selectedIndexes()]
            for ij in selectedRows:

                self.updateSegmentation(self._points_adapt_view2[ij], 'MRI', 0, 0)
                self.updateSegmentation(self._points_adapt_view1[ij], 'ECO', 0, 0)
                #self._points_adapt_view2.pop(index)
                #self._points_adapt_view1.pop(index)
                #selectedRow = self.table_widget.currentRow()
                self._num_adapted_points -=1
                index = self.table_widget.selectedIndexes()[0].row()
                self.table_widget.removeRow(index)
                self.table_widget.setRowCount(25)
            self._points_adapt_view2 = [i for n, i in enumerate(self._points_adapt_view2) if n not in selectedRows]
            self._points_adapt_view1 = [i for n, i in enumerate(self._points_adapt_view1) if n not in selectedRows]


        #self.table_widget.setItem(self._num_adapted_points, 0, QtWidgets.QTableWidgetItem(str(whiteInd[0])))

    def _linkView1_View2(self):
        """
        Lkinking MRI and US images
        :return:
        """

        if len(self._points_adapt_view2)>=3:
            self.linked_models = LinkMRI_ECO(self._points_adapt_view2, self._points_adapt_view1)

    def linkBoth(self, value):
        """
        Linking both images
        :param value:
        :return:
        """
        if self.linked_models is not None:
            if value:
                self.linked = True
                self.table_link.setText(self._translate("Main", "Linked"))
            else:
                self.linked = False
                self.table_link.setText(self._translate("Main", "Link"))
        else:
            self.table_link.setChecked(False)

    def load_plugins(self):
        """
        Finds and loads all plugins and builds the 'Plugins' menu.
        """
        print("Loading plugins...")

        plugin_dir = settings.PLUGIN_DIR

        self.plugin_manager = PluginManager(plugin_dir)
        self.plugin_manager.discover_plugins()


        categories = {}
        for plugin in self.plugin_manager.get_plugins():
            if plugin.category not in categories:
                categories[plugin.category] = []
            categories[plugin.category].append(plugin)

        for category_name, plugins_in_category in sorted(categories.items()):
            # Create a sub-menu for the category
            category_menu = self.menuPlugins.addMenu(category_name)
            for plugin in plugins_in_category:
                self.add_plugin_action(category_menu, plugin)

        print(f"Finished loading {len(self.plugin_manager.get_plugins())} plugins.")

    def add_plugin_action(self, menu: QtWidgets.QMenu, plugin):
        """Helper to create a QAction for a plugin."""
        action = QtWidgets.QAction(plugin.name, self)
        action.setToolTip(plugin.description)

        # --- This is the trigger ---
        # Connect the menu action to our "brain" function
        action.triggered.connect(lambda: self.launch_plugin(plugin))
        menu.addAction(action)

    def launch_plugin(self, plugin):
        """
        Called when a plugin's menu item is clicked.
        This "brain" function decides how to launch the plugin.
        """
        print(f"Launching plugin: {plugin.name}")
        try:
            # --- 1. Give the plugin the current data ---
            # (You must implement get_current_image_data)
            data_context = self.get_current_image_data()
            #data_context = plugin.update_data_context(current_data)

            # --- 2. Path A: CUSTOM UI (like MorphSeg) ---
            # Check if the plugin has a get_widget() method
            widget = plugin.get_widget(data_context, parent=self)

            if widget:
                print("Plugin is 'Custom UI'. Showing widget.")
                self.plugin_widgets.append(widget)
                widget.show()
                #if hasattr(widget, 'finished'):
                    #widget.finished.connect(lambda: self.plugin_widgets.remove(widget))
                widget.destroyed.connect(
                    lambda: self.plugin_widgets.remove(widget) if widget in self.plugin_widgets else None)
                # Connect the widget's 'completed' signal
                if hasattr(widget, 'completed'):
                    widget.completed.connect(self.on_plugin_complete)
                return

            # --- 3. Path B: AUTO-GUI (Simple Plugin) ---
            # Plugin has no custom widget, so we build one.
            param_spec = plugin.get_parameters()

            if param_spec is None:
                raise ValueError("Plugin provides no get_widget() or get_parameters()")

            print("Plugin is 'Simple UI'. Generating AutoGuiDialog.")

            # ****** HERE IT IS!  ******
            # This is where the AutoGuiDialog is created and used
            dialog = AutoGuiDialog(param_spec, self)

            # --- 4. Show the dialog and get results ---
            if dialog.exec_():
                # User clicked OK!
                user_parameters = dialog.get_values()

                print(f"Executing plugin with params: {user_parameters}")
                # TODO: This is where you should use a QThread
                # to run the plugin's `execute` method
                # without freezing the main window.

                # (Simple, non-threaded way for now)
                results = plugin.execute(user_parameters)

                # --- 5. Handle the plugin's results ---
                self.on_plugin_complete(results)

        except Exception as e:
            print(f"Error launching plugin {plugin.name}: {e}")
            QtWidgets.QMessageBox.critical(self, "Plugin Error", f"Error launching {plugin.name}:\n{e}")

    # --- DATA & RESULTS HANDLING (You must implement these) ---

    def get_current_image_data(self):
        """
        (Placeholder) You must implement this.
        This should return your main data object (e.g., the Nifti image).
        """
        result = {"view 1":None, "view 2":None}
        if hasattr(self.readView1, 'im'):
            result["view 1"] = self.readView1.im
        if hasattr(self.readView2, 'im'):
            result["view 2"] = self.readView2.im
        return result

    def on_plugin_complete(self, results: dict):
        """
        (Placeholder) This is called when a plugin finishes.
        """
        print(f"Plugin finished. Results: {results.keys()}")
        out_image = results.get('image', None)
        out_affine = results.get('affine', None)
        out_seg = results.get('label', None)
        view = results.get('view', None)
        if view == "view 1":
            reader = self.readView1
            updater = self.updateDispView1
        elif view == 'view 2':
            reader = self.readView2
            updater = self.updateDispView2
        else:
            return
        npSeg = reader.npSeg
        img = reader.im
        if out_seg is not None:
            if img.ndim==3:
                npSeg = out_seg.transpose(2, 1, 0)[::-1, ::-1, ::-1] # adjustment for show
            elif img.ndim == 4:
                npSeg = out_seg.transpose(2, 1, 0, 3)[::-1, ::-1, ::-1, :] # adjustment for show

        if out_image is not None:

            if out_affine is None:
                img = make_image(out_image, reader.im)
            else:
                img = make_image_using_affine(out_image, out_affine, None)
            #reader.im = img
            #reader.npImage = out_image.transpose(2, 1, 0)[::-1, ::-1, ::-1]

        reader.read_pars(im_new=img, seg_new = npSeg, reset_seg=False)

        if reader.npImage is not None:
            updater(reader.npImage, reader.npSeg, initialState=True)
        #self.changedTab()


    def updateSliceNumber(self, val):
        """
        Updating slice number
        :param val:
        :return:
        """
        try:
            sender = QtCore.QObject.sender(self)
            name = 'openGLWidget_'
            nameS = 'horizontalSlider_'
            for i in range(12):
                nameWidget = name + str(i + 1)
                if not hasattr(self, nameWidget):
                    continue
                widget = getattr(self, nameWidget)
                if sender == widget:
                    slider = getattr(self, nameS + str(i + 1))
                    if val <=slider.maximum() and val!= slider.value():
                        slider.setValue(val)
        except Exception as e:
            print(e)


    def updateSegPlanes(self, val, windowName, imtype):
        """
        Go to a segmentation plane according to 3D point location
        :param val:
        :param windowName:
        :return:
        """

        name = 'openGLWidget_'
        nameS = 'horizontalSlider_'
        if windowName is None:
            if imtype == 'eco':
                i = 11
            elif imtype == 'mri':
                i = 12
            else:
                return
            nameWidget = name + str(i)
            widget = getattr(self, nameWidget)
            if widget.isVisible():
                widget.guidelines_h = []
                widget.guidelines_v = []
            widget.makeObject()
            widget.update()
            return

        vals = [int(v) for v in val]

        widgets = select_proper_widgets(self)
        if widgets[0].imType== 'eco':
            shapes = self.readView1.npImage.shape[:3]
        else:
            shapes = self.readView2.npImage.shape[:3]

        # Unpack values and shapes for clarity
        # v0, v1, v2 correspond to indices 0, 1, 2 of your data array
        v0, v1, v2, _ = vals
        s0, s1, s2 = shapes

        if windowName == 'axial':
            # Axial View: Fixed Z (v2). Displaying X (v0) vs Y (v1).
            # Horizontal Line: Varies along X (0 to s0), Fixed Y (v1).
            # Vertical Line: Fixed X (v0), Varies along Y (0 to s1).
            line_h = generate_extrapoint_on_line([0, v1], [s0, v1], v2)
            line_v = generate_extrapoint_on_line([v0, 0], [v0, s1], v2)

            # Determine slice value for the slider
            v = v2
            self.changeToAxial(imtype)

        elif windowName == 'coronal':
            # Coronal View: Fixed Y (v1). Displaying X (v0) vs Z (v2).
            # Horizontal Line: Varies along X (0 to s0), Fixed Z (v2).
            # Vertical Line: Fixed X (v0), Varies along Z (0 to s2).
            line_h = generate_extrapoint_on_line([0, v2], [s0, v2], v1)
            line_v = generate_extrapoint_on_line([v0, 0], [v0, s2], v1)

            # Determine slice value
            v = v1
            self.changeToCoronal(imtype)

        elif windowName == 'sagittal':
            # Sagittal View: Fixed X (v0). Displaying Y (v1) vs Z (v2).
            # Horizontal Line: Varies along Y (0 to s1), Fixed Z (v2).
            # Vertical Line: Fixed Y (v1), Varies along Z (0 to s2).
            line_h = generate_extrapoint_on_line([0, v2], [s1, v2], v0)
            line_v = generate_extrapoint_on_line([v1, 0], [v1, s2], v0)

            # Determine slice value
            v = v0
            self.changeToSagittal(imtype)

        else:
            return




        if imtype=='eco':
            i = 11
        elif imtype== 'mri':
            i = 12
        else:
            return



        nameWidget = name + str(i)
        widget = getattr(self, nameWidget)
        if widget.isVisible():
            windowName = widget.currentWidnowName
            #widget.enabledGoTo= True
            widget.guidelines_h = line_h
            widget.guidelines_v = line_v
            widget.makeObject()
            widget.update()
            slider = getattr(self, nameS + str(i))
            slider.setValue(v)


    def updateAllSlices(self, val, windowName):
        """
        Updating all slices
        :param val:
        :param windowName:
        :return:
        """


        val = [int(v) for v in val]
        line_sagittal_h = []
        line_axial_h = []
        line_coronal_h = []
        line_sagittal_v = []
        line_axial_v = []
        line_coronal_v = []
        widgets = select_proper_widgets(self)
        if widgets[0].imType== 'eco':
            shapes = self.readView1.npImage.shape[:3]
        else:
            shapes = self.readView2.npImage.shape[:3]

        if windowName[0] == 'coronal':
            val = [val[2], val[0], val[1]]
            line_sagittal_h = generate_extrapoint_on_line([0,val[2]], [shapes[1],val[2]], val[1])
            line_sagittal_v = generate_extrapoint_on_line([val[0],0], [val[0],shapes[0]], val[1])

            line_axial_v = generate_extrapoint_on_line([val[1],0], [val[1],shapes[1]], val[2])
            line_axial_h = generate_extrapoint_on_line([0,val[0]], [shapes[2],val[0]], val[2])
        elif windowName[0] == 'sagittal':
            val = [val[0], val[2], val[1]]
            line_coronal_h = generate_extrapoint_on_line([0,val[2]], [shapes[2],val[2]], val[0])
            line_coronal_v = generate_extrapoint_on_line([val[1],0], [val[1],shapes[0]], val[0])

            line_axial_h = generate_extrapoint_on_line([0,val[0]], [shapes[2],val[0]], val[2])
            line_axial_v = generate_extrapoint_on_line([val[1],0], [val[1],shapes[1]], val[2])

        elif windowName[0] == 'axial':
            val = [val[1], val[0], val[2]]
            line_coronal_v = generate_extrapoint_on_line([val[1],0], [val[1],shapes[0]], val[0])
            line_coronal_h = generate_extrapoint_on_line([0,val[2]], [shapes[2],val[2]], val[0])

            line_sagittal_v = generate_extrapoint_on_line([val[0],0], [val[0],shapes[0]], val[1])
            line_sagittal_h = generate_extrapoint_on_line([0,val[2]], [shapes[1],val[2]], val[1])#shapes[2] to shapes[1]
        name = 'openGLWidget_'
        nameS = 'horizontalSlider_'

        if windowName[1]=='eco':
            for i, v in zip(range(3), val):
                nameWidget = name + str(i + 1)
                widget = getattr(self, nameWidget)
                if widget.isVisible():
                    windowName = widget.currentWidnowName
                    if windowName == 'axial':
                        widget.guidelines_h = line_axial_h
                        widget.guidelines_v = line_axial_v
                        widget.makeObject()
                        widget.update()
                    elif windowName == 'sagittal':
                        widget.guidelines_h = line_sagittal_h
                        widget.guidelines_v = line_sagittal_v
                        widget.makeObject()
                        widget.update()
                    elif windowName == 'coronal':
                        widget.guidelines_h = line_coronal_h
                        widget.guidelines_v = line_coronal_v
                        widget.makeObject()
                        widget.update()
                    slider = getattr(self, nameS + str(i + 1))
                    slider.setValue(v)
        elif windowName[1]=='mri':
            for i, v in zip(range(3), val):
                nameWidget = name + str(i + 1+3)
                widget = getattr(self, nameWidget)
                if widget.isVisible():
                    windowName = widget.currentWidnowName
                    if windowName == 'axial':
                        widget.guidelines_h = line_axial_h
                        widget.guidelines_v = line_axial_v
                        widget.makeObject()
                        widget.update()
                    elif windowName == 'sagittal':
                        widget.guidelines_h = line_sagittal_h
                        widget.guidelines_v = line_sagittal_v
                        widget.makeObject()
                        widget.update()
                    elif windowName == 'coronal':
                        widget.guidelines_h = line_coronal_h
                        widget.guidelines_v = line_coronal_v
                        widget.makeObject()
                        widget.update()
                    slider = getattr(self, nameS + str(i + 1 + 3))
                    slider.setValue(v)


    def _setFadedPix(self, val):
        """
        Fading icons
        :param val:
        :return:
        """
        self.actionContour.setIcon(self._icon_contourFaded)
        self.actionPoints.setIcon(self._icon_pointsFaded)
        self.actionCircles.setIcon(self._icon_CircleFaded)
        #self.actionGoTo.setIcon(self._icon_gotoFaded)
        self.actionContourX.setIcon(self._icon_contourXFaded)
        self.actionEraseX.setIcon(self._icon_eraseXFaded)
        self.actionArrow.setIcon(self._icon_arrowFaded)
        self.actionPaint.setIcon(self._icon_pencilFaded)
        self.actionPan.setIcon(self._icon_Hand_IXFaded)
        self.actionErase.setIcon(self._icon_EraserFaded)
        self.actionRuler.setIcon(self._icon_rulerFaded)
        #self.dock_widget_table.setVisible(False)
        #self.dock_widget_measure.setVisible(False)
        #self.main_toolbox.setCurrentIndex(0)

        if self._Xtimes == 1:

            if val == 0:
                self.actionArrow.setIcon(self._icon_arrow)
            elif val == 1:
                self.actionPaint.setIcon(self._icon_pencil)
                #self.main_toolbox.setCurrentIndex(5)
            elif val == 2:
                self.actionPan.setIcon(self._icon_Hand_IX)
            elif val == 3:

                self.actionErase.setIcon(self._icon_Eraser)
            elif val == 4:
                self.actionContour.setIcon(self._icon_contour)
            elif val == 5:
                self.actionPoints.setIcon(self._icon_pointsFaded)
                #self.dock_widget_table.setVisible(True)
                #self.changedTab()
            elif val == 6:
                self.actionRuler.setIcon(self._icon_ruler)
                #self.dock_widget_measure.setVisible(True)
            elif val == 7:
                self.actionGoTo.setIcon(self._icon_goto)
            elif val==9:
                self.actionCircles.setIcon(self._icon_CircleFaded)
                #self.main_toolbox.setCurrentIndex(5)


        else:
            if val == 4:
                self.actionContourX.setIcon(self._icon_contourX)
            elif val == 3:
                self.actionEraseX.setIcon(self._icon_eraseX)

    def setView(self, selected_view):
        selected_view = selected_view.lower()

        if not hasattr(self, 'current_view'):
            self.current_view = 'horizontal'  # initialize on first call

        if selected_view == self.current_view:
            # View is already set, do nothing
            return

        # Update the view
        if selected_view == 'horizontal':
            self.create_horizontal_mutualview()
        elif selected_view == 'vertical':
            self.create_vertical_mutualview()

        self.current_view = selected_view  # store the current view

    def setCursorsX(self, val):
        """
        Selecting correct cursor with repetition
        :param val:
        :return:
        """
        guide_lines = self.actionGoTo.isChecked()
        if val == 3:
            guide_lines = False
        #if val == 5:
        #    self._num_adapted_points = 0
        #else:
            #self._points_adapt_view1 = []
            #self._points_adapt_view2 = []
            #self.table_widget.clear()

        #icon = QtGui.QIcon()
        #icon.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/contourX.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        if abs(self._Xtimes) >1:
            self._setFadedPix(val)

            for k in range(12):
                name = 'openGLWidget_' + str(k + 1)
                if not hasattr(self, name):
                    continue
                widget = getattr(self, name)

                if widget.isVisible():
                    widget.enabledGoTo = guide_lines
                    setCursorWidget(widget, val, abs(self._Xtimes))


    def setCursors(self, val, rad_circle=None):

        """
        Selecting correct cursor
        :param val:
        :param rad_circle:
        :return:
        """
        manual_set = False
        self._Xtimes = 1
        if rad_circle is None and val==9:#circle
            #self.scrol_rad_circle.setValue(200)
            #rad_circle = self._rad_circle_dot
            manual_set = True
        self._setFadedPix(val)
        if val == 7:
            self.action_guideLines.setChecked(True)
        guide_lines = self.actionGoTo.isChecked()
        if val == 3:
            guide_lines = False

        #if val == 5:
        #    self._num_adapted_points = 0
        #else:
         #   self._points_adapt_view1 = []
         #   self._points_adapt_view2 = []
         #   self.table_widget.clear()
        #if val == 9:
            #try:
            #    self.actionCircles.triggered.connect(partial(self.setCursors, val, self._rad_circle))
            #except:
            #    pass
        for k in range(12):
            name = 'openGLWidget_' + str(k + 1)
            if not hasattr(self, name):
                continue
            widget = getattr(self, name)
            if widget.isVisible():
                #if val==9:
                    #widget._radius_circle = rad_circle
                widget.enabledGoTo = guide_lines
                if val == 1:
                    target_val = int(0.2 * self.scrol_tol_rad_circle.maximum())
                    if self.scrol_tol_rad_circle.value() == target_val:
                        # Force the logic to run because the slider won't do it
                        self.changeSizePen(target_val)
                    else:
                        # This will change the value and emit the signal automatically
                        self.scrol_tol_rad_circle.setValue(target_val)
                elif val==4:
                    self.scrol_tol_rad_circle.setValue(0)
                #    self.scrol_rad_circle.setValue(8)
                if manual_set:
                    widget._radius_circle = self._rad_circle*abs(widget.to_real_world( 1, 0)[0] - widget.to_real_world(0, 0)[0])
                    #rad_circle = self._rad_circle
                    self.changeRadiusCircle(None)
                    setCursorWidget(widget, val, abs(self._Xtimes), self._rad_circle)
                else:
                    setCursorWidget(widget, val, abs(self._Xtimes), self._rad_circle)


    def update3Dview(self, map_type, reset,  typew='eco'):
        if typew == 'eco':
            if not hasattr(self, 'readView1') or not hasattr(self.readView1, 'npImage'):
                return  # Return if npImage or npSeg does not exist for ECO
        else:
            if not hasattr(self, 'readView2') or not hasattr(self.readView2, 'npImage'):
                return  # Return if npImage or npSeg does not exist for MRI


        if reset is None:
            if typew=='eco':
                self.openGLWidget_14.paint(self.readView1.npSeg, self.readView1.npImage, None)
            else:
                self.openGLWidget_24.paint(self.readView2.npSeg, self.readView2.npImage, None)
        else:
            if typew == 'eco':
                self.openGLWidget_14.cmap_image(self.readView1.npImage, map_type, reset)
            else:
                self.openGLWidget_24.cmap_image(self.readView2.npImage, map_type, reset)




    def updateSegmentation(self, whiteInd, currentWidnowName, colorInd, sliceNum):
        """
        Updating segmentation
        :param whiteInd:
        :param currentWidnowName:
        :param colorInd:
        :param sliceNum:
        :return:
        """


        #whiteInd, edges = whiteIndEdges
        def updateWidget(widget, reader, whiteInd, colorInd):
            try:
                #reader.npSeg[tu
                # ple(zip(*whiteInd))] = colorInd
                update_last(self, reader.npSeg, colorInd, whiteInd, widget.colorInd)
                setSliceSeg(widget, reader.npSeg)
                if widget == self.openGLWidget_11:
                    self.openGLWidget_14.paint(reader.npSeg, reader.npImage, currentWidnowName, sliceNum)
                    widget.makeObject()
                    widget.update()
                elif widget == self.openGLWidget_12:
                    self.openGLWidget_24.paint(reader.npSeg, reader.npImage, currentWidnowName, sliceNum)
                    widget.makeObject()
                    widget.update()
                else:
                    widget.makeObject()
                    widget.update()
            except Exception as e:
                print('Update Widget')
                print(e)

        self._sender = QtCore.QObject.sender(self)
        sender = self._sender
        if type(colorInd) == bool:
            linking = False
            if linking:
                name = 'openGLWidget_'
                widgets = select_proper_widgets(self)
                try:
                    sender_ind =  widgets.index(sender)
                    if sender_ind<3:
                        self.table_widget.setItem(self._num_adapted_points, 0, QtWidgets.QTableWidgetItem(str(whiteInd[0])))
                        self._points_adapt_view1.append(whiteInd[0])
                    else:
                        self.table_widget.setItem(self._num_adapted_points, 1, QtWidgets.QTableWidgetItem(str(whiteInd[0])))
                        self._points_adapt_view2.append(whiteInd[0])
                    reader = self.readView1 if sender_ind < 3 else self.readView2
                    widget_name = name + str(sender_ind+1)
                    updateWidget(getattr(self, widget_name), reader, whiteInd, colorInd)
                    if self._firstSelection is None:
                        self._firstSelection = True
                        for widget in widgets:
                            widget.setDisabled(True)
                        new_ind = sender_ind+1+3 if sender_ind < 3 else sender_ind+1-3
                        next_widget = getattr(self, name + str(new_ind))
                        next_widget.setDisabled(False)
                    elif self._firstSelection == True:
                        for widget in widgets:
                            widget.setDisabled(False)
                        self._firstSelection = None
                        self._num_adapted_points += 1
                except Exception as e:
                    print('Linking Error')
                    print(e)

        else:
            if self.linked or currentWidnowName=='MRI' or currentWidnowName=='ECO':


                widgets_view2 = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6, self.openGLWidget_12]
                widgets_view1 = [self.openGLWidget_11, self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3]
                if self.tabWidget.currentIndex() == 0:
                    widgets_view2 = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6]
                    widgets_view1 = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3]
                elif self.tabWidget.currentIndex() == 1:
                    widgets_view2 = []
                    widgets_view1 = [self.openGLWidget_11]
                elif self.tabWidget.currentIndex() == 2:
                    widgets_view2 = [self.openGLWidget_12]
                    widgets_view1 = []
                if colorInd == 0:
                    if currentWidnowName =='MRI':
                        reader = self.readView2

                        for widget in widgets_view2:
                            updateWidget(widget, reader, whiteInd.reshape(-1,3), colorInd)
                    elif currentWidnowName == 'ECO':
                        reader = self.readView1
                        for widget in widgets_view1:
                            updateWidget(widget, reader, whiteInd.reshape(-1,3), colorInd)

                else:
                    if sender in widgets_view2:
                        x=self.linked_models[0].predict(whiteInd)
                        y=self.linked_models[1].predict(whiteInd)
                        z=self.linked_models[2].predict(whiteInd)
                        whiteInd_view1 = destacked(x,y,z).astype('int')
                        whiteInds = [whiteInd, whiteInd_view1]
                    elif sender in widgets_view1:
                        x=self.linked_models[3].predict(whiteInd)
                        y=self.linked_models[4].predict(whiteInd)
                        z=self.linked_models[5].predict(whiteInd)
                        whiteInd_view2 = destacked(x,y,z).astype('int')
                        whiteInds = [whiteInd_view2, whiteInd]

                    else:
                        return

                    if whiteInd is not None:
                        for reader, widgets, readerName, whiteIn in zip([self.readView2, self.readView1], [widgets_view2, widgets_view1],
                                                   ['readView2', 'readView1'], whiteInds):
                            self._lastChangedWidgest = widgets
                            self._lastReader = readerName
                            #self._lastReaderSegInd, self._lastReaderSegCol = getNoneZeroSeg(reader.npSeg)
                            print(a) # neverused
                            self._lastReaderSegInd.append(whiteIn)
                            self._lastReaderSegCol.append(colorInd)
                            self._lastReaderSegPrevCol.append(widgets[0].colorInd)
                            # self._lastReaderSeg = reader.npSeg.copy()

                            try:
                                assert (whiteIn.shape[1] == 3)
                                update_last(self, reader.npSeg, colorInd, whiteInd, widgets[0].colorInd)
                                #reader.npSeg[tuple(zip(*whiteIn))] = colorInd

                                for widget in widgets:
                                    setSliceSeg(widget, reader.npSeg)

                                    if widget == self.openGLWidget_11:
                                        self.openGLWidget_14.paint(reader.npSeg, reader.npEdge, currentWidnowName, sliceNum)
                                    elif widget == self.openGLWidget_12:
                                        self.openGLWidget_24.paint(reader.npSeg, reader.npEdge, currentWidnowName, sliceNum)

                                    widget.makeObject()
                                    widget.update()
                            except Exception as e:
                                print(e)
                                print('impossible')


            else:
                readerName = ''
                if sender == self.openGLWidget_4 or sender == self.openGLWidget_5 or sender == self.openGLWidget_6 or sender == self.openGLWidget_12:
                    # mri Image
                    readerName = 'readView2'
                    reader = self.readView2
                    widgets = []
                    if self.tabWidget.currentIndex() == 0:
                        widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6]
                    elif self.tabWidget.currentIndex()==2:
                        widgets = [self.openGLWidget_12]
                elif sender == self.openGLWidget_1 or sender == self.openGLWidget_2 or sender == self.openGLWidget_3 or sender == self.openGLWidget_11:
                    # eco
                    readerName = 'readView1'
                    reader = self.readView1
                    widgets = []
                    if self.tabWidget.currentIndex() == 0:
                        widgets = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3]
                    elif self.tabWidget.currentIndex() == 1:
                        widgets = [self.openGLWidget_11]

                elif sender in [self.actionTVSag, self.actionTVCor, self.actionTVAx]:

                    if currentWidnowName.split('_')[0]=='ECO':
                        readerName = 'readView1'
                        reader = self.readView1
                        self._sender = self.openGLWidget_1
                        if self.tabWidget.currentIndex() == 0:
                            widgets = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3]
                        elif self.tabWidget.currentIndex() == 1:
                            widgets = [self.openGLWidget_11]
                        elif self.tabWidget.currentIndex() == 2:
                            widgets = [self.openGLWidget_12]
                    elif currentWidnowName.split('_')[0]=='MRI':
                        readerName = 'readView2'
                        reader = self.readView2
                        self._sender = self.openGLWidget_5
                        widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6]
                    else:
                        return
                else:
                    return



                if whiteInd is not None:
                    if colorInd==9876:
                        return
                    self._lastChangedWidgest = widgets
                    self._lastWindowName = currentWidnowName
                    self._lastReader = readerName
                    #self._lastReaderSegInd, self._lastReaderSegCol = getNoneZeroSeg(reader.npSeg)

                    #self._lastReaderSeg = reader.npSeg.copy()


                    try:
                        if (whiteInd.shape[1] != 3):
                            return
                        if (whiteInd.shape[0] <= 0):
                            return
                        #self.dock_progressbar.setVisible(True)
                        #self.setEnabled(False)
                        is_video = False
                        if sender.imType=='mri':
                            shp = self.readView2.npImage.shape[:3]
                            self.readView2._npSeg = None
                            is_video = self.is_view2_video
                        elif sender.imType=='eco':
                            shp = self.readView1.npImage.shape[:3]
                            self.readView1._npSeg = None
                            is_video = self.is_view1_video
                        whiteInd = repetition(shp, whiteInd, self._Xtimes, currentWidnowName)
                        #edges = repetition(edges, self._Xtimes, currentWidnowName)
                        self.progressBarSaving.setValue(40)
                        if is_video:
                            update_last_video(self, reader, colorInd, whiteInd, widgets[0].colorInd, guide_lines=colorInd==1500)
                        else:
                            update_last(self, reader.npSeg, colorInd, whiteInd, widgets[0].colorInd, guide_lines=colorInd==1500)


                        #update_edges(reader, edges)
                        self.progressBarSaving.setValue(60)
                        """
                        
                        if colorInd != 0:
                            WI = getZeroSeg(reader.npSeg, whiteInd)
                            self._lastReaderSegCol.append(colorInd)
                            self._lastReaderSegInd.append(WI)
                        else:
                            self._lastReaderSegCol.append(colorInd)
                            WI = getNoneZeroSeg(reader.npSeg, whiteInd)
                            self._lastReaderSegInd.append(WI)
                        self._lastReaderSegPrevCol.append(widgets[0].colorInd)
                        self._undoTimes = 0
                        reader.npSeg[tuple(zip(*whiteInd))] = colorInd
                        if len(self._lastReaderSegInd)> self._lastMax:
                            self._lastReaderSegCol = self._lastReaderSegCol[1:]
                            self._lastReaderSegInd = self._lastReaderSegInd[1:]
                            self._lastReaderSegPrevCol = self._lastReaderSegPrevCol[1:]
                        """
                        first_entry= True
                        for widget in widgets:
                            setSliceSeg(widget, reader.npSeg)

                            if not is_video:
                                if hasattr(self, 'readView1'):
                                    if reader==self.readView1 and first_entry:
                                        self.openGLWidget_14.paint(reader.npSeg, reader.npImage, currentWidnowName, sliceNum)
                                        first_entry = False
                                if hasattr(self, 'readView2'):
                                    if reader==self.readView2 and first_entry:
                                        self.openGLWidget_24.paint(reader.npSeg, reader.npImage, currentWidnowName, sliceNum)
                                        first_entry = False
                            #else:
                            #    reader.commit_frame_segmentation_changes(reader.npSeg)
                            widget.makeObject()
                            widget.update()
                        self.progressBarSaving.setValue(100)
                        #self.setEnabled(True)
                        self.dock_progressbar.setVisible(False)
                        self.progressBarSaving.setValue(0)
                    except Exception as e:
                        self.setEnabled(True)
                        self.dock_progressbar.setVisible(False)
                        self.progressBarSaving.setValue(0)
                        print(e)
                        print('impossible')

    def update_table_measure(self,values, columnind):
        """
        Updating table measurements
        :param values:
        :param columnind:
        :return:
        """
        cont = self.table_widget_measure.rowCount()
        rw = cont
        for r in range(cont):
            itm = self.table_widget_measure.item(r, 1)
            if itm is None:
                rw = r
                break
            elif itm.text()=='':
                rw = r
                break
        if rw==cont:
            self.table_widget_measure.insertRow(rw)
        clm = self.table_widget_measure.columnCount()
        if values[1]=='0':
            values[-1] = self.filenameView1
        if values[1]=='1':
            values[-1] = self.filenameView2
        for c in range(clm):
            self.table_widget_measure.setItem(rw, c, QtWidgets.QTableWidgetItem(values[c]))



    def GenerateContour(self):
        """
        Generating contours
        :return:
        """

        if len(self.linePoints)<2:
            return
        if not hasattr(self, '_lastReader'):
            return
        sender = QtCore.QObject.sender(self)
        try:
            reader = getattr(self, self._lastReader)

            windowname = self._lastWindowName
            sliceNum = [w.sliceNum for w in self._lastChangedWidgest if w.currentWidnowName==windowname][0]
            use_additional = True
            if use_additional:
                seg = [w.segSlice for w in self._lastChangedWidgest if w.currentWidnowName == windowname][0]

                total_points, success, len_ls = SearchForAdditionalPoints(seg, sliceNum, windowname,max_lines=0, line_info=self._lineinfo,
                                                                          active_color_ind = self._colorInd)
                self._lineinfo = []
                if (len_ls['h']+len_ls['v'])>0:
                    selected_points = list(total_points) + self.linePoints
                else:
                    selected_points = self.linePoints
            else:
                selected_points = self.linePoints

            whiteInd, remps = convexhull_spline(selected_points, windowname, sliceNum, reader.npSeg)

            self.linePoints = remps
            readerName, reader, widgets = locateWidgets(self._lastChangedWidgest[0], self)
            if hasattr(self, '_colorInd'):
                update_last(self, reader.npSeg, self._colorInd, whiteInd, widgets[0].colorInd)
                reader.npSeg[reader.npSeg == 1500] = widgets[0].colorInd
                #removeLastLines(self._lastlines, reader.npSeg, self._colorInd)
                self._lastlines = []
                #reader.npSeg[tuple(zip(*whiteInd))] = self._colorInd



            for widget in widgets:
                setSliceSeg(widget, reader.npSeg)
                if widget == self.openGLWidget_11:
                    self.openGLWidget_14.paint(reader.npSeg, reader.npImage, widget.currentWidnowName, widget.sliceNum)
                elif widget == self.openGLWidget_12:
                    self.openGLWidget_24.paint(reader.npSeg, reader.npImage, widget.currentWidnowName, widget.sliceNum)
                if hasattr(widget, 'linePoints'):
                    widget.linePoints = []
                    widget.startLinePoints = []
                widget.makeObject()
                widget.update()


        except Exception as e:
            pass
    def Undo(self):
        if not hasattr(self, '_lastReader'):
            return
        num_els = len(self._lastReaderSegInd)
        if (num_els-self._undoTimes)<=0 :
            return
        reader = getattr(self, self._lastReader)
        #reader.npSeg *= 0
        curr_list = num_els-self._undoTimes-1
        _lastReaderSegInd = self._lastReaderSegInd[curr_list]
        _lastReaderSegInd, inds, us, slice = _lastReaderSegInd
        sliceNum = [slice]*len(us)
        _lastReaderSegCol = self._lastReaderSegCol[curr_list]
        _lastReaderSegPrevCol = self._lastReaderSegPrevCol[curr_list]
        if _lastReaderSegPrevCol == _lastReaderSegCol or not any([_lastReaderSegCol==0, _lastReaderSegPrevCol==0]):
            _lastReaderSegCol = 0
        else:
            _lastReaderSegCol = _lastReaderSegPrevCol
        if _lastReaderSegInd.shape[0]!=0:
            if not reader.isChunkedVideo:
                if inds is not None:
                    for ind, u in zip(inds, us, sliceNum):
                        reader.npSeg[tuple(zip(*_lastReaderSegInd[ind]))] = u
                else:
                    reader.npSeg[tuple(zip(*_lastReaderSegInd))] =  _lastReaderSegCol
            else:
                current_npseg = reader.seg_ims.get_frame(slice)
                if inds is not None:

                    for ind, u in zip(inds, us):
                        current_npseg[tuple(zip(*_lastReaderSegInd[ind]))] = u
                    reader.commit_frame_segmentation_changes(current_npseg, slice)
                else:
                    current_npseg[tuple(zip(*_lastReaderSegInd))] =  _lastReaderSegCol
                    reader.commit_frame_segmentation_changes(current_npseg, slice)
        self._undoTimes += 1
        for widget in self._lastChangedWidgest:
            if widget.isVisible():
                setSliceSeg(widget, reader.npSeg)
                widget.update()
                if widget == self.openGLWidget_11:
                    if self.openGLWidget_14.isVisible():
                        self.openGLWidget_14.paint(reader.npSeg,reader.npImage, widget.currentWidnowName, widget.sliceNum)
                elif widget == self.openGLWidget_12:
                    if self.openGLWidget_24.isVisible():
                        self.openGLWidget_24.paint(reader.npSeg,reader.npImage, widget.currentWidnowName, widget.sliceNum)
                if hasattr(widget, 'makeObject'):
                    widget.makeObject()
                widget.update()


    def Redo(self):
        """
        Redo the segmentation
        :return:
        """
        if not hasattr(self, '_lastReader'):
            return
        num_els = len(self._lastReaderSegInd)

        #if (num_els-self._undoTimes)<=0:
        #    return
        reader = getattr(self, self._lastReader)
        #reader.npSeg *= 0
        curr_list = num_els - self._undoTimes
        if curr_list>=num_els:
            return
        _lastReaderSegInd = self._lastReaderSegInd[curr_list]
        _lastReaderSegInd, inds, us, slice = _lastReaderSegInd

        _lastReaderSegCol = self._lastReaderSegCol[curr_list]
        _lastReaderSegPrevCol = self._lastReaderSegPrevCol[curr_list]
        if _lastReaderSegPrevCol == _lastReaderSegCol:
            _lastReaderSegCol = _lastReaderSegPrevCol
        else:
            _lastReaderSegCol = 0


        if _lastReaderSegInd.shape[0]!=0:
            if not reader.isChunkedVideo:
                reader.npSeg[tuple(zip(*_lastReaderSegInd))] =  _lastReaderSegCol
            else:
                current_npseg = reader.seg_ims.get_frame(slice)
                current_npseg[tuple(zip(*_lastReaderSegInd))] =  _lastReaderSegCol
                reader.commit_frame_segmentation_changes(current_npseg, slice)
        self._undoTimes -= 1
        for widget in self._lastChangedWidgest:
            if widget.isVisible():
                setSliceSeg(widget, reader.npSeg)
                widget.update()
                if widget == self.openGLWidget_11:
                    if self.openGLWidget_14.isVisible():
                        self.openGLWidget_14.paint(reader.npSeg,reader.npImage, widget.currentWidnowName, widget.sliceNum)
                elif widget == self.openGLWidget_12:
                    if self.openGLWidget_24.isVisible():
                        self.openGLWidget_24.paint(reader.npSeg,reader.npImage, widget.currentWidnowName, widget.sliceNum)
                if hasattr(widget, 'makeObject'):
                    widget.makeObject()
                widget.update()

    def showSegOnWindow(self, value):
        """
        Enable segmentation
        :param value:
        :return:
        """
        sender = QtCore.QObject.sender(self)
        if sender == self.radioButton_21:
            ## view2
            self.openGLWidget_12.showSeg = value
            self.openGLWidget_12.makeObject()
            self.openGLWidget_12.update()
        elif sender == self.radioButton_4:
            ## view1
            self.openGLWidget_11.showSeg = value
            self.openGLWidget_11.makeObject()
            self.openGLWidget_11.update()




    def changeRotAx(self, val):
        """
        rotating image
        :param val:
        :return:
        """

        sender = QtCore.QObject.sender(self)
        if sender == self.page1_rot_cor:
            if not hasattr(self.readView1, 'npImage'):
                return
            #if self.readView1.npSeg.max()>0 and val:
                #self.page1_rot_cor.setChecked(False)
                #return
            if val.lower()=='coronal':
                self.t1_5.setValue(self._rotationAngleView1_coronal)
            elif val.lower()== 'axial':
                self.t1_5.setValue(-self._rotationAngleView1_axial)
            elif val.lower() == 'sagittal':
                self.t1_5.setValue(-self._rotationAngleView1_sagittal)
        elif sender == self.page2_rot_cor:
            if not hasattr(self.readView2, 'npImage'):
                return
            if self.readView2.npSeg.max()>0 and val:
                self.page2_rot_cor.setChecked(False)
                return
            if val.lower() == 'coronal':
                self.t2_5.setValue(self._rotationAngleView2_coronal)
            elif val.lower()== 'axial':
                self.t2_5.setValue(-self._rotationAngleView2_axial)
            elif val.lower()== 'sagittal':
                self.t2_5.setValue(-self._rotationAngleView2_sagittal)



    def ColorIntensityChange(self, thrsh = 0, dtype='image'):
        """
        Changing color intensity
        :param thrsh:
        :dtype image or seg
        :return:
        """
        if thrsh<0:
            return
        not_exist1, not_exist2 = False, False
        if hasattr(self, 'readView2'):
            if not hasattr(self.readView2, 'npSeg'):
                not_exist1 = True
        if hasattr(self, 'readView1'):
            if not hasattr(self.readView1, 'npSeg'):
                not_exist2 = True
        if not_exist1 and not_exist2:
            return
        widgets_view1 = [0, 1, 2, 10, 13]
        widgets_view2 = [3, 4, 5, 11, 23]
        if self.is_view1_video:
            widgets_view1 = [10]
        if self.is_view2_video:
            widgets_view2 = [11]
        widgets_num = widgets_view1 + widgets_view2



        if dtype=='image':
            for k in widgets_num:
                name = 'openGLWidget_' + str(k + 1)
                widget = getattr(self, name)

                if widget.isVisible():#k  in [13,23]:
                    if k==13:
                        widget.intensityImg = (thrsh / 100.0) ** 2
                        self.update3Dview(None, None, typew='eco')
                    elif k==23:
                        widget.intensityImg = (thrsh/ 100.0) ** 2
                        self.update3Dview(None, None, typew='mri')
                    #widget.intensityImg = thrsh/100
                widget.update()
        elif dtype=='seg':

            for k in widgets_num:
                name = 'openGLWidget_' + str(k + 1)
                widget = getattr(self, name)
                widget.intensitySeg = thrsh/100
                if widget.isVisible() and k not in [13,23]:
                    widget.makeObject()
                    widget.update()
                elif isinstance(widget, glScientific):
                    widget.GLV.intensitySeg = thrsh/100
                    widget.GLSC.intensitySeg = thrsh / 100
                    if widget.isVisible():
                        widget.update()


    def trackDistance(self, thrsh=50):
        if not hasattr(self, 'readView2'):
            self.dw5_s1.setValue(0)
            self.dw5lb1.setText('0')
            return
        if not hasattr(self.readView2, 'npImage'):
            self.dw5_s1.setValue(0)
            self.dw5lb1.setText('0')
            return

        if not hasattr(self.readView2, 'tract'):
            self.dw5_s1.setValue(0)
            self.dw5lb1.setText('0')
            return
        oldMin, oldMax = 0, 100

        shp = self.readView2.npImage.shape[:3]
        NewMin, NewMax = 1, min(min(shp[0], shp[1]), shp[2])//6
        OldRange = (oldMax - oldMin)
        NewRange = (NewMax - NewMin)
        thrsh = (((thrsh - oldMin) * NewRange) / OldRange) + NewMin
        self.tol_trk = thrsh
        widgets_num = [3, 4, 5]
        for k in range(12):
            name = 'openGLWidget_' + str(k + 1)
            if not hasattr(self, name):
                continue
            widget = getattr(self, name)
            if k in widgets_num:
                widget.updateInfo(*getCurrentSlice(widget,
                                                   self.readView2.npImage, self.readView2.npSeg, widget.sliceNum, self.readView2.tract, tol_slice=self.tol_trk), widget.sliceNum,
                                  self.readView2.npImage.shape[:3],
                                  initialState=False, imSpacing=self.readView2.ImSpacing)

                widget.makeObject()
                widget.update()
                widget.show()

    def trackThickness(self, thrsh=50):
        """"
        Thickness in tractography images
        """
        if not hasattr(self, 'readView2'):
            self.dw5_s2.setValue(0)
            self.dw5lb2.setText('0')
            return
        if not hasattr(self.readView2, 'npImage'):
            self.dw5_s2.setValue(0)
            self.dw5lb2.setText('0')
            return
        if not hasattr(self.readView2, 'tract'):
            self.dw5_s2.setValue(0)
            self.dw5lb2.setText('0')
            return
        oldMin, oldMax = 0, 100

        NewMin, NewMax = 1, 10#min(min(shp[0], shp[1]), shp[2])//6
        OldRange = (oldMax - oldMin)
        NewRange = (NewMax - NewMin)
        thrsh = (((thrsh - oldMin) * NewRange) / OldRange) + NewMin

        widgets_num = [3, 4, 5]
        for k in range(12):
            name = 'openGLWidget_' + str(k + 1)
            if not hasattr(self, name):
                continue
            widget = getattr(self, name)
            if k in widgets_num:

                widget.width_line_tract =thrsh
                widget.makeObject()
                widget.update()
                widget.show()

    def changeMagicToolTo222l(self, value):
        widgets = find_avail_widgets(self)
        for k in widgets:
            name = 'openGLWidget_' + str(k)
            widget = getattr(self, name)

            if k in [14, 24] or not widget.isVisible():
                continue
            if widget.enabledMagicTool:
                widget._tol_magic_tool = self.scrol_tol_rad_circle.value()

    def changeRadiusCircle(self, value, slider=False):
        """
        Change circle radius
        :param value:
        :param slider:
        :return:
        """
        oldMin, oldMax = 0, 100
        NewMin, NewMax = 50, 80
        #OldRange = (oldMax - oldMin)
        #NewRange = (NewMax - NewMin)
        #value = (((value - oldMin) * NewRange) / OldRange) + NewMin
        if value is not None:
            self._rad_circle = value
        else:
            self._rad_circle = self.scrol_rad_circle.value()
        try:
            self.actionCircles.triggered.disconnect(partial(self.setCursors, 9, self._rad_circle))
        except:
            pass
        widgets = find_avail_widgets(self)
        for k in widgets:
            name = 'openGLWidget_' + str(k)
            widget = getattr(self, name)

            if k in [14, 24] or not widget.isVisible():
                continue
            if widget.enabledMagicTool:
                widget._tol_magic_tool = self.scrol_rad_circle.value()
                continue
            if not widget.enabledCircle:
                continue
            self._rad_circle_dot = self._rad_circle*abs(widget.to_real_world( 1, 0)[0] - widget.to_real_world(0, 0)[0])
            self._tol_cricle_tool =((self.scrol_tol_rad_circle.value()-self.scrol_tol_rad_circle.minimum())/(self.scrol_tol_rad_circle.maximum()-self.scrol_tol_rad_circle.minimum()))*(2-0.5)+0.5
            widget._radius_circle = self._rad_circle*abs(widget.to_real_world( 1, 0)[0] - widget.to_real_world(0, 0)[0])
            setCursorWidget(widget, 9, 1, self._rad_circle)

        self.actionCircles.triggered.connect(partial(self.setCursors, 9, self._rad_circle))

    def changeSizePen(self,value): # change color pen
        """
        change size pen
        :param value: size pen
        :return: set current pen size
        """
        rng = 30/100
        value *= rng
        val_tol = ((self.scrol_tol_rad_circle.value()-self.scrol_tol_rad_circle.minimum())/(self.scrol_tol_rad_circle.maximum()-self.scrol_tol_rad_circle.minimum()))*(2-0.5)+0.5

        widget_nums = [0, 1,2, 3,4,5,10,11]
        for k in widget_nums:
            name = 'openGLWidget_' + str(k + 1)
            widget = getattr(self, name)
            widget.widthPen = value
            widget._tol_cricle_tool = val_tol
            if widget.isVisible():
                widget.update()



    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_row_clicked_image(self, proxy_index):
        """
        Handles all logic in response to a user click by toggling the
        check state of the item in the first column of the clicked row.
        """
        # 1. Get the proxy index for the first column in the clicked row.
        check_proxy_index = proxy_index.sibling(proxy_index.row(), 0)

        # 2. Map this proxy index back to the original source model index.
        source_index = self.tree_images.model().mapToSource(check_proxy_index)

        # 3. Get the actual QStandardItem from the source model.
        item = self.tree_images.model().sourceModel().itemFromIndex(source_index)

        # 4. If the item doesn't exist or isn't checkable, do nothing.
        if not item or not item.isCheckable():
            return

        # 5. Determine the new state and TOGGLE THE CHECKBOX EXACTLY ONCE.
        new_state = QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
        old_state = item.checkState()


        # 6. Manually update the styling for all rows.

        cond = self.changeImage(item)
        try:
            if cond:
                item.setCheckState(new_state)
            else:
                item.setCheckState(old_state)
            self.style_all_rows(self.tree_images)
        except:
            pass



    def changeImage(self, value):
        """
        Change Image
        :param value:
        :return:
        """
        #parent = self.tree_colors.model().sourceModel().invisibleRootItem()

        index_row = value.index().row()
        [info, _, _, indc] = self.imported_images[index_row]#
        index_view = info[0][2]
        try:
            if value.checkState()==Qt.Unchecked:
                if info[1]<3:#image loading
                    #if info[1]<2: #eco loading
                    if index_view==0:
                        cond = self.browse_view1(info, use_dialog=False)
                        if cond:
                            self.imported_images[index_row][2] = True
                            clean_parent_image(self, index_row, [indc, indc+'_Seg'], index_view=index_view)
                        else:
                            return cond
                    else:
                        cond = self.browse_view2(info, use_dialog=False)
                        if cond:
                            self.imported_images[index_row][2] = True
                            clean_parent_image(self, index_row, [indc, indc+'_Seg'], index_view=index_view)
                        else:
                            return cond
                else: #segmentation loading
                    #if info[1]<5: #eco loading
                    if index_view == 0:
                        cond = self.importData(type_image='View1_SEG', fileObj=info[0])
                        if cond:
                            self.imported_images[index_row][2] = True
                            clean_parent_image(self, index_row, [indc], index_view=index_view)
                        else:
                            return cond
                    else:
                        cond = self.importData(type_image='View2_SEG', fileObj=info[0])
                        if cond:
                            self.imported_images[index_row][2] = True
                            clean_parent_image(self, index_row, [indc],index_view=index_view)
                        else:
                            return cond

            else: #close image
                if info[1]<3:#image loading
                    #if info[1]<2: #eco loading
                    if index_view == 0:
                        cond = self.CloseView1(message_box='on')
                        if cond:
                            if '*View 1 (loaded)' in info[0][1]:
                                self.imported_images.pop(index_row)
                                parent = self.tree_images.model().sourceModel().invisibleRootItem()
                                parent.removeRow(index_row)
                            else:
                                self.imported_images[index_row][2] = False
                                clean_parent_image(self, -1, ['Fetal_Seg','US_Seg'], index_view=index_view)
                        else:
                            return cond
                    else:
                        cond = self.CloseView2(message_box='on', dialogue=True)
                        if cond:
                            if '*View 2 (loaded)' in info[0][1]:
                                self.imported_images.pop(index_row)
                                parent = self.tree_images.model().sourceModel().invisibleRootItem()
                                parent.removeRow(index_row)
                            else:
                                self.imported_images[index_row][2] = False
                                clean_parent_image(self, -1, ['MRI_Seg'],index_view)
                        else:
                            return cond
                else: #segmentation loading
                    #if info[1] < 5:  # eco loading
                    if index_view == 0:
                        self.closeImportData(type_image='View1_SEG')
                        self.imported_images[index_row][2] = False
                    else:
                        self.closeImportData(type_image='View2_SEG')
                        self.imported_images[index_row][2] = False
        except Exception as e:
            print(e)
            return False
        return True

    # Add these three methods to your class

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_row_clicked(self, proxy_index):
        """
        Handles all logic in response to a user click by toggling the
        check state of the item in the first column of the clicked row.
        """
        # 1. Get the proxy index for the first column in the clicked row.
        check_proxy_index = proxy_index.sibling(proxy_index.row(), 0)

        # 2. Map this proxy index back to the original source model index.
        source_index = self.tree_colors.model().mapToSource(check_proxy_index)

        # 3. Get the actual QStandardItem from the source model.
        item = self.tree_colors.model().sourceModel().itemFromIndex(source_index)

        # 4. If the item doesn't exist or isn't checkable, do nothing.
        if not item or not item.isCheckable():
            return

        # 5. Determine the new state and TOGGLE THE CHECKBOX EXACTLY ONCE.
        new_state = QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
        item.setCheckState(new_state)

        # 6. Now, use this definitive new state for all follow-up logic.
        if item.text().startswith('9876'):
            # If the master was clicked, update all children
            root = self.tree_colors.model().sourceModel().invisibleRootItem()
            for row in range(root.rowCount()):
                child_item = root.child(row, 0)
                if child_item and child_item.isCheckable() and child_item != item:
                    child_item.setCheckState(new_state)
        else:
            # If a child was clicked, update the master
            self.update_master_checkbox_state(self.tree_colors)

        # 7. Manually update the styling for all rows.
        self.style_all_rows(self.tree_colors)
        self.changeColorPen(item)

    def update_master_checkbox_state(self, tree):
        """Checks children and updates the master checkbox state."""
        model = tree.model().sourceModel()
        root = model.invisibleRootItem()

        master_item = None
        checkable_children = 0
        checked_count = 0

        # Find the master item and count checked children
        for row in range(root.rowCount()):
            item = root.child(row, 0)
            if not item or not item.isCheckable():
                continue

            if item.text().startswith('9876'):
                master_item = item
            else:
                checkable_children += 1
                if item.checkState() == QtCore.Qt.Checked:
                    checked_count += 1

        if master_item:
            # Block signals here to prevent this update from causing a new signal
            model.blockSignals(True)
            if checked_count == checkable_children and checkable_children > 0:
                master_item.setCheckState(QtCore.Qt.Checked)
            else:
                master_item.setCheckState(QtCore.Qt.Unchecked)
            model.blockSignals(False)

    def style_all_rows(self, tree):
        """Loops through all rows to set the background color."""
        model = tree.model().sourceModel()
        root = model.invisibleRootItem()
        for row in range(root.rowCount()):
            item = root.child(row, 0)
            if item:
                if item.checkState() == QtCore.Qt.Checked:
                    brush = QtGui.QBrush(QtGui.QColor(212, 237, 218, 60))
                else:
                    brush = QtGui.QBrush(QtCore.Qt.transparent)

                for col in range(model.columnCount()):
                    item_in_row = model.item(row, col)
                    if item_in_row:
                        item_in_row.setBackground(brush)


    def changeColorPen(self,value): # change color pen
        """
        change color pen
        :param value: color pen
        :return: set current color
        """

        text =''
        if type(value)!=str:
            colrInds = []
            if value.checkState()==Qt.Checked:
                ind = int(float(value.text()))
                text = self.tree_colors.model().sourceModel().item(value.row(), 1).text()
            else:
                ind = None
                text = ''
            #root = self.tree_colors.model().sourceModel().invisibleRootItem()
            model = self.tree_colors.model().sourceModel()

            #if len(colrInds)==0:
                #colrInds = []
            #    ind = 9876
            #    text='Combined'


            root = model.invisibleRootItem()

            for i in range(root.rowCount()):
                signal = root.child(i)
                if signal.checkState()==Qt.Checked:
                    colrInds.append(int(float(signal.text())))





        try:
            """
            
            if value =='':
                ind = 9876#self.dw2Text.index('X_Combined') + 1
            else:
                try:
                    ind = int(self.dw2Text[self.dw2Text.index(value)].split('_')[0])
                except:
                    ind = 9876#self.dw2Text.index(value) + 1
            """
            #sender = QtCore.QObject.sender(self)
            sender_eco = False
            for f in self.widgets_view1:
                a = getattr(self, 'openGLWidget_{}'.format(f))
                if a.isVisible():
                    sender_eco = True
                    break

            #if sender_eco:
            if hasattr(self, 'readView1'):
                if hasattr(self.readView1, 'npSeg'):
                    txt = compute_volume(self.readView1, self.filenameView1, colrInds, in_txt=self.openedFileName.text(),
                                         ind_screen=0)
                    self.openedFileName.setText(txt)

            #else:
            if hasattr(self, 'readView2'):
                if hasattr(self.readView2, 'npSeg'):
                    txt = compute_volume(self.readView2, self.filenameView2, colrInds, in_txt=self.openedFileName.text(),
                                         ind_screen=1)
                    self.openedFileName.setText(txt)
            if ind is not None:
                self._selected_seg_color = ind
                #if 9876 not in colrInds:
                try:
                    colorPen = self.colorsCombinations[ind]
                except:
                    colorPen = [1, 0, 0, 1]
                #else:
                #    colorPen = [1,0,0,1]
                colorPen = [int(float(i*255)) for i in colorPen]
                self.pixmap_box_color.fill(
                    (QtGui.QColor(colorPen[0], colorPen[1], colorPen[2], 255)))
                self._icon_colorX.addPixmap(self.pixmap_box_color, QtGui.QIcon.Normal,
                                            QtGui.QIcon.On)
                self.pixmap_box_label.setText(text)
                self.actionColor.setIcon(self._icon_colorX)
                    #pixmap.fill((QtGui.QColor(colorPen[0] * 255.0, colorPen[1] * 255.0, colorPen[2] * 255.0, 1 * 255.0)))
                    #self._icon_colorX.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR + "/box.png"), QtGui.QIcon.Normal,
                    #                            QtGui.QIcon.On)
                    #self._icon_colorX.addPixmap()

            widgets = find_avail_widgets(self)
            prefix = 'openGLWidget_'
            for k in widgets:
                name = prefix + str(k)
                widget = getattr(self, name)
                if ind is not None:
                    widget.colorInd = ind
                widget.colorInds = colrInds
                if not widget.isVisible():
                    continue
                if k in [14]:
                    widget.paint(self.readView1.npSeg,
                                       self.readView1.npImage, None)
                elif k in [24]:
                    widget.paint(self.readView2.npSeg,
                          self.readView2.npImage, None)
                else:
                    if ind is not None:
                        widget.colorObject = colorPen
                    widget.makeObject()
                    widget.update()


                    #widget.show()
        except Exception as e:
            print('Imposible to change the color {}'.format(e))


    def changeBrightness(self,value): # change contrast brightns
        """
        Change contrast and brightness
        :param value:
        :return:
        """
        sender = QtCore.QObject.sender(self)
        value = (value/100)
        widgets_num = []
        if sender == self.t1_1:
            widgets_num = [0, 1, 2, 10]
        elif sender == self.t2_1:
            widgets_num = [3, 4, 5, 11]
        for k in widgets_num:
            name = 'openGLWidget_' + str(k + 1)
            widget = getattr(self, name)
            widget.brightness = value
            widget.update()

    def changeContrast(self,value): # change contrast brightns
        sender = QtCore.QObject.sender(self)
        value = (value/100)
        widgets_num = []
        if sender == self.t1_2:
            widgets_num = [0, 1, 2, 10]
        elif sender == self.t2_2:
            widgets_num = [3, 4, 5, 11]
        for k in widgets_num:
            name = 'openGLWidget_' + str(k + 1)
            widget = getattr(self, name)
            widget.contrast = value
            widget.update()


    def changeBandPass(self,value): # change threshold
        """
        Image enhancement BandPass
        :param value:
        :return:
        """
        sender = QtCore.QObject.sender(self)
        value =value/100
        if sender == self.t1_3:
            widgets = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3, self.openGLWidget_11]
            for widget in widgets:
                widget.gamma = value
                widget.update()
        elif sender == self.t1_7:
            widgets = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3, self.openGLWidget_11]
            for widget in widgets:
                widget.structure = value
                widget.update()

        elif sender == self.t2_3:
            widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6, self.openGLWidget_12]
            for widget in widgets:
                widget.gamma = value
                widget.update()
        elif sender == self.t2_7:
            widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6, self.openGLWidget_12]
            for widget in widgets:
                widget.structure = value
                widget.update()


    def changeenable_endo_enhance(self,value): # change threshold
        """
        Add enable_endo_enhance filter
        :param value:
        :return:
        """
        sender = QtCore.QObject.sender(self)
        if sender == self.toggle1_1:
            self.openGLWidget_1.enable_endo_enhance = value
            self.openGLWidget_1.makeObject()
            self.openGLWidget_1.update()
            self.openGLWidget_2.enable_endo_enhance = value
            self.openGLWidget_2.makeObject()
            self.openGLWidget_2.update()
            self.openGLWidget_3.enable_endo_enhance = value
            self.openGLWidget_3.makeObject()
            self.openGLWidget_3.update()
            self.openGLWidget_11.enable_endo_enhance = value
            self.openGLWidget_11.makeObject()
            self.openGLWidget_11.update()
        elif sender == self.toggle2_1:
            self.openGLWidget_4.enable_endo_enhance = value
            self.openGLWidget_4.makeObject()
            self.openGLWidget_4.update()
            self.openGLWidget_5.enable_endo_enhance = value
            self.openGLWidget_5.makeObject()
            self.openGLWidget_5.update()
            self.openGLWidget_6.enable_endo_enhance = value
            self.openGLWidget_6.makeObject()
            self.openGLWidget_6.update()
            self.openGLWidget_12.enable_endo_enhance = value
            self.openGLWidget_12.makeObject()
            self.openGLWidget_12.update()



    def changeSobel(self, value):
        """
        Use soble operator
        :param value:
        :return:
        """
        sender = QtCore.QObject.sender(self)
        if sender == self.t1_4:
            widgets = [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3, self.openGLWidget_11]
            for widget in widgets:
                widget.denoise = value/100
                widget.update()
        elif sender == self.t2_4:
            widgets = [self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6, self.openGLWidget_12]
            for widget in widgets:
                widget.denoise = value/100
                widget.update()



    def changeWidthPen(self):
        pass



    def Rotate(self, value):
        """
        Rotating image
        :param value:
        :return:
        """

        sender = QtCore.QObject.sender(self)
        widgets_view1 = [0, 1, 2, 10]
        widgets_view2 = [3, 4, 5, 11]
        if self.is_view1_video:
            widgets_view1 = [10]
        if self.is_view2_video:
            widgets_view2 = [11]
        widgets_num = widgets_view1 + widgets_view2


        for k in widgets_num:
            name = 'openGLWidget_' + str(k + 1)
            if not hasattr(self, name):
                continue
            widget = getattr(self, name)
            widget.zRot = 0
            widget.update()
        if sender == self.t1_5:

            if not hasattr(self, 'readView1') or self.is_view1_video:
                return
            if not hasattr(self.readView1, 'npSeg') :
                return

            uq, lenuq = len_unique(self.readView1.npSeg)
            if lenuq>2:




                #self.t1_5.valueChanged.disconnect()
                self.t1_5.blockSignals(True)
                self.lb_t1_5.blockSignals(True)
                if self.page1_rot_cor.currentText().lower() == 'coronal':
                    self.t1_5.setValue(self._rotationAngleView1_coronal)
                    self.lb_t1_5.setNum(self._rotationAngleView1_coronal)
                elif self.page1_rot_cor.currentText().lower() == 'axial':
                    self.t1_5.setValue(self._rotationAngleView1_axial)
                    self.lb_t1_5.setNum(self._rotationAngleView1_axial)
                elif self.page1_rot_cor.currentText().lower() == 'sagittal':
                    self.t1_5.setValue(self._rotationAngleView1_sagittal)
                    self.lb_t1_5.setNum(self._rotationAngleView1_sagittal)
                self.t1_5.blockSignals(False)
                self.lb_t1_5.blockSignals(False)
                #self.t1_5.valueChanged.connect(self.Rotate)
                return
            if self.page1_rot_cor.currentText().lower() == 'coronal':
                self._rotationAngleView1_coronal = value
            elif self.page1_rot_cor.currentText().lower() == 'axial':
                self._rotationAngleView1_axial = -value
            elif self.page1_rot_cor.currentText().lower() == 'sagittal':
                self._rotationAngleView1_sagittal = -value


            self.dock_progressbar.setVisible(True)
            self.setEnabled(False)
            self.progressBarSaving.setValue(20)

            rr = 0
            im, rot_mat = rotation3d(self.readView1._imChanged, self._rotationAngleView1_axial,
                            self._rotationAngleView1_coronal, self._rotationAngleView1_sagittal)
            if lenuq>1:
                if self.readView1._npSeg is None:
                    self.readView1._npSeg = self.readView1.npSeg

                npSeg = self.readView1._npSeg.copy()

                Segm = make_image(npSeg, self.readView1._imChanged)

                npSeg, _ = rotation3d(Segm, self._rotationAngleView1_axial,
                                self._rotationAngleView1_coronal, self._rotationAngleView1_sagittal)
                #for rot, axs in zip([self._rotationAngleView1_axial,self._rotationAngleView1_sagittal, self._rotationAngleView1_coronal],
                #               [[0,0,1], [1,0,0],[0,1,0]]):
                npSeg[npSeg > 0] = uq.max()

                self.readView1.npSeg = npSeg
            self.readView1.metadata['rot_axial'] = self._rotationAngleView1_axial
            self.readView1.metadata['rot_sagittal'] = self._rotationAngleView1_sagittal
            self.readView1.metadata['rot_coronal'] = self._rotationAngleView1_coronal



            self.progressBarSaving.setValue(80)
            self.readView1.updateData(im, rot_mat, type='eco')
            self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)

            self.progressBarSaving.setValue(100)
            self.setEnabled(True)
            self.dock_progressbar.setVisible(False)
            self.progressBarSaving.setValue(0)
        elif sender == self.t2_5  or self.is_view1_video:
            if not hasattr(self, 'readView2'):
                return
            if not hasattr(self.readView2, 'npSeg') :
                return

            #if not hasattr(self, 'readView2'):
            #    return
            #if not hasattr(self.readView2, 'npSeg') or value == self._rotationAngleView2_axial or value == self._rotationAngleView2_coronal:
            #    return
            if self.readView2.npSeg.max()>0:
                #MessageBox = QtWidgets.QMessageBox(self)
                #MessageBox.setText('You are not allowed to change the image after segmentation')
                #MessageBox.setWindowTitle('Warning')
                #MessageBox.show()

                self.t2_5.blockSignals(True)
                if self.page2_rot_cor.currentText().lower() == 'coronal':
                    self.t2_5.setValue(self._rotationAngleView2_coronal)
                    self.lb_t2_5.setNum(self._rotationAngleView2_coronal)
                elif self.page2_rot_cor.currentText().lower() == 'axial':
                    self.t2_5.setValue(self._rotationAngleView2_axial)
                    self.lb_t2_5.setNum(self._rotationAngleView2_axial)
                elif self.page2_rot_cor.currentText().lower() == 'sagittal':
                    self.t2_5.setValue(self._rotationAngleView2_sagittal)
                    self.lb_t2_5.setNum(self._rotationAngleView2_sagittal)
                self.t2_5.blockSignals(False)
                return
            if self.page2_rot_cor.currentText().lower() == 'coronal':
                self._rotationAngleView2_coronal = value
            if self.page2_rot_cor.currentText().lower() == 'axial':
                self._rotationAngleView2_axial = -value
            if self.page2_rot_cor.currentText().lower() == 'sagittal':
                self._rotationAngleView2_sagittal = -value

            self.dock_progressbar.setVisible(True)
            self.setEnabled(False)
            self.progressBarSaving.setValue(20)
            im, rot_mat = rotation3d(self.readView2._imChanged, self._rotationAngleView2_axial,
                            self._rotationAngleView2_coronal, self._rotationAngleView2_sagittal)

            self.readView2.metadata['rot_axial'] = self._rotationAngleView2_axial
            self.readView2.metadata['rot_sagittal'] = self._rotationAngleView2_sagittal
            self.readView2.metadata['rot_coronal'] = self._rotationAngleView2_coronal


            self.progressBarSaving.setValue(80)

            self.readView2.updateData(im, rot_mat, type='t1')
            self.updateDispView2(self.readView2.npImage, self.readView2.npSeg, initialState=True, tract=self.readView2.tract)

            self.progressBarSaving.setValue(100)
            self.setEnabled(True)
            self.dock_progressbar.setVisible(False)
            self.progressBarSaving.setValue(0)

    def create_cursors(self):
        """
        Create desired cursors
        :return:
        """
        bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/Hand.png")
        self.cursorOpenHand = QtGui.QCursor(bitmap)


        bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/Handsqueezed.png")
        self.cursorClosedHand = QtGui.QCursor(bitmap)


        bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/zoom_in.png")
        self.cursorZoomIn = QtGui.QCursor(bitmap)


        bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/zoom_out.png")
        self.cursorZoomOut = QtGui.QCursor(bitmap)



        bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/rotate.png")
        self.cursorRotate = QtGui.QCursor(bitmap)


        #bitmap = QtGui.QPixmap(settings.RESOURCE_DIR+"/arrow.png")
        self.cursorArrow = QtGui.QCursor(Qt.ArrowCursor)

        self.setCursor(self.cursorArrow)



    def init_state(self):
        """
        Initial state
        :return:
        """
        self.toolBar.setVisible(True)
        self.actionMain_Toolbar.setChecked(True)
        self.actionrotate.setChecked(False)




    def save(self):
        """
        saving current state
        :return:
        """
        if self._basefileSave != '':
            self.dock_progressbar.setVisible(True)
            self.setEnabled(False)
            self.progressBarSaving.setValue(0)

            self.saveChanges()

            self.progressBarSaving.setValue(100)
            self.setEnabled(True)
            self.dock_progressbar.setVisible(False)
            self.progressBarSaving.setValue(0)


    def saveas(self):
        """
        Save as
        :return:
        """
        filters = "BrainNeonatal (*.bn)"
        opts =QtWidgets.QFileDialog.DontUseNativeDialog
        fileObj = QtWidgets.QFileDialog.getSaveFileName( self, "Open File", settings.DEFAULT_USE_DIR, filters, options=opts)
        if fileObj[0] == '':
            return
        self._basefileSave, _ = os.path.splitext(fileObj[0])
        self.dock_progressbar.setVisible(True)
        self.setEnabled(False)

        self.progressBarSaving.setValue(0)

        self.saveChanges()

        self.setEnabled(True)
        self.dock_progressbar.setVisible(False)
        self.progressBarSaving.setValue(0)



    def convert(self):
        """
        Data Conversion
        :return:
        """


        filters = "DICOM (*.dcm);;Nifti (*.nia *.nii *.nii.gz *.hdr *.img *.img.gz *.mgz);;NRRD (*.nrrd *.nhdr)"
        opts =QtWidgets.QFileDialog.DontUseNativeDialog
        fileObj = QtWidgets.QFileDialog.getSaveFileName( self, "Open File", settings.DEFAULT_USE_DIR, filters, options=opts)

        if  fileObj[1] != '' and hasattr(self, 'readView1'):
            outfile_format = filters.split(';;').index(fileObj[1])
            if self.readView1.success:
                file_path, file_extension = os.path.splitext(fileObj[0])
                self.setCursor(QtCore.Qt.WaitCursor)
                if outfile_format == 0: # DICOM
                    save_as_dicom(self.npImage, self.readView1.metadata_dict, file_path)
                elif outfile_format == 1: # NIFTI
                    save_as_nifti(self.npImage, self.readView1.metadata_dict, file_path)
                elif outfile_format == 2: # NRRD
                    save_as_nrrd(self.npImage, self.readView1.metadata_dict, file_path)
                self.setCursor(QtCore.Qt.ArrowCursor)


    def save_changes_auto(self):
        """
        Automatic saving project
        :return:
        """
        tm = time.time() - self.startTime
        if tm  > self.expectedTime:
            try:
                self.dock_progressbar.setVisible(True)
                self.setEnabled(False)
                self.progressBarSaving.setValue(0)
                print(tm, 'Automatic Saving')
                self.saveChanges()
                self.setEnabled(True)
                self.dock_progressbar.setVisible(False)
                self.progressBarSaving.setValue(0)
                self.startTime = time.time()

            except Exception as e:
                print(e)
                print('Save changes error')
                self.setEnabled(True)
                self.dock_progressbar.setVisible(False)
                self.progressBarSaving.setValue(0)

    #def mouseReleaseEvent(self, event):

        ##################################################self.save_changes_auto()
        #return super(Ui_Main, self).mousePressEvent(event)


     #   pass
     #   if event.type() == Qt.RightButton and self.cursorOpenHand.mask().cacheKey() == self.cursor().mask().cacheKey(): # Open hand cursor
      #      self.pan(True, cursor_open = False)


    #def mouseReleaseEvent(self, event):
     #   if self.cursorClosedHand.mask().cacheKey() == self.cursor().mask().cacheKey():  # closed hand cursor
      #      self.pan(True, cursor_open = True)

    def changedTab(self):
        """
        Changing the tab
        """
        # --- 1. Determine which widgets to SHOW ---

        if self.tabWidget.currentIndex() == 0:
            # Tab 0: Multi-View
            widgets_num_view1 = [0, 1, 2]
            widgets_num_view2 = [3, 4, 5]

            # First, HIDE everything to start clean
            for k in widgets_num_view1 + widgets_num_view2:
                name = 'openGLWidget_' + str(k + 1)
                nameS = 'horizontalSlider_' + str(k + 1)
                label = getattr(self, f'label_{k + 1}', None)  # If you have labels

                # Safe getattr
                slider = getattr(self, nameS, None)
                widget = getattr(self, name, None)
                if label : label.setVisible(False)
                if widget: widget.setVisible(False)
                if slider: slider.setVisible(False)

            # Filter out slices if they are Video
            if self.is_view1_video:
                widgets_num_view1 = []
            if getattr(self, 'is_view2_video', False):
                widgets_num_view2 = []

            widgets_num = widgets_num_view1 + widgets_num_view2

        elif self.tabWidget.currentIndex() == 1:
            # Tab 1: View 1 Focus
            widgets_num = [10, 13]
            if self.is_view1_video:
                widgets_num = [10]  # Hide 3D view (13) for video

        elif self.tabWidget.currentIndex() == 2:
            # Tab 2: View 2 Focus
            widgets_num = [11, 23]
            # FIX: Check is_view2_video here, not is_view1_video!
            if getattr(self, 'is_view2_video', False):
                widgets_num = [11]  # Hide 3D view (23) for video

        else:
            widgets_num = []

        # --- 2. Update the Selected Widgets ---
        for k in widgets_num:
            name = 'openGLWidget_' + str(k + 1)
            nameS = 'horizontalSlider_' + str(k + 1)
            label = getattr(self, f'label_{k + 1}', None)  # If you have labels
            widget = getattr(self, name, None)
            slider = getattr(self, nameS, None)

            if widget is None: continue

            # --- VIDEO HANDLING ---
            # Only apply video logic if this specific view is actually video
            is_video_widget = False
            if k in [0, 1, 2, 10, 13] and self.is_view1_video:
                is_video_widget = True
            elif k in [3, 4, 5, 11, 23] and getattr(self, 'is_view2_video', False):
                is_video_widget = True

            if is_video_widget:
                if hasattr(widget, 'imSlice') and widget.imSlice is not None:
                    widget.setVisible(True)
                    if slider:
                        slider.setVisible(True)
                        slider.setValue(widget.sliceNum)
                    if label:
                        label.setVisible(True)

                    #widget.UpdatePaintInfo()

                    #widget.makeObject()
                    widget.update()
                continue

            # --- STANDARD 3D VIEW HANDLING ---
            if k == 23:
                try:
                    # 3D View 2
                    self.openGLWidget_24.paint(self.readView2.npSeg, self.readView2.npImage, None)
                except Exception as e:
                    pass
            elif k == 13:
                try:
                    # 3D View 1
                    self.openGLWidget_14.paint(self.readView1.npSeg, self.readView1.npImage, None)
                except Exception as e:
                    pass

            # --- STANDARD SLICE VIEW HANDLING ---
            else:
                if hasattr(widget, 'imSlice') and widget.imSlice is not None:
                    if slider:
                        slider.blockSignals(True)
                        slider.setRange(0, widget.imDepth)
                        slider.blockSignals(False)
                        slider.setValue(widget.sliceNum)
                        slider.setVisible(True)
                    if label:
                        label.setVisible(True)

                    widget.setVisible(True)
                    #widget.UpdatePaintInfo()

                    #widget.makeObject()
                    widget.update()
        if self.tabWidget.currentIndex()>0 and len(widgets_num)>1:
            self.changeToCoronal()


    def eventFilter(self, obj, event):

        if event.type() == QEvent.KeyPress:
            # Check for CTRL modifier
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            is_ctrl = (modifiers == Qt.ControlModifier)
            #if event.key() == Qt.Key_Control:
            #    if self.openGLWidget_1.hasFocus():
            #        self.openGLWidget_2.zRot = self.openGLWidget_1.zRot
            #        self.openGLWidget_2.update()
            # --- UNDO (Ctrl + Z) ---
            if is_ctrl and event.key() == Qt.Key_Z:
                # Check for Shift (Optional: Some apps use Ctrl+Shift+Z for Redo)
                if modifiers == (Qt.ControlModifier | Qt.ShiftModifier):
                    self.Redo()
                else:
                    self.Undo()
                return True  # Consume event

            # --- REDO (Ctrl + Y) ---
            if is_ctrl and event.key() == Qt.Key_Y:
                self.Redo()
                return True
        if event.type() == QEvent.UpdateRequest:
            pass
        elif event.type() == QEvent.MouseButtonRelease:
            self.MouseButtonRelease = True


        return QtWidgets.QWidget.eventFilter(self, obj, event)


    def manual(self):

        """
        HELP MELAGE
        :return:
        """
        url = os.path.join(settings.DOCS_DIR, 'README.html')
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(url))


        """
        
        try:
            import webbrowser
 
            webbrowser.open(url,new=2)
        except:
            pass
        """




    def about(self):
        """
        About MELAGE
        :return:
        """

        try:
            dialog = about_dialog(self, settings.RESOURCE_DIR)
            dialog.show()
        except Exception as e:
            print(e)

        #MessageBox = QtWidgets.QMessageBox(self)
        #MessageBox.setText(' melage \n Hospital Puerta del Mar\n March 2021')
        #MessageBox.show()

    def main_toolbar_visibility(self, value):
        self.toolBar.setVisible(value)
        if self.actionMain_Toolbar.isChecked():
            self.toolBar.setVisible(True)
        else:
            self.toolBar.setVisible(False)


    def changeToCoronal(self, typw='eco'):
        """
        Change to coronal
        :param typw:
        :return:
        """

        if typw == 'eco':
            windoe_name = 'video' if self.is_view1_video else 'coronal'
            changeCoronalSagittalAxial(self.horizontalSlider_11, self.openGLWidget_11,
                                       self.readView1, windoe_name, 3, self.label_11, initialState = True, tol_slice=self.tol_trk)

            self.radioButton_1.setChecked(True)
            self.radioButton_2.setChecked(False)
            self.radioButton_3.setChecked(False)
        elif typw == 'mri':
            windoe_name = 'video' if self.is_view2_video else 'coronal'
            changeCoronalSagittalAxial(self.horizontalSlider_12, self.openGLWidget_12,
                                       self.readView2, windoe_name, 3, self.label_12, initialState = True, tol_slice=self.tol_trk)

            self.radioButton_21_1.setChecked(True)
            self.radioButton_21_2.setChecked(False)
            self.radioButton_21_3.setChecked(False)


    def changeToSagittal(self, typw='eco'):
        """
        Changing to sagittal
        :param typw:
        :return:
        """
        if typw == 'eco':
            windoe_name = 'video' if self.is_view1_video else 'sagittal'
            changeCoronalSagittalAxial(self.horizontalSlider_11, self.openGLWidget_11,
                                   self.readView1, windoe_name, 1, self.label_11, initialState = True, tol_slice=self.tol_trk)
            self.radioButton_1.setChecked(False)
            self.radioButton_2.setChecked(True)
            self.radioButton_3.setChecked(False)
        elif typw == 'mri':
            windoe_name = 'video' if self.is_view2_video else 'sagittal'
            changeCoronalSagittalAxial(self.horizontalSlider_12, self.openGLWidget_12,
                                       self.readView2, windoe_name, 1, self.label_12, initialState = True, tol_slice=self.tol_trk)

            self.radioButton_21_1.setChecked(False)
            self.radioButton_21_2.setChecked(True)
            self.radioButton_21_3.setChecked(False)


    def changeToAxial(self, typw='eco'):
        if typw == 'eco':
            windoe_name = 'video' if self.is_view1_video else 'axial'
            changeCoronalSagittalAxial(self.horizontalSlider_11, self.openGLWidget_11,
                                   self.readView1, windoe_name, 5, self.label_11, initialState = True, tol_slice=self.tol_trk)
            self.radioButton_1.setChecked(False)
            self.radioButton_2.setChecked(False)
            self.radioButton_3.setChecked(True)
        elif typw == 'mri':
            windoe_name = 'video' if self.is_view2_video else 'axial'
            changeCoronalSagittalAxial(self.horizontalSlider_12, self.openGLWidget_12,
                                       self.readView2, windoe_name, 5, self.label_12, initialState=True, tol_slice=self.tol_trk)

            self.radioButton_21_1.setChecked(False)
            self.radioButton_21_2.setChecked(False)
            self.radioButton_21_3.setChecked(True)


    def _udpate_video_reader(self, widget, reader, sliceNum, tol_slice):
        if reader.current_frame == sliceNum:
            return

        reader.update_video_and_seg_frame(sliceNum)


        widget.points = []
        widget.selectedPoints = []

        widget.updateInfo(*getCurrentSlice(widget,
                                           reader.npImage, reader.npSeg,
                                           sliceNum, reader.tract, tol_slice=tol_slice), sliceNum, reader.npImage.shape,
                          imSpacing=reader.ImSpacing)
        widget.makeObject()
        widget.update()

    def changeSightTab3(self, value):
        """

        :param value:
        :return:
        """
        tol_slice = self.tol_trk
        if self.allowChangeScn:
            if not self.is_view1_video:
                updateSight(self.horizontalSlider_11, self.openGLWidget_11, self.readView1, value, tol_slice=tol_slice)
            else:
                slider = None
                if self.tabWidget.currentIndex()==1:
                    slider = self.horizontalSlider_11
                    widget = self.openGLWidget_11
                    reader = self.readView1
                if slider is not None:
                    sliceNum = value #slider.value()
                    self._udpate_video_reader(widget, reader, sliceNum, tol_slice)

    def changeSightTab4(self, value):
        tol_slice = self.tol_trk
        if self.allowChangeScn:
            if    not self.is_view2_video:
                updateSight(self.horizontalSlider_12, self.openGLWidget_12, self.readView2, value, tol_slice=self.tol_trk)
            else:
                if self.tabWidget.currentIndex()==2:
                    slider = self.horizontalSlider_12
                    widget = self.openGLWidget_12
                    reader = self.readView2
                if slider is not None:
                    sliceNum = value #slider.value()
                    self._udpate_video_reader(widget, reader, sliceNum, tol_slice)

    def changeSight1(self, value):
        if self.allowChangeScn:
            updateSight(self.horizontalSlider_1, self.openGLWidget_1, self.readView1, value, tol_slice=self.tol_trk)


    def changeSight2(self, value):
        if self.allowChangeScn:
            updateSight(self.horizontalSlider_2, self.openGLWidget_2, self.readView1, value, tol_slice=self.tol_trk)

    def changeSight3(self, value):
        if self.allowChangeScn:
            updateSight(self.horizontalSlider_3, self.openGLWidget_3, self.readView1, value, tol_slice=self.tol_trk)

    def changeSight4(self, value):
        if self.allowChangeScn:
            updateSight(self.horizontalSlider_4, self.openGLWidget_4, self.readView2, value, tol_slice=self.tol_trk)


    def changeSight5(self, value):
        if self.allowChangeScn:
            updateSight(self.horizontalSlider_5, self.openGLWidget_5, self.readView2, value, tol_slice=self.tol_trk)



    def changeSight6(self, value):
        if self.allowChangeScn:
            updateSight(self.horizontalSlider_6, self.openGLWidget_6, self.readView2, value, tol_slice=self.tol_trk)

    def toggle_video_playback(self, index=0):
        if index==0:
            video_timer = self.video_timer_view1
            button = self.btn_play_view1
        elif index==1:
            video_timer = self.video_timer_view2
            button = self.btn_play_view2
        if video_timer.isActive():
            video_timer.stop()
            button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        else:
            video_timer.start(33)  # ~30 FPS
            button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))

    def advance_video_frame(self, index = 0):
        if index==0:
            slider = self.horizontalSlider_11
        elif index==1:
            slider = self.horizontalSlider_12
        if slider.value() < slider.maximum():
            slider.setValue(slider.value() + 1)
        else:
            slider.setValue(0)  # Loop

    def updateDispView1(self, npImage = None, npSeg = None, initialState= False):
        """
        Updating US image
        :param npImage:
        :param npSeg:
        :param initialState:
        :return:
        """
        if npImage is None:
            return
        #if not '9876_Combined' in self.dw2Text:
        #    self.dw2Text.append('9876_Combined')


        if self.is_view1_video:
            ind_widgets = [11]
            ind_3d_widget = []
        else:
            ind_widgets = [1, 2, 3, 11]
            ind_3d_widget = [14]


        self.allowChangeScn = False
        self.ImageEnh_view1.setVisible(True)
        self.tree_colors.setVisible(True)
        #ind, colorPen = self.colorsCombinations[self.dw2Text.index(self.dw2_cb.currentText())]
        ind = 9876#self.dw2Text.index(self.dw2_cb.currentText())+1

        colorPen = [1, 0, 0, 1]
        if not self.is_view1_video:

            # 1. HIDE Play Button and STOP Timer
            #if hasattr(self, 'btn_play_view1'):
                #self.video_timer_view1.stop()  # Safety: stop video if switching tabs
                #self.btn_play_view1.setVisible(False)
                #self.btn_play_view1.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

            def update_slider(slider, label, extent_val):
                if slider.maximum() != extent_val - 1:
                    mid_val = extent_val // 2
                    slider.blockSignals(True)
                    slider.setRange(0, extent_val)
                    slider.blockSignals(False)
                    slider.setValue(mid_val)
                    label.setText(str_conv(mid_val))

            # Usage:
            ext = self.readView1.ImExtent
            update_slider(self.horizontalSlider_1, self.label_1, ext[3])
            update_slider(self.horizontalSlider_2, self.label_2, ext[1])
            update_slider(self.horizontalSlider_3, self.label_3, ext[5])
            currentWidnowName_in = self.openGLWidget_11.currentWidnowName
            view_to_ext_index = {
                'coronal': 3,
                'sagittal': 1,
                'axial': 5,
            }

            ext_v = ext[view_to_ext_index[currentWidnowName_in]]
            update_slider(self.horizontalSlider_11, self.label_11, ext_v)


            self.openGLWidget_1.updateCurrentImageInfo(npImage.shape[:3])
            self.openGLWidget_2.updateCurrentImageInfo(npImage.shape[:3])
            self.openGLWidget_3.updateCurrentImageInfo(npImage.shape[:3])
            #self.openGLWidget_11.updateCurrentImageInfo(npImage.shape[:3])

            self.radioButton_1.setVisible(True)
            self.radioButton_3.setVisible(True)
            self.radioButton_2.setVisible(True)
            self.radioButton_4.setVisible(True)

            # 1. Restore the Right Side
            self.openGLWidget_14.setVisible(True)
            self.openGLWidget_14.show()
            #self.widget.show()
            self.set_view_mode_show_all_view1()
            # 2. Restore Splitter Ratio (e.g., 2/3 Left, 1/3 Right)
            # Reset Vertical Sizes to something reasonable
            # e.g., Top takes majority, others take what they need
            #self.splitter_left_view1.setSizes([2000, 50, 50])
            #self.splitter_left_view1.setStretchFactor(0, 0)  # Standard resizing behavior
            #self.splitter_left_view1.setStretchFactor(1, 0)
            #self.splitter_main_view1.setSizes([2000, 1000])
        else:
            # 1. SHOW Play Button
            if hasattr(self, 'btn_play_view1'):
                self.btn_play_view1.setVisible(True)
            self.openGLWidget_14.hide()
            self.set_view_mode_focused_view1()
            #self.splitter_left_view1.setSizes([10000, 50, 0])

            # 3. Ensure the Horizontal Splitter gives space to left (as discussed before)
            #self.splitter_main_view1.setSizes([10000, 0])
            #self.splitter_left_view1.setStretchFactor(0, 1)  # Video expands
            #self.splitter_left_view1.setStretchFactor(1, 0)  # Slider stays fixed
            #self.splitter_left_view1.setStretchFactor(2, 0)  # Radio stays fixed

            self.radioButton_1.setVisible(False)
            self.radioButton_2.setVisible(False)
            self.radioButton_3.setVisible(False)
            self.radioButton_4.setVisible(False)







        name_lbl = 'label_'
        name_widg = 'openGLWidget_'
        name_slider = 'horizontalSlider_'
        name_radio = 'radioButton_'

        for i in ind_widgets:
            widget = getattr(self, name_widg+str(i))
            slider = getattr(self, name_slider+str(i))
            #if i==11:
                #widget.currentWidnowName = 'coronal' #TODO
            widget.updateCurrentImageInfo(npImage.shape[:3])

            widget.colorObject = colorPen
            widget.colorInd = ind
            widget.setVisible(True)
            slider.setVisible(True)
            widget.points = []
            sliceNum = slider.value()
            widget.updateInfo(*getCurrentSlice(widget, npImage, npSeg, sliceNum, tol_slice=self.tol_trk), sliceNum, npImage.shape[:3],
                                            initialState=initialState, imSpacing=self.readView1.ImSpacing)
            widget.update()
        for _widget_id in ind_3d_widget:
            widget = getattr(self, name_widg + str(_widget_id))
            if not initialState:

                widget.clear()
                widget._seg_im = None
                widget.paint(self.readView1.npSeg,
                                           self.readView1.npImage, None)
                widget.colorInd = ind
                widget.paint(self.readView1.npSeg, self.readView1.npImage, None)
            else:
                widget._image = self.readView1.npImage

        self.allowChangeScn = True




    def updateDispView2(self, npImage = None, npSeg = None, initialState= False):
        """
        Updating US image
        :param npImage:
        :param npSeg:
        :param initialState:
        :return:
        """
        if npImage is None:
            return
        #if not '9876_Combined' in self.dw2Text:
        #    self.dw2Text.append('9876_Combined')


        if self.is_view2_video:
            ind_widgets = [12]
            ind_3d_widget = []
        else:
            ind_widgets = [4, 5, 6, 12]
            ind_3d_widget = [24]


        self.allowChangeScn = False
        self.ImageEnh_view1.setVisible(True)
        self.tree_colors.setVisible(True)
        #ind, colorPen = self.colorsCombinations[self.dw2Text.index(self.dw2_cb.currentText())]
        ind = 9876#self.dw2Text.index(self.dw2_cb.currentText())+1
        colorPen = [1, 0, 0, 1]

        if not self.is_view2_video:

            # 1. HIDE Play Button and STOP Timer
            #if hasattr(self, 'btn_play_view2'):
                #self.video_timer_view2.stop()  # Safety: stop video if switching tabs
                #self.btn_play_view2.setVisible(False)
                #self.btn_play_view2.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

            def update_slider(slider, label, extent_val):
                if slider.maximum() != extent_val - 1:
                    mid_val = extent_val // 2
                    slider.blockSignals(True)
                    slider.setRange(0, extent_val)
                    slider.blockSignals(False)
                    slider.setValue(mid_val)
                    label.setText(str_conv(mid_val))


            ext = self.readView2.ImExtent
            update_slider(self.horizontalSlider_4, self.label_1, ext[3])
            update_slider(self.horizontalSlider_5, self.label_2, ext[1])
            update_slider(self.horizontalSlider_6, self.label_3, ext[5])
            currentWidnowName_in = self.openGLWidget_12.currentWidnowName
            view_to_ext_index = {
                'coronal': 3,
                'sagittal': 1,
                'axial': 5,
            }

            ext_v = ext[view_to_ext_index[currentWidnowName_in]]
            update_slider(self.horizontalSlider_12, self.label_12, ext_v)



            self.openGLWidget_4.updateCurrentImageInfo(npImage.shape[:3])
            self.openGLWidget_5.updateCurrentImageInfo(npImage.shape[:3])
            self.openGLWidget_6.updateCurrentImageInfo(npImage.shape[:3])
            #self.openGLWidget_12.updateCurrentImageInfo(npImage.shape[:3])





            self.label_5.setVisible(True)
            self.label_4.setVisible(True)
            self.label_6.setVisible(True)



            self.radioButton_21.setVisible(True)
            self.radioButton_21_1.setVisible(True)
            self.radioButton_21_2.setVisible(True)
            self.radioButton_21_3.setVisible(True)

            # 1. Restore the Right Side
            self.openGLWidget_24.setVisible(True)
            self.openGLWidget_24.show()
            #self.widget.show()
            # 2. Restore Splitter Ratio (e.g., 2/3 Left, 1/3 Right)
            # Reset Vertical Sizes to something reasonable
            # e.g., Top takes majority, others take what they need
            self.set_view_mode_show_all_view2()
            #self.splitter_left_view2.setSizes([2000, 50, 50])
            #self.splitter_left_view2.setStretchFactor(0, 0)  # Standard resizing behavior
            #self.splitter_left_view2.setStretchFactor(1, 0)
            #self.splitter_main_view2.setSizes([2000, 1000])
        else:
            # 1. SHOW Play Button
            if hasattr(self, 'btn_play_view2'):
                self.btn_play_view2.setVisible(True)
            self.openGLWidget_24.hide()
            self.set_view_mode_focused_view2()
            #self.splitter_left_view2.setSizes([10000, 50, 0])

            # 3. Ensure the Horizontal Splitter gives space to left (as discussed before)
            #self.splitter_main_view2.setSizes([10000, 0])
            #self.splitter_left_view2.setStretchFactor(0, 1)  # Video expands
            #self.splitter_left_view2.setStretchFactor(1, 0)  # Slider stays fixed
            #self.splitter_left_view2.setStretchFactor(2, 0)  # Radio stays fixed

            self.radioButton_21.setVisible(False)
            self.radioButton_21_1.setVisible(False)
            self.radioButton_21_2.setVisible(False)
            self.radioButton_21_3.setVisible(False)







        name_lbl = 'label_'
        name_widg = 'openGLWidget_'
        name_slider = 'horizontalSlider_'
        name_radio = 'radioButton_'

        for i in ind_widgets:
            widget = getattr(self, name_widg+str(i))
            slider = getattr(self, name_slider+str(i))
            #if i==11:
                #widget.currentWidnowName = 'coronal' #TODO
            widget.updateCurrentImageInfo(npImage.shape[:3])

            widget.colorObject = colorPen
            widget.colorInd = ind
            widget.setVisible(True)
            slider.setVisible(True)
            widget.points = []
            sliceNum = slider.value()
            widget.updateInfo(*getCurrentSlice(widget, npImage, npSeg, sliceNum, tol_slice=self.tol_trk), sliceNum, npImage.shape[:3],
                                            initialState=initialState, imSpacing=self.readView2.ImSpacing)
            widget.update()
        for _widget_id in ind_3d_widget:
            widget = getattr(self, name_widg + str(_widget_id))
            if not initialState:

                widget.clear()
                widget._seg_im = None
                widget.paint(self.readView2.npSeg,
                                           self.readView2.npImage, None)
                widget.colorInd = ind
                widget.paint(self.readView2.npSeg, self.readView2.npImage, None)
            else:
                widget._image = self.readView2.npImage

        self.allowChangeScn = True




    def retranslateUi(self, Main):
        self._translate = QtCore.QCoreApplication.translate
        Main.setWindowTitle(self._translate("Main", "MELAGE"))
        self.menuFile.setTitle(self._translate("Main", "File"))
        self.menuAbout.setTitle(self._translate("Main", "Help"))
        self.menuView.setTitle(self._translate("Main", "View"))
        self.menuToolbar.setTitle(self._translate("Main", "Toolbars"))
        self.menuWidgets.setTitle(self._translate("Main", "Widgets"))
        self.actionOpenView1.setText(self._translate("Main", "Image View 1"))
        self.actionOpenView2.setText(self._translate("Main", "Image View 2"))
        self.actionOpenFA.setText(self._translate("Main", "OpenFA"))
        self.actionLoad.setText(self._translate("Main", "Load project"))
        self.actionNew.setText(self._translate("Main", "New project"))
        self.actionFile_info.setText(self._translate("Main", "Settings"))
        self.actionFile_changeIM.setText(self._translate("Main", "Change IM"))
        self.actionfile_iminfo.setText(self._translate("Main", "Images Info."))
        #self.actionconvert.setText(self._translate("Main", "Convert"))
        #self.actionsaveModified.setText(self._translate("Main", "SAVE MODIFIED"))
        self.actionexit.setText(self._translate("Main", "Exit"))
        self.actionabout.setText(self._translate("Main", "About"))
        self.actionmanual.setText(self._translate("Main", "Help"))
        self.actionVersion.setText(self._translate("Main", f"Version {__VERSION__}"))
        #self.pushButton.setText(self._translate("Main", "PushButton"))
        self.toolBar.setWindowTitle(self._translate("Main", "Main ToolBar"))
        self.toolBar2.setWindowTitle(self._translate("Main", "Interaction"))


        self.action_interaction_Toolbar.setText(self._translate("Main", "Interaction"))
        self.action_guideLines.setText(self._translate("Main", "Guides"))
        self.action_axisLines.setText(self._translate("Main", "Axis"))
        self.actionPan.setText(self._translate("Main", "Pan"))
        self.actionContour.setText(self._translate("Main", "Contour"))
        self.actionPoints.setText(self._translate("Main", "Point Selection"))
        self.actionCircles.setText(self._translate("Main", "Circle Selection"))
        self.actionGoTo.setText(self._translate("Main", "Link"))
        self.action3D.setText(self._translate("Main", "3D"))
        self.actionZoomIn.setText(self._translate("Main", "Zoom In"))
        self.actionZoomOut.setText(self._translate("Main", "Zoom Out"))
        self.actionContourX.setText(self._translate("Main", "Contour X times"))
        self.actionEraseX.setText(self._translate("Main", "Eraser X times"))
        self.actionPaint.setText(self._translate("Main", "Paint"))
        self.actionErase.setText(self._translate("Main", "Erase"))
        self.actionLazyContour.setText(self._translate("Main", "Zoom Out"))



        self.actionHistImage.setText(self._translate("Main", "Image Histogram"))

        self.actionImageThresholding.setText(self._translate("Main", "Image Thresholding"))
        self.actionImageRegistration.setText(self._translate("Main", "Image Registration"))
        self.actionImageTransformation.setText(self._translate("Main", "Image Transformation"))
        self.actionMasking.setText(self._translate("Main", "Image Masking"))
        self.actionOperationMask.setText(self._translate("Main", "Masking Operation"))

        self.actionrotate.setText(self._translate("Main", "Rotate"))
        self.actionArrow.setText(self._translate("Main", "Arrow"))
        self.menuTools.setTitle(self._translate("Main", "Tools"))
        self.menuPlugins.setTitle(self._translate("Main", "Plugins"))
        self.menuBasicInfo.setTitle(self._translate("Main", "Basic Info"))
        self.menuCalc.setTitle(self._translate("Main", "Calc"))
        self.menuPreprocess.setTitle(self._translate("Main", "Preprocessing"))
        self.menuRegistration.setTitle(self._translate("Main", "Registeration"))

        self.menuExport.setTitle(self._translate("Main", "Export"))
        self.menuImport.setTitle(self._translate("Main", "Import"))
        self.menuTV.setTitle(self._translate("Main", "Total Volume"))
        self.actionTVCor.setText(self._translate("Main", "Coronal"))
        self.actionTVSag.setText(self._translate("Main", "Sagital"))
        self.actionTVAx.setText(self._translate("Main", "Axial"))
        self.menuSeg.setTitle( self._translate("Main", "Ventricle Segmentation") )
        self.actionMain_Toolbar.setText(self._translate("Main", "Main Toolbar"))
        self.actionsave.setText(self._translate("Main", "Save"))
        self.actionsaveas.setText(self._translate("Main", "Save as"))
        self.actionLine.setText(self._translate("Main", "Draw Line"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.mutulaViewTab), self._translate("Main", "Mutual view"))
        #self.tabWidget.setTabText(self.tabWidget.indexOf(self.reservedTab), self._translate("Main", "Rserved"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.segmentationTab), self._translate("Main", "View 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.MRISegTab), self._translate("Main", "View 2"))
        #self.tabWidget.setTabVisible(1, True)
        self.tabWidget.setStyleSheet("QTabBar::tab {"+"height: {}px; width: {}px;".format(self.height()//52,self.width()//12)+"}");
        #self.tabWidget.setTabVisible(2, False)
        #self.tabWidget.setTabVisible(3, False)


        self.radioButton_21_1.setText(self._translate("Main", "Coronal"))
        self.radioButton_21_2.setText(self._translate("Main", "Sagittal"))
        self.radioButton_21_3.setText(self._translate("Main", "Axial"))
        self.radioButton_21.setText(self._translate("Main", "Show Seg"))

        self.radioButton_1.setText(self._translate("Main", "Coronal"))
        self.radioButton_2.setText(self._translate("Main", "Sagittal"))
        self.radioButton_3.setText(self._translate("Main", "Axial"))
        self.radioButton_4.setText(self._translate("Main", "Show Seg"))



        self.actionUndo.setText(self._translate("Main", "Undo"))
        self.actionRedo.setText(self._translate("Main", "Redo"))
        #self.actionContourGen.setText(self._translate("Main", "Contour Gen from line"))
        manually_check_tree_item(self,'9876')

    def get_dialog(self, source_dir, filters, opts, title = "Open File"):

        # 1. Create a QFileDialog instance instead of using the static method
        dialog = QtWidgets.QFileDialog(self, title, source_dir, filters)

        # 2. Set the options you need
        dialog.setOptions(opts)
        # Note: If using PyQt5/PySide2, you might need:
        # dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)

        # 3. Get the parent window's size
        parent_size = self.size()
        # You can also use self.geometry() for more complex positioning

        # 4. Set the dialog's size relative to the parent's size
        # For example, 80% of the parent's width and 70% of its height
        dialog_width = int(parent_size.width() * 0.4)
        dialog_height = int(parent_size.height() * 0.5)
        dialog.resize(dialog_width, dialog_height)
        return dialog

    def loadProject(self, base_file=None):
        def _get_from_ui():
            filters = "BrainNeonatal (*.bn)"
            opts = QtWidgets.QFileDialog.DontUseNativeDialog
            dialog = self.get_dialog(settings.DEFAULT_USE_DIR, filters, opts, title="Open File")
            # if fileObj[0] != '':
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                selected_files = dialog.selectedFiles()
                if selected_files:
                    # getOpenFileName returns a tuple (filepath, filter)
                    fileObj = (selected_files[0], dialog.selectedNameFilter())
                else:
                    return False
                self._basefileSave, _ = os.path.splitext(fileObj[0])
                return True
            return False
        """
        Loading saved project
        :return:
        """
        #filters = "BrainNeonatal (*.bn)"
        #opts =QtWidgets.QFileDialog.DontUseNativeDialog
        #fileObj = QtWidgets.QFileDialog.getOpenFileName( self, "Open File", settings.DEFAULT_USE_DIR, filters, options=opts)
        if base_file is None or type(base_file) is bool:
            succes = _get_from_ui()
            base_file = None
        else:
            self._basefileSave = base_file
            succes = True
        if succes:
            self.dock_progressbar.setVisible(True)
            self.setEnabled(False)

            self.progressBarSaving.setValue(0)
            self._loaded = False
            self.CloseView1(message_box='off')
            self.CloseView2(message_box='off')
            self.loadChanges(base_file)

            self.activateGuidelines(False)
            self.progressBarSaving.setValue(100)
            self.setEnabled(True)
            self.dock_progressbar.setVisible(False)
            self.progressBarSaving.setValue(0)

            if self._openUSEnabled:
                self.actionOpenView1.setDisabled(False)
            self.actionOpenView2.setDisabled(False)
            self.actionOpenFA.setDisabled(True)
            self.actionOpenTract.setDisabled(True)
            if rhasattr(self, 'readView1.npImage'):
                self.actionImportSegView1.setDisabled(False)
                self.actionExportImView1.setDisabled(False)
                self.actionExportSegView1.setDisabled(False)

            if rhasattr(self, 'readView2.npImage'):
                self.actionImportSegView2.setDisabled(False)
                self.actionExportImView2.setDisabled(False)
                self.actionExportSegView2.setDisabled(False)
            #self.actionImportSegView2.setDisabled(False)
            self.actionsave.setDisabled(False)
            self.actionsaveas.setDisabled(False)
            self.toolBar2.setDisabled(False)
            self.startTime = time.time()

        else:
            return

    def newProject(self):
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        #fileObj = QtWidgets.QFileDialog.getSaveFileName(self, "Open File", settings.DEFAULT_USE_DIR, options=opts)

        dialog = self.get_dialog( settings.DEFAULT_USE_DIR, filters=None, opts=opts, title="Open File")
        #if fileObj[0] != '':
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                # getOpenFileName returns a tuple (filepath, filter)
                fileObj = (selected_files[0], dialog.selectedNameFilter())
            else:
                return
        #if fileObj[0] != '':

            self._basefileSave, _ = os.path.splitext(fileObj[0])
            if self._openUSEnabled:
                self.actionOpenView1.setDisabled(False)
            self.actionOpenView2.setDisabled(False)

            self.actionOpenFA.setDisabled(True)
            self.actionOpenTract.setDisabled(True)
            self.actionsave.setDisabled(False)
            self.actionsaveas.setDisabled(False)
            self.reset_view_pages(index=0)
            self.reset_view_pages(index=1)
            widgets_num = [0, 1, 2, 3, 4, 5, 10, 11]
            for k in widgets_num:
                name = 'openGLWidget_' + str(k + 1)
                widget = getattr(self, name)
                widget.resetInit()
                widget.initialState()
            self.openGLWidget_14.clear()
            self.openGLWidget_24.clear()
            self.Main.setWindowTitle(self._translate("Main", os.path.basename(self._basefileSave)))
            self.save()

    def CloseView1(self, message_box=True):
        """
        Closes the Ultrasound (ECO) image and resets the associated UI elements.

        :param show_confirmation: If True, asks the user for confirmation before closing.
        :return: True if the view was closed, False if the user cancelled.
        """
        # 1. Check state: Is there anything to close?
        if not self.readView1:
            # print("Ultrasound view is already closed.")
            return True  # Already closed

        proceed = False
        if message_box:
            # 2. Ask for confirmation
            qm = QtWidgets.QMessageBox(self)
            ret = qm.question(self, 'Confirm Close',
                              "Are you sure you want to close View 1?",
                              qm.Yes | qm.No)
            if ret == qm.Yes:
                proceed = True
        else:
            # 3. No confirmation needed
            proceed = True

        # 4. Perform cleanup only if confirmed
        if proceed:
            self._perform_view1_cleanup()

        return proceed

    def _set_view1_widgets_visible(self, is_visible):
        """Helper method to show/hide all widgets related to the US view."""

        # Disable actions
        self.actionImportSegView1.setDisabled(True)
        self.actionExportImView1.setDisabled(True)
        self.actionExportSegView1.setDisabled(True)

        # Toggle radio buttons
        for name in self.View1_RADIO_NAMES:
            radio_button = getattr(self, name)
            radio_button.setVisible(is_visible)

        # Toggle sliders, labels, and OpenGL widgets
        for k in self.VIEW1_INDICES:
            slider = getattr(self, 'horizontalSlider_' + str(k))
            label = getattr(self, 'label_' + str(k))
            widget = getattr(self, 'openGLWidget_' + str(k))

            label.setVisible(is_visible)
            slider.setVisible(is_visible)
            widget.setVisible(is_visible)

            if not is_visible:
                # If hiding, also clear and reset the widget
                widget.imType = 'eco'
                widget.clear()
                widget.resetInit()
                widget.initialState()

    def _perform_view1_cleanup(self):
        """Helper method to consolidate all cleanup tasks for closing the US view."""

        # 1. Hide all the main US widgets
        self._set_view1_widgets_visible(False)

        # 2. Reset other related UI elements
        self.reset_view_pages(index=0)
        self.openGLWidget_14.clear()
        self.ImageEnh_view1.setVisible(False)
        # self.tree_colors.setVisible(False) # Removed commented-out code

        # 3. Clear the image data
        self.readView1 = []
        self.btn_play_view1.setVisible(False)
        # 4. Update tab state
        self.changedTab()

        # 5. Resize other (MRI) widgets
        for k in self.VIEW2_INDICES:
            name = 'openGLWidget_' + str(k)
            widget = getattr(self, name)
            event = QtGui.QResizeEvent(widget.size(), widget.size())
            widget.resizeEvent(event)
            widget.resize(QtCore.QSize(widget.size().width(), widget.size().height() * 2))

        # 6. Reset the ComboBox
        self.actionComboBox_visible.setVisible(False)
        self.actionComboBox_visible.setDisabled(True)
        self.actionComboBox.setObjectName("View1")

        # 7. Safely disconnect the signal
        try:
            self.actionComboBox.currentTextChanged.disconnect(self.changeVolume)
        except (TypeError, RuntimeError):
            # Catches specific errors if it's not connected
            pass

        self.actionComboBox.clear()

        # 8. Call the final cleanup utility
        clean_parent_image(self, -1, ['View 1'], index_view=0)
        for plugin in self.plugin_widgets:
            data_context = self.get_current_image_data()
            plugin.update_data_context(data_context)
        self.full_path_view1 = None

    def CloseView2(self, message_box=True):
        """
        Closes the View2 image and resets the associated UI elements.

        :param show_confirmation: If True, asks the user for confirmation before closing.
        :return: True if the view was closed, False if the user cancelled.
        """
        # 1. Check state: Is there anything to close?
        #    (Assuming self.readView2 = [] is the "closed" state)
        if not self.readView2:

            return True  # Already closed, operation is "successful"

        proceed = False
        if message_box:
            # 2. Ask for confirmation
            # (Uncommented the MessageBox logic)
            qm = QtWidgets.QMessageBox(self)
            ret = qm.question(self, 'Confirm Close',
                              "Are you sure you want to close View 2?",
                              qm.Yes | qm.No)
            if ret == qm.Yes:
                proceed = True
        else:
            # 3. No confirmation needed, just proceed
            proceed = True

        # 4. Perform cleanup only if user confirmed or no confirmation was needed
        if proceed:
            self._perform_view2_cleanup()

        return proceed

    def _set_view2_widgets_visible(self, is_visible):
        """Helper method to show/hide all widgets related to the MRI view."""

        # Disable actions
        self.actionImportSegView2.setDisabled(True)
        self.actionExportImView2.setDisabled(True)
        self.actionExportSegView2.setDisabled(True)

        # Toggle radio buttons
        for name in self.View2_RADIO_NAMES:
            radio_button = getattr(self, name)
            radio_button.setVisible(is_visible)

        # Clean up the opened file name text
        txt = self.openedFileName.text()
        division_ind = txt.find(' ; ')
        if division_ind != -1:
            # Use split to robustly get the part before ' ; '
            kept_part = txt.split(' ; ', 1)[0]
            self.openedFileName.setText(f'{kept_part} ; ')

        # Toggle sliders, labels, and OpenGL widgets
        for k in self.VIEW2_INDICES:
            slider = getattr(self, 'horizontalSlider_' + str(k))
            label = getattr(self, 'label_' + str(k))
            widget = getattr(self, 'openGLWidget_' + str(k))

            label.setVisible(is_visible)
            slider.setVisible(is_visible)
            widget.setVisible(is_visible)

            if not is_visible:
                # If hiding, also clear and reset the widget
                widget.imType = 'mri'
                widget.clear()
                widget.resetInit()
                widget.initialState()

    def _perform_view2_cleanup(self):
        """Helper method to consolidate all cleanup tasks for closing the MRI view."""

        # 1. Hide all the main MRI widgets
        self._set_view2_widgets_visible(False)

        # 2. Reset other related UI elements
        self.reset_view_pages(index=1)
        self.openGLWidget_24.clear()
        self.page1_mri.setVisible(False)
        # self.tree_colors.setVisible(False) # Removed commented-out code

        # 3. Clear the image data
        self.readView2 = []
        # self.save() # Removed commented-out code
        self.btn_play_view2.setVisible(False)
        # 4. Resize other widgets
        for k in self.VIEW1_INDICES:
            name = 'openGLWidget_' + str(k)
            widget = getattr(self, name)
            event = QtGui.QResizeEvent(widget.size(), widget.size())
            widget.resizeEvent(event)
            widget.resize(QtCore.QSize(widget.size().width(), widget.size().height() * 2))

        # 5. Reset the ComboBox
        self.actionComboBox_visible.setVisible(False)
        self.actionComboBox_visible.setDisabled(True)
        self.actionComboBox.setObjectName("View2")

        # 6. Safely disconnect the signal
        try:
            self.actionComboBox.currentTextChanged.disconnect(self.changeVolume)
        except (TypeError, RuntimeError):
            # This is safer than a bare 'except:'
            # It catches the specific errors disconnect() might raise
            pass

        self.actionComboBox.clear()

        # 7. Call the final cleanup utility
        clean_parent_image(self, -1, ['View 2'], index_view=1)
        for plugin in self.plugin_widgets:
            data_context = self.get_current_image_data()
            plugin.update_data_context(data_context)

        self.full_path_view2 = None

    def _check_status_warning_view1(self):

        if not hasattr(self, 'readView1'):
            self.warning_msgbox('There is no UltraSound image')
            return False
        if not hasattr(self.readView1, 'npImage'):
            self.warning_msgbox('There is no UltraSound image')
            return False
        return True

    def _check_status_warning_view2(self):
        if not hasattr(self, 'readView2'):
            self.warning_msgbox('There is no View2 image')
            return False
        if not hasattr(self.readView2, 'npImage'):
            self.warning_msgbox('There is no View2 image')
            return False
        return True

    def _filesave_dialog(self, filters,opts, pref='', currentCS=None):
        """
        Dialogue to save files
        :param filters:
        :param opts:
        :param pref:
        :param currentCS:
        :return:
        """
        #fileObj = QtWidgets.QFileDialog.getSaveFileName(self, "Open File", settings.DEFAULT_USE_DIR+'/'+pref, filters, options=opts)
        from melage.dialogs.helpers import QFileSaveDialogPreview
        #from melage.utils.utils import getCurrentCoordSystem
        check_save = True
        if hasattr(self, '_last_state_save_csv' ):
            check_save=self._last_state_save_csv
        dialg = QFileSaveDialogPreview(self, "Open File", settings.DEFAULT_USE_DIR+'/'+pref, filters, options=opts, check_state_csv=check_save)

        dialg.setCS(currentCS)
        parent_size = self.size()
        dialog_width = int(parent_size.width() * 0.6)
        dialog_height = int(parent_size.height() * 0.7)
        dialg.resize(dialog_width, dialog_height)

        #if dialg.exec_() == QFileSaveDialogPreview.Accepted:
        dialg.exec_()
        cs, from_to, save_csv = dialg.getInfo()
        self._last_state_save_csv = dialg.checkBox_csv.isChecked()
        fileObj = dialg.getFileSelected()
        fileObj[1] = dialg.selectedNameFilter()

        if fileObj[0] == '':
            self.warning_msgbox('No file name is selected')
        return fileObj, [cs, from_to, save_csv]

    def exportData(self, export_type):
        """
        Unified Export Manager for MRI, Ultrasound (Volumetric), and Video Segmentation.
        Handles NIfTI, TIFF, and AVI formats.
        """


        # --- 1. CONFIGURATION MAPPING ---
        # Maps the input 'type' string to the actual object attributes
        config_map = {
            'View1_IM': {'reader': 'readView1', 'attr': 'npImage', 'suffix': '_new', 'dtype': np.float32,
                     'check': '_check_status_warning_view1'},
            'View1_SEG': {'reader': 'readView1', 'attr': 'npSeg', 'suffix': '_seg', 'dtype': np.int16,
                      'check': '_check_status_warning_view1'},
            'View2_IM': {'reader': 'readView2', 'attr': 'npImage', 'suffix': '_new', 'dtype': np.float32,
                      'check': '_check_status_warning_view2'},
            'View2_SEG': {'reader': 'readView2', 'attr': 'npSeg', 'suffix': '_seg', 'dtype': np.int16,
                       'check': '_check_status_warning_view2'},
        }

        # Validate Input
        export_type = export_type.lower()
        if export_type not in config_map:
            print(f"Error: Unknown export type {export_type}")
            return

        conf = config_map[export_type]

        # Run Safety Checks (e.g., is data loaded?)
        check_func = getattr(self, conf['check'], None)
        if check_func and not check_func():
            return

        # Get the Reader Object (e.g., self.readView1)
        reader = getattr(self, conf['reader'], None)
        if reader is None: return

        # --- 2. DETECT MODE: VIDEO vs STATIC 3D ---
        # Check if this is a video proxy or a standard 3D volume
        is_video_proxy = reader.isChunkedVideo


        # --- 3. PREPARE FILE DIALOG ---
        # Determine Filters based on mode
        if is_video_proxy:
            filters = "Video (*.avi);;NifTi (*.nii *.nii.gz)"
        else:
            filters = "NifTi (*.nii *.nii.gz *.mgz);;tif(*.tif)"

        # Generate Default Filename
        original_path = getattr(self, 'filenameView1' if 'us' in export_type else 'filenameView2', "output")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        default_name = base_name + conf['suffix']

        # Coordinate System
        current_cs = getattr(reader, 'source_system', 'RAS')

        # Show Dialog
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, selected_filter = self._filesave_dialog(filters, opts, default_name, currentCS=current_cs)

        # User Cancelled
        if not file_path:
            return


        file_path, selected_filter = file_path  # unpack if needed

        # --- 4. EXECUTE SAVE ---
        try:
            self.dock_progressbar.setVisible(False)
            self.progressBarSaving.setValue(0)
            self.setEnabled(False)
            self.app.processEvents()  # Force UI update

            # A. VIDEO EXPORT
            if is_video_proxy and 'video' in selected_filter.lower():
                if not file_path.endswith('.avi'):
                    file_path = file_path + '.avi'
                self._export_video_data(reader, file_path, export_type)

            # B. STATIC 3D IMAGE EXPORT (NIfTI / TIF)
            else:
                # Get the actual data array
                # If it's a proxy, we might need to convert it to a full 3D array first for NIfTI
                if is_video_proxy:
                    print("Converting Video Proxy to 3D Volume for NIfTI export...")
                    # Note: This loads the WHOLE video into RAM. Be careful.
                    # Assuming get_fdata() or similar exists, or stack frames manually.
                    data_to_save = self._stack_video_to_array(reader, conf['attr'])
                else:
                    data_to_save = getattr(reader, conf['attr'])

                # Cast type
                data_to_save = data_to_save.astype(conf['dtype'])

                # Determine Format Index (0=Nifti, 1=Tif) based on filter string
                is_tif = 'tif' in selected_filter
                format_code = 1 if is_tif else 0
                type_im = 'eco' if 'us' in export_type else 'mri'

                # Save Logic
                if format_code == 1:  # TIF
                    save_3d_img(reader, file_path, data_to_save, 'tif', type_im=type_im, cs=current_cs)
                    export_tables(self, file_path[:-4] + "_table")
                else:  # NIfTI
                    save_3d_img(reader, file_path, data_to_save, format='nifti', type_im=type_im, cs=current_cs)
                    export_tables(self, file_path + "_table")

        except Exception as e:
            print(f"Export Failed: {e}")

            traceback.print_exc()

        finally:
            # Cleanup UI
            self.setEnabled(True)
            self.dock_progressbar.setVisible(False)
            self.progressBarSaving.setValue(0)

    # --- HELPER FUNCTIONS (Add these to your class) ---

    def _export_video_data(self, reader, output_path, export_type):
        """Helper to handle AVI export logic."""

        # Manual Loop Writer (if proxy.save() didn't handle it)
        # 1. Get FPS
        fps = 30.0
        if hasattr(reader, 'video_im') and hasattr(reader.video_im, 'fps'):
            fps = reader.video_im.fps


        # Determine which object to save (Image vs Segmentation)
        if 'seg' in export_type:
            # Saving Segmentation Proxy
            if hasattr(reader, 'seg_ims'):
                self.setEnabled(False)
                # Use the proxy's built-in save if available
                reader.seg_ims.save(output_path, fps)
                self.setEnabled(True)
                return
            else:
                # Fallback if no proxy (standard array)
                target_obj = reader.npSeg
                is_seg = True
        else:
            # Saving Video Image
            if hasattr(reader, 'video_im'):
                # If video_im is a proxy that supports save, use it.
                # Otherwise, we might need to copy the original file or re-encode.
                print("Exporting Image Video...")
                target_obj = reader.video_im
                is_seg = False
            else:
                target_obj = reader.npImage
                is_seg = False



        # 2. Setup Writer
        # Use Lossless PNG for segmentation, MJPG/MP4V for images
        codec = 'png ' if is_seg else 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*codec)

        # Dimensions: (Width, Height)
        if hasattr(target_obj, 'shape'):
            h, w = target_obj.shape[:2]
            depth = target_obj.shape[2]
        else:
            return  # Cannot determine shape

        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=not is_seg)

        # 3. Write Frames
        total_frames = depth
        for i in range(total_frames):
            # Progress Bar Update
            if i % 10 == 0:
                prog = int((i / total_frames) * 100)
                self.progressBarSaving.setValue(prog)
                self.app.processEvents()

            # Get Frame
            if hasattr(target_obj, 'get_frame'):
                frame = target_obj.get_frame(i)
            elif target_obj.ndim == 3:  # (H, W, T) or (H, W, Channels)? Assuming (H, W, T) for seg
                frame = target_obj[..., i]
            elif target_obj.ndim == 4:  # (H, W, T, C)
                frame = target_obj[:, :, i, :]

            # Ensure format
            if is_seg:
                frame = frame.astype(np.uint8)
            else:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV needs BGR

            writer.write(frame)

        writer.release()
        print(f"Video saved: {output_path}")

    def _stack_video_to_array(self, reader, attr_name):
        """
        Helpers to convert a lazy Video Proxy into a numpy RAM array for NIfTI saving.
        WARNING: Memory Intensive.
        """

        if attr_name == 'npSeg' and hasattr(reader, 'seg_proxy'):
            proxy = reader.seg_proxy
        elif hasattr(reader, 'video_im'):
            proxy = reader.video_im
        else:
            return getattr(reader, attr_name)

        d, w, h = proxy.shape[2], proxy.shape[1], proxy.shape[0]

        print(f"Allocating full volume in RAM: {h}x{w}x{d}")

        # Pre-allocate
        if attr_name == 'npSeg':
            full_vol = np.zeros((h, w, d), dtype=np.uint8)
            for i in range(d):
                full_vol[..., i] = proxy.get_frame(i)
        else:
            # Image might be RGB, NIfTI usually expects 4D for RGB or 3D for Gray
            # Let's assume grayscale for NIfTI compatibility or 4D
            full_vol = np.zeros((h, w, d, 3), dtype=np.uint8)
            for i in range(d):
                full_vol[:, :, i, :] = proxy.get_frame(i)

        return full_vol

    def exportData2(self, type):
        """
        Export data
        :param type:
        :return:
        """

        def save_as_image(reader, file, img, format=0, type_im = 'mri', cs=['RAS', 'AS']):

            try:
                self.dock_progressbar.setVisible(True)
                self.setEnabled(False)

                self.progressBarSaving.setValue(0)
                if format==1:
                    save_3d_img(reader, file, img, 'tif', type_im=type_im, cs=cs)
                    export_tables(self, file[:-7] + "_table")
                elif format==0:

                    save_3d_img(reader, file, img, format='nifti', type_im=type_im, cs=cs)
                    export_tables(self, file+"_table")

                self.setEnabled(True)
                self.dock_progressbar.setVisible(False)
                self.progressBarSaving.setValue(0)
            except Exception as e:
                self.setEnabled(True)
                self.dock_progressbar.setVisible(False)
                self.progressBarSaving.setValue(0)
                print('save 3d image')

        def save_video(reader, output_path):
            """
            Saves segmentation as a Lossless Grayscale AVI using original video's FPS.
            """
            if not output_path.endswith('.avi'):
                output_path = output_path.rsplit('.', 1)[0] + '.avi'

            # 1. GET FPS FROM PARENT VIDEO (Critical)
            # Assuming parent_video_proxy stored the cap info or has access to it.
            # If you didn't store it in __init__, we can read it again quickly.
            fps = 30.0  # Default fallback
            if hasattr(reader.video_im, 'fps'):
                fps = reader.video_im.fps
            else:
                # Quick check if we need to retrieve it manually
                temp_cap = cv2.VideoCapture(self.video_im.file_path)
                fps = temp_cap.get(cv2.CAP_PROP_FPS)
                temp_cap.release()

            print(f"Saving to {output_path} at {fps} FPS...")

            fourcc = cv2.VideoWriter_fourcc(*'png ')  # Lossless

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,  # <--- Use the exact source FPS
                (reader.video_im.shape[1], reader.video_im.shape[0]),
                isColor=False
            )

            if not writer.isOpened():
                print(f"Error: Could not open writer for {output_path}")
                return

            for i in range(reader.video_im.shape[2]):
                mask_2d = reader.video_im.get_frame(i)
                writer.write(mask_2d)
                if i % 50 == 0: print(f"Saving frame {i}/{reader.video_im.shape[2]}", end='\r')

            writer.release()
            print(f"\nSaved successfully.")



        filters = "NifTi (*.nii *.nii.gz *.mgz);;tif(*.tif)"
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        currentCS = 'RAS'
        if type.lower()=='View1_IM':
            status = self._check_status_warning_view1()
            if not status:
                return
            try:
                #fl = '.'.join(self.filenameView1.split('.')[:-1])
                fl = self.filenameView1
                flfmt = [el for el in self._availableFormats if el in self.filenameView1]
                flfmt = flfmt[flfmt.index(max(flfmt))]
                fl = fl.replace(flfmt, '')
                newn = fl + '_new'
            except:
                newn = ''
            if hasattr(self.readView1, 'source_system'):
                currentCS = self.readView1.source_system
            fileObj, cs = self._filesave_dialog(filters, opts, newn, currentCS=currentCS)
            if fileObj[0] != '':
                save_as_image(self.readView1, fileObj[0], self.readView1.npImage.astype(np.float32), format=filters.split(';;').index(fileObj[1]), type_im='eco', cs=cs)
        elif type.lower()=='View1_SEG':
            status = self._check_status_warning_view1()
            if not status:
                return
            try:
                #fl = '.'.join(self.filenameView1.split('.')[:-1])
                fl = self.filenameView1
                flfmt = [el for el in self._availableFormats if el in self.filenameView1]
                flfmt = flfmt[flfmt.index(max(flfmt))]
                fl = fl.replace(flfmt, '')
                newn = fl + '_seg'
            except:
                newn = ''
            if hasattr(self.readView1, 'source_system'):
                currentCS = self.readView1.source_system
            fileObj, cs = self._filesave_dialog(filters, opts, newn, currentCS=currentCS)
            if fileObj[0] != '':
                save_as_image(self.readView1, fileObj[0], self.readView1.npSeg.astype(np.int16), format=filters.split(';;').index(fileObj[1]), type_im='eco', cs=cs)
        elif type.lower()=='View2_IM':
            status = self._check_status_warning_view2()
            if not status:
                return
            try:
                #fl = '.'.join(self.filenameView2.split('.')[:-1])
                fl = self.filenameView2
                flfmt = [el for el in self._availableFormats if el in self.filenameView2]
                flfmt = flfmt[flfmt.index(max(flfmt))]
                fl = fl.replace(flfmt, '')
                newn = fl + '_new'
            except:
                newn = ''
            if hasattr(self.readView2, 'source_system'):
                currentCS = self.readView2.source_system
            fileObj, currentCS = self._filesave_dialog(filters, opts, newn, currentCS=currentCS)
            if fileObj[0] != '':
                save_as_image(self.readView2, fileObj[0], self.readView2.npImage.astype(np.float32),format=filters.split(';;').index(fileObj[1]), cs=currentCS)
        elif type.lower()=='View2_SEG':
            status = self._check_status_warning_view2()
            if not status:
                return
            try:
                fl = self.filenameView2
                flfmt = [el for el in self._availableFormats if el in self.filenameView2]
                flfmt = flfmt[flfmt.index(max(flfmt))]
                fl = fl.replace(flfmt, '')
                #fl = '.'.join(self.filenameView2.split('.')[:-1])
                newn = fl + '_seg'
            except:
                newn = ''

            if hasattr(self.readView2, 'source_system'):
                currentCS = self.readView2.source_system
            fileObj, cs = self._filesave_dialog(filters, opts, newn, currentCS=currentCS)
            if fileObj[0]!='':
                save_as_image(self.readView2, fileObj[0], self.readView2.npSeg.astype(np.int16), format=filters.split(';;').index(fileObj[1]), cs=cs)



    def closeImportData(self, type_image):
        """

        :param type_image:
        :return:
        """
        if type_image.lower()=='View1_SEG':
            if hasattr(self, 'readView1'):
                if hasattr(self.readView1, 'npImage'):
                    self.readView1.npSeg = self.readView1.npImage*0
                    self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)
        elif type_image.lower()=='View2_SEG':
            if hasattr(self, 'readView2'):
                if hasattr(self.readView2, 'npImage'):
                    self.readView2.npSeg = self.readView2.npImage*0
                    self.updateDispView2(self.readView2.npImage, self.readView2.npSeg, initialState=True,
                                       tract=self.readView2.tract)



    def importData(self, type_image, fileObj=None):
        """
        Importing image data
        :param type_image:
        :param fileObj:
        :return:
        """

        update_color_s = False # do not update color scheme
        if fileObj is None or type(fileObj)==bool:
            update_color_s = True
            opts = QtWidgets.QFileDialog.DontUseNativeDialog
            dialog = self.get_dialog(settings.DEFAULT_USE_DIR, self._filters, opts, title="Open File")
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                selected_files = dialog.selectedFiles()
                if selected_files:
                    # getOpenFileName returns a tuple (filepath, filter)
                    fileObj = (selected_files[0], dialog.selectedNameFilter())
                else:
                    return False
            else:
                return False
            #if not dialog.exec() == QtWidgets.QDialog.Accepted:
            if fileObj[0]=='':
                return False

            #return True
        try:
            if type_image.lower()=='view1_seg':
                if hasattr(self, 'readView1'):
                    if hasattr(self.readView1, 'npImage'):
                        if self.is_view1_video:
                            reader = self.readView1
                            npSeg, readable, equalDim = reader.read_video_segmentations(fileObj[0])
                            if len(self.readView1.npImage.shape[:2]) != len(npSeg.shape):
                                return False
                        else:
                            npSeg, readable, equalDim = read_segmentation_file(self, fileObj[0], self.readView1, update_color_s=update_color_s)
                            if len(self.readView1.npImage.shape[:3]) != len(npSeg.shape):
                                return False
                        if not equalDim:
                            self.warning_msgbox(
                                'Expected segmentation with dimensions {}, but the segmentation has {}'.format(self.readView1.npImage.shape[:3], npSeg.shape))
                            return False
                        if not readable:
                            self.warning_msgbox('The number of colors are less than the segmentated parts. Unable to read the file.')
                            return False


                        self.readView1.npSeg = npSeg.astype('int')
                        #manually_check_tree_item(self, '9876')

                        self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)
                        #make_all_seg_visibl(self)
                        ls = manually_check_tree_item(self, '9876')
                        self.changeColorPen(self.tree_colors.model().sourceModel().invisibleRootItem().child(ls[0]))
                        return True
            elif type_image.lower()=='view2_seg':
                if hasattr(self, 'readView2'):
                    if hasattr(self.readView2, 'npImage'):
                        if self.is_view2_video:
                            reader = self.readView2
                            npSeg, readable, equalDim = reader.read_video_segmentations(fileObj[0])
                            if len(self.readView2.npImage.shape[:2]) != len(npSeg.shape):
                                return False
                        else:
                            npSeg, readable, equalDim = read_segmentation_file(self, fileObj[0], self.readView2, update_color_s=update_color_s)
                            if len(self.readView2.npImage.shape[:3]) != len(npSeg.shape):
                                return False
                        if not equalDim:
                            self.warning_msgbox(
                                'Expected segmentation with dimensions {}, but the segmentation has {}'.format(self.readView2.npImage.shape[:3], npSeg.shape))
                            return False
                        if not readable:
                            self.warning_msgbox(
                                 'The number of colors are less than the segmentated parts. Unable to read the file.')
                            return False


                        self.readView2.npSeg = npSeg.astype('int')
                        #manually_check_tree_item(self, '9876')

                        self.updateDispView2(self.readView2.npImage, self.readView2.npSeg, initialState=True)
                        ls = manually_check_tree_item(self, '9876')
                        self.changeColorPen(self.tree_colors.model().sourceModel().invisibleRootItem().child(ls[0]))
                        #make_all_seg_visibl(self)
                        return True
        except Exception as e:
            self.warning_msgbox(e)

    def save_screenshot(self, q_img, filename):
        """
        This function export segmentation results to a file.
        :return:
        """

        self.dock_progressbar.setVisible(True)
        self.setEnabled(False)

        self.progressBarSaving.setValue(0)

        save_snapshot(q_img, filename)

        self.setEnabled(True)
        self.dock_progressbar.setVisible(False)
        self.progressBarSaving.setValue(0)



    def loadChanges(self, file_base=None):
        """
        This function loads all previous values.
        Refactored to continue execution even if individual sections fail.
        """
        # --- 1. Imports and Setup ---


        sender = QtCore.QObject.sender(self)

        if file_base is not None and type(file_base) is not bool:
            self._basefileSave = file_base

        if type(self._basefileSave) == bool or self._basefileSave == '' or self._loaded:
            return

        # Initialize settings and UI flags
        try:
            self.settings = QSettings(self._basefileSave + '.ini', self.settings.IniFormat)
            self.openGLWidget_14._updatePaint = False
            self.openGLWidget_24._updatePaint = False
            self.activate3d(True)
        except Exception as e:
            print(f"Error initializing settings: {e}")

        if file_base is None:
            file = self._basefileSave + '.bn'
        else:
            file = file_base

        dic = None
        self.progressBarSaving.setValue(20)

        # --- 2. CRITICAL: Load File Data ---
        # This is the only block that stops the function if it fails,
        # because without 'dic', there is nothing to load.
        if os.path.exists(file) and os.path.getsize(file) > 0:
            try:
                with open(file, 'rb') as inputs:
                    unpickler = pickle.Unpickler(inputs)
                    dic = unpickler.load()
            except Exception as e:
                print(f"Critical Error: Failed to load or decrypt file: {e}")
                return  # Cannot proceed without data

        if dic is None:
            return

        # =========================================================
        # INDEPENDENT LOADING BLOCKS
        # Failures here will NOT stop the rest of the function
        # =========================================================
        try:
            loadAttributeWidget(self, 'main', dic, self.progressBarSaving)
        except Exception as e:
            print(f"Error loading main attributes: {e}")
        if self.full_path_view1:
            self.browse_view1(fileObj=[self.full_path_view1, 'view1'])
        if self.full_path_view2:
            self.browse_view2(fileObj=[self.full_path_view2, 'view2'])
        # --- 3. Load Measurements ---
        try:
            if 'measurements' in dic:
                vals = dic['measurements']
                self.table_widget_measure.setRowCount(len(vals))
                self.table_widget_measure.setColumnCount(8)
                r = 0
                for row in range(len(vals)):
                    for col in range(len(vals[row])):
                        self.table_widget_measure.setItem(row, col, QtWidgets.QTableWidgetItem(vals[row][col]))
                    r += 1
            else:
                self.table_widget_measure.setRowCount(0)
        except Exception as e:
            print(f"Error loading measurements: {e}")

        self.progressBarSaving.setValue(65)

        # --- 4. Load Widgets (Iterative) ---
        name = 'openGLWidget_'
        nameS = 'horizontalSlider_'
        widgets_num = [0, 1, 2, 3, 4, 5, 10, 11, 13, 23]
        self.scroll_intensity.setValue(50)

        for i in widgets_num:
            try:
                nameWidget = name + str(i + 1)
                if hasattr(self, nameWidget):
                    widget = getattr(self, nameWidget)

                    # UI Visibility Logic
                    if i < 13:
                        slider = getattr(self, nameS + str(i + 1))
                        slider.setVisible(True)
                    if i == 10:
                        self.radioButton_1.setVisible(True)
                        self.radioButton_2.setVisible(True)
                        self.radioButton_3.setVisible(True)
                        self.radioButton_4.setVisible(True)
                    elif i == 11:
                        self.radioButton_21_1.setVisible(True)
                        self.radioButton_21_2.setVisible(True)
                        self.radioButton_21_3.setVisible(True)
                        self.radioButton_21.setVisible(True)

                    # Load Widget Data
                    loadAttributeWidget(widget, nameWidget, dic, self.progressBarSaving)

                    # Update Widget View
                    if i < 13:
                        if widget.imSlice is not None:
                            widget.setVisible(True)
                            widget.makeObject()
                            widget.update()
                        else:
                            widget.setVisible(False)
                            slider.setVisible(False)
            except Exception as e:
                print(f"Error loading widget {i + 1}: {e}")

        self.progressBarSaving.setValue(75)
        self.progressBarSaving.setValue(80)
        self.progressBarSaving.setValue(95)


        self.progressBarSaving.setValue(98)

        # --- 10. Final UI Cleanups & Settings ---
        try:


            set_new_color_scheme(self)
            update_widget_color_scheme(self)

            self.openGLWidget_4.imType = 'mri'
            self.openGLWidget_5.imType = 'mri'
            self.openGLWidget_6.imType = 'mri'

            # Reset Color Indices
            widgets = [1, 2, 3, 4, 5, 6, 11, 12, 14, 24]
            prefix = 'openGLWidget_'
            for k in widgets:
                if hasattr(self, prefix + str(k)):
                    getattr(self, prefix + str(k)).colorInds = []

            self.progressBarSaving.setVisible(False)
            self.openGLWidget_14._updatePaint = True
            self.openGLWidget_24._updatePaint = True

            if hasattr(self.settingsBN, 'auto_save_spinbox'):
                self.expectedTime = self.settingsBN.auto_save_spinbox.value() * 60



            # Load Settings Dict
            if "settings" in dic:
                dic_settings = dic["settings"]
                allowed_keys = ["auto_save_interval", "DEFAULT_MODELS_DIR", "DEFAULT_USE_DIR"]
                # Assuming 'settings' is a module or object available in context
                # If 'settings' refers to self.settings, adjust accordingly
                for key in allowed_keys:
                    if key in dic_settings:
                        # Be careful if 'settings' here refers to QSettings or a global module
                        pass

            self.current_view = 'horizontal'

        except Exception as e:
            print(f"Error in final UI cleanup: {e}")

        # --- Finalize ---
        self._loaded = True
        self.openGLWidget_14._updatePaint = True
        self.openGLWidget_24._updatePaint = True

    def loadChanges2(self, file_base=None):
        """
        This function loads all previous values.
        Refactored to continue execution even if individual sections fail.
        """
        # --- 1. Imports and Setup ---


        sender = QtCore.QObject.sender(self)

        if file_base is not None and type(file_base) is not bool:
            self._basefileSave = file_base

        if type(self._basefileSave) == bool or self._basefileSave == '' or self._loaded:
            return

        # Initialize settings and UI flags
        try:
            self.settings = QSettings(self._basefileSave + '.ini', self.settings.IniFormat)
            self.openGLWidget_14._updatePaint = False
            self.openGLWidget_24._updatePaint = False
            self.activate3d(True)
        except Exception as e:
            print(f"Error initializing settings: {e}")

        if file_base is None:
            file = self._basefileSave + '.bn'
        else:
            file = file_base

        dic = None
        self.progressBarSaving.setValue(20)

        # --- 2. CRITICAL: Load File Data ---
        # This is the only block that stops the function if it fails,
        # because without 'dic', there is nothing to load.
        if os.path.exists(file) and os.path.getsize(file) > 0:
            try:
                with open(file, 'rb') as inputs:
                    unpickler = pickle.Unpickler(inputs)
                    dic = unpickler.load()
            except Exception as e:
                print(f"Critical Error: Failed to load or decrypt file: {e}")
                return  # Cannot proceed without data

        if dic is None:
            return

        # =========================================================
        # INDEPENDENT LOADING BLOCKS
        # Failures here will NOT stop the rest of the function
        # =========================================================

        # --- 3. Load Measurements ---
        try:
            if 'measurements' in dic:
                vals = dic['measurements']
                self.table_widget_measure.setRowCount(len(vals))
                self.table_widget_measure.setColumnCount(8)
                r = 0
                for row in range(len(vals)):
                    for col in range(len(vals[row])):
                        self.table_widget_measure.setItem(row, col, QtWidgets.QTableWidgetItem(vals[row][col]))
                    r += 1
            else:
                self.table_widget_measure.setRowCount(0)
        except Exception as e:
            print(f"Error loading measurements: {e}")

        self.progressBarSaving.setValue(65)

        # --- 4. Load Widgets (Iterative) ---
        name = 'openGLWidget_'
        nameS = 'horizontalSlider_'
        widgets_num = [0, 1, 2, 3, 4, 5, 10, 11, 13, 23]
        self.scroll_intensity.setValue(50)

        for i in widgets_num:
            try:
                nameWidget = name + str(i + 1)
                if hasattr(self, nameWidget):
                    widget = getattr(self, nameWidget)

                    # UI Visibility Logic
                    if i < 13:
                        slider = getattr(self, nameS + str(i + 1))
                        slider.setVisible(True)
                    if i == 10:
                        self.radioButton_1.setVisible(True)
                        self.radioButton_2.setVisible(True)
                        self.radioButton_3.setVisible(True)
                        self.radioButton_4.setVisible(True)
                    elif i == 11:
                        self.radioButton_21_1.setVisible(True)
                        self.radioButton_21_2.setVisible(True)
                        self.radioButton_21_3.setVisible(True)
                        self.radioButton_21.setVisible(True)

                    # Load Widget Data
                    loadAttributeWidget(widget, nameWidget, dic, self.progressBarSaving)

                    # Update Widget View
                    if i < 13:
                        if widget.imSlice is not None:
                            widget.setVisible(True)
                            widget.makeObject()
                            widget.update()
                        else:
                            widget.setVisible(False)
                            slider.setVisible(False)
            except Exception as e:
                print(f"Error loading widget {i + 1}: {e}")

        self.progressBarSaving.setValue(75)

        # --- 5. Initialize Image Readers ---
        try:
            names = ['readView1', 'readView2']
            for name in names:
                # Re-initialize readers
                if name == 'readView1':
                    imtype = 'eco'
                else:
                    imtype = 't1'

                # Note: readData needs to be available in context
                # Assuming it is a method of self or imported globally
                setattr(self, name, self.readData(type=imtype) if hasattr(self, 'readData') else None)

                readD = getattr(self, name)
                if readD:
                    loadAttributeWidget(readD, name, dic, self.progressBarSaving)
        except Exception as e:
            print(f"Error initializing image readers: {e}")

        self.progressBarSaving.setValue(80)
        self.app.processEvents()

        # --- 6. Setup ECO (View 1) ---
        try:
            if hasattr(self, 'readView1') and hasattr(self.readView1, 'npImage'):
                self.tree_colors.setVisible(True)
                self.readView1.npSeg = self.readView1.npSeg.astype('int')

                self.ImageEnh_view1.setVisible(True)
                self.readView1.manuallySetIms('eco')
                self.setNewImage_view1.emit(self.readView1.npImage.shape[:3])
                self.openGLWidget_14.load_paint(self.readView1.npSeg)
                self.tabWidget.setTabVisible(1, True)

                # Sync Affine
                for i in widgets_num:
                    widget = getattr(self, 'openGLWidget_' + str(i + 1))
                    if i < 13 and widget.imSlice is not None and hasattr(widget, 'affine') and hasattr(self.readView2,
                                                                                                       'affine'):
                        if widget.imType == 'eco':
                            widget.affine = self.readView2.affine

                self.updateDispView1(self.readView1.npImage, self.readView1.npSeg, initialState=True)

                if not hasattr(self.readView1, 'npEdge'):
                    self.readView1.npEdge = []
        except Exception as e:
            print(f"Error setting up ECO/View1: {e}")

        self.app.processEvents()

        # --- 7. Setup MRI (View 2) ---
        try:
            if hasattr(self, 'readView2') and hasattr(self.readView2, 'npImage'):
                self.readView2.npSeg = self.readView2.npSeg.astype('int')

                self.tabWidget.setTabVisible(2, True)
                self.page1_mri.setVisible(True)
                self.tree_colors.setVisible(True)
                self.readView2.manuallySetIms('t1')
                self.setNewImage_view2.emit(self.readView2.npImage.shape[:3])

                # Sync Affine
                for i in widgets_num:
                    widget = getattr(self, 'openGLWidget_' + str(i + 1))
                    if i < 13 and widget.imSlice is not None and hasattr(widget, 'affine') and hasattr(self.readView2,
                                                                                                       'affine'):
                        if widget.imType == 'mri':
                            widget.affine = self.readView2.affine

                self.updateDispView2(self.readView2.npImage, self.readView2.npSeg, initialState=True,
                                   tract=self.readView2.tract)

                if not hasattr(self.readView2, 'npEdge'):
                    self.readView2.npEdge = []

                # Setup ComboBox for Dimensions
                if hasattr(self.readView2, 'ims'):
                    shape = self.readView2.ims.shape
                    self.actionComboBox.setObjectName("View2")
                    try:
                        self.actionComboBox.currentTextChanged.disconnect(self.changeVolume)
                    except:
                        pass

                    self.actionComboBox.clear()
                    for r in range(shape[-1]):
                        self.actionComboBox.addItem("{}".format(r + 1))

                    try:
                        self.actionComboBox.currentTextChanged.connect(self.changeVolume)
                    except:
                        pass

                    self.actionComboBox_visible.setVisible(True)
                    self.actionComboBox_visible.setDisabled(False)
                else:
                    self.actionComboBox_visible.setVisible(False)
                    self.actionComboBox_visible.setDisabled(True)
        except Exception as e:
            print(f"Error setting up MRI/View2: {e}")

        self.progressBarSaving.setValue(95)
        self.app.processEvents()

        # --- 8. Load Main Attributes & Reset Tree ---
        try:
            loadAttributeWidget(self, 'main', dic, self.progressBarSaving)
            self.imported_images = []
            self.tree_images.model().sourceModel().clear()
            self.tree_images.model().sourceModel().setColumnCount(2)
            self.tree_images.model().sourceModel().setHorizontalHeaderLabels(['Index', 'Name'])
        except Exception as e:
            print(f"Error loading main attributes: {e}")

        # --- 9. Update Info Dialogs ---
        try:

            # MRI Info
            if hasattr(self, 'readView2') and hasattr(self.readView2, 'im'):
                if self.filenameView1:
                    file_out = 'View 1: {}, View 2: {}'.format(self.filenameView1, self.filenameView2)
                else:
                    file_out = 'View 2: {}'.format(self.filenameView2)

                info1, color1 = [[[self.filenameView2], "*View 2 (loaded)"], 2, 1], [1, 1, 0]
                update_image_sch(self, info=info1, color=color1, loaded=True)
                self.iminfo_dialog.updata_name_iminfo(self.filenameView2, 1)

                if hasattr(self.readView2, 'im_metadata'):
                    self.iminfo_dialog.set_tag_value(self.readView2, ind=1)

            # ECO Info
            if hasattr(self, 'readView1') and hasattr(self.readView1, 'im'):
                self.iminfo_dialog.updata_name_iminfo(self.filenameView1, 0)
                info1, color1 = [[[self.filenameView1], "*View 1 (loaded)"], 0, 0], [0, 1, 1]
                update_image_sch(self, info=info1, color=color1, loaded=True)

                if hasattr(self.readView1, 'im_metadata'):
                    self.iminfo_dialog.set_tag_value(self.readView1, ind=0)
        except Exception as e:
            print(f"Error updating info dialogs: {e}")

        self.progressBarSaving.setValue(98)

        # --- 10. Final UI Cleanups & Settings ---
        try:

            adapt_previous_versions(self)
            set_new_color_scheme(self)
            update_widget_color_scheme(self)

            self.t1_5.setValue(0)
            self.lb_t1_5.setText('0')
            self.t2_5.setValue(0)
            self.lb_t2_5.setText('0')

            self.openGLWidget_4.imType = 'mri'
            self.openGLWidget_5.imType = 'mri'
            self.openGLWidget_6.imType = 'mri'

            # Reset Color Indices
            widgets = [1, 2, 3, 4, 5, 6, 11, 12, 14, 24]
            prefix = 'openGLWidget_'
            for k in widgets:
                if hasattr(self, prefix + str(k)):
                    getattr(self, prefix + str(k)).colorInds = []

            self.progressBarSaving.setVisible(False)
            self.openGLWidget_14._updatePaint = True
            self.openGLWidget_24._updatePaint = True

            if hasattr(self.settingsBN, 'auto_save_spinbox'):
                self.expectedTime = self.settingsBN.auto_save_spinbox.value() * 60

            if hasattr(self, 'lb_scrol_rad_circle'):
                self.scrol_rad_circle.setValue(int(self.lb_scrol_rad_circle.text()))

            # Load Settings Dict
            if "settings" in dic:
                dic_settings = dic["settings"]
                allowed_keys = ["auto_save_interval", "DEFAULT_MODELS_DIR", "DEFAULT_USE_DIR"]
                # Assuming 'settings' is a module or object available in context
                # If 'settings' refers to self.settings, adjust accordingly
                for key in allowed_keys:
                    if key in dic_settings:
                        # Be careful if 'settings' here refers to QSettings or a global module
                        pass

            try:
                self.scrol_tol_rad_circle.setValue(int(self.lb_scrol_tol_rad_circle.text()))
            except ValueError:
                self.scrol_tol_rad_circle.setValue(0)

            self.current_view = 'horizontal'

        except Exception as e:
            print(f"Error in final UI cleanup: {e}")

        # --- Finalize ---
        self._loaded = True
        self.openGLWidget_14._updatePaint = True
        self.openGLWidget_24._updatePaint = True


    def browse_view1(self, fileObj=None, use_dialog=True):
        """
        Browsing/Loading MRI, Ultrasound, or Video files (Refactored).
        """
        self.init_state()

        # --- STEP 1: LOAD FILE ---
        if use_dialog or fileObj is not None:
            if not isinstance(fileObj, list):
                # Show Dialog if no file provided
                fileObj, index = self._show_file_dialog()
                if not fileObj: return False

                # Save state

            else:
                # File object passed directly (e.g. from Drag & Drop)
                fileObj, index = fileObj, fileObj[1] if len(fileObj) > 1 else 2
            self._last_index_select_image_view1 = index
            # Read Data based on selected type
            success = self._load_data_by_type(fileObj, index, index_view = 0)
            if not success: return False

        # --- STEP 2: SETUP VIDEO VS STANDARD MODE ---
        # Check if loaded data is a Video Proxy
        self.is_view1_video = getattr(self.readView1, 'isVideo', False) or \
                              getattr(self.readView1, 'isChunkedVideo', False)

        if self.is_view1_video:
            self._configure_video_ui(self.readView1, name_tab= 'segmentationTab', ind_use = [0, 1, 2, 10, 13], index_tab=0)
        else:
            self._configure_standard_ui(self.readView2, index_tab=0) # update second view too


        # --- STEP 3: CONFIGURE VIEWERS ---
        # Handle multi-dimensional volumes (4D)
        self._setup_multidim_combobox(self.readView1, use_dialog)

        # If data loaded successfully, initialize display
        if hasattr(self.readView1, 'npImage'):

            self._initialize_display_widgets(self.readView1, str_name_file = 'filenameView1', fileObj = fileObj, use_dialog = use_dialog,
            index_tab = 0, widgets_to_reset = [0, 1, 2, 10], view_text = 'View 1')

            # Trigger external plugins / signals
            self.setNewImage_view1.emit(self.readView1.npImage.shape[:3])
            self.toolBar2.setDisabled(False)

            # Initial text update
            self._update_volume_text_info(index_view = 0)

        else:
            # Hide everything if load failed logically
            self._set_widgets_visible(False, index_view = 0)

        # --- STEP 4: CLEANUP & PLUGINS ---
        self.activateGuidelines(self._last_state_guide_lines)
        thrsh = 50
        if self.scroll_intensity.value()==thrsh:
            self.ColorIntensityChange(thrsh, 'seg')
        else:
            self.scroll_intensity.setValue(thrsh)
        # Notify Plugins
        data_context = self.get_current_image_data()
        for plugin in self.plugin_widgets:
            plugin.update_data_context(data_context)

        return True


    def browse_view2(self, fileObj=None, use_dialog=True):
        """
        Browsing/Loading MRI, Ultrasound, or Video files (Refactored).
        """
        self.init_state()

        # --- STEP 1: LOAD FILE ---
        if use_dialog or fileObj is not None:
            if not isinstance(fileObj, list):
                # Show Dialog if no file provided
                fileObj, index = self._show_file_dialog()
                if not fileObj: return False

                # Save state

            else:
                # File object passed directly (e.g. from Drag & Drop)
                fileObj, index = fileObj, fileObj[1] if len(fileObj) > 1 else 2
            self._last_index_select_image_view1 = index
            # Read Data based on selected type
            success = self._load_data_by_type(fileObj, index, index_view = 1)
            if not success: return False

        # --- STEP 2: SETUP VIDEO VS STANDARD MODE ---
        # Check if loaded data is a Video Proxy
        self.is_view2_video = getattr(self.readView2, 'isVideo', False) or \
                              getattr(self.readView2, 'isChunkedVideo', False)

        if self.is_view2_video:
            self._configure_video_ui(self.readView2, name_tab= 'MRISegTab', ind_use = [3, 4, 5, 11, 23], index_tab=1)
        else:
            self._configure_standard_ui(self.readView1, index_tab=1) # update second view too

        # --- STEP 3: CONFIGURE VIEWERS ---
        # Handle multi-dimensional volumes (4D)
        self._setup_multidim_combobox(self.readView2, use_dialog)

        # If data loaded successfully, initialize display
        if hasattr(self.readView2, 'npImage'):

            self._initialize_display_widgets(self.readView2, str_name_file = 'filenameView2', fileObj = fileObj, use_dialog = use_dialog,
            index_tab = 1, widgets_to_reset = [3, 4, 5, 11], view_text = 'View 2')

            # Trigger external plugins / signals
            self.setNewImage_view2.emit(self.readView2.npImage.shape[:3])
            self.toolBar2.setDisabled(False)

            # Initial text update
            self._update_volume_text_info(index_view=1)

        else:
            # Hide everything if load failed logically
            self._set_widgets_visible(False, index_view = 1)

        # --- STEP 4: CLEANUP & PLUGINS ---
        self.activateGuidelines(self._last_state_guide_lines)
        thrsh = 50
        if self.scroll_intensity.value()==thrsh:
            self.ColorIntensityChange(thrsh, 'seg')
        else:
            self.scroll_intensity.setValue(thrsh)
        # Notify Plugins
        data_context = self.get_current_image_data()
        for plugin in self.plugin_widgets:
            plugin.update_data_context(data_context)

        return True


    # ------------------------------------------------------------------
    # HELPER METHODS (Add these to your class to keep code clean)
    # ------------------------------------------------------------------

    def _show_file_dialog(self):
        """Handles the File Dialog logic."""
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        dialg = QFileDialogPreview(self, "Open File", settings.DEFAULT_USE_DIR,
                                   self._filters, options=opts,
                                   index=self._last_index_select_image_view2,
                                   last_state=self._last_state_preview) #TODO
        dialg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        # Resize dialog nicely
        p_size = self.size()
        dialg.resize(int(p_size.width() * 0.6), int(p_size.height() * 0.7))

        if dialg.exec_() == QFileDialogPreview.Accepted:
            fileObj = dialg.getFileSelected()
            fileObj[1] = dialg.selectedNameFilter()
            self._last_state_preview = dialg.checkBox_preview.isChecked()
            return fileObj, dialg._combobox_type.currentIndex()

        return None, -1

    def _load_data_by_type(self, fileObj, index, index_view=0):
        """Reads the file into self.readView1 based on index type."""
        try:
            reader, Info, file_full_path = self.readD(fileObj, 'eco', target_system='RAS')
            if Info[2].lower() != 'success':
                return False
            if index_view == 0:
                self.readView1 = reader
                self.full_path_view1 = file_full_path
            elif index_view == 1:
                self.readView2 = reader
                self.full_path_view2 = file_full_path



            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def _configure_video_ui(self, reader, name_tab= 'segmentationTab', ind_use = [0, 1, 2, 13], index_tab=0):
        """Disables unnecessary tabs and sets up Video Slider."""
        main_tab = getattr(self, name_tab, None)
        main_tab_index = self.tabWidget.indexOf(main_tab)

        # Disable non-relevant tabs
        #for i in range(self.tabWidget.count()):
         #   enabled = (i == main_tab_index)
         #   self.tabWidget.setTabEnabled(i, enabled)
         #   self.tabWidget.setTabToolTip(i, "Not available for Video" if not enabled else "")

        # Hide 3D Slice Views (Axial, Sagittal, Coronal)
        for k in ind_use:  # Indices of standard views
            getattr(self, f'openGLWidget_{k + 1}').setVisible(False)

        # Set context name
        if index_tab == 0:
            self.openGLWidget_11.currentWidnowName = "video"
        elif index_tab==1:
            self.openGLWidget_12.currentWidnowName = "video"

        self.tabWidget.setCurrentIndex(main_tab_index)
        self._format = 'video'



        # Setup Video Slider
        if hasattr(reader, 'seg_ims'):
            if index_tab==0:
                slider = self.horizontalSlider_11

            elif index_tab==1:
                slider = self.horizontalSlider_12
            slider.blockSignals(True)
            slider.setRange(0, reader.seg_ims.shape[-1])
            slider.blockSignals(False)
            slider.setValue(reader.current_frame)




    def _configure_standard_ui(self, reader, index_tab=0):
        """Restores UI for standard 3D MRI/Ultrasound."""
        # Re-enable all tabs
        for i in range(self.tabWidget.count()):
            self.tabWidget.setTabEnabled(i, True)
            self.tabWidget.setTabToolTip(i, "")

        # Ensure View2 widgets are visible if data exists
        if index_tab==0:
            if hasattr(self, 'readView2') and hasattr(self.readView2, 'npImage'):
                for k in [3, 4, 5]:
                    getattr(self, f'openGLWidget_{k + 1}').setVisible(True)
                self.updateDispView2(reader.npImage, reader.npSeg, initialState=True)
            self.openGLWidget_11.currentWidnowName = "coronal"
        elif index_tab==1:
            if hasattr(self, 'readView1') and hasattr(self.readView1, 'npImage'):
                for k in [0, 1, 2]:
                    getattr(self, f'openGLWidget_{k + 1}').setVisible(True)
                self.updateDispView1(reader.npImage, reader.npSeg, initialState=True)
            self.openGLWidget_12.currentWidnowName = "coronal"

    def _setup_multidim_combobox(self, reader, use_dialog):
        """Handles the combobox for 4D datasets."""
        if use_dialog:
            self.actionComboBox_visible.setVisible(False)
            self.actionComboBox_visible.setDisabled(True)

        if hasattr(reader, 'ims') and use_dialog:
            _num_dims = reader._num_dims
            if _num_dims > 1:
                try:
                    self.actionComboBox.currentTextChanged.disconnect(self.changeVolume)
                except:
                    pass

                self.actionComboBox.clear()
                for r in range(_num_dims):
                    self.actionComboBox.addItem(str(r + 1))

                self.actionComboBox_visible.setDisabled(False)
                self.actionComboBox_visible.setVisible(True)
                self.actionComboBox.currentTextChanged.connect(self.changeVolume)

    def _initialize_display_widgets(self, reader, str_name_file = 'filenameView1', fileObj=None, use_dialog=None,
                                    index_tab = 0, widgets_to_reset = [0, 1, 2, 10], view_text = 'View 1'):
        """Sets up the OpenGL Widgets, Colors, and Metadata."""
        self.format_view1 = self._format
        self._set_widgets_visible(True, index_view = index_tab)

        # Set Filename
        if fileObj is not None:
            setattr(self, str_name_file, getattr(reader, '_fileDicom', basename(fileObj[0])))
        name_file = getattr(self, str_name_file)

        # Update Metadata Info Dialog
        self.iminfo_dialog.updata_name_iminfo(name_file, index_tab)
        if hasattr(reader, 'im_metadata'):
            self.iminfo_dialog.set_tag_value(reader, ind=index_tab)

        # Set Colors
        ls = manually_check_tree_item(self, '9876')
        if ls:
            item = self.tree_colors.model().sourceModel().invisibleRootItem().child(ls[0])
            self.changeColorPen(item)

        # Reset & Update OpenGL Widgets
        self.tabWidget.setTabVisible(1, True)


        for k in widgets_to_reset:
            widget = getattr(self, f'openGLWidget_{k + 1}')
            if widget.isVisible():
                widget.resetInit()
                widget.initialState()
                # Apply affine if available
                widget.affine = getattr(reader, "im_metadata", {}).get("Affine", widget.affine)

        # Final Paint
        if index_tab==0:
            self.updateDispView1(reader.npImage, reader.npSeg, initialState=True)
        elif index_tab==1:
            self.updateDispView2(reader.npImage, reader.npSeg, initialState=True)
        self.changedTab()
        self.reset_view_pages(index= index_tab)

        # History / Recent Files Update
        if use_dialog and fileObj:
            self.HistImage.UpdateName(name_file, None)
            from melage.utils.utils import update_image_sch, clean_parent_image2
            info1 = [[[fileObj[0]], fileObj[1]], 2, 0]
            update_image_sch(self, info=info1, color=[1, 1, 0], loaded=True)
            clean_parent_image2(self, fileObj[0], view_text, index_view=index_tab)

    def _set_widgets_visible(self, val, index_view=0):
        """Helper to show/hide standard control groups."""
        if index_view ==0:
            self.actionImportSegView1.setDisabled(not val)
            self.actionExportImView1.setDisabled(not val)
            self.actionExportSegView1.setDisabled(not val)
            self.btn_play_view1.setVisible(True)
            for rb in [self.radioButton_1, self.radioButton_2, self.radioButton_3, self.radioButton_4]:
                rb.setVisible(val)

            for k in [1, 2, 3, 11]:
                getattr(self, f'label_{k}').setVisible(val)
                getattr(self, f'horizontalSlider_{k}').setVisible(val)
                getattr(self, f'openGLWidget_{k}').setVisible(val)
            self.openGLWidget_14.setVisible(val)
            if val:  # Reset rotations if showing
                self._rotationAngleView1_coronal = 0
                self._rotationAngleView1_axial = 0
                self._rotationAngleView1_sagittal = 0
                self.t1_5.setValue(0)
        elif index_view ==1:
            self.actionImportSegView2.setDisabled(not val)
            self.actionExportImView2.setDisabled(not val)
            self.actionExportSegView2.setDisabled(not val)
            self.btn_play_view2.setVisible(True)

            for rb in [self.radioButton_21, self.radioButton_21_1, self.radioButton_21_2, self.radioButton_21_2]:
                rb.setVisible(val)

            for k in [4, 5, 6, 12]:
                getattr(self, f'label_{k}').setVisible(val)
                getattr(self, f'horizontalSlider_{k}').setVisible(val)
                getattr(self, f'openGLWidget_{k}').setVisible(val)
            self.openGLWidget_24.setVisible(val)
            if val:  # Reset rotations if showing
                self._rotationAngleView2_coronal = 0
                self._rotationAngleView2_axial = 0
                self._rotationAngleView2_sagittal = 0
                self.t2_5.setValue(0)

    def _update_volume_text_info(self, index_view =0):
        """Updates the text showing volume calculation."""
        if index_view == 0:
            txt = compute_volume(self.readView1, self.filenameView1, [9876],
                                 in_txt=self.openedFileName.text(), ind_screen=index_view)
        elif index_view == 1:
            txt = compute_volume(self.readView2, self.filenameView2, [9876],
                                 in_txt=self.openedFileName.text(), ind_screen=index_view)
        self.openedFileName.setText(txt)



    def browseTractoGraphy(self, fileObj=None):
        """
        Browsing Tractography images
        :param fileObj:
        :return:
        """

        # Browse Tractography
        if not hasattr(self,'readView2'):
            return
        if not hasattr(self.readView2, 'npImage'):
            return
        if fileObj is None or type(fileObj) is bool:
            opts =QtWidgets.QFileDialog.DontUseNativeDialog
            fileObj = QtWidgets.QFileDialog.getOpenFileName( self, "Open File", settings.DEFAULT_USE_DIR, 'track(*.trk)', options=opts)

        if fileObj[0]=='':
            return
        stk, success = load_trk(fileObj[0])
        #from melage.utils.utils import get_affine_shape
        if not hasattr(self.readView2, 'affine'):
            self.readView2.afine = self.readView2.im.affine
        if not success:
            return

        vox_world = get_world_from_trk(stk.streamlines, self.readView2.affine, inverse=True)
        self.readView2.tract = vox_world


        if hasattr(self, 'readView2'):
            if hasattr(self.readView2, 'npSeg'):
                widgets_num = [0, 1, 2, 11]
                for k in widgets_num:
                    name = 'openGLWidget_' + str(k + 1)
                    widget = getattr(self, name)
                    widget.resetInit()
                    widget.initialState()
                widgets_num = [3, 4, 5]
                name_slider = 'horizontalSlider_'
                for k in widgets_num:
                    name = 'openGLWidget_' + str(k+1)
                    slider = getattr(self, name_slider + str(k+1))
                    widget = getattr(self, name)
                    sliceNum = slider.value()

                    widget.updateInfo(*getCurrentSlice(widget,
                                                       self.readView2.npImage, self.readView2.npSeg, sliceNum, self.readView2.tract, tol_slice=self.tol_trk), sliceNum, self.readView2.npImage.shape[:3],
                                      initialState=False, imSpacing=self.readView2.ImSpacing)
                    widget.makeObject()
                    widget.update()



    def browseFA(self, fileObj=None):

        # Browse Fractional Anisotropy
        if fileObj is None or type(fileObj) is bool:
            opts =QtWidgets.QFileDialog.DontUseNativeDialog
            fileObj = QtWidgets.QFileDialog.getOpenFileName( self, "Open File", settings.DEFAULT_USE_DIR, self._filters, options=opts)
        if fileObj[0]=='':
            return
        Img, _ = self.readD(fileObj, type='t1')
        npImage = Img.npImage.astype('int')
        npImage[Img.npImage <=20] = 0
        npImage[(Img.npImage <= 40) * (Img.npImage > 20)] = 1
        npImage[(Img.npImage <= 60) * (Img.npImage > 40)] = 2
        npImage[(Img.npImage <= 80) * (Img.npImage > 60)] = 3
        npImage[(Img.npImage <= 100) * (Img.npImage > 80)] = 4
        npImage[(Img.npImage > 100)] = 5
        #npImage[Img.npImage <= 10] = 0
        if hasattr(self, 'readView2'):
            if hasattr(self.readView2, 'npSeg'):
                widgets_num = [0, 1, 2, 11]
                for k in widgets_num:
                    name = 'openGLWidget_' + str(k + 1)
                    widget = getattr(self, name)
                    widget.resetInit()
                    widget.initialState()
                self.readView2.npSeg = npImage
                widgets_num = [3, 4, 5]
                name_slider = 'horizontalSlider_'
                for k in widgets_num:
                    name = 'openGLWidget_' + str(k+1)
                    slider = getattr(self, name_slider + str(k+1))
                    widget = getattr(self, name)
                    sliceNum = slider.value()

                    widget.updateInfo(*getCurrentSlice(widget,
                                                       self.readView2.npImage, self.readView2.npSeg, sliceNum, self.readView2.tract, tol_slice=self.tol_trk), sliceNum, self.readView2.npImage.shape[:3],
                                      initialState=False, imSpacing=self.readView2.ImSpacing)
                    widget.makeObject()
                    widget.update()


    def warning_msgbox(self, text= None): # warning message box
        if text is None:
            text = 'There is an error.'
        MessageBox = QtWidgets.QMessageBox(self)
        MessageBox.setText(text)
        MessageBox.setWindowTitle('Warning')
        MessageBox.show()

    def readInfo(self, Info):
        """
        Reading information form images
        :param Info:
        :return:
        """
        if Info[2] == 'No file':
            MessageBox = QtWidgets.QMessageBox(self)
            if Info[0] and not Info[1]:
                MessageBox.setText('Read JSON metadata, Image data not found')
                MessageBox.setWindowTitle('Reading Warning')
            elif not Info[0] and not Info[1]:
                MessageBox.setText('JSON metadata data not found, Image data not found')
                MessageBox.setWindowTitle('Reading Error')
            elif not Info[0] and Info[1]:
                MessageBox.setText('JSON metadata data not found, Read Image data')

                MessageBox.setWindowTitle('Reading Warning')
            MessageBox.setWindowTitle('Reading Error')
            MessageBox.show()

    def readD(self, fileObj, type = 'eco', target_system='IPL'):
        if '_loaded' in fileObj[1]:
            return [], [False, False, 'No file']
        if fileObj[1] != '':
            self.settingsBN.update_use_dir(dirname(fileObj[0]))
            matched_filter = get_filter_for_file(fileObj[0], self._filters)
            filters = self._filters.split(';;')
            index_sel = filters.index(matched_filter)
            outfile_format = self._filters.split(';;')[index_sel].lower()
            filters[0], filters[index_sel] = filters[index_sel], filters[0]
            self._filters = ';;'.join(filters)
            #if type == 'eco':
            #    self.filenameView1 = basename(fileObj[0])
            #else:
            #    self.filenameView2 = basename(fileObj[0])
            file_read_name = fileObj[0]
            filename = basename(file_read_name)
            _, file_extension = os.path.splitext(fileObj[0])
            self.file_extension = file_extension
            self.setCursor(QtCore.Qt.WaitCursor)
            readIM = readData(target_system=target_system)
            format = 'None'
            try:
                if 'vol' in outfile_format: # Kretz data
                    Info = readIM.readKretz(join(settings.DEFAULT_USE_DIR, filename))
                    format = 'VOL'
                elif 'nii' in outfile_format: # read NIFTI
                    Info = readIM.readNIFTI(join(settings.DEFAULT_USE_DIR, filename), type)
                    format = 'NIFTI'
                elif 'nrrd' in outfile_format: # Read NRRD
                    Info = readIM.readNRRD(join(settings.DEFAULT_USE_DIR, filename), type)
                    format = 'NRRD'
                elif 'dcm' in outfile_format or "dicomdir" in outfile_format: # READ DICOM
                    Info = readIM.readDICOM(join(settings.DEFAULT_USE_DIR, filename), type)
                    format = 'DICOM'
                elif "video" in outfile_format:
                    Info = readIM.readNIFTI(join(settings.DEFAULT_USE_DIR, filename), type)
                    format = 'Video'
                else:
                    Info = [False, False, 'No file']
                if Info[1]==True:
                    self._format = format
                self.setCursor(QtCore.Qt.ArrowCursor)
            except Exception as e:
                print(e)
                Info = [False, False, 'No file']
            self.setCursor(QtCore.Qt.ArrowCursor)

            return readIM, Info, file_read_name
        else:
            return [], [False, False, 'No file'], None







class MainWindow0(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self, *args, obj = None, **kwargs):
        super(MainWindow0, self).__init__(*args, **kwargs)
        QWidget.__init__(self)
        self.setupUi(self)
        #QtCore.QTimer.singleShot(5000, self.showChildWindow)






if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow0()
    window.show()
    sys.exit(app.exec_())
