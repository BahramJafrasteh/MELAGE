__AUTHOR__ = 'Bahram Jafrasteh'


from melage.utils.utils import help_dialogue_open_image
import re
from PyQt5.QtGui import QPixmap, QImage
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QCheckBox, QSpinBox, QGridLayout, QComboBox, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal


class QFileDialogPreview(QFileDialog):
    """
    Custom File Dialog that parses the input filter string to automatically
    create an "All Supported Files" option.
    """

    def __init__(self, parent=None, caption="", directory="", filter="", options=None, index=0, last_state=False):
        # 1. Initialize the base QFileDialog
        # We pass the parent, caption, and directory.
        # We do NOT pass 'filter' yet, because we want to modify it first.
        super().__init__(parent, caption, directory, "")

        # 2. Auto-Detect Logic: Create "All Supported Files"
        if filter:
            # Step A: Find everything inside parentheses e.g. "*.dcm **"
            # r'\(([^)]+)\)' captures all text between ( and )
            raw_groups = re.findall(r'\(([^)]+)\)', filter)

            valid_extensions = []

            for group in raw_groups:
                # Step B: Split the group into individual items
                # " *.dcm  ** " -> ["*.dcm", "**"]
                exts = group.split()

                for ext in exts:
                    # Step C: STRICTLY EXCLUDE '**'
                    if ext.strip() == '**':
                        continue

                    valid_extensions.append(ext)

            # Step D: Remove duplicates and join
            # Using set() removes duplicates, sorted() keeps the order consistent
            unique_exts = sorted(list(set(valid_extensions)))
            all_string = " ".join(unique_exts)

            if all_string:
                final_filter = f"All Supported Files ({all_string});;{filter}"
                self.setNameFilter(final_filter)
            else:
                # Fallback if no valid extensions found
                self.setNameFilter(filter)

        # 3. Handle Options (passed as kwarg in your call)
        if options:
            self.setOptions(options)
            # Ensure we keep specific flags like DontUseNativeDialog if options overwrote them
            self.setOption(QFileDialog.DontUseNativeDialog, True)
        else:
            self.setOption(QFileDialog.DontUseNativeDialog, True)

        # 4. Standard Setup
        self.setFileMode(QFileDialog.ExistingFile)
        self.setAcceptMode(QFileDialog.AcceptOpen)

        # 3. UI Setup (Preview Panel)
        self._init_preview_ui(last_state, index)

        # 4. Connections
        self.currentChanged.connect(self.onChange)
        self.fileSelected.connect(self.onFileSelected)
        self.filesSelected.connect(self.onFilesSelected)

        # Internal state
        self._fileSelected = None
        self._filesSelected = None

        for combo in self.findChildren(QtWidgets.QComboBox):
            if "All Supported" in combo.currentText():
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
                sizePolicy.setHorizontalStretch(1)
                combo.setSizePolicy(sizePolicy)
                combo.setMinimumWidth(150)  # Stop it from disappearing

    def _init_preview_ui(self, last_state, index):
        """Helper to build the side panel UI"""

        # Container Layout for the side panel
        self.side_panel_layout = QHBoxLayout()
        self.side_panel_layout.setContentsMargins(0, 0, 0, 0)



        # C. Advanced Options Group
        adv_layout = QHBoxLayout()

        self.checkBox_advanced = QtWidgets.QCheckBox('Advanced', self)
        self.checkBox_advanced.setObjectName("checkBox")
        self.checkBox_advanced.stateChanged.connect(self.activate_combobox)

        self._combobox_type = QtWidgets.QComboBox(self)
        self.systems = ['US (neonatal)', 'US (fetal)', 'MRI']
        for ss in self.systems:
            self._combobox_type.addItem(f"   {ss}   ")
        self._combobox_type.setCurrentIndex(index)
        self._combobox_type.setEnabled(False)  # Default disabled until checked

        adv_layout.addWidget(self.checkBox_advanced)
        adv_layout.addWidget(self._combobox_type)

        self.side_panel_layout.addLayout(adv_layout)

        # A. Preview Label
        self.mpPreview = QLabel("Preview", self)
        self.mpPreview.setMinimumSize(300, 300)
        self.mpPreview.setAlignment(Qt.AlignCenter)
        self.mpPreview.setStyleSheet("border: 1px solid #aaa; background-color: #eee;")  # Visual cue
        self.mpPreview.setScaledContents(False)  # We handle scaling manually for aspect ratio


        self.mpPreview.setVisible(last_state)

        # B. Preview Checkbox
        self.checkBox_preview = QtWidgets.QCheckBox('Show Preview', self)
        self.checkBox_preview.setChecked(last_state)

        self.checkBox_preview.stateChanged.connect(self.on_toggle_preview)

        self.side_panel_layout.addWidget(self.checkBox_preview)
        self.side_panel_layout.addWidget(self.mpPreview)


        self.side_panel_layout.addStretch()  # Push everything up

        # D. Inject into QFileDialog Grid
        # QFileDialog uses a QGridLayout. (1, 3) places it to the right of the file list.
        # We assume the internal layout is a QGridLayout (standard in PyQt5).
        layout = self.layout()
        if layout:
            layout.addLayout(self.side_panel_layout, 4, 1, 1, 1)

    def on_toggle_preview(self, state):
        """
        Slot called when 'Show Preview' is checked/unchecked.
        state: 0 (Unchecked) or 2 (Checked)
        """
        is_checked = (state == Qt.Checked)

        # 1. Hide/Show the actual label widget
        self.mpPreview.setVisible(is_checked)

        # 2. If turning ON, force an update immediately
        # (Otherwise the preview would be empty until the user clicks a new file)
        if is_checked:
            # Get the file currently highlighted in the dialog
            current_files = self.selectedFiles()
            if current_files:
                self.onChange(current_files[0])
            else:
                self.mpPreview.setText("No file selected")



    def onChange(self, path):
        if not self.checkBox_preview.isChecked():
            return
        """Handle file selection change for preview"""
        if not self.checkBox_preview.isChecked() or not path or not os.path.isfile(path):
            self.mpPreview.setText("Preview")
            return

        try:
            # CALL YOUR EXTERNAL LOADING FUNCTION
            # Assuming help_dialogue_open_image returns a numpy array-like object
            s = help_dialogue_open_image(path)

            if s is None:
                raise ValueError("Image data is None")

            s = help_dialogue_open_image(path)
            height, width,_ = s.shape

            bytesPerLine = 3 * width
            #qImg = QImage(normal.data, width, height, bytesPerLine, QImage.Format_RGB888)
            #gray_color_table = [qRgb(i, i, i) for i in range(256)]
            qImg = QImage(s.data, s.shape[1]//3, s.shape[0]//3, bytesPerLine, QImage.Format_RGB888)
            #qImg.setColorTable(gray_color_table)
            #from PyQt5.QtCore import QByteArray
            #qb = QByteArray(np.ndarray.tobytes(s))
            #qImg = QImage.fromData(qb)
            pixmap01 = QPixmap.fromImage(qImg)
            pixmap = QPixmap(pixmap01)

            if pixmap.isNull():
                self.mpPreview.setText("Preview unavailable")
            else:
                # Scale smoothly to fit the label area
                scaled_pix = pixmap.scaled(
                    self.mpPreview.width(),
                    self.mpPreview.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.mpPreview.setPixmap(scaled_pix)

        except Exception as e:
            # Fail silently or show error text so dialogue doesn't crash
            # print(f"Preview Error: {e}")
            self.mpPreview.setText("No Preview")

    def activate_combobox(self, state):
        # state is integer (0: Unchecked, 2: Checked)
        self._combobox_type.setEnabled(state == Qt.Checked)

    def onFileSelected(self, file):
        self._fileSelected = file

    def onFilesSelected(self, files):
        self._filesSelected = files

    def getFileSelected(self):
        return [self._fileSelected, '']

    def getFilesSelected(self):
        return self._filesSelected





class QFileSaveDialogPreview(QFileDialog):
    """
    Customizing save dialogue to include Coordinate systems (RAS),
    FPS controls, and a 'Segmented only' filter.
    """
    changCS = pyqtSignal()

    def __init__(self, parent=None, caption="", directory="", filter="", options=None, default_fps=30):
        # Pass the standard arguments to the super init
        super().__init__(parent, caption, directory, filter)
        if options:
            self.setOptions(options)

        # --- Basic Setup ---
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setWindowTitle('Saving...')

        # --- Sizing ---
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

        _translate = QtCore.QCoreApplication.translate
        layout = self.layout()

        # =========================================================
        # 1. Segmented Only Option (Photos)
        # =========================================================
        self.checkBox_seg_only = QtWidgets.QCheckBox(self)
        self.checkBox_seg_only.setObjectName("checkBox_seg_only")
        self.checkBox_seg_only.setText('Segmented only')
        self.checkBox_seg_only.setToolTip("Only save frames that contain segmentation labels")
        self.checkBox_seg_only.setVisible(False)  # Default Hidden


        if isinstance(layout, QGridLayout):
            # Row 4, Column 2 (Right side)
            layout.addWidget(self.checkBox_seg_only, 4, 2, 1, 1)

        # =========================================================
        # 2. Coordinate Systems & Advanced Toggle (RAS - NIfTI)
        # =========================================================
        self.ras_container_widget = QtWidgets.QWidget(self)
        ras_layout = QHBoxLayout(self.ras_container_widget)
        ras_layout.setContentsMargins(0, 0, 0, 0)

        self.label_ras = QtWidgets.QLabel(self)
        self.label_ras.setText(_translate("Dialog", "RAS"))
        self.label_ras.setAlignment(Qt.AlignCenter)
        self.label_ras.setStyleSheet('color: Red')

        self.combbox_asto = QtWidgets.QComboBox(self)
        for i, ss in enumerate(['as', 'to']):
            self.combbox_asto.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self.combbox_asto.addItem(" " * 9 + ss + " " * 9)

        self._combobox_coords = QtWidgets.QComboBox(self)
        self.systems = ['RAS', 'RSA', 'RPI', 'RPS', 'RIP', 'RIA', 'RSP', 'RSA',
                        'LAS', 'LAI', 'LPI', 'LPS', 'LSA', 'LIA', 'LIP', 'LSP',
                        'PIL', 'PIR', 'PSL', 'PSR', 'PLI', 'PRI', 'PLS', 'PRS',
                        'AIL', 'AIR', 'ASL', 'ASR', 'ALI', 'ARI', 'ALS', 'ARS',
                        'IPL', 'IPR', 'IAL', 'IAR', 'ILP', 'IRP', 'ILA', 'IRA',
                        'SPL', 'SPR', 'SAL', 'SAR', 'SLP', 'SRP', 'SLA', 'SRA']

        for i, ss in enumerate(self.systems):
            self._combobox_coords.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_coords.addItem(" " * 9 + ss + " " * 9)

        self._combobox_coords.setEnabled(False)
        self.combbox_asto.setEnabled(False)

        self.checkBox_adv = QtWidgets.QCheckBox(self)
        self.checkBox_adv.setText('Advanced')
        self.checkBox_adv.stateChanged.connect(self.activate_combobox)

        ras_layout.addWidget(self.checkBox_adv)
        ras_layout.addWidget(self.label_ras)
        ras_layout.addWidget(self.combbox_asto)
        ras_layout.addWidget(self._combobox_coords)

        self.ras_container_widget.setVisible(False)

        if isinstance(layout, QGridLayout):
            layout.addWidget(self.ras_container_widget, 4, 1, 1, 1)

        # =========================================================
        # 3. FPS Controls (Video)
        # =========================================================
        self.fps_container_widget = QtWidgets.QWidget(self)
        fps_layout = QHBoxLayout(self.fps_container_widget)
        fps_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_fps = QLabel("FPS:", self)
        self.chk_default_fps = QCheckBox("Default", self)
        self.chk_default_fps.setToolTip("Use default frame rate")
        self.chk_default_fps.setChecked(True)

        self.spin_fps = QSpinBox(self)
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(default_fps)
        self.spin_fps.setEnabled(False)
        self.spin_fps.setSuffix(" fps")

        self.chk_default_fps.toggled.connect(self._toggle_fps_manual)

        fps_layout.addWidget(self.lbl_fps)
        fps_layout.addWidget(self.spin_fps)
        fps_layout.addWidget(self.chk_default_fps)
        fps_layout.addStretch()

        if isinstance(layout, QGridLayout):
            layout.addWidget(self.fps_container_widget, 5, 1, 1, 2)

        # =========================================================
        # 4. Connections & Logic
        # =========================================================

        # Extensions definitions
        self.ext_video = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        self.ext_nifti = ['.nii', '.nii.gz']
        self.ext_photo = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        self.currentChanged.connect(self._update_ui_visibility)
        self.filterSelected.connect(self._update_ui_visibility)

        self._fileSelected = ''
        self.fileSelected.connect(self.onFileSelected)

        # --- FIXED INITIALIZATION ---
        # Pass the incoming 'filter' string so the logic can see
        # "Images (*.png *.jpg)" and activate the photo checkboxes immediately.
        self._update_ui_visibility(filter.split(";;")[0])

    # --- Methods ---

    def _toggle_fps_manual(self, checked):
        self.spin_fps.setEnabled(not checked)

    def _update_ui_visibility(self, path_or_filter):
        """
        Determines which UI elements to show based on:
        1. The 'Save as type' dropdown (Primary)
        2. The currently selected file (Secondary)
        """

        # --- Build a robust string to check against ---
        # 1. Start with the input argument (could be a path or a filter string)
        check_str = str(path_or_filter).lower()

        # 2. Add the currently selected Name Filter (The Dropdown)
        # This is CRITICAL: It ensures options stay visible even when navigating folders
        check_str += " " + self.selectedNameFilter().lower()

        # 3. Add the currently selected file (if any)
        # This helps if the user clicked a file that matches an extension
        current_files = self.selectedFiles()
        if current_files:
            check_str += " " + current_files[0].lower()

        # --- Check Types based on the combined string ---

        # Video Logic
        is_video = any(ext in check_str for ext in self.ext_video)
        # Also check for explicit keywords in the filter name like "Video"
        if "video" in check_str or "movie" in check_str:
            is_video = True

        # NIfTI Logic
        is_nifti = any(ext in check_str for ext in self.ext_nifti)
        if "nii" in check_str:
            is_nifti = True

        # Photo Logic
        is_photo = any(ext in check_str for ext in self.ext_photo)
        # Check if "image" or "photo" is in the filter name (optional safety net)
        if "image" in check_str or "photo" in check_str:
            is_photo = True

        # --- Apply Visibility ---

        # 1. Video -> FPS Controls
        self.fps_container_widget.setVisible(is_video)

        # 2. NIfTI -> RAS Options
        self.ras_container_widget.setVisible(is_nifti)

        # 3. Photo -> Segmented Only
        self.checkBox_seg_only.setVisible(is_photo)

        self.checkBox_seg_only.setChecked(is_photo)

        # Auto-tick 'Segmented only' if it becomes visible and wasn't manually unchecked?
        # (Optional: Logic below keeps it ticked if it's visible)
        #if is_photo and not self.checkBox_seg_only.isChecked():
        #    self.checkBox_seg_only.setChecked(True)
    def setCS(self, cs):
        try:
            index = [i for i, el in enumerate(self.systems) if el == cs][0]
            self._combobox_coords.setCurrentIndex(0)
            self.label_ras.setText(cs)
        except:
            pass

    def activate_combobox(self, value):
        self._combobox_coords.setEnabled(value)
        self.combbox_asto.setEnabled(value)

    def getInfo(self):
        return (
            self._combobox_coords.currentText().strip(' '),
            self.combbox_asto.currentText().strip(' '),
            self.spin_fps.value(),
            self.chk_default_fps.isChecked(),
            self.checkBox_seg_only.isChecked()
        )

    def getFileSelected(self):
        return [self._fileSelected, '']

    def onFileSelected(self, file):
        self._fileSelected = file



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QFileDialogPreview()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()