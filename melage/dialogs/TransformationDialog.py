import sys
import os
import ants
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import List
from pathlib import Path
from PyQt5.QtCore import pyqtSignal
MNI_FILE = 'mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz'

# --- Backend ANTsPy Function (Unchanged) ---

def apply_ants_transforms_from_paths(
        fixed_path: str,
        moving_path: str,
        transform_list: List[str],
        output_path: str,
        interpolation: str = 'linear'
) -> str:
    """
    Applies a list of ANTs transforms to a moving image using file paths.
    Includes robust error checking for file existence.

    Args:
        fixed_path (str): Path to the fixed (reference) image.
        moving_path (str): Path to the moving image to be transformed.
        transform_list (List[str]): A list of paths to the transform files.
        output_path (str): Path to save the warped output image.
        interpolation (str): Interpolation method ('linear', 'nearestNeighbor', etc.).

    Returns:
        str: The path to the saved output image.

    Raises:
        FileNotFoundError: If any of the input files do not exist.
        RuntimeError: If the ANTsPy transformation fails for any reason.
    """
    for path in [fixed_path, moving_path] + transform_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    print("--- Applying ANTsPy Transformation ---")
    print(f"Reference (Fixed): {fixed_path}")
    print(f"Target (Moving): {moving_path}")
    print(f"Transforms: {transform_list}")
    print(f"Interpolation: {interpolation}")

    try:
        fixed_image = ants.image_read(fixed_path)
        moving_image = ants.image_read(moving_path)

        warped_image = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=transform_list,
            interpolator=interpolation,
            verbose=True
        )

        ants.image_write(warped_image, output_path)
        print(f"--- Transformation Complete ---")
        print(f"Output saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"ERROR: An error occurred during transformation: {e}")
        raise RuntimeError(f"ANTsPy failed to apply transform. Reason: {e}")


# --- PyQt5 Frontend (Updated) ---

class TransformationDialog(QtWidgets.QDialog):
    """
    A PyQt5 dialog for applying pre-computed ANTsPy transformations.
    """
    closeSig = pyqtSignal()
    datachange = pyqtSignal()
    def __init__(self, parent=None, source_dir=os.path.expanduser("~")):
        super().__init__(parent)
        self.source_dir = source_dir

        self.fixed_path = None
        self.moving_path = None
        self.transform_paths = []
        self.output_path = None
        # Get MNI path once on initialization for efficiency
        self.mni_template_path = ants.get_ants_data('mni')

        self._setup_ui()

    def _setup_ui(self):
        """Initializes the UI components, layout, and connections."""
        self.setWindowTitle("Apply ANTsPy Transformation")
        self.setMinimumSize(700, 280)

        self._create_widgets()
        self._create_layouts()
        self._connect_signals()

        # Set the initial state of the MNI checkbox to off
        self._toggle_mni_template(False)
    def set_source(self, source_dir):
        self.source_dir = source_dir
    def _create_widgets(self):
        """Creates all the widgets needed for the dialog."""
        # --- Updated MNI Widgets ---
        self.check_use_mni = QtWidgets.QCheckBox("Use standard MNI152 template as Fixed Image")
        self.lineEdit_fixed = QtWidgets.QLineEdit()
        self.lineEdit_fixed.setReadOnly(True)
        self.button_fixed = QtWidgets.QPushButton("Browse...")

        self.label_moving = QtWidgets.QLabel("<b>2. Input Image (Moving):</b>")
        self.lineEdit_moving = QtWidgets.QLineEdit("Select the image or label map to transform...")
        self.lineEdit_moving.setReadOnly(True)
        self.button_moving = QtWidgets.QPushButton("Browse...")

        self.label_transforms = QtWidgets.QLabel("<b>3. Transform File(s):</b>")
        self.lineEdit_transforms = QtWidgets.QLineEdit("Select .mat, .h5, or .nii.gz transform files...")
        self.lineEdit_transforms.setReadOnly(True)
        self.button_transforms = QtWidgets.QPushButton("Browse...")

        self.label_output = QtWidgets.QLabel("<b>4. Output Warped Image:</b>")
        self.lineEdit_output = QtWidgets.QLineEdit("Select where to save the output file...")
        self.lineEdit_output.setReadOnly(True)
        self.button_output = QtWidgets.QPushButton("Save As...")

        self.label_interp = QtWidgets.QLabel("<b>5. Interpolation Method:</b>")
        self.combobox_interp = QtWidgets.QComboBox()
        self.combobox_interp.addItems(['linear', 'nearestNeighbor', 'genericLabel'])
        self.combobox_interp.setToolTip(
            "Use 'linear' for anatomical images.\n"
            "Use 'nearestNeighbor' or 'genericLabel' for label maps."
        )

        self.button_run = QtWidgets.QPushButton("Run Transformation")
        self.button_run.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold; padding: 5px;")

    def _create_layouts(self):
        """Lays out the widgets in a grid."""
        main_layout = QtWidgets.QVBoxLayout(self)
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setSpacing(10)

        grid_layout.addWidget(self.check_use_mni, 0, 0, 1, 3)
        grid_layout.addWidget(self.lineEdit_fixed, 1, 0, 1, 2)
        grid_layout.addWidget(self.button_fixed, 1, 2)

        grid_layout.addWidget(self.label_moving, 2, 0)
        grid_layout.addWidget(self.lineEdit_moving, 2, 1)
        grid_layout.addWidget(self.button_moving, 2, 2)

        grid_layout.addWidget(self.label_transforms, 3, 0)
        grid_layout.addWidget(self.lineEdit_transforms, 3, 1)
        grid_layout.addWidget(self.button_transforms, 3, 2)

        grid_layout.addWidget(self.label_output, 4, 0)
        grid_layout.addWidget(self.lineEdit_output, 4, 1)
        grid_layout.addWidget(self.button_output, 4, 2)

        interp_layout = QtWidgets.QHBoxLayout()
        interp_layout.addWidget(self.label_interp)
        interp_layout.addWidget(self.combobox_interp)
        interp_layout.addStretch()
        grid_layout.addLayout(interp_layout, 5, 1)

        main_layout.addLayout(grid_layout)
        main_layout.addStretch()
        main_layout.addWidget(self.button_run)

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self.check_use_mni.toggled.connect(self._toggle_mni_template)
        self.button_fixed.clicked.connect(self._browse_fixed)
        self.button_moving.clicked.connect(self._browse_moving)
        self.button_transforms.clicked.connect(self._browse_transforms)
        self.button_output.clicked.connect(self._browse_output)
        self.button_run.clicked.connect(self._run_transformation)

    # --- UI Logic Methods ---

    def _toggle_mni_template(self, checked: bool):
        """Handles using the built-in MNI template."""
        if checked:
            try:
                current_dir = Path(__file__).parent
                project_root = (current_dir / "../").resolve()
                mni_file_path = project_root / "MNI" / MNI_FILE
                self.fixed_path = str(mni_file_path)
            except Exception:
                self.fixed_path = []
                print("Warning: Could not determine model file paths automatically.")
                return
            self.lineEdit_fixed.setText(self.fixed_path)
            self.button_fixed.setEnabled(False)
        else:
            self.fixed_path = None
            self.lineEdit_fixed.setText("Select custom fixed image...")
            self.button_fixed.setEnabled(True)

    # --- File Browser Methods ---

    def _browse_fixed(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Image", self.source_dir,
                                                        "NIfTI Files (*.nii *.nii.gz)")
        if path:
            self.fixed_path = path
            self.lineEdit_fixed.setText(path)
            self.source_dir = os.path.dirname(path)

    def _browse_moving(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Input Image", self.source_dir,
                                                        "NIfTI Files (*.nii *.nii.gz)")
        if path:
            self.moving_path = path
            self.lineEdit_moving.setText(path)

    def _browse_transforms(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Transform Files", self.source_dir,
                                                          "Transform Files (*.mat)")
        if paths:
            self.transform_paths = paths
            self.lineEdit_transforms.setText(
                f"{len(paths)} file(s) selected: {', '.join([os.path.basename(p) for p in paths])}")

    def _browse_output(self):
        if self.moving_path:
            moving_name = os.path.basename(self.moving_path).split('.')[0]
            default_name = os.path.join(self.source_dir, f"{moving_name}_warped.nii.gz")
        else:
            default_name = self.source_dir
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Warped Image", default_name,
                                                        "NIfTI Files (*.nii.gz)")
        if path:
            self.output_path = path
            self.lineEdit_output.setText(path)

    # --- Execution Method ---

    def _run_transformation(self):
        """Validates inputs and triggers the ANTsPy transformation."""
        if not all([self.fixed_path, self.moving_path, self.transform_paths, self.output_path]):
            QtWidgets.QMessageBox.warning(self, "Missing Information",
                                          "Please select all required files (reference, input, transform(s), and output).")
            return

        interpolation_method = self.combobox_interp.currentText()

        try:
            self.button_run.setEnabled(False)
            self.button_run.setText("Transforming...")
            QtWidgets.QApplication.processEvents()

            output_file = apply_ants_transforms_from_paths(
                fixed_path=self.fixed_path,
                moving_path=self.moving_path,
                transform_list=self.transform_paths,
                output_path=self.output_path,
                interpolation=interpolation_method
            )

            QtWidgets.QMessageBox.information(
                self, "Success", f"Transformation complete!\nWarped image saved to:\n{output_file}"
            )

        except (FileNotFoundError, RuntimeError, Exception) as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during transformation:\n\n{str(e)}")
        finally:
            self.button_run.setEnabled(True)
            self.button_run.setText("Run Transformation")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = TransformationDialog()
    dialog.show()
    sys.exit(app.exec_())