import sys
import os
import ants
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Literal, Optional, Tuple
#from MELAGE.utils.source_folder import *
from PyQt5.QtCore import pyqtSignal
from pathlib import Path

MNI_FILE = 'mni_icbm152_t1_tal_nlin_sym_09a_masked.nii.gz'

class StreamRedirector(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
        # Force the GUI to process events to update the text box in real-time
        QtWidgets.QApplication.processEvents()

    def flush(self):
        # This flush method is needed for stream-like objects
        pass

def run_ants_registration(
    fixed_path: str,
    moving_path: str,
    output_prefix: str,
    transform_type: Literal['Rigid', 'Affine', 'SyN'] = 'SyN'
) -> dict:
    """
    Performs image registration using ANTsPy with robust default parameters.
    """
    print("--- Starting ANTs Registration ---")
    print(f"Fixed Image: {fixed_path}")
    print(f"Moving Image: {moving_path}")
    print(f"Transform Type: {transform_type}")

    fixed_image = ants.image_read(fixed_path, pixeltype='float')
    moving_image = ants.image_read(moving_path, pixeltype='float')


    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=transform_type,
        outprefix=output_prefix,
        verbose=True
    )
    """

    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=transform_type,  # Affine includes Rigid + Affine transformations in one step
        reg_iterations=[1000, 500, 250, 100],  # Corresponds to your iterations [1000x500x250x100]
        grad_step=0.01,  # Corresponds to the gradient step size [Rigid[0.01], Affine[0.01]]
        aff_sampling=32,  # Corresponds to the MI sampling points [1,32,Regular,0.25]
        metric='MI',  # Mutual Information
        metric_params=[1, 32],  # Matches MI[${file_in_atlas},${file_in_t1},1,32,Regular,0.25]
        smoothing_sigmas=(3, 2, 1, 0),  # Smoothing at different resolutions
        shrink_factors=(8, 4, 2, 1),  # Downsampling at different resolutions
        transform_parameters=(0.01,),  # Corresponds to transform step size for Rigid and Affine
        output_prefix=output_prefix,
        verbose = True
    )
    """


    print(f"--- Registration Complete ---")
    file_out = output_prefix+'warped.nii.gz'
    print(f'Transofrmed file {file_out}')
    print(f"Forward Transforms: {registration['fwdtransforms']}")
    ants.image_write(registration['warpedmovout'], file_out)
    return registration, file_out


# --- PyQt5 Frontend (Updated) ---

class RegistrationDialog(QtWidgets.QDialog):
    """
    A PyQt5 dialog for performing image-to-image registration using ANTsPy.
    """
    closeSig = pyqtSignal()
    datachange = pyqtSignal()
    def __init__(self, parent=None, source_dir=os.path.expanduser("~")):
        super().__init__(parent)
        self.source_dir = source_dir
        self.fixed_path = None
        self.moving_path = None
        self.output_prefix = None
        #self.mni_template_path = ants.get_ants_data('mni') # Get MNI path once
        self._curent_weight_dir = os.path.dirname(os.path.join(os.getcwd()))
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("ANTs Image Registration")
        self.setMinimumSize(650, 250)
        self._create_widgets()
        self._create_layouts()
        self._connect_signals()

        # Set initial UI state
        self._toggle_mni_template(False)
        self.combobox_methods.setCurrentIndex(2)

    def _create_widgets(self):
        # --- Updated MNI Template Widgets ---
        self.toggle_log_button = QtWidgets.QPushButton("Show Verbose Output ▶")
        self.toggle_log_button.setCheckable(True)
        self.toggle_log_button.setChecked(False)
        # Style the button to look like a section header
        self.toggle_log_button.setStyleSheet("""
            QPushButton {
                text-align: left; padding: 5px; background-color: #E0E0E0;
                border: 1px solid #C0C0C0; border-radius: 4px;
                color: green; /* Default text color */
            }
            QPushButton:checked {
                background-color: #D0D0D0;
                color: red; /* Text color when button is checked */
            }
        """)
        self.log_stack = QtWidgets.QStackedWidget()

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #2b2b2b; color: #f0f0f0; font-family: Monospace;")
        self.log_output.setVisible(True)

        self.check_use_mni = QtWidgets.QCheckBox("Use standard MNI152 template as Fixed Image")
        self.lineEdit_fixed = QtWidgets.QLineEdit()
        self.lineEdit_fixed.setReadOnly(True)
        self.button_fixed = QtWidgets.QPushButton("Browse...")

        self.label_moving = QtWidgets.QLabel("<b>Moving Image:</b>")
        self.lineEdit_moving = QtWidgets.QLineEdit("Select subject's image...")
        self.lineEdit_moving.setReadOnly(True)
        self.button_moving = QtWidgets.QPushButton("Browse...")

        self.label_output = QtWidgets.QLabel("<b>Output Prefix:</b>")
        self.lineEdit_output = QtWidgets.QLineEdit("Select output location and prefix...")
        self.lineEdit_output.setReadOnly(True)
        self.button_output = QtWidgets.QPushButton("Save As...")

        self.label_method = QtWidgets.QLabel("<b>Method:</b>")
        self.combobox_methods = QtWidgets.QComboBox()
        self.combobox_methods.addItems(['Rigid', 'Affine', 'SyN'])


        self.button_run = QtWidgets.QPushButton("Run Registration")
        self.button_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")



    def _create_layouts(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setSpacing(10)

        grid_layout.addWidget(self.check_use_mni, 0, 0, 1, 4)
        grid_layout.addWidget(self.lineEdit_fixed, 1, 0, 1, 3)
        grid_layout.addWidget(self.button_fixed, 1, 3)

        grid_layout.addWidget(self.label_moving, 2, 0)
        grid_layout.addWidget(self.lineEdit_moving, 2, 1, 1, 2)
        grid_layout.addWidget(self.button_moving, 2, 3)

        grid_layout.addWidget(self.label_output, 3, 0)
        grid_layout.addWidget(self.lineEdit_output, 3, 1, 1, 2)
        grid_layout.addWidget(self.button_output, 3, 3)

        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(self.label_method)
        method_layout.addWidget(self.combobox_methods)
        method_layout.addStretch()
        grid_layout.addLayout(method_layout, 4, 1)



        self.log_stack.addWidget(QtWidgets.QWidget()) # Page 0: Empty widget for "collapsed"
        self.log_stack.addWidget(self.log_output)     # Page 1: The log viewer

        # Add components to the main vertical layout
        main_layout.addLayout(grid_layout)
        main_layout.addWidget(self.button_run)
        main_layout.addWidget(self.toggle_log_button) # Add the toggle button
        main_layout.addWidget(self.log_stack)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    # This is the new slot that will receive text from the redirector
    @QtCore.pyqtSlot(str)
    def _append_log_text(self, text):
        """Appends text to the log viewer."""
        self.log_output.insertPlainText(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())


    def _connect_signals(self):
        self.toggle_log_button.toggled.connect(self._animate_log_toggle)
        self.button_fixed.clicked.connect(self._browse_fixed)
        self.button_moving.clicked.connect(self._browse_moving)
        self.button_output.clicked.connect(self._browse_output)
        self.button_run.clicked.connect(self._run_registration)
        self.check_use_mni.toggled.connect(self._toggle_mni_template)

    def _animate_log_toggle(self, checked):
        """Animates the expansion and collapse of the log container."""
        # The animation now targets the container, not the text box itself
        #animation = QtCore.QPropertyAnimation(self.log_container, b"maximumHeight")
        #animation.setDuration(300)

        if checked:
            self.log_stack.setCurrentIndex(1) # Show log viewer page
            self.toggle_log_button.setText("Hide Progress ▼")
        else:
            self.log_stack.setCurrentIndex(0) # Show empty page
            self.toggle_log_button.setText("Show Progress ▶")


    def set_source(self, source_dir):
        self.source_dir = source_dir

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



    def _browse_fixed(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Custom Fixed Image", self.source_dir, "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.fixed_path = file_path
            self.lineEdit_fixed.setText(file_path)
            self.source_dir = os.path.dirname(file_path)

    def _browse_moving(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Moving Image", self.source_dir, "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.moving_path = file_path
            self.lineEdit_moving.setText(file_path)

    def _browse_output(self):
        if self.moving_path:
            moving_name = os.path.basename(self.moving_path).split('.')[0]
            fixed_name = "MNI152" if self.check_use_mni.isChecked() else "CustomFixed"
            default_name = os.path.join(self.source_dir, f"{moving_name}_to_{fixed_name}_")
        else:
            default_name = self.source_dir
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Set Output Prefix", default_name, "All Files (*)")
        if file_path:
            self.output_prefix = file_path
            self.lineEdit_output.setText(file_path)

    def _run_registration(self):
        if not all([self.fixed_path, self.moving_path, self.output_prefix]):
            QtWidgets.QMessageBox.warning(self, "Missing Information", "Please specify fixed, moving, and output files.")
            return
        self.toggle_log_button.setChecked(True)
        # Clear the log viewer before starting
        self.log_output.clear()

        transform_type = self.combobox_methods.currentText()

        redirector = StreamRedirector()
        redirector.textWritten.connect(self._append_log_text)
        original_stdout = sys.stdout
        sys.stdout = redirector

        try:
            self.button_run.setEnabled(False)
            self.button_run.setText("Registration in progress...")
            QtWidgets.QApplication.processEvents()

            reg_results, file_out = run_ants_registration(
                fixed_path=self.fixed_path, moving_path=self.moving_path,
                output_prefix=self.output_prefix, transform_type=transform_type,
            )

            QtWidgets.QMessageBox.information(
                self, "Success", f"Registration complete! Warped image saved to:\n{file_out}"
            )
        except Exception as e:
            # Also print the error to the log viewer
            print(f"\n--- CRITICAL ERROR ---\n{str(e)}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
        finally:
            sys.stdout = original_stdout
            self.button_run.setEnabled(True)
            self.button_run.setText("Run Registration")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = RegistrationDialog()
    dialog.show()
    sys.exit(app.exec_())