# -*- coding: utf-8 -*-
"""
__AUTHOR__ = 'Bahram Jafrasteh'
Refactored Code: This script has been improved to offer a clearer user
interface, cleaner code, and more robust logic for mask operations.
"""
import sys
from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt


class MaskOperationsDialog(QtWidgets.QDialog):
    """
    A dialog for performing simple arithmetic operations (+, -) between two masks.
    """
    # Emits [view_index, mask1_index, mask2_index, operation_str]
    apply_pressed = pyqtSignal(list)
    closeSig = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.connect_signals()

        # Ensure the dialog is properly deleted when closed to free memory
        #self.setAttribute(Qt.WA_DeleteOnClose)

    def setupUi(self):
        """Creates and arranges all the UI widgets."""
        self.setWindowTitle("Mask Operations")
        self.setMinimumSize(450, 150)

        # --- Main Layout ---
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Options Widgets ---
        options_group = QtWidgets.QGroupBox("Operation")
        options_layout = QtWidgets.QHBoxLayout(options_group)

        self.mask1_combo = QtWidgets.QComboBox()
        self.mask2_combo = QtWidgets.QComboBox()
        self.operation_combo = QtWidgets.QComboBox()
        self.operation_combo.addItems(["+", "-"])
        self.operation_combo.setFixedWidth(50)  # Keep the operation box small

        # Add widgets to the horizontal layout to create the "Mask1 + Mask2" look
        options_layout.addWidget(self.mask1_combo)
        options_layout.addWidget(self.operation_combo)
        options_layout.addWidget(self.mask2_combo)

        # --- View Selector ---
        view_layout = QtWidgets.QFormLayout()
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["View 1", "View 2"])
        view_layout.addRow("Apply to View:", self.view_combo)

        # --- Buttons ---
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setDefault(True)

        # Use a standard button box for consistent button placement (Apply/Cancel)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply
        )

        # --- Assemble the Main Layout ---
        main_layout.addLayout(view_layout)
        main_layout.addWidget(options_group)
        main_layout.addStretch()  # Add space
        main_layout.addWidget(button_box)

    def connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        button_box = self.findChild(QtWidgets.QDialogButtonBox)
        button_box.accepted.connect(self.on_apply)
        button_box.rejected.connect(self.reject)

    def set_color_options(self, mask_names: List[str]):
        """
        Populates both mask selection comboboxes with a list of names.

        Args:
            mask_names (List[str]): A list of strings representing mask names.
        """
        self.mask1_combo.clear()
        self.mask2_combo.clear()

        self.mask1_combo.addItems(mask_names)
        self.mask2_combo.addItems(mask_names)

    @QtCore.pyqtSlot()
    def on_apply(self):
        """
        Gathers user selections, validates them, and emits the result.
        """
        view_index = self.view_combo.currentIndex()
        mask1_index = self.mask1_combo.currentIndex()
        mask2_index = self.mask2_combo.currentIndex()
        operation = self.operation_combo.currentText()

        # --- Validation ---
        if mask1_index == mask2_index:
            self.show_warning_message(
                "Cannot perform an operation on the same mask. Please select two different masks.")
            return  # Stop the process

        # --- Emit Signal ---
        self.apply_pressed.emit([view_index, mask1_index, mask2_index, operation])
        self.accept()  # Close the dialog with an "OK" status

    def show_warning_message(self, message: str):
        """Displays a non-critical warning message to the user."""
        QtWidgets.QMessageBox.warning(self, "Validation Error", message)

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Emits a signal when the dialog is closed by the user."""
        self.closeSig.emit()
        super().closeEvent(event)


# --- Example of how to use the dialog ---
def run_example():
    app = QtWidgets.QApplication(sys.argv)

    window = MaskOperationsDialog()

    # Populate the mask options
    available_masks = ["Brain Mask", "Tumor Segmentation", "Ventricles"]
    window.set_mask_options(available_masks)

    def handle_results(params: list):
        view, idx1, idx2, op = params
        print("Applying operation with the following settings:")
        print(f"  - View: {view + 1}")
        print(f"  - Formula: {available_masks[idx1]} {op} {available_masks[idx2]}")

    window.apply_pressed.connect(handle_results)

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_example()