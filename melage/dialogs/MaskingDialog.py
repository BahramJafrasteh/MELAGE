# -*- coding: utf-8 -*-
"""
__AUTHOR__ = 'Bahram Jafrasteh'
Refactored Code: This script has been cleaned up for better readability,
simpler UI management, and adherence to modern PyQt practices.
"""
import sys
from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal


class Masking(QtWidgets.QDialog):
    """
    A dialog to get user choices for applying a segmentation mask.

    It emits the selected view, color index, and whether to keep or remove
    the masked area.
    """
    # Emits [view_index, color_index, keep_flag]
    apply_pressed = pyqtSignal(list)
    closeSig = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.connect_signals()

        # Ensure the dialog is properly deleted when closed to free memory
        #self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def setupUi(self):
        """Creates and arranges all the UI widgets."""
        self.setWindowTitle("Apply Mask")
        self.setMinimumSize(400, 150)

        # --- Main Layout ---
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Options Form ---
        options_layout = QtWidgets.QFormLayout()
        options_layout.setContentsMargins(10, 10, 10, 10)

        # Widget for selecting the view
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["View 1", "View 2"])

        # Widget for choosing to keep or remove the mask
        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.addItems(["Keep", "Remove"])

        # Widget for selecting the mask color/label
        self.color_combo = QtWidgets.QComboBox()

        # Add widgets to the form layout for a clean, aligned look
        options_layout.addRow("View:", self.view_combo)
        options_layout.addRow("Action:", self.action_combo)
        options_layout.addRow("Mask Color:", self.color_combo)

        # --- Buttons ---
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setDefault(True)

        # Use a standard button box for consistent button placement
        button_box = QtWidgets.QDialogButtonBox()
        button_box.addButton(self.apply_button, QtWidgets.QDialogButtonBox.AcceptRole)

        # --- Assemble the Main Layout ---
        main_layout.addLayout(options_layout)
        main_layout.addWidget(button_box)

    def connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self.apply_button.clicked.connect(self.on_apply)

    def set_color_options(self, color_names: List[str]):
        """
        Populates the color selection combobox with a list of names.

        Args:
            color_names (List[str]): A list of strings representing color or label names.
        """
        self.color_combo.clear()
        self.color_combo.addItems(color_names)

    @QtCore.pyqtSlot()
    def on_apply(self):
        """
        Gathers user selections and emits them in a signal.
        """
        view_index = self.view_combo.currentIndex()
        color_index = self.color_combo.currentIndex()

        # Determine if the user chose to keep the mask area
        keep_flag = self.action_combo.currentText().lower() == 'keep'

        # Emit all selections in a single list
        self.apply_pressed.emit([view_index, color_index, keep_flag])

        # Optionally close the dialog after applying
        self.accept()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Emits a signal when the dialog is closed by the user."""
        self.closeSig.emit()
        super().closeEvent(event)


# --- Example of how to use the dialog ---
def run_example():
    app = QtWidgets.QApplication(sys.argv)

    # Create the dialog window
    window = Masking()

    # Populate the color options
    available_colors = ["Region 1 (Red)", "Region 2 (Green)", "Region 3 (Blue)"]
    window.set_color_options(available_colors)

    # Define a function to receive the results from the dialog
    def handle_results(params: list):
        view, color, keep = params
        action = "Keep" if keep else "Remove"
        print(f"Applying mask with the following settings:")
        print(f"  - View Index: {view}")
        print(f"  - Color/Label Index: {color} ({available_colors[color]})")
        print(f"  - Action: {action}")

    # Connect the dialog's signal to our handler function
    window.buttonpressed.connect(handle_results)

    # Show the dialog
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_example()