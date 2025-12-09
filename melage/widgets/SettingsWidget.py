from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QLineEdit, QPushButton
from pathlib import Path

# Import the global settings instance
from melage.config import settings


class SettingsDialog(QDialog):
    """
    A dialog for editing application settings.
    It reads from and saves to the global SettingsManager.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MELAGE Settings")
        self.setModal(True)  # Block other windows while open
        self.setupUi()
        self.load_settings()

    def setupUi(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # --- Create a form layout ---
        form_layout = QtWidgets.QFormLayout()

        # --- Auto-save setting ---
        self.auto_save_spinbox = QtWidgets.QDoubleSpinBox()
        self.auto_save_spinbox.setSuffix(" minutes")
        form_layout.addRow("Auto-save interval:", self.auto_save_spinbox)

        # --- Default Save Directory ---
        self.save_dir_edit = QLineEdit()
        self.save_dir_button = QPushButton("Browse...")
        self.save_dir_button.clicked.connect(self.browse_save_dir)
        save_dir_layout = QtWidgets.QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(self.save_dir_button)
        form_layout.addRow("Default Save Directory:", save_dir_layout)

        # --- Models Directory ---
        self.models_dir_edit = QLineEdit()
        self.models_dir_button = QPushButton("Browse...")
        self.models_dir_button.clicked.connect(self.browse_models_dir)
        models_dir_layout = QtWidgets.QHBoxLayout()
        models_dir_layout.addWidget(self.models_dir_edit)
        models_dir_layout.addWidget(self.models_dir_button)
        form_layout.addRow("Models Directory:", models_dir_layout)

        self.main_layout.addLayout(form_layout)

        # --- OK / Cancel Buttons ---
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)  # Calls our save_settings method
        button_box.rejected.connect(self.reject)

        self.main_layout.addWidget(button_box)

    def browse_save_dir(self):
        """Lets user pick a default save directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Default Save Directory", self.save_dir_edit.text()
        )
        if directory:
            self.save_dir_edit.setText(directory)

    def browse_models_dir(self):
        """Lets user pick a models directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", self.models_dir_edit.text()
        )
        if directory:
            self.models_dir_edit.setText(directory)

    def load_settings(self):
        """Load current settings from the manager into the UI."""
        self.auto_save_spinbox.setValue(settings.auto_save_interval)
        self.save_dir_edit.setText(str(settings.DEFAULT_USE_DIR))
        self.models_dir_edit.setText(str(settings.DEFAULT_MODELS_DIR))

    def update_use_dir(self, new_dir):
        settings.DEFAULT_USE_DIR = str(Path(new_dir))

    def save_settings(self):
        """Save UI values back to the settings manager and persist."""
        settings.auto_save_interval = self.auto_save_spinbox.value()
        settings.DEFAULT_USE_DIR = str(Path(self.save_dir_edit.text()))
        settings.DEFAULT_MODELS_DIR = str(Path(self.models_dir_edit.text()))

        settings.save()  # This saves to the JSON file

    def accept(self):
        """Called when OK is clicked."""
        self.save_settings()
        super().accept()  # Closes the dialog
