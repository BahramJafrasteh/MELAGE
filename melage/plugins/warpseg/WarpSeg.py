from PyQt5.QtWidgets import QMessageBox
from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .WarpSeg_schema import get_schema
from PyQt5 import QtWidgets
from .warpseg_main.test import build_model, get_inference
import torch
from PyQt5.QtCore import Qt
from melage.config import settings
import os
from PyQt5.QtCore import pyqtSignal
# --- THE LOGIC CLASS ---
class WarpSegLogic(DynamicDialog):
    """
    This class handles the BRAIN of the WarpSeg tool.
    The LOOKS are handled automatically by DynamicDialog + Schema.
    """
    completed = pyqtSignal(object)
    def __init__(self, data_context,parent=None):
        # 1. Initialize DynamicDialog with the Schema
        # This single line builds the entire window!

        super().__init__(parent)
        self.create_main_ui(schema=get_schema())
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.data_context = data_context

        # 2. AUTOMATICALLY WIDGET BINDING
        # Iterate over all widgets created by the schema and bind them to 'self'.
        for widget_id, widget_obj in self.widgets.items():
            setattr(self, widget_id, widget_obj)

        # 3. GENERAL PURPOSE SIGNAL CONNECTION (Auto-Connect)
        # Instead of a hardcoded dictionary, we iterate through every widget from the schema
        # and look for a matching method in this class named: on_<ID>_<Signal>
        # This works for ANY schema changes automatically.
        supported_signals = [
            "clicked", "toggled", "currentIndexChanged",
            "textChanged", "valueChanged"
        ]

        for widget_id, widget_obj in self.widgets.items():
            for signal_name in supported_signals:
                # 1. Check if the widget has this signal (e.g., Button has 'clicked')
                if hasattr(widget_obj, signal_name):
                    # 2. Check if WE have a handler method (e.g., 'on_btn_apply_clicked')
                    handler_name = f"on_{widget_id}_{signal_name}"
                    if hasattr(self, handler_name):
                        # 3. Connect them!
                        signal = getattr(widget_obj, signal_name)
                        handler = getattr(self, handler_name)
                        # Disconnect first to be safe (idempotent), then connect
                        try:
                            signal.disconnect(handler)
                        except TypeError:
                            pass
                        signal.connect(handler)
                        # print(f"Auto-connected: {widget_id}.{signal_name} -> {handler_name}")

        # 4. Set Initial State
        #self.update_ui_state()

    @property
    def ui_schema(self):
        # We call the function to get the dictionary
        return get_schema()
    # We alias these to 'update_ui_state' so the auto-connector finds them
    def on_check_adult_toggled(self, checked=None):
        return None
        #self.update_ui_state()

    def on_check_infant_toggled(self, checked=None):
        return None
        #self.update_ui_state()

    def update_ui_state(self):
        """Handle the Adult vs Infant mutual exclusion logic."""
        sender = self.sender()

        # Mutual Exclusion Logic
        if sender == self.check_adult and self.check_adult.isChecked():
            self.check_infant.blockSignals(True)
            self.check_infant.setChecked(False)
            self.check_infant.blockSignals(False)
        elif sender == self.check_infant and self.check_infant.isChecked():
            self.check_adult.blockSignals(True)
            self.check_adult.setChecked(False)
            self.check_adult.blockSignals(False)

        # Enable/Disable Groups
        self.group_adult.setEnabled(self.check_adult.isChecked())
        self.group_infant.setEnabled(self.check_infant.isChecked())

        # Update Label (using the auto-bound ID 'lbl_model_info')
        mode = "Adult" if self.check_adult.isChecked() else "Infant"
        self.lbl_model_info.setText(f"Model Info: Ready for {mode} segmentation")

    def browse_weights(self, widget):
        """
        Opens file dialog and updates the LineEdit.
        """
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Model Weights",
            "",
            "PyTorch Models (*.pth *.pt);;All Files (*)",
            options=options
        )

        if file_path:
            # 1. Update the LineEdit text to show the path
            widget['txt_weights_path'].setText(file_path)

            # 2. Update the status label to give feedback
            filename = os.path.basename(file_path)
            widget['lbl_model_info'].setText(f"Selected: {filename}")


    # Renamed from 'run_process' to match the schema ID 'btn_apply'
    def on_btn_apply_clicked(self):
        view = self.combo_view.currentText()
        data_view = self.data_context[view]
        if  data_view is None:
            QMessageBox.information(self, "Error", "No image data available for the selected view.")
            return

        try:
            """The main execution function."""
            # Get values using auto-bound attributes
            use_cuda = self.check_cuda.isChecked()


            self.progress_bar.setValue(10)
            print(f"Running WarpSeg: View={view}, CUDA={use_cuda}")

            if use_cuda and torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using CUDA for computation.")
            else:
                device = torch.device("cpu")
                print("Using CPU for computation.")
            #if self.check_adult.isChecked():
            model_path = getattr(self, 'weights_path_custom', None) or os.path.join(settings.DEFAULT_MODELS_DIR, "WarpSeg_Adult.pth")
            #else:
            #    model_path = getattr(self, 'weights_path_custom', None) or os.path.join(settings.DEFAULT_MODELS_DIR, "WarpSeg_Infant.pth")
            self.progress_bar.setValue(30)
            # Build model
            model = build_model(model_path=model_path, device=device)
            self.progress_bar.setValue(50)
            # Here you would load pre-trained weights if available
            # For simplicity, we skip that step
            # Run inference

            data = data_view.get_fdata().copy()
            if len(data.shape)<=1:
                raise ValueError("Input image must be at least 2D NIfTI.")
            seg_tissue, seg = get_inference(model, data, device, post_processing=False)
            if self.radio_adult_whole.isChecked():
                seg = seg
            else:
                seg = seg_tissue
            result_package = {
                "image": None,
                "label": seg,
                "view": view
            }
            self.completed.emit(result_package)
            self.progress_bar.setValue(100)
            #QMessageBox.information(self, "Done", "Segmentation Complete")

        except Exception as e:
            QMessageBox.information(self, "Error", f"{e}")
            self.progress_bar.setValue(0)

    def on_context_action(self, text, widget):
        """Handle the Right-Click Context Menu defined in schema"""
        if text == "Reset Adult Options":
            # Access radio button directly by ID
            self.radio_adult_whole.setChecked(True)
            print("Options reset.")


# --- THE PLUGIN WRAPPER ---
class WarpSegPlugin(MelagePlugin):
    @property
    def name(self) -> str: return "WarpSeg (Dynamic)"

    @property
    def category(self) -> str: return "Segmentation"

    def get_widget(self, data_context =None,parent=None):
        logic = WarpSegLogic(data_context, parent)

        return logic