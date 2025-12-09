from PyQt5.QtWidgets import QMessageBox
from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .tissue_segmentation_schema import get_schema
from PyQt5 import QtWidgets
from .main.test import get_inference
import torch
from PyQt5.QtCore import Qt
from melage.config import settings
import os
from PyQt5.QtCore import pyqtSignal
# --- THE LOGIC CLASS ---
class esFCMLogic(DynamicDialog):
    """
    This class handles the BRAIN of the WarpSeg tool.
    The LOOKS are handled automatically by DynamicDialog + Schema.
    """
    completed = pyqtSignal(object)
    def __init__(self, data_context,parent=None):
        # 1. Initialize DynamicDialog with the Schema
        # This single line builds the entire window!

        super().__init__(parent)
        self.create_main_ui(schema=get_schema(), default_items=False)
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


    @property
    def ui_schema(self):
        # We call the function to get the dictionary
        return get_schema()




    # Renamed from 'run_process' to match the schema ID 'btn_apply'
    def on_btn_apply_clicked(self):
        view = self.combo_view.currentText()
        data_view = self.data_context[view]
        if  data_view is None:
            QMessageBox.information(self, "Error", "No image data available for the selected view.")
            return

        try:
            """The main execution function."""
            self.progress_bar.setValue(10)

            data = data_view.get_fdata().copy()
            if len(data.shape)<=1:
                raise ValueError("Input image must be at least 2D NIfTI.")

            model_name = self.combo_method.currentText()
            num_tissues = int(self.spin_num_classes.value())
            max_iter = int(self.spin_max_iter.value())
            seg = get_inference(model_name, self.progress_bar, data, affine=None, num_tissues=num_tissues,
                                post_correction=True, max_iter=max_iter)
            seg = seg.astype('int')
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
    def name(self) -> str: return "FCM"

    @property
    def category(self) -> str: return "UnSupervised Segmentation"


    def get_widget(self, data_context =None,parent=None):
        logic = esFCMLogic(data_context, parent)

        return logic