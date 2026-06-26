import numpy as np
from PyQt5.QtWidgets import QMessageBox
from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .mo_schema import get_schema
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal


def combine_label_masks(npSeg, label1, op, label2, output_label=None):
    """
    Combine two existing segmentation labels with a boolean set operation.

    "+" = union, "-" = difference (label1 minus label2), "*" = intersection,
    "/" = symmetric difference. The result replaces output_label (defaults
    to label1); all other labels are left untouched.
    """
    if output_label is None:
        output_label = label1

    mask1 = npSeg == label1
    mask2 = npSeg == label2

    if op == "+":
        result = mask1 | mask2
    elif op == "-":
        result = mask1 & ~mask2
    elif op == "*":
        result = mask1 & mask2
    elif op == "/":
        result = mask1 ^ mask2
    else:
        raise ValueError(f"Unknown operation {op!r}; expected one of '+', '-', '*', '/'.")

    new_seg = npSeg.copy()
    new_seg[new_seg == output_label] = 0
    new_seg[result] = output_label
    return new_seg


# --- THE LOGIC CLASS ---
class MainLogic(DynamicDialog):
    """
    This class handles the BRAIN of the Masking Operation tool.
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

        self._populate_label_combos()

    @property
    def ui_schema(self):
        # We call the function to get the dictionary
        return get_schema()

    def on_combo_view_currentIndexChanged(self):
        self._populate_label_combos()

    def _populate_label_combos(self):
        """Fill combo_1/combo_2 with the labels currently present in the
        selected view's segmentation."""
        view   = self.combo_view.currentText()
        reader = self.data_context.get(view) if self.data_context else None

        labels = []
        if reader is not None and getattr(reader, "npSeg", None) is not None:
            labels = [str(int(l)) for l in np.unique(reader.npSeg) if l != 0]

        for combo in (self.combo_1, self.combo_2):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    # Renamed from 'run_process' to match the schema ID 'btn_apply'
    def on_btn_apply_clicked(self):
        view = self.combo_view.currentText()
        data_obj = self.data_context[view]
        if data_obj is None:
            QMessageBox.information(self, "Error", "No image data available for the selected view.")
            return
        if getattr(data_obj, "npSeg", None) is None:
            QMessageBox.information(self, "Error", "No segmentation available for the selected view.")
            return

        label1_txt = self.combo_1.currentText()
        label2_txt = self.combo_2.currentText()
        if not label1_txt or not label2_txt:
            QMessageBox.information(self, "Error", "No labels available to combine.")
            return

        try:
            self.progress_bar.setValue(10)
            label1 = int(label1_txt)
            label2 = int(label2_txt)
            op     = self.combo_operation.currentText()

            new_seg = combine_label_masks(data_obj.npSeg, label1, op, label2)
            self.progress_bar.setValue(70)

            result_package = {
                "image": None,
                "affine": None,
                "label": new_seg,
                "view": view
            }
            self.completed.emit(result_package)
            self.progress_bar.setValue(100)
            #QMessageBox.information(self, "Done", "Segmentation Complete")

        except Exception as e:
            QMessageBox.information(self, "Error", f"{e}")
            self.progress_bar.setValue(0)




# --- THE PLUGIN WRAPPER ---
class AuxPlugin(MelagePlugin):
    @property
    def name(self) -> str: return "Masking Operation"

    @property
    def category(self) -> str: return "Basic"


    def get_widget(self, data_context =None,parent=None):
        logic = MainLogic(data_context, parent)
        return logic
