import numpy as np
from PyQt5.QtWidgets import QMessageBox
from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .change_coord_schema import get_schema
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from .main.utils import changeCoordSystem
# --- THE LOGIC CLASS ---
class ChangeCoordLogic(DynamicDialog):
    """
    This class handles the BRAIN of the WarpSeg tool.
    The LOOKS are handled automatically by DynamicDialog + Schema.
    """
    completed = pyqtSignal(object)
    AXIS_GROUPS = [
        {'R', 'L'},
        {'A', 'P'},
        {'S', 'I'}
    ]
    ALL_OPTIONS = ['R', 'L', 'A', 'P', 'S', 'I']

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
        self.update_2nd_combo()

    @property
    def ui_schema(self):
        # We call the function to get the dictionary
        return get_schema()

    def get_axis_group(self, letter):
        """Helper: Returns the set containing the letter (e.g., 'R' -> {'R', 'L'})"""
        for group in self.AXIS_GROUPS:
            if letter in group:
                return group
        return set()

    def update_2nd_combo(self):
        """
        Filters Combo 2 based on Combo 1 selection.
        """
        selection_1 = self.combo_1st.currentText()
        forbidden_group = self.get_axis_group(selection_1)

        # Calculate allowed options for 2nd box
        # Logic: All Options MINUS the axis used in box 1
        allowed = [opt for opt in self.ALL_OPTIONS if opt not in forbidden_group]

        # Repopulate 2nd Combo
        self._repopulate_combo(self.combo_2nd, allowed)

        # Force update of 3rd combo because 2nd changed
        self.update_3rd_combo()

    def update_3rd_combo(self):
        """
        Filters Combo 3 based on Combo 1 AND Combo 2 selection.
        """
        selection_1 = self.combo_1st.currentText()
        selection_2 = self.combo_2nd.currentText()

        group_1 = self.get_axis_group(selection_1)
        group_2 = self.get_axis_group(selection_2)

        # Calculate allowed options for 3rd box
        # Logic: All Options MINUS axis 1 MINUS axis 2
        allowed = [
            opt for opt in self.ALL_OPTIONS
            if opt not in group_1 and opt not in group_2
        ]

        # Repopulate 3rd Combo
        self._repopulate_combo(self.combo_3rd, allowed)

    def _repopulate_combo(self, combo_widget, allowed_options):
        """
        Safely updates items while trying to preserve the current selection.
        """
        current_val = combo_widget.currentText()

        # Block signals to prevent infinite recursion loops during update
        combo_widget.blockSignals(True)
        combo_widget.clear()
        combo_widget.addItems(allowed_options)

        # Try to restore previous selection, otherwise pick index 0
        index = combo_widget.findText(current_val)
        if index >= 0:
            combo_widget.setCurrentIndex(index)
        else:
            combo_widget.setCurrentIndex(0)

        combo_widget.blockSignals(False)

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

            el_1 = self.combo_1st.currentText()
            el_2 = self.combo_2nd.currentText()
            el_3 = self.combo_3rd.currentText()
            target = f"{el_1}{el_2}{el_3}"
            used_axes = [self.get_axis_group(x) for x in [el_1, el_2, el_3]]
            unique_axes = set(tuple(s) for s in used_axes)

            if len(unique_axes) != 3:
                raise ValueError(f"Invalid orientation {target}. Axes must be orthogonal.")
            if np.unique([el_1, el_2, el_3]).shape[0]!=3:
                QMessageBox.information(self, "INVALID COORDINATION SYSTEM", f"{target} is not a valid coorrdiantion system")
                self.progress_bar.setValue(0)
                return
            im = changeCoordSystem(im=data_view, target=target)
            result_package = {
                "image": im.get_fdata(),
                "affine": im.affine,
                "label": np.zeros_like(im.get_fdata()),
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
class ChangeCoordPlugin(MelagePlugin):
    @property
    def name(self) -> str: return "Change Coord Sys."

    @property
    def category(self) -> str: return "Basic"


    def get_widget(self, data_context =None,parent=None):
        logic = ChangeCoordLogic(data_context, parent)
        return logic