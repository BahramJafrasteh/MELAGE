import numpy as np
from PyQt5.QtWidgets import QMessageBox
from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .resize_schema import get_schema
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
import SimpleITK as sitk
from melage.utils.utils import resample_to_spacing


#from .main.utils import changeCoordSystem
# --- THE LOGIC CLASS ---
class MainLogic(DynamicDialog):
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
        self.spin_x.valueChanged.connect(self.synchronizeSpacings)
        self.spin_y.valueChanged.connect(self.synchronizeSpacings)
        self.spin_z.valueChanged.connect(self.synchronizeSpacings)
        self.combo_view.currentIndexChanged.connect(self.compute_current_spcaing)
        self.compute_current_spcaing()

    def compute_current_spcaing(self, data_view=None):
        if data_view is None:
            view = self.combo_view.currentText()
            data_view = self.data_context[view]
        try:
            spacing = data_view.header['pixdim'][1:4]
            label_text = f"X {spacing[0]:0.2f}, Y {spacing[1]:0.2f}, Z {spacing[2]:0.2f}"
        except (KeyError, TypeError, AttributeError):
            label_text = f"Unknown"
        self.lbl_spacing.setText(label_text)

    def synchronizeSpacings(self, value):
        if self.check_iso.isChecked():
            # Set the value of all spin boxes to the changed value
            self.spin_x.setValue(value)
            self.spin_y.setValue(value)
            self.spin_z.setValue(value)

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
            x_v = self.spin_x.value()
            y_v = self.spin_y.value()
            z_v = self.spin_z.value()
            method = self.combo_method.currentText().lower()
            im = resample_to_spacing(data_view, [x_v, y_v, z_v], method)
            self.compute_current_spcaing(im)
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




# --- THE PLUGIN WRAPPER ---
class AuxPlugin(MelagePlugin):
    @property
    def name(self) -> str: return "Resize"

    @property
    def category(self) -> str: return "Basic"


    def get_widget(self, data_context =None,parent=None):
        logic = MainLogic(data_context, parent)
        return logic