import numpy as np
from PyQt5.QtWidgets import QMessageBox
from melage.widgets import MelagePlugin
from melage.dialogs.dynamic_gui import DynamicDialog
from .bet_schema import get_schema
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from .main.BET import BET
#from .main.utils import changeCoordSystem
# --- THE LOGIC CLASS ---
class BETLogic(DynamicDialog):
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
        self.check_advanced.stateChanged.connect(self.activate_advanced)
        self.check_thresholding.stateChanged.connect(self.change_hist_threshold)

    @property
    def ui_schema(self):
        # We call the function to get the dictionary
        return get_schema()

    def change_hist_threshold(self, val):
        value = not val
        if not val:
            self.hist_thresh_max.setEnabled(value)
            self.hist_thresh_min.setMaximum(100)
            self.hist_thresh_min.setMinimum(0)
            self.hist_thresh_min.setValue(2)
        else:
            self.hist_thresh_max.setEnabled(value)
            vl = self.hist_thresh_min.value()
            if vl>10:
                self.hist_thresh_min.setValue(10)
            elif vl<4:
                self.hist_thresh_min.setValue(6)
            self.hist_thresh_min.setMaximum(10)
            self.hist_thresh_min.setMinimum(4)


    def get_params(self, img):
        state = self.check_thresholding.isChecked()
        if state:
            from melage.utils.utils import Threshold_MultiOtsu
            numc = int(self.hist_thresh_min.value())
            thresholds = Threshold_MultiOtsu(img, numc)
            t02t = thresholds[0]
            t98t = thresholds[-1]
        else:
            t02t = self.hist_thresh_min.value()/100
            t98t = self.hist_thresh_max.value()/100


        bt = self.fractional_threshold.value()/100
        d1 = self.search_dist_max.value()
        d2 = self.search_dist_min.value()
        rmin = self.rad_curv_min.value()
        rmax = self.rad_curv_max.value()
        n_iter = np.int64(self.iteration.value())
        return [t02t, t98t, bt, d1, d2, rmin, rmax,n_iter]



    def activate_advanced(self, value):
        self.group_advanced.setEnabled(value)


    # Renamed from 'run_process' to match the schema ID 'btn_apply'
    def on_btn_apply_clicked(self):
        view = self.combo_view.currentText()
        data_view = self.data_context[view]
        if  data_view is None:
            QMessageBox.information(self, "Error", "No image data available for the selected view.")
            return

        try:
            """The main execution function."""
            self.lbl_status.setVisible(True)
            self.lbl_status.setText('Initialization...')
            self.progress_bar.setValue(0)
            state = self.check_thresholding.isChecked()
            data = data_view.get_fdata().copy()
            try:
                spacing = data_view.header['pixdim'][1:4]
                spacing = np.max(spacing)
            except (KeyError, TypeError, AttributeError):
                # Fallback value
                spacing = 1  # or simply spacing = 1

            bet = BET(data, spacing)
            info = self.get_params(data)
            bet.update_params(info, state)
            self.progress_bar.setValue(5)
            self._progress = 5
            bet.initialization(self.progress_bar, self.lbl_status)
            self.lbl_status.setText('preparation...')
            bet.run(self.progress_bar, self.lbl_status)
            self.progress_bar.setValue(98)
            self.lbl_status.setText('Computing mask...')
            mask = bet.compute_mask()
            self.lbl_status.setVisible(False)
            self._progress = 100
            self.progress_bar.setValue(self._progress)
            self._progress =0
            result_package = {
                "image": None,
                "label": mask,
                "view": view
            }
            self.completed.emit(result_package)
            self.progress_bar.setValue(100)
            #QMessageBox.information(self, "Done", "Segmentation Complete")

        except Exception as e:
            QMessageBox.information(self, "Error", f"{e}")
            self.progress_bar.setValue(0)




# --- THE PLUGIN WRAPPER ---
class BETPlugin(MelagePlugin):
    @property
    def name(self) -> str: return "BET"

    @property
    def category(self) -> str: return "UnSupervised Segmentation"


    def get_widget(self, data_context =None,parent=None):
        logic = BETLogic(data_context, parent)
        return logic