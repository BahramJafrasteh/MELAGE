
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker, SpinBox, Reference

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():
    UI_SCHEMA = {
        "title": "Enhanced spatial Fuzzy C-Means algorithm ...",
        "min_width": 500,
        "layout": "vbox",
        "items": [
            #Label(id="lbl_model_info", text="Model Info: Ready for Ifant brain segmentation"),
            Reference(
                "<b>Jafrasteh et al. (2024)</b>. <i>'Enhanced Spatial Fuzzy C-Means...'</i>. "
                '<a href="https://link.springer.com/article/10.1007/s12021-024-09661-x">(paper)</a>'

            ),
            Group(id="group_mode", layout="hbox", title="Destination",children=[
                Combo(id="combo_1st", label="", options=["R", "L", "P", "A", "I", "S"], default="R", function_name="update_2nd_combo"),
                Combo(id="combo_2nd", label="", options=["R", "L", "P", "A", "I", "S"], default="A", function_name="update_3rd_combo"),
                Combo(id="combo_3rd", label="", options=["R", "L", "P", "A", "I", "S"], default="S"),
                ]),

            HBox([
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])

        ]
    }
    return UI_SCHEMA
