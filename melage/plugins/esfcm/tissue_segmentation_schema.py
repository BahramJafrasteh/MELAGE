
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
            Combo(id="combo_method", label="Select Method:", options=["esFCM", "FCM"]),
            Group(id="group_mode", layout="hbox", children=[
                SpinBox(id="spin_num_classes", label="Number of classes:", value=3, min_val=2, max_val=50,
                        step=1, decimals=0),
                SpinBox(id="spin_max_iter", label="Max No of Iter:", value=50, min_val=2, max_val=500,
                        step=1, decimals=0),
            ]),

            HBox([
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])

        ]
    }
    return UI_SCHEMA
