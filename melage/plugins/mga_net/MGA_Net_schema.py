
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker, SpinBox, Reference

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():
    UI_SCHEMA = {
        "title": "MGA-Net Infant Deep Learning Segmentation",
        "min_width": 500,
        "layout": "vbox",
        "items": [
            #Label(id="lbl_model_info", text="Model Info: Ready for Ifant brain segmentation"),
            Reference(
                "<b>Jafrasteh et al. (2024)</b>. <i>'A novel mask-guided attention...'</i>. "
                '<a href="https://www.sciencedirect.com/science/article/pii/S1053811924003690">(paper)</a>'

            ),
            SpinBox(id="spin_threshold", label="Segmentation Threshold:", value=0.5, min_val=-4.0, max_val=4.0, step=0.1, decimals=1),
            # Context menu allows for extra actions like resetting options

            Group(id="group_mode", layout="hbox", children=[
                Radio(id="radio_mri", text="MRI Segmentation", checked=True),
                Radio(id="radio_us", text="US Segmentation")
            ]),


        ]
    }
    return UI_SCHEMA
