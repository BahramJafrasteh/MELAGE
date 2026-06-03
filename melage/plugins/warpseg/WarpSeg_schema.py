
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():
    UI_SCHEMA = {
        "title": "MorphSeg Deep Learning Segmentation",
        "min_width": 500,
        "layout": "vbox",
        "items": [

            # 1. Combobox with a Label
            # The 'Combo' helper with a 'label' argument automatically creates
            # an HBox with a Label and the ComboBox inside.


            # 2. Adult Section
            # Context menu allows for extra actions like resetting options
            #Check(id="check_adult", text="<b>Adult Segmentation</b>", checked=True,
            #      context=["Reset Adult Options"]),
            Label(id = "lbl_model_info", text="Model Info: Ready for Adult segmentation"),
            Group(id="group_adult", layout="hbox", children=[
                Radio(id="radio_adult_whole", text="Whole Segmentation", checked=True),
                Radio(id="radio_adult_tissue", text="Tissue Segmentation")
            ]),

            # 3. Infant Section
            #Check(id="check_infant", text="<b>Infant Segmentation</b>"),

            # This group starts disabled because Infant mode is not active by default
            #Group(id="group_infant", layout="hbox", enabled=False, children=[
            #    Radio(id="radio_infant_whole", text="Whole Segmentation", checked=True),
            #    Radio(id="radio_infant_tissue", text="Tissue Segmentation", enabled=False)
            #]),

            # 4. Configuration Section (UPDATED)

        ]
    }
    return UI_SCHEMA
