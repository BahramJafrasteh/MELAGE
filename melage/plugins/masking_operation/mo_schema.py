
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker, SpinBox, Reference

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():


    UI_SCHEMA = {
        "title": "N4 Bias Field Correction",
        "min_width": 310,
        "layout": "vbox",
        "items": [



            Group(id="group_mode", layout="hbox", title="Operation",children=[
                Combo(id="combo_1", label="Label 1", options=[],),
                Combo(id="combo_operation", label="", options=["/", "*", "-", "+"], default="+"),
                Combo(id="combo_2", label="Label 2", options=[]),
                ]),




            # --- Bottom Row: Progress + Apply ---
            HBox(children=[
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])
        ]
    }
    return UI_SCHEMA
