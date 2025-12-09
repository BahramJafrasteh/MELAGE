
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker, SpinBox, Reference

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():


    UI_SCHEMA = {
        "title": "N4 Bias Field Correction",
        "min_width": 310,
        "layout": "vbox",
        "items": [

            HBox(children=[
                Label(text="Iteration:"),
                SpinBox(id="iteration", value=50, min_val=1, max_val=10000, decimals=0),
                Check(id="check_otsu", text="Otsu", checked=True),
            ]),

            HBox(children=[
                Label(text="Fitting Level:"),
                SpinBox(id="fit_lvl", value=1, min_val=1, max_val=12, decimals=0)
            ]),
            HBox(children=[
                Label(text="Shrink Factor:"),
                SpinBox(id="shrink_fct", value=1, min_val=1, max_val=4, decimals=0)
            ]),




            # --- Bottom Row: Progress + Apply ---
            HBox(children=[
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])
        ]
    }
    return UI_SCHEMA
