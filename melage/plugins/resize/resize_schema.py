
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker, SpinBox, Reference

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():


    UI_SCHEMA = {
        "title": "Image resizing",
        "min_width": 420,
        "layout": "vbox",
        "items": [


            HBox(children=[
                Combo(id="combo_method", label="Select Method:", options=["Spline", "Linear"]),
                Check(id="check_iso", text="Isotropic", checked=True),  # Connected to activate_advanced

            ]),
            # 1. Iteration (Splitter 3)
            HBox(children=[
                Label(text="New Spacing:"),
                SpinBox(id="spin_x", value=1, min_val=0.1, max_val=10, decimals=2, step=0.1),
                SpinBox(id="spin_y", value=1, min_val=0.1, max_val=10, decimals=2, step=0.1),
                SpinBox(id="spin_z", value=1, min_val=0.1, max_val=10, decimals=2, step=0.1)
            ]),
            HBox(children=[
                Label(text="Current Spacing:"),
                Label(id="lbl_spacing", text="", visible=True),
            ]),


            # --- Bottom Row: Progress + Apply ---
            HBox(children=[
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])
        ]
    }
    return UI_SCHEMA
