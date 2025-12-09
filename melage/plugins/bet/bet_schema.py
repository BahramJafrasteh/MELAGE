
from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, FilePicker, SpinBox, Reference

# The simplified schema using helper functions.
# The main layout defaults to 'vbox', so items stack automatically.
def get_schema():


    UI_SCHEMA = {
        "title": "Brain Extaction Tools (BET)",
        "min_width": 310,
        "layout": "vbox",
        "items": [

            # --- Top Row: Checkbox (Advanced) + Image Selector ---
            HBox(children=[
                Check(id="check_advanced", text="Advanced Settings", checked=False),  # Connected to activate_advanced
            ]),

            # --- Advanced Settings Widget (Disabled by default) ---
            # We group all the advanced settings here.
            # The ID 'group_advanced' corresponds to 'self.widget' in your code.
            Group(id="group_advanced", title="Advanced Parameters", enabled=False, children=[

                # 1. Iteration (Splitter 3)
                HBox(children=[
                    Label(text="Iteration:"),
                    SpinBox(id="iteration", value=50, min_val=1, max_val=10000, decimals=0)
                ]),

                # 2. Thresholding (Splitter)
                # This logic is tricky: checked=True enables min, disables max.
                # We map this behavior using 'enable_targets' logic.
                HBox(children=[
                    Check(id="check_thresholding", text="Auto Threshold", checked=True),

                    # These two spinboxes behave differently based on the check
                    SpinBox(id="hist_thresh_min", value=2.0, min_val=0.0, max_val=10.0),
                    SpinBox(id="hist_thresh_max", value=98.0, min_val=0.0, max_val=100.0, enabled=False)
                ]),

                # 3. Fractional Threshold (Splitter 2)
                HBox(children=[
                    Label(text="Fractional Thresh(%):"),
                    SpinBox(id="fractional_threshold", value=1.0, min_val=0.0, max_val=100.0)
                ]),

                # 4. Search Distance (Splitter 4)
                HBox(children=[
                    Label(text="Search Dist (Min/Max):"),
                    SpinBox(id="search_dist_min", value=10.0, max_val=10000),
                    SpinBox(id="search_dist_max", value=25.0, max_val=10000)
                ]),

                # 5. Radius Curvature (Splitter 5)
                HBox(children=[
                    Label(text="Rad Curvature (Min/Max):"),
                    SpinBox(id="rad_curv_min", value=3.3, max_val=10000),
                    SpinBox(id="rad_curv_max", value=10.0, max_val=10000)
                ])
            ]),

            # --- Hidden Status Label ---
            Label(id="lbl_status", text="", visible=False),

            # --- Bottom Row: Progress + Apply ---
            HBox(children=[
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])
        ]
    }
    return UI_SCHEMA
