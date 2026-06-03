from ..ui_helpers import Label, Combo, Check, Radio, Group, Button, Progress, HBox, SpinBox


def get_schema():
    UI_SCHEMA = {
        "title": "SAM 2 Video/3D Segmentation",
        "min_width": 480,
        "layout": "vbox",
        "items": [
            Label(text="<h3>Interactive 3D Segmentation</h3>"),


            # 2. SUB-VOLUME RANGE (New!)
            Group(id="group_range", title="Slice Range (Crop)", layout="hbox", children=[
                Check(id="check_limit_range", text="Limit Range", checked=True),
                SpinBox(id="spin_start", label="Start:", min_val=0, max_val=99999, value=0),
                SpinBox(id="spin_end", label="End:", min_val=0, max_val=99999, value=100)
            ]),

            # 3. PROMPTING
            Group(id="group_prompt", title="Initialization Prompt", layout="vbox", children=[
                Combo(id="combo_mode", label="Mode:",
                      options=["Existing Labels (Proxy)"],#"Point Click (Coordinates)", "Auto-Mask (Slice)", ],
                      default="Existing Labels (Proxy)"),

                # Point/Slice UI
                Group(id="group_coords", layout="hbox", children=[
                    SpinBox(id="spin_prompt_slice", label="Slice (Z):", min_val=0, max_val=99999, value=0),
                    SpinBox(id="spin_x", label="X:", min_val=0, max_val=9999, value=256),
                    SpinBox(id="spin_y", label="Y:", min_val=0, max_val=9999, value=256),
                ]),
                Label(text="<i>Tip: 'Existing Labels' uses masks already loaded in the input proxy.</i>")
            ]),

            # 4. CONFIGURATION
            Group(id="group_config", title="Model Settings", layout="vbox", children=[
                Combo(id="combo_model_size", label="Model Size:",
                      options=["large", "base_plus", "small", "tiny"],
                      default="tiny"),
                Check(id="check_cuda", text="Use CUDA (GPU)", checked=True)
            ]),

            # 5. EXECUTION
            Button(id="btn_apply", text="Run SAM 2"),
            Progress(id="progress_bar", value=0)
        ]
    }
    return UI_SCHEMA