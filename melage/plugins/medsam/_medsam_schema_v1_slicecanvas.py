from melage.plugins.ui_helpers import (
    HBox, Group, Label, Combo, Check, SpinBox,
    Button, Progress, Reference,
)


def get_schema():
    return {
        "title": "MedSAM – Medical Segment Anything",
        "min_width": 520,
        "layout": "vbox",
        "items": [
            # --- Weights ---
            Group(title="Model Weights", layout="vbox", children=[
                Label(id="lbl_weights_status", text="Checking for weights…"),
                HBox([
                    Button(id="btn_download", text="Download Weights (~2.4 GB)"),
                    Progress(id="progress_download"),
                ]),
            ]),
            # --- Draw mode + clear (canvas is inserted here programmatically) ---
            HBox([
                Combo(id="combo_mode", label="Draw mode:",
                      options=["Bounding Box", "Point  ✚  positive", "Point  ✖  negative"]),
                Button(id="btn_clear", text="Clear"),
            ]),
            # --- Slice navigation ---
            HBox([
                Combo(id="combo_axis", label="Axis:",
                      options=["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]),
                SpinBox(id="spin_slice", label="Slice:", value=0,
                        min_val=0, max_val=9999, step=1, decimals=0),
                Check(id="check_all_slices", text="All slices"),
            ]),
            Check(id="check_cuda", text="Use CUDA if available", checked=True),
            HBox([
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Segment", default=True),
            ]),
            Reference(
                text=(
                    'Ma et al. (2024). <i>Segment anything in medical images.</i> '
                    '<a href="https://doi.org/10.1038/s41467-024-44824-z">Nature Communications</a>.<br>'
                    'Weights: <a href="https://zenodo.org/records/10689643">Zenodo 10689643</a> '
                    '— downloaded automatically on first use.'
                ),
                title="Reference",
            ),
        ],
    }
