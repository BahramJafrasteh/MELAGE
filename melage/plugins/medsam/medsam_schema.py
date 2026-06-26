from melage.plugins.ui_helpers import (
    HBox, Group, Label, Combo, Check, SpinBox,
    Button, Progress, Reference,
)

# MedSAM: SAM ViT-B fine-tuned on medical images.
_MODEL_REGISTRY = {
    "MedSAM (ViT-B)": {
        "arch": "vit_b",
        "filename": "medsam_vit_b.pth",
        "urls": ["https://zenodo.org/records/10689643/files/medsam_vit_b.pth"],
        "manual_url": "https://zenodo.org/records/10689643",
    },
}
_DEFAULT_MODEL = "MedSAM (ViT-B)"


def get_schema():
    return {
        "title": "MedSAM – Medical Segment Anything",
        "min_width": 480,
        "layout": "vbox",
        "items": [
            # ── Activate / deactivate ─────────────────────────────────────
            Button(id="btn_toggle_active",
                   text="▶  Activate",
                   default=False),
            Group(title="Model Weights", layout="vbox", children=[
                Label(id="lbl_weights_status", text="Checking for weights…"),
                HBox([
                    Button(id="btn_download", text="Download Weights (~2.4 GB)"),
                    Progress(id="progress_download"),
                ]),
            ]),
            HBox([
                Combo(id="combo_mode", label="Draw mode:",
                      options=["Bounding Box",
                               "Point  ✚  positive",
                               "Point  ✖  negative"]),
                Button(id="btn_clear", text="Clear"),
            ]),
            HBox([
                Combo(id="combo_axis", label="View plane:",
                      options=["Axial", "Coronal", "Sagittal"]),
                Combo(id="combo_scope", label="Scope:",
                      options=["Current slice",
                               "All slices — fixed box",
                               "All slices — propagate",
                               "Propagate current label mask"],
                      default="Propagate current label mask"),
            ]),
            # Propagation options (only relevant for "propagate" scope)
            Group(title="Propagation settings", layout="vbox", children=[
                HBox([
                    Label("Stop when mask area <"),
                    SpinBox(id="spin_min_area", value=50, min_val=1,
                            max_val=99999, step=10, decimals=0),
                    Label("px"),
                ]),
                HBox([
                    Label("Stop when IoU confidence <"),
                    SpinBox(id="spin_iou_threshold", value=50, min_val=1,
                            max_val=99, step=5, decimals=0),
                    Label("% (lower = keep going longer)"),
                ]),
            ]),
            HBox([
                Check(id="check_cuda", text="Use CUDA if available", checked=True),
                Label(id="lbl_device", text=""),
            ]),
            HBox([
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Segment", default=True),
                Button(id="btn_stop", text="Stop", enabled=False),
            ]),
            Reference(
                text=(
                    'Ma et al. (2024). <i>Segment anything in medical images.</i> '
                    '<a href="https://doi.org/10.1038/s41467-024-44824-z">Nature Communications</a>.<br>'
                    'Weights: <a href="https://zenodo.org/records/10689643">Zenodo 10689643</a>'
                    ' — downloaded automatically on first use.'
                ),
                title="Reference",
            ),
        ],
    }
