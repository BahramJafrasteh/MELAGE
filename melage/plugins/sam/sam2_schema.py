from melage.plugins.ui_helpers import (
    HBox, Group, Label, Combo, Check, SpinBox,
    Button, Progress, Reference,
)

# SAM 2.1 (Hiera) backbones. Checkpoints are downloaded on first use into
# MELAGE's shared weights directory (see _weights_dir in sam2.py).
_MODEL_REGISTRY = {
    "SAM 2.1 (Tiny)": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "filename": "sam2.1_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    },
    "SAM 2.1 (Small)": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "filename": "sam2.1_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    },
    "SAM 2.1 (Base+)": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "filename": "sam2.1_hiera_base_plus.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    },
    "SAM 2.1 (Large)": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "filename": "sam2.1_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    },
}
_DEFAULT_MODEL = "SAM 2.1 (Small)"


def get_schema():
    return {
        "title": "SAM 2 – Video/3D Segmentation",
        "min_width": 480,
        "layout": "vbox",
        "items": [
            # ── Activate / deactivate ─────────────────────────────────────
            Button(id="btn_toggle_active",
                   text="▶  Activate",
                   default=False),
            Group(title="Model Weights", layout="vbox", children=[
                Combo(id="combo_model", label="Model:",
                      options=list(_MODEL_REGISTRY.keys()),
                      default=_DEFAULT_MODEL),
                Label(id="lbl_weights_status", text="Checking for weights…"),
                HBox([
                    Button(id="btn_download", text="Download Weights"),
                    Progress(id="progress_download"),
                ]),
            ]),
            HBox([
                Label(id="lbl_label_status", text="Label 1"),
                Button(id="btn_new_label", text="New Label →"),
                Button(id="btn_clear_label", text="Clear Label"),
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
                               "Refine from existing mask (current slice)",
                               "Propagate current label mask"],
                      default="Propagate current label mask"),
            ]),
            Group(title="Propagation range", layout="hbox", children=[
                Check(id="check_limit_range", text="Limit range to", checked=True),
                SpinBox(id="spin_prop_range", value=60, min_val=1,
                        max_val=99999, step=10, decimals=0),
                Label("slices/frames each direction"),
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
                    'Ravi et al. (2024). <i>SAM 2: Segment Anything in Images and '
                    'Videos.</i> <a href="https://ai.meta.com/sam2/">Meta AI</a>.<br>'
                    'Propagation uses SAM 2\'s native memory-attention engine — '
                    'each new frame is segmented using the model\'s memory of '
                    'previously segmented frames, not as an isolated 2-D slice.<br>'
                    'Weights are downloaded automatically on first use.'
                ),
                title="Reference",
            ),
        ],
    }
