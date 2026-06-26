from melage.plugins.ui_helpers import (
    HBox, Group, Label, Combo, Check,
    Button, Progress, Reference,
)


def get_schema():
    return {
        "title": "nnInteractive – 3D Interactive Segmentation",
        "layout": "vbox",
        "items": [
            # ── Activate / deactivate ─────────────────────────────────────
            Button(id="btn_toggle_active",
                   text="▶  Activate",
                   default=False),
            # ── Weights ───────────────────────────────────────────────────
            Group(title="Weights", layout="vbox", children=[
                Label(id="lbl_weights_status", text="Checking…"),
                HBox([
                    Button(id="btn_download", text="Download (~400 MB)"),
                    Progress(id="progress_download"),
                ]),
            ]),
            # ── Label management ─────────────────────────────────────────
            Group(title="Label", layout="vbox", children=[
                Label(id="lbl_label_status", text="Label 1"),
                HBox([
                    Button(id="btn_new_label",   text="New Label →"),
                    Button(id="btn_clear_label", text="Clear"),
                ]),
            ]),
            # ── Interaction controls ─────────────────────────────────────
            Combo(id="combo_mode", label="Mode:",
                  options=["Positive ✚ (include)", "Negative ✖ (exclude)"]),
            Check(id="check_live", text="Live update", checked=True),
            HBox([
                Check(id="check_cuda", text="CUDA", checked=True),
                Label(id="lbl_device", text=""),
            ]),
            HBox([
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Segment", default=True),
            ]),
            Reference(
                text=(
                    'Isensee et al. (2024). <i>nnInteractive.</i> '
                    '<a href="https://arxiv.org/abs/2411.19414">arXiv 2411.19414</a>. '
                    'Weights: <a href="https://huggingface.co/nnInteractive/nnInteractive">'
                    'HuggingFace</a>.'
                ),
                title="Reference",
            ),
        ],
    }
