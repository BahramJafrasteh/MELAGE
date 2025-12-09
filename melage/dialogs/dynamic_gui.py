from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QDialog, QWidget, QLabel, QComboBox, QCheckBox,
                             QGroupBox, QRadioButton, QPushButton, QProgressBar,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QMenu, QAction)
from melage.plugins.ui_helpers import *

class DynamicDialog(QDialog):
    """
    A Dialog that builds its own UI based on a dictionary schema.

    It supports:
    - Recursive layouts (Groups, Containers)
    - Widget Registry (access widgets by ID)
    - Automatic Context Menus
    """

    def __init__(self, schema, parent=None):
        super().__init__(parent)
        self.schema = schema
        self.widgets = {}  # Registry to find widgets by ID: self.widgets["my_id"]

        # 1. Setup Window Properties

    def update_data_context(self, data_context):
        """
        Update the data context for the dialog.
        """
        self.data_context = data_context
    def create_main_ui(self, schema, default_items=True):
        """
        Main method to create the UI based on the provided schema dictionary.
        This method can be called from __init__ or overridden in subclasses.
        """
        self.setWindowTitle(schema.get("title", "Dynamic Dialog"))
        if "min_width" in schema:
            self.setMinimumWidth(schema["min_width"])
        if "min_height" in schema:
            self.setMinimumHeight(schema["min_height"])
        self.layout_type = schema.get("layout", "vbox")
        self.main_layout = self._create_layout(self.layout_type, self)

        schema["items"].insert(0,
                Combo(id="combo_view", label="Image View:", options=["view 1", "view 2"], function_name=None),
                                   )


        if default_items:
            schema["items"].append(
            Group(title="Configuration", children=[
                Check(id="check_cuda", text="Use CUDA (GPU) if available"),

                # --- NEW: Custom Weights Loader ---
                FilePicker(
                    id="weights",
                    check_label="Custom Weights (.pth):", # it can be label or check lab
                    placeholder="Default (Pre-trained)",
                    file_filter="PyTorch Models (*.pth *.pt);;All Files (*)"
                ),
                # ----------------------------------

                #Label(id="lbl_model_info", text="Model Info: Ready")
            ]))
            schema["items"].append(
            # 5. Bottom Row (Progress Bar + Apply Button)
            HBox([
                Progress(id="progress_bar"),
                Button(id="btn_apply", text="Apply", default=True)
            ])
            )
        # 3. Build Items
        # We iterate through the list of items in the schema and build them
        for item in schema.get("items", []):
            self._build_item(item, self.main_layout)

    def get_widget(self, widget_id):
        """
        Public method to retrieve a specific widget object by its ID string.
        Example: self.get_widget("btn_apply")
        """
        return self.widgets.get(widget_id)

    def _create_layout(self, layout_type, parent_widget):
        """Helper to create the correct layout object."""
        if layout_type == "hbox":
            l = QHBoxLayout(parent_widget)
        elif layout_type == "grid":
            l = QGridLayout(parent_widget)
        else:
            # Default to Vertical
            l = QVBoxLayout(parent_widget)

        # If this layout is inside a generic Container (QWidget), remove margins
        # so it aligns perfectly with the rest of the UI.
        # We keep margins if it's the Main Dialog or a GroupBox.
        if isinstance(parent_widget, QWidget) and \
                not isinstance(parent_widget, QDialog) and \
                not isinstance(parent_widget, QGroupBox):
            l.setContentsMargins(0, 0, 0, 0)

        return l

    # Inside your DynamicDialog class

    def _handle_browse_click(self, target_id, file_filter):
        """
        Opens the file dialog and updates the target LineEdit.
        """
        # 1. Open the Dialog
        dir_used = ""
        if hasattr(self, "_last_dir"):
            dir_used = self._last_dir
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            dir_used,
            file_filter
        )

        # 2. If user selected a file...
        if path:
            # 3. Find the target LineEdit by ID
            target_widget = self.widgets.get(target_id)

            if isinstance(target_widget, QtWidgets.QLineEdit):
                target_widget.setText(path)
                setattr(self, target_id+"_custom", path)
                # Optional: Move focus away so the text isn't highlighted
                target_widget.clearFocus()
            else:
                print(f"Error: Target widget {target_id} not found or not a LineEdit")

    def _toggle_widgets(self, is_checked, target_ids):
        """
        Enables or disables a list of widgets based on a checkbox state.
        """
        for tid in target_ids:
            widget = self.widgets.get(tid)
            if widget:
                widget.setEnabled(is_checked)
            else:
                # Safe fail: might happen if targets are built AFTER the checkbox
                # (Though usually not an issue in VBox layouts)
                pass




    def _build_item(self, item_def, parent_layout):
        """
        Recursively builds a widget from a definition dictionary
        and adds it to the parent_layout.
        """
        widget = None
        w_type = item_def.get("type")
        w_id = item_def.get("id")
        #print(w_type)
        # --- A. Widget Factory ---
        if w_type == "label":
            widget = QLabel(item_def.get("text", ""))

            if item_def.get('mode') == 'reference':
                # 1. Allow HTML links (<a href="...">) to open browser
                widget.setOpenExternalLinks(True)

                # 2. Allow user to highlight/copy text with mouse
                widget.setTextInteractionFlags(
                    QtCore.Qt.TextBrowserInteraction |
                    QtCore.Qt.TextSelectableByMouse
                )

                # 3. Optional styling (make it look like a footnote)
                widget.setStyleSheet("""
                                QLabel {
                                    background-color: white;
                                    color: #555;
                                    font-style: italic;
                                    margin: 5px;
                                    padding: 5px;
                                    border: 1px solid #ddd; /* Optional: adds a subtle border */
                                }
                            """)
            else:
                widget.setWordWrap(item_def.get('word_wrap', False))
            visible = item_def.get("visible", True)
            widget.setVisible(visible)


        elif w_type == "combobox":
            widget = QComboBox()
            widget.addItems(item_def.get("options", []))
            default_option = item_def.get("default", None)
            if default_option and default_option in item_def.get("options", []):
                widget.setCurrentIndex(item_def.get("options", []).index(default_option))
            #func = getattr(self, item_def.get('id'))
            if item_def.get('function_name') is not None:
                widget.currentIndexChanged.connect(getattr(self, item_def.get('function_name')))


        elif w_type == 'spinbox':
            # Create QDoubleSpinBox (handles both floats and ints well)
            widget = QtWidgets.QDoubleSpinBox()
            # Configure properties
            enabled = item_def.get('enabled', True)
            widget.setMinimum(item_def.get('min', 0.0))
            widget.setMaximum(item_def.get('max', 100.0))
            widget.setSingleStep(item_def.get('step', 1.0))
            widget.setValue(item_def.get('value', 0.0))

            # Handle decimals (set to 0 for integer behavior)
            decimals = item_def.get('decimals', 2)
            widget.setDecimals(decimals)
            widget.setEnabled(enabled)

        elif w_type == "checkbox":
            widget = QCheckBox(item_def.get("text", ""))
            if item_def.get("checked"):
                widget.setChecked(True)
            targets = item_def.get('enable_targets')
            if targets:
                # Connect the toggle signal to our helper
                # rigid=False means: checked=True -> Enabled, checked=False -> Disabled
                widget.toggled.connect(lambda checked: self._toggle_widgets(checked, targets))

        elif w_type == "radio":
            widget = QRadioButton(item_def.get("text", ""))
            if item_def.get("checked"):
                widget.setChecked(True)

        elif w_type == "button":
            widget = QPushButton(item_def.get("text", "Button"))
            if item_def.get("default"):
                widget.setDefault(True)

            # 2. Handle FILE BROWSE Logic
            if item_def.get('special_mode') == 'file_browse':
                target_id = item_def.get('target_id')
                #w_id = target_id
                file_filter = item_def.get('file_filter', "All Files (*)")

                # Connect the signal to a generic handler in this class
                widget.clicked.connect(lambda: self._handle_browse_click(target_id, file_filter))
        elif w_type=='line_edit':
            widget = QtWidgets.QLineEdit()
            widget.setText(item_def.get("text", ""))
            widget.setPlaceholderText(item_def.get("placeholder", ""))
            widget.setReadOnly(item_def.get("read_only", False))
            if "width" in item_def:
                if item_def['width'] is not None:
                    widget.setFixedWidth(item_def["width"])


        elif w_type == "progressbar":
            widget = QProgressBar()
            widget.setRange(item_def.get("min", 0), item_def.get("max", 100))
            widget.setValue(item_def.get("value", 0))

        # --- B. Container Recursion ---
        # GroupBoxes and Containers have 'children', so we recurse.

        elif w_type == "groupbox":
            widget = QGroupBox(item_def.get("title", ""))
            # Create a layout inside the groupbox
            layout_str = item_def.get("layout", "vbox")
            layout = self._create_layout(layout_str, widget)
            widget.setLayout(layout)

            # Build children into this new layout
            for child in item_def.get("children", []):
                self._build_item(child, layout)

        elif w_type == "container":
            # A generic QWidget used to hold a sub-layout (like HBox inside VBox)
            widget = QWidget()
            layout_str = item_def.get("layout", "vbox")
            layout = self._create_layout(layout_str, widget)
            widget.setLayout(layout)

            for child in item_def.get("children", []):
                self._build_item(child, layout)

        # --- C. Common Properties & Registration ---
        if widget:
            # 1. Register ID
            if w_id:
                widget.setObjectName(w_id)
                self.widgets[w_id] = widget

            # 2. Handle Enabled/Disabled state
            if "enabled" in item_def and not item_def["enabled"]:
                widget.setEnabled(False)

            # 3. Context Menu Wiring
            if "context_menu" in item_def:
                widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
                # We use a lambda to pass the specific menu items list to the handler
                widget.customContextMenuRequested.connect(
                    lambda pos, w=widget, items=item_def["context_menu"]:
                    self._show_context_menu(pos, w, items)
                )

            # 4. Add to Parent Layout
            # If the parent is a Grid, we look for row/col, otherwise just addWidget
            if isinstance(parent_layout, QGridLayout):
                row = item_def.get("row", 0)
                col = item_def.get("col", 0)
                row_span = item_def.get("row_span", 1)
                col_span = item_def.get("col_span", 1)
                parent_layout.addWidget(widget, row, col, row_span, col_span)
            else:
                parent_layout.addWidget(widget)

    def _show_context_menu(self, pos, widget, items):
        """Internal handler to generate the right-click menu."""
        menu = QMenu(self)
        for text in items:
            action = QAction(text, self)
            # Connect the action to the overridable method 'on_context_action'
            action.triggered.connect(lambda _, t=text: self.on_context_action(t, widget))
            menu.addAction(action)
        menu.exec_(widget.mapToGlobal(pos))

    def on_context_action(self, text, widget):
        """
        Virtual method.
        Override this in your subclass (e.g., MorphSegLogic) to handle menu clicks.
        """
        print(f"DEBUG: Context action '{text}' on widget '{widget.objectName()}'")