"""
Helper functions to make UI schemas simple and readable.
"""

def Reference(text, title="Reference / Paper"):
    """
    Creates a collapsible group containing the citation.
    """
    # We use a Label with special flags (handled in the builder next)
    ref_label = {
        "type": "label",
        "text": text,
        "mode": "reference",
        "word_wrap": True
    }

    # Return it wrapped in a Group so it doesn't clutter the UI
    # 'collapsed=True' requires your Group logic to support it (optional)
    return Group(title=title, children=[ref_label])

def Label(text, id=None, visible=True):
    return {"type": "label", "text": text, "id": id, "visible": visible}

def Combo(id, options, label=None, function_name=None, default=None):
    """
    Creates a Combobox.
    If 'label' is provided, it returns a Container with the Label and Combo side-by-side.
    """
    combo = {"type": "combobox", "id": id, "options": options,
             "function_name": function_name, "default": default}
    if label:
        return HBox([Label(label), combo])
    return combo

def Check(id, text, checked=False, context=None):
    return {
        "type": "checkbox", "id": id, "text": text,
        "checked": checked, "context_menu": context
    }

def Radio(id, text, checked=False, enabled=True):
    return {
        "type": "radio", "id": id, "text": text,
        "checked": checked, "enabled": enabled
    }

def Group(children, id=None, title="", layout="vbox", enabled=True):
    return {
        "type": "groupbox", "id": id, "title": title,
        "layout": layout, "children": children, "enabled": enabled
    }


def SpinBox(id, label=None, value=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=2,
            enabled = True):
    """
    Creates a DoubleSpinBox definition.
    For integers, just set decimals=0.
    """
    item = {
        "type": "spinbox",
        "id": id,
        "value": value,
        "min": min_val,
        "max": max_val,
        "step": step,
        "decimals": decimals,
        "enabled": enabled
    }

    # If a label is requested, wrap it in a vertical group
    if label:
        return Group(layout="vbox", children=[
            Label(text=label),
            item
        ])

    return item

def Button(id, text, default=False, enabled=True):
    return {"type": "button", "id": id, "text": text, "default": default, "enabled": enabled}

def Progress(id, min=0, max=100, value=0):
    return {"type": "progressbar", "id": id, "min": min, "max": max, "value": value}

def HBox(children):
    """Puts items side-by-side"""
    return {"type": "container", "layout": "hbox", "children": children}

def VBox(children):
    """Puts items top-to-bottom"""
    return {"type": "container", "layout": "vbox", "children": children}

def LineEdit(id=None, text="", placeholder="", read_only=False, width=None):
    return {
        "type": "line_edit",
        "id": id,
        "text": text,
        "placeholder": placeholder,
        "read_only": read_only,
        "width": width
    }


def FilePicker(id, label=None, check_label =None, placeholder="Select file...", button_text="Browse...", file_filter="All Files (*)"):
    """
    Creates a composite widget: Label (optional) + [LineEdit | Button]
    """
    # 1. Create unique IDs for the internal parts based on the main ID
    line_id = f"{id}_path"
    btn_id = f"{id}_btn"
    check_id = f"{id}_check"
    is_enabled = False if check_label else True

    # 2. Define the Browse Button
    # We add a custom key 'file_filter' here.
    # Your UI renderer will ignore it, but we will read it in the Python logic.
    browse_btn = Button(id=btn_id, text=button_text, enabled=is_enabled)
    browse_btn['special_mode'] = 'file_browse'
    browse_btn['file_filter'] = file_filter  # <--- Storing the filter string here
    browse_btn['target_id'] = line_id  # <--- Storing which line edit to update

    # 3. Define the HBox (Line Edit + Button)
    picker_row = HBox(children=[
        LineEdit(id=line_id, placeholder=placeholder, read_only=True),
        browse_btn
    ])

    # 4. If a label is requested, wrap it all in a VBox (or list), otherwise return just the row
    if check_label:
        # We add a custom key 'enable_targets' that our builder will read
        toggle_check = Check(id=check_id, text=check_label, checked=False)
        toggle_check['enable_targets'] = [line_id, btn_id]
        return Group(layout="vbox", children=[
            toggle_check,
            picker_row
        ])
    if label:
        return Group(layout="vbox", children=[
            Label(text=label),
            picker_row
        ])

    return picker_row