# ui_schema.py

def Label(text, id=None, visible=True, style=None):
    return {"type": "label", "text": text, "id": id, "visible": visible, "style": style}



def Combo(id, options, label=None, default=None):
    combo = {"type": "combobox", "id": id, "options": options, "default": default}
    if label:
        return {"type": "container", "layout": "hbox", "children": [
            Label(label), combo
        ]}
    return combo

def Check(id, text, checked=False):
    return {"type": "checkbox", "id": id, "text": text, "checked": checked}

def Toggle(id, text, checked=False, color="#FFB000"):
    """For your AnimatedToggle"""
    return {"type": "toggle", "id": id, "text": text, "checked": checked, "color": color}

def Group(id, title, layout="vbox", children=None):
    if children is None: children = []
    return {"type": "group", "id": id, "title": title, "layout_type": layout, "children": children}

def Separator():
    return {"type": "separator"}



def Slider(id, label, label_id=None, min_val=0, max_val=100, default=0):
    """
    id: The core ID (e.g. 't2_1').
        -> Creates slider 'hs_t2_1'
        -> Creates value label 'lb_t2_1'
    label_id: The ID for the title text (e.g. 'lb_ft2_1').
    """
    return {
        "type": "slider_group",
        "label": label,
        "label_id": label_id, # <--- Added this
        "slider": {"id": id, "min": min_val, "max": max_val, "value": default}
    }


###  Layout Helpers ---
def TabWidget(id, tabs):
    """
    tabs: List of tuples/dicts:
          [{"title": "Tab 1", "layout": "vbox", "children": [...]}, ...]
    """
    return {"type": "tab_widget", "id": id, "tabs": tabs}

def Splitter(orientation, children, id=None, sizes=None):
    """
    orientation: 'h' (horizontal) or 'v' (vertical)
    """
    return {
        "type": "splitter",
        "id": id,
        "orientation": orientation,
        "children": children,
        "sizes": sizes
    }
def Spacer(orientation='v'):
    """
    Adds a spacer to push content (vertical or horizontal).
    """
    return {"type": "spacer", "orientation": orientation}

def Radio(id, text, checked=False, enabled=True):
    return {
        "type": "radio", "id": id, "text": text,
        "checked": checked, "enabled": enabled
    }
def Grid(children, id=None):
    """
    children: List of dicts, each having a 'widget' schema and grid coords.
    Example: [ {"row": 0, "col": 0, "item": Label(...)}, ... ]
    """
    return {"type": "grid", "id": id, "children": children}

def GridItem(row, col, item, row_span=1, col_span=1, alignment=None):
    """Helper to wrap an item for the Grid"""
    return {"row": row, "col": col, "row_span": row_span, "col_span": col_span, "item": item, "alignment": alignment}

# --- Custom GL Helpers ---
def GLSurface(id, gl_type, window_name, colors, parent_override=None):
    """
    Definition for your custom GLWidget.
    parent_override: needed if your GLWidget logic strictly requires a specific parent object
                     during init (like self.mutulaViewTab in your code).
    """
    return {
        "type": "gl_surface",
        "id": id,
        "gl_type": gl_type,
        "window_name": window_name,
        "colors": colors,
        "parent_override": parent_override
    }

def GLScientific(id, colors):
    return {
        "type": "gl_scientific",
        "id": id,
        "colors": colors
    }


def MedicalView(id, window_name, img_type, colors):
    """
    Defines a composite widget containing:
    1. GLWidget (The Image)
    2. Label (The Overlay/Text)
    3. Custom Slider (For slicing/cutting)

    The builder will auto-generate IDs:
    - openGLWidget_{id}
    - horizontalSlider_{id}
    - label_{id}
    """
    return {
        "type": "medical_view",
        "id": id,
        "window_name": window_name,
        "img_type": img_type,
        "colors": colors
    }