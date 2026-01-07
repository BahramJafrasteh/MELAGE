# ui_builder.py
from PyQt5 import QtWidgets, QtCore, QtGui
from melage.widgets.toggle import AnimatedToggle  # Import your custom toggle
from melage.rendering.DisplayIm import GLWidget
from melage.rendering.glScientific import glScientific
from melage.dialogs.helpers import custom_qscrollbar
class UIBuilder:
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.widgets = {}

    def build(self, schema, layout, context=None):
        """
        Args:
            schema: List of UI definitions.
            layout: The layout to add widgets to.
            context: The object (usually 'self') to attach variables to.
                     This fixes the AttributeError.
        """
        row = 0

        for item in schema:
            widget = None
            ########## OPEN GL ##########


            # --- NEW: MEDICAL VIEW COMPONENT ---
            if item['type'] == 'medical_view':
                view_id = item['id']


                # A. Create a Container for this view
                #container = QtWidgets.QWidget()

                # CRITICAL FIX: Ensure the container expands to fill the grid cell
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                #container.setSizePolicy(sizePolicy)

                # CRITICAL FIX: Remove all padding inside the container
                #container_layout = QtWidgets.QVBoxLayout(container)
                #container_layout.setContentsMargins(0, 0, 0, 0)
                #container_layout.setSpacing(0)

                # B. Create GLWidget
                gl_widget = GLWidget(
                    item['colors'],
                    self.parent,
                    imdata=None,
                    currentWidnowName=item['window_name'],
                    type=item['img_type'],
                    id=view_id
                )
                gl_widget.setObjectName(f"openGLWidget_{view_id}")
                gl_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
                # Ensure GLWidget itself tries to expand
                gl_widget.setSizePolicy(sizePolicy)
                gl_widget.setVisible(False)

                if context: setattr(context, f"openGLWidget_{view_id}", gl_widget)

                # C. Create Custom Slider
                slider = custom_qscrollbar(self.parent, id=view_id)
                slider.setObjectName(f"horizontalSlider_{view_id}")
                slider.setOrientation(QtCore.Qt.Vertical)

                if context: setattr(context, f"horizontalSlider_{view_id}", slider)

                # D. Create Label
                label = QtWidgets.QLabel(self.parent)
                label.setObjectName(f"label_{view_id}")
                label.setAlignment(QtCore.Qt.AlignCenter)

                if context: setattr(context, f"label_{view_id}", label)

                # E. Layout Organization (Label Top, GL Middle, Slider Right/Bottom?)
                # Stacking vertical is usually safest for 2x3 grids
                #container_layout.addWidget(label)
                #container_layout.addWidget(gl_widget)
                #container_layout.addWidget(slider)

                # Add to parent
                #layout.addWidget(container)



            # ... inside the build loop ...



            elif item['type'] == 'tab_widget':
                tab_widget = QtWidgets.QTabWidget()
                if item.get('id') and context: setattr(context, item['id'], tab_widget)

                for tab_data in item['tabs']:
                    page = QtWidgets.QWidget()
                    # Determine layout for the tab page
                    if tab_data.get('layout') == 'grid':
                        page_layout = QtWidgets.QGridLayout(page)
                    else:
                        page_layout = QtWidgets.QVBoxLayout(page)

                    # Recursive Build
                    self.build(tab_data['children'], page_layout, context)
                    tab_widget.addTab(page, tab_data['title'])

                layout.addWidget(tab_widget)

            # --- 2. SPLITTERS ---
            elif item['type'] == 'splitter':
                orientation = QtCore.Qt.Horizontal if item['orientation'] == 'h' else QtCore.Qt.Vertical
                splitter = QtWidgets.QSplitter(orientation)

                # Splitters are tricky: they add widgets directly, not layouts.
                # We need a temporary "layout-like" mechanics or just loop children.
                for child_schema in item['children']:
                    # We wrap each child schema in a list so we can pass it to build
                    # But build expects a layout. Splitter takes widgets.
                    # Solution: Create a container for the child.
                    container = QtWidgets.QWidget()
                    # Default to VBox for splitter panes unless specified
                    container_layout = QtWidgets.QVBoxLayout(container)
                    container_layout.setContentsMargins(0, 0, 0, 0)

                    self.build([child_schema], container_layout, context)
                    splitter.addWidget(container)

                if item.get('id') and context: setattr(context, item['id'], splitter)
                layout.addWidget(splitter)

            # --- 3. GRID LAYOUTS ---
            elif item['type'] == 'grid':
                # Note: parent_layout must be a QGridLayout or we create a container
                if isinstance(layout, QtWidgets.QGridLayout):
                    target_layout = layout
                else:
                    container = QtWidgets.QWidget()
                    target_layout = QtWidgets.QGridLayout(container)
                    layout.addWidget(container)

                for cell in item['children']:
                    # We need to build the widget first.
                    # We can use a temporary dummy layout to catch the built widget,
                    # then move it to the grid.
                    dummy = QtWidgets.QWidget()
                    dummy_layout = QtWidgets.QVBoxLayout(dummy)
                    self.build([cell['item']], dummy_layout, context)

                    # Extract the widget we just built
                    if dummy_layout.count() > 0:
                        built_widget = dummy_layout.itemAt(0).widget()
                        if built_widget:
                            # Add to Grid
                            target_layout.addWidget(built_widget, cell['row'], cell['col'], cell['row_span'],
                                                    cell['col_span'])

            # --- 4. CUSTOM GL WIDGETS ---
            elif item['type'] == 'gl_surface':
                # Note: You need to ensure GLWidget is imported
                # In your code, you passed 'self.mutulaViewTab' as parent.
                # Here we use the builder's parent or the layout's parent.
                parent = item.get('parent_override')

                # Create the widget
                # Assuming constructor: GLWidget(colors, parent, imdata, windowName, type, id)
                widget = GLWidget(
                    item['colors'],
                    parent,
                    imdata=None,
                    currentWidnowName=item['window_name'],
                    type=item['gl_type'],
                    id=item['id']
                )
                widget.setFocusPolicy(QtCore.Qt.StrongFocus)
                widget.setVisible(False)  # As per your code

                # Naming and Saving
                widget_name = f"openGLWidget_{item['id']}"
                widget.setObjectName(widget_name)
                if context: setattr(context, widget_name, widget)

                layout.addWidget(widget)

            elif item['type'] == 'gl_scientific':
                # Assuming constructor: glScientific(colors, parent, id)
                widget = glScientific(item['colors'], None, id=item['id'])
                widget.initiate_actions()
                widget.setFocusPolicy(QtCore.Qt.StrongFocus)

                widget_name = f"openGLWidget_{item['id']}"
                widget.setObjectName(widget_name)
                if context: setattr(context, widget_name, widget)

                layout.addWidget(widget)

            # --- 1. NESTED GROUPS & CONTAINERS (Recursive Step) ---
            if item['type'] in ['group', 'container']:
                # Create the container widget
                if item['type'] == 'group':
                    # Visible GroupBox with Title
                    container = QtWidgets.QGroupBox(item['title'])
                    if item.get('id') and context: setattr(context, item['id'], container)
                else:
                    # Invisible Container (QWidget)
                    container = QtWidgets.QWidget()

                # Create the Layout for this container
                if item['layout_type'] == 'hbox':
                    container_layout = QtWidgets.QHBoxLayout(container)
                else:
                    container_layout = QtWidgets.QVBoxLayout(container)

                # Remove margins for cleaner nesting (optional)
                container_layout.setContentsMargins(5, 5, 5, 5)

                # *** RECURSION *** # Build the children into this new layout
                self.build(item['children'], container_layout, context)

                # Add the finished container to the parent
                layout.addWidget(container)
                continue
            ##########OTHER WIDGET TYPES ##########
            # --- 2. RADIO BUTTONS ---
            elif item['type'] == 'radio':
                widget = QtWidgets.QRadioButton(item['text'])
                widget.setChecked(item['checked'])

                if item.get('id') and context:
                    setattr(context, item['id'], widget)
                    # For radio buttons, we often want strict saving/loading
                    widget.setObjectName(item['id'])

                layout.addWidget(widget)

            # --- Separator ---
            if item['type'] == 'separator':
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Sunken)
                line.setStyleSheet("background-color: rgb(50,50,50)")
                layout.addWidget(line)#, row, 0, 1, 1)
                row += 1
                continue

            # --- Label ---
            elif item['type'] == 'label':
                widget = QtWidgets.QLabel(item['text'])
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
                widget.setSizePolicy(sizePolicy)
                if item.get('style'): widget.setStyleSheet(item['style'])

                layout.addWidget(widget)#, row, 0, 1, 1)

                # Assign to self (Fixes 'has no attribute lb_ft2_1')
                if item.get('id') and context:
                    setattr(context, item['id'], widget)

                row += 1

            # --- Slider Group (Title + Value + Slider) ---
            elif item['type'] == 'slider_group':
                # 1. Title Label
                lbl_title = QtWidgets.QLabel(item['label'])
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

                lbl_title.setSizePolicy(sizePolicy)
                layout.addWidget(lbl_title)#, row, 0, 1, 1)

                # Assign Title Label ID (e.g., lb_ft2_1)
                if item.get('label_id') and context:
                    setattr(context, item['label_id'], lbl_title)
                row += 1

                # 2. Value Label
                val = item['slider'].get('value', 0)
                lbl_val = QtWidgets.QLabel(str(val))
                lbl_val.setAlignment(QtCore.Qt.AlignCenter)

                layout.addWidget(lbl_val)#, row, 0, 1, 1)

                # Assign Value Label ID (e.g., lb_t2_1)
                # We assume standard naming "lb_" + slider_id if not provided
                slider_id = item['slider'].get('id')
                val_lbl_id = f"lb_{slider_id}" if slider_id else None

                if val_lbl_id and context:
                    setattr(context, val_lbl_id, lbl_val)

                row += 1

                # 3. Slider
                slider_data = item['slider']
                slider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
                slider.setRange(slider_data['min'], slider_data['max'])

                slider.setObjectName(slider_id)  # Important for getAttributeWidget

                # Connect signal
                slider.valueChanged.connect(lbl_val.setNum)


                layout.addWidget(slider)#, row, 0, 1, 1)
                #slider.setValue(-1001)
                slider.setValue(val)
                # Assign Slider ID (e.g., hs_t2_1)
                # We assume standard naming "hs_" + slider_id if not provided
                slider_var_name = f"{slider_id}" if slider_id else None

                if slider_var_name and context:
                    setattr(context, slider_var_name, slider)

                row += 1


            # --- Combobox ---
            elif item['type'] == 'combobox':
                widget = QtWidgets.QComboBox()
                widget.addItems(item['options'])
                cbstyle = """
                    QComboBox QAbstractItemView {border: 1px solid grey; background: white; selection-background-color: #03211c;} 
                    QComboBox {background: #03211c; margin-right: 1px;}
                    QComboBox::drop-down {subcontrol-origin: margin;}
                """
                widget.setStyleSheet(cbstyle)
                layout.addWidget(widget)#, row, 0, 1, 1)

                if item.get('id') and context:
                    setattr(context, item['id'], widget)
                row += 1

            # --- Toggle / Checkbox ---
            elif item['type'] in ['toggle', 'checkbox']:
                if item['type'] == 'toggle':
                    widget = AnimatedToggle()#checked_color=item.get('color', "#FFB000")
                else:
                    widget = QtWidgets.QCheckBox(item['text'])

                widget.setChecked(item['checked'])
                layout.addWidget(widget)#, row, 0, 1, 1)

                if item.get('id') and context:
                    setattr(context, item['id'], widget)
                row += 1

        return self.widgets