__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from melage.rendering.DisplayIm import GLWidget
from melage.rendering.glScientific import glScientific
from melage.utils.utils import update_color_scheme
from .ui_schema import *
from functools import partial
from .ui_builder import UIBuilder
from melage.dialogs.helpers import custom_qscrollbar

class openglWidgets():
    """
    Maing OPENGL WIDGETS
    """
    def __init__(self):
        pass

    def create_mutual_view(self, colorsCombinations):
        """
        Creates the 6-view grid (Eco + MRI) using the UIBuilder schema.
        Matches the 'Full Size' layout of the original code.
        """
        self.mutulaViewTab = QtWidgets.QWidget()
        self.mutulaViewTab.setObjectName("tab1")  # ID matches your old code

        # 1. Setup Grid with Zero Margins (Matches your old code)
        self.gridLayout = QtWidgets.QGridLayout(self.mutulaViewTab)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)




        # 2. Define the Schema (2 Rows x 3 Cols)
        # Row 0: ECO (Coronal, Sagittal, Axial) -> IDs 1, 2, 3
        # Row 1: MRI (Coronal, Sagittal, Axial) -> IDs 4, 5, 6
        schema = [

            # --- Row 0: ECO ---
            MedicalView(id=1, window_name="coronal", img_type="eco", colors=colorsCombinations),
            MedicalView(id=2, window_name="sagittal", img_type="eco", colors=colorsCombinations),
            MedicalView(id=3, window_name="axial", img_type="eco", colors=colorsCombinations),

            # --- Row 1: MRI ---
            MedicalView(id=4, window_name="coronal", img_type="mri", colors=colorsCombinations),
            MedicalView(id=5, window_name="sagittal", img_type="mri", colors=colorsCombinations),
            MedicalView(id=6, window_name="axial", img_type="mri", colors=colorsCombinations),

        ]

        # 3. Build it
        builder = UIBuilder(self.mutulaViewTab)
        builder.build(schema, self.gridLayout, context=self)

        # 4. Re-connect Signals (Since we replaced the manual loop)
        for i in range(1, 7):
            try:
                slider = getattr(self, f"horizontalSlider_{i}")
                # Connect the cut_limit signal
                slider.cut_limit.connect(lambda val, vid=i: self._cutIM(val, vid))
            except AttributeError:
                pass  # Handle case where widget creation might have failed
        #self.create_vertical_mutualview()

        #self.create_horizontal_mutualview()

    def create_mutual_view0(self, colorsCombinations):
        """
        Initializes the 6 views (3 ECO, 3 MRI) and their controllers.
        """
        self.mutulaViewTab = QtWidgets.QWidget()
        self.mutulaViewTab.setObjectName("tab1")

        # Initialize the main grid
        self.gridLayout = QtWidgets.QGridLayout(self.mutulaViewTab)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        # Define configuration for the 6 views
        # Format: (ID, WindowName, Type)
        view_configs = [
            (1, 'coronal', 'eco'),
            (2, 'sagittal', 'eco'),
            (3, 'axial', 'eco'),
            (4, 'coronal', 'mri'),
            (5, 'sagittal', 'mri'),
            (6, 'axial', 'mri')
        ]

        # Loop to create all widgets dynamically
        for view_id, window_name, img_type in view_configs:
            self._init_single_view(colorsCombinations, view_id, window_name, img_type)




        # Now that widgets exist, build the layout
        self.create_vertical_mutualview()

        self.create_horizontal_mutualview()
    def _init_single_view(self, colorsCombinations, view_id, window_name, img_type):
        """
        Helper to initialize one set of (OpenGL + Slider + Label).
        Sets attributes like self.openGLWidget_1, self.horizontalSlider_1, etc.
        """


        # 1. Create OpenGL Widget
        gl_widget = GLWidget(
            colorsCombinations,
            self.mutulaViewTab,
            imdata=None,
            currentWidnowName=window_name,
            type=img_type,
            id=view_id
        )
        gl_widget.setObjectName(f"openGLWidget_{view_id}")
        gl_widget.setFocusPolicy(Qt.StrongFocus)
        gl_widget.setEnabled(True)
        gl_widget.setVisible(False)  # Start hidden until data loads

        # 2. Create Slider (Pre-configured for Vertical Layout)
        slider = custom_qscrollbar(self.mutulaViewTab, id=view_id)
        slider.setObjectName(f"horizontalSlider_{view_id}")
        slider.setOrientation(QtCore.Qt.Vertical)  # <--- Vertical by default now

        # Connect signal (assuming _cutIM exists)
        # Using default arg 'vid=view_id' to capture the value in the lambda closure
        slider.cut_limit.connect(lambda val, vid=view_id: self._cutIM(val, vid))

        # 3. Create Label
        label = QtWidgets.QLabel(self.mutulaViewTab)
        label.setObjectName(f"label_{view_id}")
        label.setAlignment(QtCore.Qt.AlignCenter)

        # 4. Attach to 'self' dynamically
        # This ensures self.openGLWidget_1 etc. still work in the rest of your app
        setattr(self, f"openGLWidget_{view_id}", gl_widget)
        setattr(self, f"horizontalSlider_{view_id}", slider)
        setattr(self, f"label_{view_id}", label)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _make_control_pair(self, view_id, is_vertical, align_right=True):
        """
        Returns a Layout containing [Label + Slider].

        Args:
            view_id: ID of the view (1-6)
            is_vertical: True for vertical sidebar, False for bottom bar
            align_right: (Vertical only) True = Label | Slider. False = Slider | Label.
        """
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)  # Small gap between label and slider

        slider = getattr(self, f"horizontalSlider_{view_id}")
        label = getattr(self, f"label_{view_id}")

        if is_vertical:
            # Vertical Sidebar Logic
            if align_right:
                # Left Side: [ Label | Slider ] -> Pushes against Image
                label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                layout.addWidget(label)
                layout.addWidget(slider)
                layout.setAlignment(QtCore.Qt.AlignRight)
            else:
                # Right Side: [ Slider | Label ] -> Pushes against Image
                label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                layout.addWidget(slider)
                layout.addWidget(label)
                layout.setAlignment(QtCore.Qt.AlignLeft)
        else:
            # Horizontal Bottom Bar Logic: [ Slider -------- Label ]
            # We add the slider first, then the label
            layout.addWidget(slider)
            layout.addWidget(label)

        return layout

    def create_horizontal_mutualview(self):
        #self.clear_layout(self.gridLayout)

        # 1. Configure widgets for Horizontal orientation
        # (Sliders become wide, labels align left/center)
        self._configure_widgets_for_orientation(is_vertical=False)

        # 2. Setup Grid Settings to minimize gaps
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)

        # 3. Define the pairs for the 3 columns
        # Format: (Column Index, Top_ID, Bottom_ID)
        columns = [
            (0, 1, 4),  # Col 0: Coronal
            (1, 2, 5),  # Col 1: Sagittal
            (2, 3, 6)  # Col 2: Axial
        ]

        for col, top_id, bot_id in columns:
            # --- Top Block (Label -> Slider -> Image) ---
            # Note: I am mapping your indices (0, 2, 4) to (0, 1, 2) to ensure tight packing
            self.gridLayout.addWidget(getattr(self, f"label_{top_id}"), 0, col, 1, 1)
            self.gridLayout.addWidget(getattr(self, f"horizontalSlider_{top_id}"), 1, col, 1, 1)
            self.gridLayout.addWidget(getattr(self, f"openGLWidget_{top_id}"), 2, col, 1, 1)

            # --- Bottom Block (Image -> Slider -> Label) ---
            # Note: Mapping indices (5, 7, 9) to (3, 4, 5)
            self.gridLayout.addWidget(getattr(self, f"openGLWidget_{bot_id}"), 3, col, 1, 1)
            self.gridLayout.addWidget(getattr(self, f"horizontalSlider_{bot_id}"), 4, col, 1, 1)
            self.gridLayout.addWidget(getattr(self, f"label_{bot_id}"), 5, col, 1, 1)

        # 4. Set Stretch Factors to remove black spaces
        # All columns share width equally
        """

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 1)

        # Rows 2 and 3 (the Images) get all the height
        self.gridLayout.setRowStretch(0, 0)  # Label
        self.gridLayout.setRowStretch(1, 0)  # Slider
        self.gridLayout.setRowStretch(2, 10)  # Top Image (Expand!)
        self.gridLayout.setRowStretch(3, 10)  # Bot Image (Expand!)
        self.gridLayout.setRowStretch(4, 0)  # Slider
        self.gridLayout.setRowStretch(5, 0)  # Label
                """
        self.setup_layout_expansion()

    def create_vertical_mutualview(self):
        #self.clear_layout(self.gridLayout)

        # 1. Configure widgets for Vertical Mode
        # (Sliders become tall/thin, Labels align correctly)
        self._configure_widgets_for_orientation(is_vertical=True)
        #height = self.height()
        width = round(self.width()*0.1)
        # 2. Setup Grid Settings
        self.gridLayout.setContentsMargins(width, 0, width, 0)
        self.gridLayout.setSpacing(0)

        # 3. Add Widgets (Strictly following your provided rule)

        # --- ROW 0 (Coronal) ---
        self.gridLayout.addWidget(self.label_1, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_1, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_1, 0, 2, 1, 1)  # Image
        self.gridLayout.addWidget(self.openGLWidget_4, 0, 3, 1, 1)  # Image
        self.gridLayout.addWidget(self.horizontalSlider_4, 0, 4, 1, 1)
        self.gridLayout.addWidget(self.label_4, 0, 5, 1, 1)

        # --- ROW 1 (Sagittal) ---
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_2, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_2, 1, 2, 1, 1)  # Image
        self.gridLayout.addWidget(self.openGLWidget_5, 1, 3, 1, 1)  # Image
        self.gridLayout.addWidget(self.horizontalSlider_5, 1, 4, 1, 1)
        self.gridLayout.addWidget(self.label_5, 1, 5, 1, 1)

        # --- ROW 2 (Axial) ---
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_3, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_3, 2, 2, 1, 1)  # Image
        self.gridLayout.addWidget(self.openGLWidget_6, 2, 3, 1, 1)  # Image
        self.gridLayout.addWidget(self.horizontalSlider_6, 2, 4, 1, 1)
        self.gridLayout.addWidget(self.label_6, 2, 5, 1, 1)
        """

        # 4. Set Stretch Factors (The Fix for Black Spaces)
        # Give all horizontal space to Columns 2 and 3 (the Images)
        self.gridLayout.setColumnStretch(0, 0)  # Label Left
        self.gridLayout.setColumnStretch(1, 0)  # Slider Left
        self.gridLayout.setColumnStretch(2, 10)  # Image Left (EXPAND)
        self.gridLayout.setColumnStretch(3, 10)  # Image Right (EXPAND)
        self.gridLayout.setColumnStretch(4, 0)  # Slider Right
        self.gridLayout.setColumnStretch(5, 0)  # Label Right

        # Give equal vertical space to all rows
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 1)
        """
        self.setup_layout_expansion()

    def _configure_widgets_for_orientation(self, is_vertical):
        """
        Helper ensures sliders are rotated and labels are aligned correctly.
        """
        ids = [1, 2, 3, 4, 5, 6]
        for i in ids:
            slider = getattr(self, f"horizontalSlider_{i}")
            label = getattr(self, f"label_{i}")

            if is_vertical:
                # Vertical View: Slider is Vertical, Fixed Width
                slider.setOrientation(QtCore.Qt.Vertical)
                slider.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

                # Label aligns to the slider
                label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
                # Align Right for Left controls (ids 1,2,3), Left for Right controls (ids 4,5,6)
                align = QtCore.Qt.AlignRight if i <= 3 else QtCore.Qt.AlignLeft
                label.setAlignment(align | QtCore.Qt.AlignVCenter)
            else:
                # Horizontal View: Slider is Horizontal
                slider.setOrientation(QtCore.Qt.Horizontal)
                slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                label.setAlignment(QtCore.Qt.AlignCenter)

    def _add_vertical_row(self, row_idx, left_id, right_id):
        """Helper for Vertical View Row Construction"""
        # Left Side: [Label | Slider] ... [Image]
        l_grp = self._make_control_pair(left_id, is_vertical=True, align_right=True)
        self.gridLayout.addLayout(l_grp, row_idx, 0)
        self.gridLayout.addWidget(getattr(self, f"openGLWidget_{left_id}"), row_idx, 1)

        # Right Side: [Image] ... [Slider | Label]
        r_grp = self._make_control_pair(right_id, is_vertical=True, align_right=False)
        self.gridLayout.addWidget(getattr(self, f"openGLWidget_{right_id}"), row_idx, 2)
        self.gridLayout.addLayout(r_grp, row_idx, 3)

    def _add_row_to_grid(self, row, left_ids, right_ids):
        """
        Helper to assemble one row.
        left_ids = (label_id, slider_id, gl_id)
        right_ids = (gl_id, slider_id, label_id)
        """
        # --- Left Side (Control + Image) ---
        l_label = getattr(self, f"label_{left_ids[0]}")
        l_slider = getattr(self, f"horizontalSlider_{left_ids[1]}")
        l_gl = getattr(self, f"openGLWidget_{left_ids[2]}")

        # Group Label+Slider tightly
        left_control = self._make_control_group(l_label, l_slider, is_left=True)

        self.gridLayout.addLayout(left_control, row, 0)  # Col 0
        self.gridLayout.addWidget(l_gl, row, 1)  # Col 1

        # --- Right Side (Image + Control) ---
        r_gl = getattr(self, f"openGLWidget_{right_ids[0]}")
        r_slider = getattr(self, f"horizontalSlider_{right_ids[1]}")
        r_label = getattr(self, f"label_{right_ids[2]}")

        # Group Slider+Label tightly
        right_control = self._make_control_group(r_label, r_slider, is_left=False)

        self.gridLayout.addWidget(r_gl, row, 2)  # Col 2
        self.gridLayout.addLayout(right_control, row, 3)  # Col 3

    def setup_layout_expansion(self):
        # 1. Ensure the Container allows itself to grow
        # This tells the tab/window: "I want to be as big as possible"
        self.mutulaViewTab.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # 2. Ensure the Layout has no dead zones
        # Remove margins so the grid hits the exact edges of the window
        #self.gridLayout.setContentsMargins(0, 0, 0, 0)

        # CRITICAL: Do NOT set alignment on the layout itself.
        # If you do self.gridLayout.setAlignment(Qt.AlignCenter),
        # it forces the grid to SHRINK to its minimum size.
        # By default (without alignment), a layout tries to fill the space.

        # 3. Ensure the Inner Widgets (OpenGL) want to grow
        # You likely did this, but double-check:
        for gl_widget in [self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3,
        self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6]:
            gl_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )


    def _make_control_group(self, label, slider, is_left):
        """Creates the tight QHBoxLayout for controls."""
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        slider.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)

        if is_left:
            # [Label | Slider]
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.setAlignment(QtCore.Qt.AlignRight)
        else:
            # [Slider | Label]
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            layout.addWidget(slider)
            layout.addWidget(label)
            layout.setAlignment(QtCore.Qt.AlignLeft)

        return layout




    def create_view1_tab_old(self, colorsCombinations):
        self.segmentationTab = QtWidgets.QWidget()
        self.segmentationTab.setObjectName("segmentationTab")
        self.gridLayout_seg = QtWidgets.QGridLayout(self.segmentationTab)
        self.gridLayout_seg.setObjectName("gridLayout_seg")
        self.splitter_main_view1 = QtWidgets.QSplitter(self.segmentationTab)
        self.splitter_main_view1.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_main_view1.setObjectName("splitter_main")
        self.splitter_left_view1 = QtWidgets.QSplitter(self.splitter_main_view1)
        self.splitter_left_view1.setOrientation(QtCore.Qt.Vertical)
        self.splitter_left_view1.setObjectName("splitter_left")




        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.openGLWidget_11 = GLWidget(colorsCombinations,self.splitter_left_view1, imdata=None, currentWidnowName='coronal', type='eco', id=11)





        #self.openGLWidget_11.setSizePolicy(sizePolicy)
        #self.openGLWidget_11.setMaximumSize(QtCore.QSize(self.width(), self.height()))
        #self.openGLWidget_11.setObjectName("openGLWidget_11")

        self.splitter_slider = QtWidgets.QSplitter(self.splitter_left_view1)
        self.splitter_slider.setOrientation(QtCore.Qt.Vertical)
        self.splitter_slider.setObjectName("splitter_slider")
        self.label_11 = QtWidgets.QLabel(self.splitter_slider)
        self.splitter_slider.setMinimumHeight(40)
        self.splitter_slider.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(10, 10))
        self.label_11.setMaximumSize(QtCore.QSize(self.width()-self.width()//4, self.height()//44))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")



        self.splitterRadioButton = QtWidgets.QSplitter(self.splitter_left_view1)
        self.splitterRadioButton.setOrientation(QtCore.Qt.Horizontal)
        self.splitterRadioButton.setObjectName("splitterRadioButton")
        self.splitterRadioButton.setSizePolicy(sizePolicy)
        self.splitterRadioButton.setMaximumSize(QtCore.QSize(self.width(),
                                                               self.height() // 44))


        self.splitterRadioButton_3 = QtWidgets.QSplitter(self.splitterRadioButton)
        self.splitterRadioButton_3.setMaximumSize(QtCore.QSize(self.width(),
                                                               self.height() // 44))
        self.splitterRadioButton_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitterRadioButton_3.setObjectName("splitterRadioButton_3")


        self.radioButton_4 = QtWidgets.QCheckBox(self.splitterRadioButton)
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_4.setMaximumSize(QtCore.QSize(self.width(),
                                                               self.height() // 44))
        self.radioButton_4.setChecked(True)
        self.radioButton_4.setSizePolicy(sizePolicy)

        self.radioButton_1 = QtWidgets.QRadioButton(self.splitterRadioButton_3)
        self.radioButton_1.setMaximumSize(QtCore.QSize(self.width(),
                                                               self.height() // 44))
        self.radioButton_1.setSizePolicy(sizePolicy)
        self.radioButton_1.setObjectName("radioButton_1")
        self.radioButton_1.setChecked(True)

        self.radioButton_2 = QtWidgets.QRadioButton(self.splitterRadioButton_3)
        self.radioButton_2.setMaximumSize(QtCore.QSize(self.width(),
                                                               self.height() // 44))
        self.radioButton_2.setSizePolicy(sizePolicy)
        self.radioButton_2.setObjectName("radioButton_2")


        self.radioButton_3 = QtWidgets.QRadioButton(self.splitterRadioButton_3)
        self.radioButton_3.setMaximumSize(QtCore.QSize(self.width(),
                                                               self.height() // 44))
        self.radioButton_3.setSizePolicy(sizePolicy)
        self.radioButton_3.setObjectName("radioButton_3")


        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        splitter_right = QtWidgets.QSplitter(self.splitter_main_view1)
        splitter_right.setOrientation(QtCore.Qt.Vertical)
        splitter_right.setObjectName("splitter_right")
        self.openGLWidget_14 = glScientific(colorsCombinations,splitter_right, id=0)
        self.openGLWidget_14.initiate_actions()
        self.openGLWidget_14.setObjectName("openGLWidget_14")
        self.openGLWidget_14.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_14.setSizePolicy(sizePolicy)
        width_3d, height_3d = self.width()//3, int(self.height()/1.2)
        #self.openGLWidget_14.setFixedSize(QtCore.QSize(width_3d, self.height() - self.height()//3))
        self.splitter_main_view1.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])
        self.openGLWidget_11.setFocusPolicy(Qt.StrongFocus)



        self.widget = QtWidgets.QWidget(splitter_right)
        self.widget.setObjectName("widget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")


        spacerItem = QtWidgets.QSpacerItem(14, 118, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_seg.addWidget(self.splitter_main_view1, 0, 0, 1, 1)



        # 1. Create a Container Widget to hold Button + Slider
        self.slider_container = QtWidgets.QWidget(self.splitter_slider)
        self.slider_container.setObjectName("slider_container")

        # 2. Create Horizontal Layout (No margins so it fits tight)
        layout_hbox = QtWidgets.QHBoxLayout(self.slider_container)
        layout_hbox.setContentsMargins(0, 0, 0, 0)
        layout_hbox.setSpacing(5)

        # 3. Create the Small Play Button
        self.btn_play_view1 = QtWidgets.QPushButton(self.slider_container)
        self.btn_play_view1.setObjectName("btn_play")
        # Use standard Play Icon
        self.btn_play_view1.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        # Make it small and square (e.g., 25x25 pixels)
        self.btn_play_view1.setFixedSize(25, 25)
        self.btn_play_view1.setToolTip("Play/Pause Video")
        self.btn_play_view1.clicked.connect(partial(self.toggle_video_playback, 0))

        # 4. Create the Slider (Parent is now the container, not the splitter)
        self.horizontalSlider_11 = QtWidgets.QScrollBar(self.slider_container)
        self.horizontalSlider_11.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_11.setObjectName("horizontalSlider_11")

        # Apply your existing sizing policies to the slider
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.horizontalSlider_11.setSizePolicy(sizePolicy)
        self.horizontalSlider_11.setMaximumSize(QtCore.QSize(16777215, self.height() // 44))

        # 5. Add them to the Horizontal Layout
        layout_hbox.addWidget(self.btn_play_view1)
        layout_hbox.addWidget(self.horizontalSlider_11)

        # 6. Initialize Timer
        self.video_timer_view1 = QtCore.QTimer(self)
        self.video_timer_view1.timeout.connect(partial(self.advance_video_frame, 0))

        # Default: Hide button (it will show only when video loads)
        self.btn_play_view1.setVisible(False)


        self.tabWidget.addTab(self.segmentationTab, "")






    def set_view_mode_show_all_view1(self):
        """
        Mode 1: Show Everything (80% Left, 20% Right)
        """
        # 1. Make components visible
        self.radio_row_widget_view1.setVisible(True)
        self.right_panel_widget_view1.setVisible(True)

        # 2. Force Splitter Sizes
        # We give the left side 4 parts and right side 1 part (approx 80/20)
        total_w = self.splitter_main_view1.width()
        if total_w == 0: total_w = 1000  # Fallback if not rendered yet
        self.splitter_main_view1.setSizes([int(total_w * 0.8), int(total_w * 0.2)])

    def set_view_mode_focused_view1(self):
        """
        Mode 2: Focused View (100% Left OpenGL, No Radio Buttons)
        """
        # 1. Hide components
        self.radio_row_widget_view1.setVisible(False)
        self.right_panel_widget_view1.setVisible(False)

        # 2. Force Splitter Sizes
        # We give the left side ALL the width, and right side 0
        # Even though right side is hidden, this helps reset the handle position
        total_w = self.splitter_main_view1.width()
        self.splitter_main_view1.setSizes([total_w, 0])

    def create_view1_tab(self, colorsCombinations):


        self.segmentationTab = QtWidgets.QWidget()
        self.segmentationTab.setObjectName("segmentationTab")

        # Main Layout
        self.gridLayout_seg_1 = QtWidgets.QGridLayout(self.segmentationTab)
        self.gridLayout_seg_1.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_seg_1.setObjectName("gridLayout_seg")

        # --- Main Splitter (Divides Left and Right Panels) ---
        self.splitter_main_view1 = QtWidgets.QSplitter(self.segmentationTab)
        self.splitter_main_view1.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_main_view1.setChildrenCollapsible(
            False)  # Prevents widgets from disappearing completely unless we hide them
        self.splitter_main_view1.setObjectName("splitter_main")

        # ========================================================
        # LEFT SIDE SETUP
        # ========================================================
        self.left_panel_widget_view1 = QtWidgets.QWidget(self.splitter_main_view1)

        # Main Layout for Left Panel (Vertical)
        self.left_panel_layout_view1 = QtWidgets.QVBoxLayout(self.left_panel_widget_view1)
        self.left_panel_layout_view1.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout_view1.setSpacing(0)

        # 1. Main OpenGL Widget (Expanding)
        self.openGLWidget_11 = GLWidget(colorsCombinations, self.left_panel_widget_view1, imdata=None,
                                        currentWidnowName='coronal', type='eco', id=11)

        sizePolicy_gl = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy_gl.setHorizontalStretch(1)
        sizePolicy_gl.setVerticalStretch(1)
        self.openGLWidget_11.setSizePolicy(sizePolicy_gl)
        self.openGLWidget_11.setMaximumSize(16777215, 16777215)
        self.openGLWidget_11.setObjectName("openGLWidget_11")
        self.openGLWidget_11.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.left_panel_layout_view1.addWidget(self.openGLWidget_11, 1)

        # 2. Controls Container
        self.controls_container_view1 = QtWidgets.QWidget()
        self.controls_container_view1.setObjectName("controls_container")

        # Vertical Layout for Controls: [Label] -> [Slider Row] -> [Radio Row]
        self.controls_layout_view1 = QtWidgets.QVBoxLayout(self.controls_container_view1)
        self.controls_layout_view1.setContentsMargins(5, 5, 5, 5)
        self.controls_layout_view1.setSpacing(2)  # Small spacing between Label and Slider

        # Don't let controls expand vertically
        self.controls_container_view1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        # --- A. Label (Top, Centered) ---
        self.label_11 = QtWidgets.QLabel("Slice Info")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)  # Center the text
        self.label_11.setMinimumSize(QtCore.QSize(10, 15))
        self.label_11.setObjectName("label_11")

        # Add Label directly to vertical layout
        self.controls_layout_view1.addWidget(self.label_11)

        # --- B. Slider Row (Button + Slider) ---
        self.slider_row_widget_view1 = QtWidgets.QWidget()
        layout_hbox_slider = QtWidgets.QHBoxLayout(self.slider_row_widget_view1)
        layout_hbox_slider.setContentsMargins(0, 0, 0, 0)
        layout_hbox_slider.setSpacing(5)

        # Play Button
        self.btn_play_view1 = QtWidgets.QPushButton()
        self.btn_play_view1.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.btn_play_view1.setFixedSize(25, 25)
        self.btn_play_view1.clicked.connect(partial(self.toggle_video_playback, 0))
        self.btn_play_view1.setVisible(False)

        # Slider
        self.horizontalSlider_11 = QtWidgets.QScrollBar()
        self.horizontalSlider_11.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_11.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        layout_hbox_slider.addWidget(self.btn_play_view1)
        layout_hbox_slider.addWidget(self.horizontalSlider_11)

        self.video_timer_view1 = QtCore.QTimer(self)
        self.video_timer_view1.timeout.connect(partial(self.advance_video_frame, 0))
        # Add Slider Row to vertical layout
        self.controls_layout_view1.addWidget(self.slider_row_widget_view1)

        # --- C. Radio Buttons Row ---
        self.radio_row_widget_view1 = QtWidgets.QWidget()
        self.radio_row_widget_view1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        layout_hbox_radios = QtWidgets.QHBoxLayout(self.radio_row_widget_view1)
        layout_hbox_radios.setContentsMargins(0, 5, 0, 0)  # Add a little top margin to separate from slider

        self.radioButton_4 = QtWidgets.QCheckBox("Overlay")
        self.radioButton_4.setChecked(True)
        self.radioButton_1 = QtWidgets.QRadioButton("Option 1")
        self.radioButton_1.setChecked(True)
        self.radioButton_2 = QtWidgets.QRadioButton("Option 2")
        self.radioButton_3 = QtWidgets.QRadioButton("Option 3")

        layout_hbox_radios.addWidget(self.radioButton_4)
        layout_hbox_radios.addWidget(self.radioButton_1)
        layout_hbox_radios.addWidget(self.radioButton_2)
        layout_hbox_radios.addWidget(self.radioButton_3)
        layout_hbox_radios.addStretch()

        # Add Radio Row to vertical layout
        self.controls_layout_view1.addWidget(self.radio_row_widget_view1)

        # Add the entire controls container to the main left panel
        self.left_panel_layout_view1.addWidget(self.controls_container_view1, 0)

        # ========================================================
        # RIGHT SIDE SETUP
        # ========================================================
        self.right_panel_widget_view1 = QtWidgets.QWidget(self.splitter_main_view1)
        self.right_panel_layout_view1 = QtWidgets.QVBoxLayout(self.right_panel_widget_view1)
        self.right_panel_layout_view1.setContentsMargins(0, 0, 0, 0)

        self.openGLWidget_14 = glScientific(colorsCombinations, self.right_panel_widget_view1, id=1)
        self.openGLWidget_14.initiate_actions()
        self.openGLWidget_14.setSizePolicy(sizePolicy_gl)  # Use same greedy policy

        self.right_panel_layout_view1.addWidget(self.openGLWidget_14)

        # Add to splitter
        self.splitter_main_view1.addWidget(self.left_panel_widget_view1)
        self.splitter_main_view1.addWidget(self.right_panel_widget_view1)

        # Add to Tab
        self.gridLayout_seg_1.addWidget(self.splitter_main_view1, 0, 0, 1, 1)
        self.tabWidget.addTab(self.segmentationTab, "MRI Segmentation")

        # Start in full mode
        self.set_view_mode_show_all_view1()




    def set_view_mode_show_all_view2(self):
        """
        Mode 1: Show Everything (80% Left, 20% Right)
        """
        # 1. Make components visible
        self.radio_row_widget.setVisible(True)
        self.right_panel_widget.setVisible(True)

        # 2. Force Splitter Sizes
        # We give the left side 4 parts and right side 1 part (approx 80/20)
        total_w = self.splitter_main_view2.width()
        if total_w == 0: total_w = 1000  # Fallback if not rendered yet
        self.splitter_main_view2.setSizes([int(total_w * 0.8), int(total_w * 0.2)])

    def set_view_mode_focused_view2(self):
        """
        Mode 2: Focused View (100% Left OpenGL, No Radio Buttons)
        """
        # 1. Hide components
        self.radio_row_widget.setVisible(False)
        self.right_panel_widget.setVisible(False)

        # 2. Force Splitter Sizes
        # We give the left side ALL the width, and right side 0
        # Even though right side is hidden, this helps reset the handle position
        total_w = self.splitter_main_view2.width()
        self.splitter_main_view2.setSizes([total_w, 0])

    def create_view2_tab(self, colorsCombinations):
        self.MRISegTab = QtWidgets.QWidget()
        self.MRISegTab.setObjectName("segmentationTab")

        # Main Layout
        self.gridLayout_seg_2 = QtWidgets.QGridLayout(self.MRISegTab)
        self.gridLayout_seg_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_seg_2.setObjectName("gridLayout_seg")

        # --- Main Splitter (Divides Left and Right Panels) ---
        self.splitter_main_view2 = QtWidgets.QSplitter(self.MRISegTab)
        self.splitter_main_view2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_main_view2.setChildrenCollapsible(
            False)  # Prevents widgets from disappearing completely unless we hide them
        self.splitter_main_view2.setObjectName("splitter_main")

        # ========================================================
        # LEFT SIDE SETUP
        # ========================================================
        self.left_panel_widget = QtWidgets.QWidget(self.splitter_main_view2)

        # Main Layout for Left Panel (Vertical)
        self.left_panel_layout = QtWidgets.QVBoxLayout(self.left_panel_widget)
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout.setSpacing(0)

        # 1. Main OpenGL Widget (Expanding)
        self.openGLWidget_12 = GLWidget(colorsCombinations, self.left_panel_widget, imdata=None,
                                        currentWidnowName='coronal', type='mri', id=12)

        sizePolicy_gl = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy_gl.setHorizontalStretch(1)
        sizePolicy_gl.setVerticalStretch(1)
        self.openGLWidget_12.setSizePolicy(sizePolicy_gl)
        self.openGLWidget_12.setMaximumSize(16777215, 16777215)
        self.openGLWidget_12.setObjectName("openGLWidget_11")
        self.openGLWidget_12.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.left_panel_layout.addWidget(self.openGLWidget_12, 1)

        # 2. Controls Container
        self.controls_container = QtWidgets.QWidget()
        self.controls_container.setObjectName("controls_container")

        # Vertical Layout for Controls: [Label] -> [Slider Row] -> [Radio Row]
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_container)
        self.controls_layout.setContentsMargins(5, 5, 5, 5)
        self.controls_layout.setSpacing(2)  # Small spacing between Label and Slider

        # Don't let controls expand vertically
        self.controls_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        # --- A. Label (Top, Centered) ---
        self.label_12 = QtWidgets.QLabel("Slice Info")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)  # Center the text
        self.label_12.setMinimumSize(QtCore.QSize(10, 15))
        self.label_12.setObjectName("label_11")

        # Add Label directly to vertical layout
        self.controls_layout.addWidget(self.label_12)

        # --- B. Slider Row (Button + Slider) ---
        self.slider_row_widget = QtWidgets.QWidget()
        layout_hbox_slider = QtWidgets.QHBoxLayout(self.slider_row_widget)
        layout_hbox_slider.setContentsMargins(0, 0, 0, 0)
        layout_hbox_slider.setSpacing(5)

        # Play Button
        self.btn_play_view2 = QtWidgets.QPushButton()
        self.btn_play_view2.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.btn_play_view2.setFixedSize(25, 25)
        self.btn_play_view2.clicked.connect(partial(self.toggle_video_playback, 1))
        self.btn_play_view2.setVisible(False)

        # Slider
        self.horizontalSlider_12 = QtWidgets.QScrollBar()
        self.horizontalSlider_12.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_12.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        layout_hbox_slider.addWidget(self.btn_play_view2)
        layout_hbox_slider.addWidget(self.horizontalSlider_12)

        self.video_timer_view2 = QtCore.QTimer(self)
        self.video_timer_view2.timeout.connect(partial(self.advance_video_frame, 1))
        # Add Slider Row to vertical layout
        self.controls_layout.addWidget(self.slider_row_widget)

        # --- C. Radio Buttons Row ---
        self.radio_row_widget = QtWidgets.QWidget()
        self.radio_row_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        layout_hbox_radios = QtWidgets.QHBoxLayout(self.radio_row_widget)
        layout_hbox_radios.setContentsMargins(0, 5, 0, 0)  # Add a little top margin to separate from slider

        self.radioButton_21 = QtWidgets.QCheckBox("Overlay")
        self.radioButton_21.setChecked(True)
        self.radioButton_21_1 = QtWidgets.QRadioButton("Option 1")
        self.radioButton_21_1.setChecked(True)
        self.radioButton_21_2 = QtWidgets.QRadioButton("Option 2")
        self.radioButton_21_3 = QtWidgets.QRadioButton("Option 3")

        layout_hbox_radios.addWidget(self.radioButton_21)
        layout_hbox_radios.addWidget(self.radioButton_21_1)
        layout_hbox_radios.addWidget(self.radioButton_21_2)
        layout_hbox_radios.addWidget(self.radioButton_21_3)
        layout_hbox_radios.addStretch()

        # Add Radio Row to vertical layout
        self.controls_layout.addWidget(self.radio_row_widget)

        # Add the entire controls container to the main left panel
        self.left_panel_layout.addWidget(self.controls_container, 0)

        # ========================================================
        # RIGHT SIDE SETUP
        # ========================================================
        self.right_panel_widget = QtWidgets.QWidget(self.splitter_main_view2)
        self.right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel_widget)
        self.right_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.openGLWidget_24 = glScientific(colorsCombinations, self.right_panel_widget, id=1)
        self.openGLWidget_24.initiate_actions()
        self.openGLWidget_24.setSizePolicy(sizePolicy_gl)  # Use same greedy policy

        self.right_panel_layout.addWidget(self.openGLWidget_24)

        # Add to splitter
        self.splitter_main_view2.addWidget(self.left_panel_widget)
        self.splitter_main_view2.addWidget(self.right_panel_widget)

        # Add to Tab
        self.gridLayout_seg_2.addWidget(self.splitter_main_view2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.MRISegTab, "MRI Segmentation")

        # Start in full mode
        self.set_view_mode_show_all_view2()



    def create_view2_tab_old(self, colorsCombinations):
        self.MRISegTab = QtWidgets.QWidget()
        self.MRISegTab.setObjectName("segmentationTab")
        self.gridLayout_seg_2 = QtWidgets.QGridLayout(self.MRISegTab)
        self.gridLayout_seg_2.setObjectName("gridLayout_seg")
        self.splitter_main_view2 = QtWidgets.QSplitter(self.MRISegTab)
        self.splitter_main_view2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_main_view2.setObjectName("splitter_main")
        self.splitter_left_view2 = QtWidgets.QSplitter(self.splitter_main_view2)
        self.splitter_left_view2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_left_view2.setObjectName("splitter_left")


        width_3d, height_3d = self.width() // 3, int(self.height() // 1.2)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.openGLWidget_12 = GLWidget(colorsCombinations, self.splitter_left_view2, imdata=None, currentWidnowName='coronal',
                                        type='mri', id=12)

        self.openGLWidget_12.setSizePolicy(sizePolicy)
        self.openGLWidget_12.setMaximumSize(QtCore.QSize(self.width(), self.height()))
        self.openGLWidget_12.setObjectName("openGLWidget_11")
        self.openGLWidget_12.setFocusPolicy(Qt.StrongFocus)

        self.splitter_slider_view2 = QtWidgets.QSplitter(self.splitter_left_view2)
        self.splitter_slider_view2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_slider_view2.setObjectName("splitter_slider")
        self.label_12 = QtWidgets.QLabel(self.splitter_slider_view2)

        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(10, 10))
        self.label_12.setMaximumSize(QtCore.QSize(self.width() - self.width() // 4, self.height() // 44))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_11")


        
        
        
        # 1. Create a Container Widget to hold Button + Slider
        self.slider_container_view2 = QtWidgets.QWidget(self.splitter_slider_view2)
        self.slider_container_view2.setObjectName("slider_container")

        # 2. Create Horizontal Layout (No margins so it fits tight)
        layout_hbox = QtWidgets.QHBoxLayout(self.slider_container_view2)
        layout_hbox.setContentsMargins(0, 0, 0, 0)
        layout_hbox.setSpacing(5)

        # 3. Create the Small Play Button
        self.btn_play_view2 = QtWidgets.QPushButton(self.slider_container_view2)
        self.btn_play_view2.setObjectName("btn_play")
        # Use standard Play Icon
        self.btn_play_view2.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        # Make it small and square (e.g., 25x25 pixels)
        self.btn_play_view2.setFixedSize(25, 25)
        self.btn_play_view2.setToolTip("Play/Pause Video")
        self.btn_play_view2.clicked.connect(partial(self.toggle_video_playback, 1))


        # 4. Create the Slider (Parent is now the container, not the splitter)
        self.horizontalSlider_12 = QtWidgets.QScrollBar(self.slider_container_view2)
        self.horizontalSlider_12.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_12.setObjectName("horizontalSlider_11")

        # Apply your existing sizing policies to the slider
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.horizontalSlider_12.setSizePolicy(sizePolicy)
        self.horizontalSlider_12.setMaximumSize(QtCore.QSize(16777215, self.height() // 44))

        # 5. Add them to the Horizontal Layout
        layout_hbox.addWidget(self.btn_play_view2)
        layout_hbox.addWidget(self.horizontalSlider_12)

        # 6. Initialize Timer
        self.video_timer_view2 = QtCore.QTimer(self)
        self.video_timer_view2.timeout.connect(partial(self.advance_video_frame, 1))

        # Default: Hide button (it will show only when video loads)
        self.btn_play_view2.setVisible(False)
        
        
        
        
        

        splitterRadioButton = QtWidgets.QSplitter(self.splitter_left_view2)
        splitterRadioButton.setOrientation(QtCore.Qt.Horizontal)
        splitterRadioButton.setObjectName("splitterRadioButton")
        splitterRadioButton.setSizePolicy(sizePolicy)
        splitterRadioButton.setMaximumSize(QtCore.QSize(self.width(),
                                                        self.height() // 44))

        splitterRadioButton_3 = QtWidgets.QSplitter(splitterRadioButton)
        splitterRadioButton_3.setMaximumSize(QtCore.QSize(self.width(),
                                                          self.height() // 44))
        splitterRadioButton_3.setOrientation(QtCore.Qt.Horizontal)
        splitterRadioButton_3.setObjectName("splitterRadioButton_3")

        self.radioButton_21 = QtWidgets.QCheckBox(splitterRadioButton)
        self.radioButton_21.setObjectName("radioButton_4")
        self.radioButton_21.setMaximumSize(QtCore.QSize(self.width(),
                                                        self.height() // 44))
        self.radioButton_21.setChecked(True)
        self.radioButton_21.setSizePolicy(sizePolicy)

        self.radioButton_21_1 = QtWidgets.QRadioButton(splitterRadioButton_3)
        self.radioButton_21_1.setMaximumSize(QtCore.QSize(self.width(),
                                                          self.height() // 44))
        self.radioButton_21_1.setSizePolicy(sizePolicy)
        self.radioButton_21_1.setObjectName("radioButton_1")
        self.radioButton_21_1.setChecked(True)

        self.radioButton_21_2 = QtWidgets.QRadioButton(splitterRadioButton_3)
        self.radioButton_21_2.setMaximumSize(QtCore.QSize(self.width(),
                                                          self.height() // 44))
        self.radioButton_21_2.setSizePolicy(sizePolicy)
        self.radioButton_21_2.setObjectName("radioButton_2")

        self.radioButton_21_3 = QtWidgets.QRadioButton(splitterRadioButton_3)
        self.radioButton_21_3.setMaximumSize(QtCore.QSize(self.width(),
                                                          self.height() // 44))
        self.radioButton_21_3.setSizePolicy(sizePolicy)
        self.radioButton_21_3.setObjectName("radioButton_3")

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        splitter_right = QtWidgets.QSplitter(self.splitter_main_view2)
        splitter_right.setOrientation(QtCore.Qt.Vertical)
        splitter_right.setObjectName("splitter_right")
        self.openGLWidget_24 = glScientific(colorsCombinations, splitter_right, id=1)
        self.openGLWidget_24.initiate_actions()
        self.openGLWidget_24.setObjectName("openGLWidget_14")
        self.openGLWidget_24.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_24.setSizePolicy(sizePolicy)
        self.openGLWidget_24.setFixedSize(QtCore.QSize(width_3d, self.height() - self.height() // 3))

        self.widget_2 = QtWidgets.QWidget(splitter_right)
        self.widget_2.setObjectName("widget")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setObjectName("verticalLayout")

        spacerItem = QtWidgets.QSpacerItem(14, 118, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.gridLayout_seg_2.addWidget(self.splitter_main_view2, 0, 0, 1, 1)

        self.tabWidget.addTab(self.MRISegTab, "")



    def createOpenGLWidgets(self, centralwidget, colorsCombinations):
        """
        Creating main opengl widgets with its characteristics
        :param centralwidget:
        :param colorsCombinations:
        :return:
        """
        self.gridLayout_main = QtWidgets.QGridLayout(centralwidget)
        self.gridLayout_main.setObjectName("gridLayout_main")
        self.tabWidget = QtWidgets.QTabWidget(centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(100, 100))
        self.tabWidget.setObjectName("tabWidget")

        #self.create_horizontal_mutual_view(colorsCombinations)
        self.create_mutual_view(colorsCombinations)

        self.create_horizontal_mutualview()


        self.tabWidget.addTab(self.mutulaViewTab, "fdfdfdfdf")


        #self.tabWidget.addTab(self.reservedTab, "")

        ################# LEFT SIDE ######################################

        self.create_view1_tab(colorsCombinations)

        ########################################## MRI TAB #########################################
        ################# LEFT SIDE ######################################
        self.create_view2_tab(colorsCombinations)




        self.gridLayout_main.addWidget(self.tabWidget, 2, 0, 1, 1)

        self.openedFileName = QtWidgets.QLabel(centralwidget)
        self.openedFileName.setAlignment(QtCore.Qt.AlignCenter)
        self.openedFileName.setObjectName("FileName")
        self.openedFileName.setText('US:NONE, MRI:NONE')
        self.openedFileName.setVisible(True)
        self.gridLayout_main.addWidget(self.openedFileName, 0, 0, 1, 1)


        # mouse press event
        self.openGLWidget_1.mousePress.connect(
            lambda obj: self.mousePressEvent(obj)
        )

        self.openGLWidget_1.NewPoints.connect(
            lambda totalPs, chInd: self.openGLWidget_2.subpaintGL(totalPs, chInd))
        self.openGLWidget_1.NewPoints.connect(
            lambda totalPs, chInd: self.openGLWidget_3.subpaintGL(totalPs, chInd))

        update_color_scheme(self, None, dialog=False, update_widget=False)
        #self.tabWidget.addTab(self.tab, "")
        #self.gridLayout_main.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider_1.valueChanged.connect(self.label_1.setNum)
        self.horizontalSlider_2.valueChanged.connect(self.label_2.setNum)
        self.horizontalSlider_3.valueChanged.connect(self.label_3.setNum)
        self.horizontalSlider_4.valueChanged.connect(self.label_4.setNum)
        self.horizontalSlider_5.valueChanged.connect(self.label_5.setNum)
        self.horizontalSlider_6.valueChanged.connect(self.label_6.setNum)
        #self.horizontalSlider_7.valueChanged.connect(self.label_7.setNum)
        #self.horizontalSlider_8.valueChanged.connect(self.label_8.setNum)
        #self.horizontalSlider_9.valueChanged.connect(self.label_9.setNum)
        #self.horizontalSlider_10.valueChanged.connect(self.label_10.setNum)
        self.horizontalSlider_11.valueChanged.connect(self.label_11.setNum)
        self.horizontalSlider_12.valueChanged.connect(self.label_12.setNum)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)

        try:

            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
            self.label_1.setSizePolicy(sizePolicy)
            self.label_2.setSizePolicy(sizePolicy)
            self.label_3.setSizePolicy(sizePolicy)
            self.label_4.setSizePolicy(sizePolicy)
            self.label_5.setSizePolicy(sizePolicy)
            self.label_6.setSizePolicy(sizePolicy)
            #self.label_7.setSizePolicy(sizePolicy)
            #self.label_8.setSizePolicy(sizePolicy)
            #self.label_9.setSizePolicy(sizePolicy)
            #self.label_10.setSizePolicy(sizePolicy)
            self.label_11.setSizePolicy(sizePolicy)
            self.label_12.setSizePolicy(sizePolicy)
        except AttributeError:
            print("Warning: Labels not found. Skipping label size policy setup.")
        self.openGLWidget_11.setVisible(False)
        self.openGLWidget_12.setVisible(False)
        for i in [0,1,2,3,4,5,6,7,8,9,10,11]:
            try:
                a='openGLWidget_{}'.format(i+1)
                ats = getattr(self, a)
                #ats.setSizePolicy(sizePolicy)
                ats.clicked.connect(self.save_changes_auto)
            except:
                pass



