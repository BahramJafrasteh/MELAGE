__AUTHOR__ = 'Bahram Jafrasteh'

from pathlib import Path
import sys
sys.path.append("../")
from PyQt5 import QtWidgets, QtCore, QtGui
from functools import partial
from melage.utils.utils import generate_color_scheme_info
from PyQt5 import Qt
import numpy as np
import os
from .ui_schema import *
from .ui_builder import UIBuilder
from .SideBar import VSCodeSidebar, CollapsibleBox
from melage.utils.utils import read_txt_color, set_new_color_scheme, addTreeRoot, update_color_scheme, addLastColor, update_image_sch
from melage.config import settings
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMenu, QAction
from melage.dialogs.helpers import QFileDialogPreview
from melage.utils.utils import export_tables
from PyQt5.QtWidgets import QMenu, QAction
from collections import defaultdict

colorNames =("#FFCC08","darkRed","red", "darkOrange", "orange", "#8b8b00","yellow",
             "darkGreen","green","darkCyan","cyan",
             "darkBlue","blue","magenta","darkMagenta", 'red')




class dockWidgets():
    """
    This class has been implemented for dock widgets in MELAGE
    """
    def __init__(self):
        self.setAcceptDrops(True)

    # This function should be a method in your main class
    def style_row_by_checkstate(self, item):
        """
        Styles the entire row based on the check state of the item in the first column.
        """
        # Get the model from self, not from an argument
        model = self.tree_colors.model().sourceModel()
        if not model:
            return

        row = item.row()
        # Get the item in the first column to read the consistent check state for the row
        check_item = model.item(row, 0)

        if check_item and check_item.checkState() == QtCore.Qt.Checked:
            brush = QtGui.QBrush(QtGui.QColor(212, 237, 218, 60))  # Light green
        else:
            brush = QtGui.QBrush(QtCore.Qt.transparent)  # Default/transparent

        # Loop through all columns in the row and apply the style
        for col in range(model.columnCount()):
            item_in_row = model.item(row, col)
            if item_in_row:
                item_in_row.setBackground(brush)

    def _create_segDock(self, Main):
        """
        Create the Intensity settings widget and add it to the Sidebar.
        Refactored to use UIBuilder schema.
        """
        # 1. Create the container
        self.content_segInt = QtWidgets.QWidget(Main)
        self.content_segInt.setObjectName("content_segInt")

        # 2. Setup Layout (Vertical)
        layout = QtWidgets.QVBoxLayout(self.content_segInt)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)  # slightly increased spacing for better readability

        # 3. Define the UI Schema
        schema = [
            Group(
                id="group_segmentation",
                title="Intensity Settings",
                layout="vbox",
                children=[
            # --- Section 1: Segmentation Intensity ---
            Slider(
                id="scroll_intensity",  # Base ID
                label="Segmentation Intensity",  # Title Text
                label_id="label_seg_intensity_title",  # Title Variable Name
                min_val=0,
                max_val=100,
                default=48
            ),


            # --- Section 2: Image Intensity ---
            Slider(
                id="scroll_image_intensity",
                label="Image Intensity",
                label_id="label_image_intensity_value",
                min_val=0,
                max_val=100,

                default=100
            ),
            ] ),
            Separator(),
            Group(
                id="group_tolerance",
                title="Tolerance Settings",
                layout="vbox",
                children=[
                    Slider(
                        id="scrol_rad_circle",  # Creates self.scrol_rad_circle
                        label="Circle Radius",
                        label_id="label_assigned_rad_circle",
                        min_val=50, max_val=1000, default=50
                    ),
                    Slider(
                        id="scrol_tol_rad_circle",  # Creates self.scrol_tol_rad_circle
                        label="Tolerance Level",
                        label_id="label_assigned_tol_rad_circle",
                        min_val=0, max_val=100, default=0
                    )
                ]
            ),
        ]

        # 4. Build the UI
        builder = UIBuilder(self.content_segInt)
        builder.build(schema, layout, context=self)

        # 5. Push widgets to top
        layout.addStretch()

        # --- ALIASING FOR BACKWARD COMPATIBILITY ---
        # The builder creates 'self.scrol_intensity' and 'self.label_intensity'
        # Your old code used 'self.scroll_intensity' and 'self.label_intensity_value'
        # We map them here so you don't have to change your logic elsewhere.

        # Map Segmentation Intensity


        # Map Image Intensity
        #self.scroll_image_intensity = self.scrol_image_intensity
        #self.label_image_intensity_value = self.label_image_intensity

        # 6. Add to Sidebar
        self.vscode_widget.add_tab(
            self.content_segInt,
            settings.RESOURCE_DIR + "/seg_intensity.png",
            "Image Intensity"
        )


    def on_file_double_clicked(self, index):
        file_path = self.file_model.filePath(index)

        if self.file_model.isDir(index):
            # Allow expanding/collapsing folders (default behavior usually works,
            # but sometimes you might want custom logic here)
            return

        print(f"User selected file: {file_path}")
        # Logic to load the image/file goes here
        # e.g., self.loadImage(file_path)

    def dragEnterEvent(self, event):
        """
        Called when a file is dragged OVER the widget.
        """
        # Check if the dragged object contains file paths (URLs)
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Called when the file is DROPPED.
        """
        urls = event.mimeData().urls()
        if urls:
            # Convert URL to local system path (e.g., /home/user/image.nii)
            file_path = urls[0].toLocalFile()
            if file_path[-3:]=='.bn':
                self.loadProject(file_path)
            else:
                self.browse_view1([[file_path, "None"], 2], use_dialog=False)
            print(f"Dropped file: {file_path}")

            # Call your loading function here
            # e.g., self.load_new_image(file_path)

    def _dock_folder(self, Main):
        # ==============================================================================
        # TAB: File Explorer (With Navigation)
        # ==============================================================================

        self.page_explorer = QtWidgets.QWidget()
        self.layout_explorer = QtWidgets.QVBoxLayout(self.page_explorer)
        self.layout_explorer.setContentsMargins(0, 0, 0, 0)
        self.layout_explorer.setSpacing(0)


        if hasattr(settings, 'DEFAULT_USE_DIR') and os.path.exists(settings.DEFAULT_USE_DIR):
            home_path = settings.DEFAULT_USE_DIR
        else:
            home_path = QtCore.QDir.homePath()  # Fallback

        # --- 1. NAVIGATION BAR (The new part) ---
        self.nav_bar = QtWidgets.QWidget()
        self.nav_bar.setStyleSheet("background-color: #252526; border-bottom: 1px solid #3E3E42;")
        self.nav_layout = QtWidgets.QHBoxLayout(self.nav_bar)
        self.nav_layout.setContentsMargins(5, 5, 5, 5)
        self.nav_layout.setSpacing(5)

        # Button Style
        nav_btn_style = """
                QToolButton { border: none; color: #CCCCCC; font-weight: bold; }
                QToolButton:hover { background-color: #3E3E42; border-radius: 3px; }
            """

        # "Home" Button (Go back to Desktop)
        self.btn_home = QtWidgets.QToolButton()
        self.btn_home.setText("🏠")  # Or use an icon
        self.btn_home.setToolTip(f"Go to Default ({os.path.basename(home_path)})")
        self.btn_home.setStyleSheet(nav_btn_style)

        # "Up" Button (Go to parent folder)
        self.btn_up = QtWidgets.QToolButton()
        self.btn_up.setText("⬆")  # Or use an icon like 'icons/up_arrow.png'
        self.btn_up.setToolTip("Up one level")
        self.btn_up.setStyleSheet(nav_btn_style)

        # Current Folder Label
        self.lbl_current_path = QtWidgets.QLabel(os.path.basename(home_path))
        self.lbl_current_path.setStyleSheet("color: #CCCCCC; font-size: 11px;")

        self.nav_layout.addWidget(self.btn_home)
        self.nav_layout.addWidget(self.btn_up)
        self.nav_layout.addWidget(self.lbl_current_path)
        self.nav_layout.addStretch()  # Push everything to left

        self.layout_explorer.addWidget(self.nav_bar)



        # --- 2. FILE SYSTEM MODEL ---
        self.file_model = QtWidgets.QFileSystemModel()

        # Set Root Path to "/" (or Drives on Windows) so the model knows about everything
        # If we restrict this, we can't go up!
        self.file_model.setRootPath("")

        # Filters
        filters = ["*.bn", "*.nii", "*.nii.gz"]
        self.file_model.setNameFilters(filters)
        self.file_model.setNameFilterDisables(False)

        # --- 3. TREE VIEW ---
        self.tree_explorer = QtWidgets.QTreeView()
        self.tree_explorer.setModel(self.file_model)
        start_index = self.file_model.index(home_path)
        # Start at Desktop
        #desktop_path = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation)
        #start_index = self.file_model.index(desktop_path)
        self.tree_explorer.setRootIndex(start_index)

        # Tree Settings
        self.tree_explorer.setDragEnabled(True)
        self.tree_explorer.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.tree_explorer.setHeaderHidden(True)
        self.tree_explorer.setColumnHidden(1, True)
        self.tree_explorer.setColumnHidden(2, True)
        self.tree_explorer.setColumnHidden(3, True)
        self.tree_explorer.setAnimated(True)
        self.tree_explorer.setIndentation(20)
        self.tree_explorer.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.layout_explorer.addWidget(self.tree_explorer)

        # Double click to open file
        self.tree_explorer.doubleClicked.connect(self.on_file_double_clicked)

        # --- 4. NAVIGATION LOGIC ---

        def go_home():
            """Reset view to Desktop"""
            idx = self.file_model.index(home_path)
            self.tree_explorer.setRootIndex(idx)
            self.lbl_current_path.setText("Home")

        def go_up():
            """Move root index to parent directory"""
            current_idx = self.tree_explorer.rootIndex()
            parent_idx = self.file_model.parent(current_idx)

            # Check if parent is valid (stops at Root Drive)
            if parent_idx.isValid():
                self.tree_explorer.setRootIndex(parent_idx)

                # Update Label
                path = self.file_model.filePath(parent_idx)
                folder_name = QtCore.QDir(path).dirName()
                # If empty (Root drive), show full path
                self.lbl_current_path.setText(folder_name if folder_name else path)

        def on_folder_entered(index):
            """If user double clicks a FOLDER inside the tree, treat it as new root? (Optional)"""
            # Usually VS Code Explorer just expands.
            # But if you want "Drill down" behavior, uncomment this:
            # if self.file_model.isDir(index):
            #     self.tree_explorer.setRootIndex(index)
            #     self.lbl_current_path.setText(self.file_model.fileName(index))
            pass

        # Connect Buttons
        self.btn_home.clicked.connect(go_home)
        self.btn_up.clicked.connect(go_up)

        # Add to Sidebar
        self.vscode_widget.add_tab(self.page_explorer,
                                   settings.RESOURCE_DIR+ "/explorer.png", "Explorer")
    def _create_main_dock(self, Main):
        self.MainDock = QtWidgets.QDockWidget(Main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MainDock.sizePolicy().hasHeightForWidth())

        self.MainDock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.MainDock.setSizePolicy(sizePolicy)
        #self.MainDock.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))
        self.MainDock.setObjectName("MainDock")
        self.content_imageEnh = QtWidgets.QWidget()
        self.content_imageEnh.setObjectName("content_imageEnh")

        self.content_imageEnh = QtWidgets.QWidget()
        self.layout_imageEnh = QtWidgets.QVBoxLayout(self.content_imageEnh)
        self.layout_imageEnh.setContentsMargins(0, 0, 0, 0)

        # Initialize VS Code Sidebar
        self.vscode_widget = VSCodeSidebar(self.content_imageEnh)
        self.vscode_widget.setObjectName("main_toolbox")
        #self.vscode_widget.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))
        self.vscode_widget.setMinimumWidth(20)
        self.layout_imageEnh.addWidget(self.vscode_widget)
        self.MainDock.setWidget(self.content_imageEnh)
        self.MainDock.setVisible(True)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.MainDock)


    def _create_dockColor(self, Main):
        self.page1_color = QtWidgets.QWidget(Main)
        self.page1_color.setGeometry(QtCore.QRect(0, 0, self.width()//8, self.height()//2))
        self.page1_color.setObjectName("page")
        self.gridLayout_color = QtWidgets.QVBoxLayout(self.page1_color)
        self.gridLayout_color.setObjectName("gridLayout_view1")
        # controls
        self.line_text = QtWidgets.QLineEdit()
        self.line_text.setPlaceholderText('Search...')

        self.tags_model = SearchProxyModel()
        self.tags_model.setSourceModel(QtGui.QStandardItemModel())
        self.tags_model.setDynamicSortFilter(True)
        self.tags_model.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)


        self.tree_colors = QtWidgets.QTreeView()
        self.tree_colors.setSortingEnabled(True)
        self.tree_colors.sortByColumn(0, QtCore.Qt.AscendingOrder)
        # self.tree_colors.setColumnCount(2)
        # self.tree_colors.setHeaderLabels(['', ''])
        self.tree_colors.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_colors.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tree_colors.setHeaderHidden(False)
        self.tree_colors.setRootIsDecorated(True)
        self.tree_colors.setUniformRowHeights(True)
        self.tree_colors.setModel(self.tags_model)
        self.tree_colors.setStyleSheet("""
            QTreeView::item {
                padding-top: 5px;      /* 5px space above the content */
                padding-bottom: 5px;   /* 5px space below the content */
            }
        """)

        # layout
        main_layout = QtWidgets.QVBoxLayout()

        self.gridLayout_color.addWidget(self.line_text)
        self.gridLayout_color.addWidget(self.tree_colors)



        # signals
        self.tree_colors.doubleClicked.connect(self._double_clicked)
        self.line_text.textChanged.connect(partial(self.searchTreeChanged, 'color'))
        self.tree_colors.itemDelegate().closeEditor.connect(self._on_closeEditor)
        self.tree_colors.customContextMenuRequested.connect(self.ShowContextMenu_tree)
        # init
        model = self.tree_colors.model().sourceModel()
        model.setColumnCount(2)
        model.setHorizontalHeaderLabels(['Index', 'Name'])
        self.tree_colors.sortByColumn(0, QtCore.Qt.AscendingOrder)


        # Add to Sidebar
        self.vscode_widget.add_tab(self.page1_color,
                                   settings.RESOURCE_DIR+"/palette.png", "Color Settings")


    def _create_dock_imEnhance_view1(self, Main):
        self.ImageEnh_view1 = QtWidgets.QWidget(Main)
        self.ImageEnh_view1.setObjectName("page")
        layout = QtWidgets.QGridLayout(self.ImageEnh_view1)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        schema = [
            # --- Group 1: Essential Visuals ---
            Group(
                id="grp_basics",
                title="Color & Contrast",
                layout="vbox",
                children=[
                    Slider(id="t1_1", label="Brightness", label_id="lb_ft1_1", min_val=-100, max_val=100, default=0),
                    Slider(id="t1_2", label="Contrast", label_id="lb_ft1_2", min_val=1, max_val=500, default=100),
                    Slider(id="t1_3", label="Gamma", label_id="lb_ft1_3", min_val=10, max_val=300, default=100),
                ]
            ),

            # --- Group 2: Image Filters ---
            Group(
                id="grp_filters",
                title="Filters & Quality",
                layout="vbox",
                children=[
                    Slider(id="t1_7", label="Vascular Enhancement", label_id="lb_ft1_7", min_val=0, max_val=100, default=0),
                    Slider(id="t1_4", label="Denoise", label_id="lb_ft1_4", min_val=0, max_val=100, default=0),
                ]
            ),

            # --- Group 3: Geometry (Rotation) ---
            Group(
                id="grp_geo",
                title="Geometry & Correction",
                layout="vbox",
                children=[
                    Label("Correction Mode", id="lb_ft1_5"),
                    Combo(id="page1_rot_cor", options=["", "Option 1", "Option 2"]),

                    # Small separator inside the group if needed
                    Separator(),

                    Slider(id="t1_5", label="Rotation Fine Tune", label_id="lb_t1_5", min_val=-25, max_val=25),


                ]
            ),

            # --- Group 4: Advanced ---
            # Using HBox to keep the toggle compact
            Group(
                id="grp_adv",
                title="Advanced Settings",
                layout="hbox",
                children=[
                    Label("Advanced Mode", id="lb_ft1_6"),
                    Toggle(id="toggle1_1", text="Toggle")
                ]
            )
        ]

        # Build and inject variables into 'self'
        builder = UIBuilder(self.ImageEnh_view1)
        builder.build(schema, layout, context=self)

        self.ImageEnh_view1.setVisible(True)
        self.box_view1 = CollapsibleBox("View 1 Enhancement", self.ImageEnh_view1)
        self.layout_enh_container.addWidget(self.box_view1)
        self.enhancement_sections.append(self.box_view1)


    def _create_dock_imEnhance_vew2(self, Main):
        self.page1_mri = QtWidgets.QWidget(Main)
        self.page1_mri.setObjectName("page")
        layout = QtWidgets.QGridLayout(self.page1_mri)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)


        schema = [
            # --- Group 1: Essential Visuals ---
            Group(
                id="grp_basics",
                title="Color & Contrast",
                layout="vbox",
                children=[
                    Slider(id="t2_1", label="Brightness", label_id="lb_ft2_1", min_val=-100, max_val=100, default=0),
                    Slider(id="t2_2", label="Contrast", label_id="lb_ft2_2", min_val=0, max_val=500, default=100),
                    Slider(id="t2_3", label="Gamma", label_id="lb_ft2_3", min_val=10, max_val=300, default=100),
                ]
            ),

            # --- Group 2: Image Filters ---
            Group(
                id="grp_filters",
                title="Filters & Quality",
                layout="vbox",
                children=[
                    Slider(id="t2_7", label="Vascular Enhancement", label_id="lb_ft2_7", min_val=0, max_val=100, default=0),
                    Slider(id="t2_4", label="Denoise", label_id="lb_ft2_4", min_val=0, max_val=100, default=0),
                ]
            ),

            # --- Group 3: Geometry (Rotation) ---
            Group(
                id="grp_geo",
                title="Geometry & Correction",
                layout="vbox",
                children=[
                    Label("Correction Mode", id="lb_ft2_5"),
                    Combo(id="page2_rot_cor", options=["", "Option 1", "Option 2"]),

                    # Small separator inside the group if needed
                    Separator(),

                    Slider(id="t2_5", label="Rotation Fine Tune", label_id="lb_t2_5", min_val=-25, max_val=25),

                ]
            ),

            # --- Group 4: Advanced ---
            # Using HBox to keep the toggle compact
            Group(
                id="grp_adv",
                title="Advanced Settings",
                layout="hbox",
                children=[
                    Label("Advanced Mode", id="lb_ft2_6"),
                    Toggle(id="toggle2_1", text="Toggle")
                ]
            )
        ]

        # Build and inject variables into 'self'
        builder = UIBuilder(self.page1_mri)
        builder.build(schema, layout, context=self)

        self.page1_mri.setVisible(True)
        self.box_view2 = CollapsibleBox("View 2 Enhancement", self.page1_mri)
        self.layout_enh_container.addWidget(self.box_view2)
        self.enhancement_sections.append(self.box_view2)



    def _some_intial_steps(self):
        self.color_name, self.color_index_rgb, _ = read_txt_color(settings.RESOURCE_DIR+"/color/Simple.txt", mode= '', from_one=True)
        #update_color_scheme(self, None, dialog=False, update_widget=False)

        self.colorsCombinations = defaultdict(list)

        for clrn, clr in zip(self.color_name, self.color_index_rgb):
            colr = clr[1:]
            colr[-1] = 1  # clr[0]
            colr = list(colr)
            self.colorsCombinations[int(clrn.split('_')[0])] = colr
        last_color = '9876_Combined'
        if last_color not in self.color_name:
            addLastColor(self, last_color)

        #self.populate_tree(tags, model.invisibleRootItem())
        set_new_color_scheme(self)

        #self.tree_colors.model().sourceModel().itemChanged.connect(self.Upper_changeColorPen)
        self.tree_colors.clicked.connect(self.on_row_clicked)
        # In your setup method, after populating the model...

        # This connection is correct, assuming the function signature is fixed
        #self.tree_colors.model().sourceModel().itemChanged.connect(self.style_row_by_checkstate)

        # This loop must be corrected
        model = self.tree_colors.model().sourceModel()
        root = model.invisibleRootItem()
        for row in range(root.rowCount()):
            # Get the item in the first column (where the checkbox is)
            item = root.child(row, 0)
            if item:
                # Call the function with only the item
                self.style_row_by_checkstate(item)

    def _dock_progress_bar(self, Main):
        self.dock_progressbar = QtWidgets.QDockWidget(Main)
        self.dock_progressbar.setObjectName("dock_progressbar")
        #self.dock_progressbar.setWindowState(QtCore.Qt.WindowMinimized)
        #self.dock_progressbar.setFloating(True)
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.dockWidgetContents_3)
        self.formLayout_3.setObjectName("formLayout_3")
        self.progressBarSaving = QtWidgets.QProgressBar(self.dockWidgetContents_3)
        self.progressBarSaving.setProperty("value", 24)

        #self.progressBarSaving.setWindowState(QtCore.Qt.WindowMinimized)
        self.progressBarSaving.setObjectName("progressBar")

        #self.progressBarSaving.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        #self.dock_progressbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.progressBarSaving)
        self.dock_progressbar.setWidget(self.dockWidgetContents_3)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dock_progressbar)
        self.dock_progressbar.setVisible(False)


    def _unknown_dock(self, Main):
        self.dockWidget_4 = QtWidgets.QDockWidget(Main)
        self.dockWidget_4.setObjectName("dockWidget_4")
        self.dockWidgetContents_4 = QtWidgets.QWidget()
        self.dockWidgetContents_4.setObjectName("dockWidgetContents_3")
        self.formLayout_4 = QtWidgets.QFormLayout(self.dockWidgetContents_4)
        self.formLayout_4.setObjectName("formLayout_4")


        self.dw4_flb1 = QtWidgets.QLabel(self.dockWidgetContents_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dw4_flb1.sizePolicy().hasHeightForWidth())
        self.dw4_flb1.setSizePolicy(sizePolicy)
        self.dw4_flb1.setAlignment(QtCore.Qt.AlignCenter)
        self.dw4_flb1.setObjectName("dw4_flb1")

        self.dw4lb1 = QtWidgets.QLabel(self.dockWidgetContents_4)
        self.dw4lb1.setAlignment(QtCore.Qt.AlignCenter)
        self.dw4lb1.setObjectName("dw4lb1")
        self.dw4lb1.setText('50')

        self.dw4_s1 = QtWidgets.QScrollBar(self.dockWidgetContents_4)
        self.dw4_s1.setOrientation(QtCore.Qt.Horizontal)
        #self.dw4_s1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.dw4_s1.setObjectName("dw4_s1")
        self.dw4_s1.setRange(0,100)
        self.dw4_s1.setValue(50)


        self.dw4_s1.setSingleStep(1)


        self.line4_11 = QtWidgets.QFrame(self.dockWidgetContents_4)
        self.line4_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line4_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line4_11.setObjectName("line4_11")
        #self.formLayout_4.addWidget(self.line4_11, 29, 0, 1, 1)
        self.formLayout_4.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.line4_11)
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dw4_s1)

        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.dw4lb1)
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.dw4_flb1)

        self.dockWidget_4.setWidget(self.dockWidgetContents_4)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_4)
        self.dockWidget_4.setVisible(False)
        self.dw4_s1.valueChanged.connect(self.dw4lb1.setNum)


    def _tract_dock(self, Main):
        self.dockWidget_5 = QtWidgets.QDockWidget(Main)
        self.dockWidget_5.setObjectName("dockWidget_5")
        self.dockWidgetContents_5 = QtWidgets.QWidget()
        self.dockWidgetContents_5.setObjectName("dockWidgetContents_5")
        self.formLayout_5 = QtWidgets.QFormLayout(self.dockWidgetContents_5)
        self.formLayout_5.setObjectName("formLayout_5")

        self.dw5_flb1 = QtWidgets.QLabel(self.dockWidgetContents_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dw5_flb1.sizePolicy().hasHeightForWidth())
        self.dw5_flb1.setSizePolicy(sizePolicy)
        self.dw5_flb1.setAlignment(QtCore.Qt.AlignCenter)
        self.dw5_flb1.setObjectName("dw5_flb1")

        self.dw5lb1 = QtWidgets.QLabel(self.dockWidgetContents_5)
        self.dw5lb1.setAlignment(QtCore.Qt.AlignCenter)
        self.dw5lb1.setObjectName("dw5lb1")
        self.dw5lb1.setText('0')

        self.dw5_s1 = QtWidgets.QScrollBar(self.dockWidgetContents_5)
        self.dw5_s1.setOrientation(QtCore.Qt.Horizontal)
        #self.dw5_s1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.dw5_s1.setObjectName("dw5_s1")
        self.dw5_s1.setRange(0,100)
        self.dw5_s1.setValue(0)


        self.dw5_s1.setSingleStep(1)


        self.line5_11 = QtWidgets.QFrame(self.dockWidgetContents_5)
        self.line5_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line5_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line5_11.setObjectName("line5_11")



        self.dw5_flb2 = QtWidgets.QLabel(self.dockWidgetContents_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dw5_flb2.sizePolicy().hasHeightForWidth())
        self.dw5_flb2.setSizePolicy(sizePolicy)
        self.dw5_flb2.setAlignment(QtCore.Qt.AlignCenter)
        self.dw5_flb2.setObjectName("dw5_flb2")

        self.dw5lb2 = QtWidgets.QLabel(self.dockWidgetContents_5)
        self.dw5lb2.setAlignment(QtCore.Qt.AlignCenter)
        self.dw5lb2.setObjectName("dw5lb1")
        self.dw5lb2.setText('0')

        self.dw5_s2 = QtWidgets.QScrollBar(self.dockWidgetContents_5)
        self.dw5_s2.setOrientation(QtCore.Qt.Horizontal)
        #self.dw5_s1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.dw5_s2.setObjectName("dw5_s2")
        self.dw5_s2.setRange(0,100)
        self.dw5_s2.setValue(0)


        self.dw5_s2.setSingleStep(1)


        #self.formLayout_5.addWidget(self.line5_11, 29, 0, 1, 1)
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.line5_11)
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dw5_s1)

        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.dw5lb1)
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.dw5_flb1)


        self.formLayout_5.setWidget(4, QtWidgets.QFormLayout.SpanningRole, self.dw5lb2)
        self.formLayout_5.setWidget(5, QtWidgets.QFormLayout.SpanningRole, self.dw5_s2)
        self.formLayout_5.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.dw5_flb2)
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.line5_11)

        self.dockWidget_5.setWidget(self.dockWidgetContents_5)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_5)
        self.dockWidget_5.setVisible(False)
        self.dw5_s1.valueChanged.connect(self.dw5lb1.setNum)
        self.dw5_s2.valueChanged.connect(self.dw5lb2.setNum)
        self.dockWidget_5.setWindowTitle(_translate("Main", "Tracking Distance"))
        self.dw5_flb1.setText(_translate("Main", "Track Distance"))
        self.dw5_flb2.setText(_translate("Main", "Track Width"))

    def _setting_radius_dock(self, Main):
        self.Settings_widget = QtWidgets.QWidget(Main)
        self.Settings_widget.setObjectName("setting")
        self.Settings_widget.setMinimumSize(QtCore.QSize(self.width() // 8, self.height() // 2))

        # Main layout is Vertical
        layout = QtWidgets.QVBoxLayout(self.Settings_widget)
        layout.setSpacing(10)

        # Define the UI Structure
        schema = [
            # --- Group 1: Radius ---


            Separator(),

            # --- Group 2: Tolerance ---
            # (Example: Wrapping it in a GroupBox to show off the builder)
            Group(
                id="group_tolerance",
                title="Tolerance Settings",
                layout="vbox",
                children=[
                    Slider(
                        id="scrol_rad_circle",  # Creates self.scrol_rad_circle
                        label="Circle Radius",
                        label_id="label_assigned_rad_circle",
                        min_val=50, max_val=1000, default=50
                    ),
                    Slider(
                        id="scrol_tol_rad_circle",  # Creates self.scrol_tol_rad_circle
                        label="Tolerance Level",
                        label_id="label_assigned_tol_rad_circle",
                        min_val=0, max_val=100, default=0
                    )
                ]
            ),

            Separator(),

            # --- Example of Side-by-Side (HBox) for future reference ---
            # HBox(children=[
            #     Label("Left"),
            #     Label("Right")
            # ])
        ]

        # Build and Inject Variables into 'self'
        builder = UIBuilder(self.Settings_widget)
        builder.build(schema, layout, context=self)

        # Add spacing at the bottom to push widgets up
        layout.addStretch()

        # Add to Sidebar
        self.vscode_widget.add_tab(
            self.Settings_widget,
            settings.RESOURCE_DIR + "/seg_settings.png",
            "Settings"
        )


    def _table_link_doc(self, Main):
        self.dock_widget_table = QtWidgets.QWidget(Main)
        self.dock_widget_table.setObjectName("dock_widget_table")
        self.dock_widget_table.setVisible(False)

        dock_widget_content_table = QtWidgets.QWidget()

        dock_widget_content_table.setObjectName("dock_widget_content_table")
        gridLayout_table = QtWidgets.QGridLayout(dock_widget_content_table)
        gridLayout_table.setObjectName("gridLayout_table")
        # self.table_widget = QtWidgets.QTableWidget(dock_widget_content_table)
        self.table_widget = QtWidgets.QTableWidget(dock_widget_content_table)
        self.table_widget.setRowCount(25)
        self.table_widget.setColumnCount(2)
        self.table_widget.setEditTriggers(Qt.QAbstractItemView.NoEditTriggers)

        self.table_widget.setObjectName("table_widget")
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(0, 0, item)

        splitter_2 = QtWidgets.QSplitter(dock_widget_content_table)
        splitter_2.setOrientation(QtCore.Qt.Horizontal)

        self.table_update = QtWidgets.QPushButton(splitter_2)
        self.table_update.setObjectName("table_update")
        self.table_link = QtWidgets.QPushButton(splitter_2)
        self.table_link.setObjectName("table_link")
        self.table_link.setCheckable(True)

        _translate = QtCore.QCoreApplication.translate
        self.dock_widget_table.setWindowTitle(_translate("Main", "Table Link"))

        item = self.table_widget.horizontalHeaderItem(0)
        item.setText(_translate("Main", "XYZ_eco"))
        item = self.table_widget.horizontalHeaderItem(1)
        item.setText(_translate("Main", "XYZ_mri"))

        __sortingEnabled = self.table_widget.isSortingEnabled()
        self.table_widget.setSortingEnabled(False)
        self.table_widget.setSortingEnabled(__sortingEnabled)
        self.table_link.setText(_translate("Main", "Link"))
        self.table_update.setText(_translate("Main", "Update"))


        self.table_update.clicked.connect(self.linkMRIECO)
        self.table_link.clicked.connect(self.linkBoth)

        gridLayout_table.addWidget(self.table_widget, 0, 1, 1, 1)
        gridLayout_table.addWidget(splitter_2, 1, 1, 1, 1)
        #self.dock_widget_table.setWidget(dock_widget_content_table)
        #Main.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock_widget_table)
        self.vscode_widget.add_tab(dock_widget_content_table, settings.RESOURCE_DIR+"/table.png", "Tables")


    def _table_measure(self, Main):

        self.dock_widget_measure = QtWidgets.QDockWidget(Main)
        self.dock_widget_measure.setObjectName("dock_widget_table")
        self.dock_widget_measure.setVisible(False)

        dock_widget_content_table = QtWidgets.QWidget()

        dock_widget_content_table.setObjectName("dock_widget_content_table")
        gridLayout_view1 = QtWidgets.QGridLayout(dock_widget_content_table)
        gridLayout_view1.setObjectName("gridLayout_view1")
        # self.table_widget = QtWidgets.QTableWidget(dock_widget_content_table)
        self.table_widget_measure = QtWidgets.QTableWidget(dock_widget_content_table)

        self.table_widget_measure.setRowCount(5)
        self.table_widget_measure.setColumnCount(8)
        self.table_widget_measure.setEditTriggers(Qt.QAbstractItemView.NoEditTriggers)
        self.table_widget_measure.setHorizontalHeaderLabels(['Description','ImType', 'Measure1', 'Measure2', 'Slice', 'WindowName', 'CenterXY', 'FileName'])

        self.table_widget_measure.setObjectName("table_widget_measure")
        self.table_widget_measure.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table_widget_measure.customContextMenuRequested.connect(self.ShowContextMenu_table1)

        gridLayout_view1.addWidget(self.table_widget_measure, 0, 1, 1, 1)
        splitter_2 = QtWidgets.QSplitter(dock_widget_content_table)
        splitter_2.setOrientation(QtCore.Qt.Horizontal)


        gridLayout_view1.addWidget(splitter_2, 1, 1, 1, 1)

        self.dock_widget_measure.setWidget(dock_widget_content_table)

        #Main.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock_widget_measure)

        _translate = QtCore.QCoreApplication.translate
        self.dock_widget_measure.setWindowTitle(_translate("Main", "Table Measure"))


        #__sortingEnabled = self.table_widget.isSortingEnabled()
        self.table_widget_measure.setSortingEnabled(True)
        #self.table_widget_measure.setSortingEnabled(__sortingEnabled)

        self.vscode_widget.add_tab(self.table_widget_measure, settings.RESOURCE_DIR+"/table_measure.png",
                                   "Table Measure")


    def _batchImages_dock(self, Main):
        self.page1_images = QtWidgets.QWidget()

        self.page1_images.setGeometry(QtCore.QRect(0, 0, self.width()//8, self.height()//2))
        self.page1_images.setObjectName("page")
        self.gridLayout_images = QtWidgets.QVBoxLayout(self.page1_images)
        self.gridLayout_images.setObjectName("gridLayout_view1")





        # controls
        self.line_text_image = QtWidgets.QLineEdit()
        self.line_text_image.setPlaceholderText('Search...')

        self.tags_model_image = SearchProxyModel()
        self.tags_model_image.setSourceModel(QtGui.QStandardItemModel())
        self.tags_model_image.setDynamicSortFilter(True)

        self.tags_model_image.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)


        self.tree_images = QtWidgets.QTreeView()
        self.tree_images.setSortingEnabled(True)
        self.tree_images.sortByColumn(1, QtCore.Qt.AscendingOrder)
        # self.tree_colors.setColumnCount(2)
        # self.tree_colors.setHeaderLabels(['', ''])
        self.tree_images.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_images.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tree_images.setHeaderHidden(False)
        self.tree_images.setRootIsDecorated(True)
        self.tree_images.setUniformRowHeights(True)
        self.tree_images.setModel(self.tags_model_image)
        self.tree_images.setStyleSheet("""
            QTreeView::item {
                padding-top: 5px;      /* 5px space above the content */
                padding-bottom: 5px;   /* 5px space below the content */
            }
        """)
        #self.tree_images.clicked.connect(self.on_row_clicked)
        # layout
        #main_layout = QtWidgets.QVBoxLayout()

        self.gridLayout_images.addWidget(self.line_text_image)
        self.gridLayout_images.addWidget(self.tree_images)


        # signals
        self.tree_images.doubleClicked.connect(self._double_clicked)
        self.line_text_image.textChanged.connect(partial(self.searchTreeChanged, 'image'))
        self.tree_images.itemDelegate().closeEditor.connect(self._on_closeEditor)
        self.tree_images.customContextMenuRequested.connect(self.ShowContextMenu_images)
        # init
        model = self.tree_images.model().sourceModel()
        model.setColumnCount(2)
        model.setHorizontalHeaderLabels(['Index', 'Name'])
        self.tree_images.sortByColumn(1, QtCore.Qt.AscendingOrder)
        self.imported_images = []


        #self.tree_images.model().sourceModel().itemChanged.connect(self.changeImage)
        self.tree_images.clicked.connect(self.on_row_clicked_image)

        #self.gridLayout_color.addWidget(self.tree_colors, 1, 0, 1, 1)

        self.vscode_widget.add_tab(self.page1_images, settings.RESOURCE_DIR+ "/batch.png", "Batch Images")

    def _dock_enhancement(self, Main):
        self.scroll_enhancement = QtWidgets.QScrollArea()
        self.scroll_enhancement.setWidgetResizable(True)
        self.scroll_enhancement.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_enhancement.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # Only vertical scroll

        self.page_enhancement_container = QtWidgets.QWidget()
        self.layout_enh_container = QtWidgets.QVBoxLayout(self.page_enhancement_container)
        self.layout_enh_container.setContentsMargins(0, 0, 0, 0)
        self.layout_enh_container.setSpacing(1)  # Small gap between sections
        self.layout_enh_container.setAlignment(QtCore.Qt.AlignTop)
        self.enhancement_sections = []

        # --- HELPER: The Mutual Exclusion Logic ---


        self._create_dock_imEnhance_view1(Main)

        self._create_dock_imEnhance_vew2(Main)

        def on_box_toggled(is_checked, sender_box):
            # If a box was just OPENED (is_checked=True)
            if is_checked:
                # Loop through all known sections
                for box in self.enhancement_sections:
                    # Close everyone else
                    if box != sender_box:
                        box.collapse()

        self.box_view1.toggled.connect(lambda c: on_box_toggled(c, self.box_view1))
        self.box_view2.toggled.connect(lambda c: on_box_toggled(c, self.box_view2))

        self.scroll_enhancement.setWidget(self.page_enhancement_container)

        # IMPORTANT: Add a spacer at the bottom so the boxes push to the top
        self.layout_enh_container.addStretch()
        self.box_view1.expand()
        self.vscode_widget.add_tab(
            self.scroll_enhancement,
            settings.RESOURCE_DIR+"/imageEnh.png",  # Use a generic "Image" icon
            "Image Enhancement"
        )

    def createDockWidget(self, Main):
        """
        Creating main attributes for the main widgets
        :param Main:
        :return:
        """
        _translate = QtCore.QCoreApplication.translate
        self._create_main_dock(Main)
        #self._table_link_doc(Main)
        self._create_dockColor(Main)
        self._create_segDock(Main)
        #self._setting_radius_dock(Main)
        self._dock_enhancement(Main)
        self._some_intial_steps()
        self._dock_progress_bar(Main)
        self._dock_folder(Main)
        self._table_measure(Main)
        self._batchImages_dock(Main)


        #self.MainDock.setWindowTitle(_translate("Main", "Main Windows"))
        self.dock_progressbar.setWindowTitle(_translate("Main", "Progress Bar"))
        self.toggle1_1.setText(_translate("Main", "Toggle"))


        self.page1_rot_cor.setItemText(0, _translate("Main", "Coronal"))
        self.page1_rot_cor.setItemText(1, _translate("Main", "Sagittal"))
        self.page1_rot_cor.setItemText(2, _translate("Main", "Axial"))
        self.page2_rot_cor.setItemText(0, _translate("Main", "Coronal"))
        self.page2_rot_cor.setItemText(1, _translate("Main", "Sagittal"))
        self.page2_rot_cor.setItemText(2, _translate("Main", "Axial"))


        self.label_assigned_rad_circle.setText(_translate("Main", "Effect strength"))
        self.label_assigned_tol_rad_circle.setText(_translate("Main", "Tolerance AutoSeg"))
        #self.label_rad_circle.setText(_translate("Main", "0"))


    def ShowContextMenu_table1(self, pos):
        """
        Context Menu of the segmentation table
        :param pos:
        :return:
        """
        index = self.table_widget_measure.indexAt(pos)
        if not index.isValid():
            return

        menu = QMenu("Color")
        add_action = menu.addAction("&Add")
        edit_action = menu.addAction("&Edit")
        export_action = menu.addAction('&Export')
        remove_action = menu.addAction("&Remove")

        root = self.table_widget_measure
        action = menu.exec_(root.viewport().mapToGlobal(pos))
        if action == edit_action:
            root.edit(index)
        elif action==remove_action:
            # remove items with all the details
            rows = set()
            for index in root.selectedIndexes():
                rows.add(index.row())
            for row in sorted(rows, reverse=True):
                root.removeRow(row)
        elif action == add_action:
            root.insertRow(root.rowCount())
        elif action == export_action:

            filters = "CSV (*.csv)"
            opts = QtWidgets.QFileDialog.DontUseNativeDialog
            try:
                fileObj = self._filesave_dialog(filters, opts)
                export_tables(self, fileObj[0])
            except Exception as e:
                print(e)
        return


    def ShowContextMenu_images(self, pos):
        """
        Tree images Context Menu
        :param pos:
        :return:
        """
        def dialog():

            opts = QtWidgets.QFileDialog.DontUseNativeDialog
            dialg = QFileDialogPreview(self, "Open File", self.source_dir, self._filters, options=opts,
                                       index=self._last_index_select_image_mri,
                                       last_state=self._last_state_preview)

            dialg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            fileObj = [[''], '']
            if dialg.exec_() == QFileDialogPreview.Accepted:
                fileObj = dialg.getFilesSelected()
                fileObj = [fileObj, 0]
                fileObj[1] = dialg.selectedNameFilter()
            if fileObj[0][0] == '':
                return [fileObj, None]
            index = dialg._combobox_type.currentIndex()
            self.source_dir = os.path.dirname(fileObj[0][0])

            return [fileObj, index]

        index = self.tree_images.indexAt(pos)
        #if not index.isValid():
        #    return
        #if index.column()==0:
        #    return
        #_ind = self.tree_images.model().sourceModel().item(index.row(), 0).text()



        menu = QMenu("Images")

        menu_import = QtWidgets.QMenu(menu)
        menu_import.setObjectName('Import')
        menu_import.setWindowIconText('CCC')
        _translate = QtCore.QCoreApplication.translate
        menu_import.setTitle(_translate('Main', "Import"))
        import_image_action_1 = QtWidgets.QAction(self)
        icon_mri = QtGui.QIcon()
        icon_mri.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/mri.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon_mriS = QtGui.QIcon()
        icon_mriS.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/mri_seg.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        icon_eco = QtGui.QIcon()
        icon_eco.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/eco.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon_ecoS = QtGui.QIcon()
        icon_ecoS.addPixmap(QtGui.QPixmap(settings.RESOURCE_DIR+"/eco_seg.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)

        import_image_action_1.setIcon(icon_eco)
        import_image_action_1.setIconText('Images (view 1)')

        menu_import.addAction(import_image_action_1)
        import_seg_action_1 = QtWidgets.QAction(self)
        import_seg_action_1.setIconText('Segmentations (view 1)')

        import_seg_action_1.setIcon(icon_ecoS)

        menu_import.addAction(import_seg_action_1)

        menu_import.addSeparator()

        import_image_action_2 = QtWidgets.QAction(self)
        import_image_action_2.setIconText('Images (view 2)')
        import_image_action_2.setIcon(icon_mri)


        menu_import.addAction(import_image_action_2)
        import_seg_action_2 = QtWidgets.QAction(self)
        import_seg_action_2.setIconText('Segmentations (view 2)')
        import_seg_action_2.setIcon(icon_mriS)
        menu_import.addAction(import_seg_action_2)

        menu.addMenu(menu_import)

        remove_action = menu.addAction("&RemoveSelected")
        clear_action = menu.addAction("&ClearAll")

        action = menu.exec_(self.tree_images.viewport().mapToGlobal(pos))

        if action==remove_action:
            if self._basefileSave is None or self._basefileSave=='':
                return
            # remove the item with all the details
            if index.row()>=0:
                self.imported_images.pop(index.row())
                parent = self.tree_images.model().sourceModel().invisibleRootItem()
                parent.removeRow(index.row())
        elif action==clear_action:
            if self._basefileSave is None or self._basefileSave=='':
                return
            parent = self.tree_images.model().sourceModel().invisibleRootItem()
            index_to_r = []
            for i in range(parent.rowCount()):
                signal = parent.child(i)
                if signal.checkState() == QtCore.Qt.Unchecked:
                    index_to_r.append(i)
            for index in sorted(index_to_r, reverse=True):
                del self.imported_images[index]
                parent.removeRow(index)


        elif action==import_image_action_1:
            index_view = 0
            #if action==import_action:
            if self._basefileSave is None or self._basefileSave=='':
                return
            [fileObj, index] = dialog()
            filen = fileObj[0][0]
            if filen=='':
                return
            #if index<2:
            #    color = [1,0,0]
            #else:
            color = [0, 1, 1]
            update_image_sch(self, [fileObj, index, index_view], color =color)
        elif action==import_seg_action_1:
            index_view = 0
            if self._basefileSave is None or self._basefileSave=='':
                return
            [fileObj, index] = dialog()
            filen = fileObj[0][0]
            if filen=='':
                return
            index += 3
            #if index<5:
            #    color = [0.5,1,0]
            #else:
            color = [1, 0, 1]
            update_image_sch(self, [fileObj, index, index_view], color = color)


        elif action==import_image_action_2:
            index_view = 1
            #if action==import_action:
            if self._basefileSave is None or self._basefileSave=='':
                return
            [fileObj, index] = dialog()
            filen = fileObj[0][0]
            if filen=='':
                return
            #if index<2:
            #    color = [1,0,0]
            #else:
            color = [1, 1, 0]
            update_image_sch(self, [fileObj, index, index_view], color =color)
        elif action==import_seg_action_2:
            index_view = 1
            if self._basefileSave is None or self._basefileSave=='':
                return
            [fileObj, index] = dialog()
            filen = fileObj[0][0]
            if filen=='':
                return
            index += 3
            #if index<5:
            #    color = [0.5,1,0]
            #else:
            color = [0.5, 0.5, 1]
            update_image_sch(self, [fileObj, index, index_view], color = color)

    def ShowContextMenu_tree(self, pos):
        """
        Context Menu of color
        :param pos:
        :return:
        """
        index = self.tree_colors.indexAt(pos)
        if not index.isValid():
            return
        if index.column()==0:
            return
        _ind = self.tree_colors.model().sourceModel().item(index.row(), 0).text()
        if _ind == '9876':
            return

        from PyQt5.QtWidgets import QMenu, QAction
        menu = QMenu("Color")
        edit_action = menu.addAction("&Edit")
        remove_action = menu.addAction("&Remove")
        import_action = menu.addAction("&Import")
        export_action = menu.addAction("&Export")

        # --- Dynamic Submenu Creation ---
        menu_color = QtWidgets.QMenu("Scheme", menu)

        scheme_action_group = QtWidgets.QActionGroup(self)
        scheme_action_group.setExclusive(True)

        # 1. Define the directory to scan
        color_dir = Path(settings.RESOURCE_DIR) / 'color'

        # 2. Loop through each .txt file and create an action for it
        if color_dir.is_dir():
            # Sort files alphabetically for a consistent menu order
            for file_path in sorted(color_dir.glob('*.txt')):
                # Create an action for this file
                action = QtWidgets.QAction(self)

                # Use the filename without extension as the action's text (e.g., "Simple")
                action.setText(file_path.stem)

                action.setCheckable(True)
                scheme_action_group.addAction(action)

                # Store the full file path in the action's data field. This is key!
                action.setData(str(file_path))

                menu_color.addAction(action)

        if not menu_color.isEmpty():
            menu.addMenu(menu_color)

        action = menu.exec_(self.tree_colors.viewport().mapToGlobal(pos))
        if action == edit_action:
            #item = self.tree_colors.itemFromIndex(index)
            #self._ind, self._txt = item.text(0), item.text(1)
            self.tree_colors.edit(index)
        elif action==remove_action:
            # remove the item with all the details
            txt = ''
            for i in [0, 1]:
                txt += self.tree_colors.model().sourceModel().item(index.row(), i).text()+ '_'
            colr_n = txt[:-1]
            self.color_name.remove(colr_n)
            colrnum = int(colr_n.split('_')[0])
            self.color_index_rgb = self.color_index_rgb[self.color_index_rgb[:, 0] != colrnum,:]
            self.colorsCombinations.pop(colrnum, None)
            root = self.tree_colors.model().sourceModel().invisibleRootItem()
            root.removeRow(index.row())
        else:
            if action is None:
                return
            if action==import_action:

                filters = "LUT(*.txt *.lut)"
                opts = QFileDialog.DontUseNativeDialog
                fileObj = QFileDialog.getOpenFileName(self, "Open COLOR File", settings.DEFAULT_USE_DIR, filters, options=opts)
                filen = fileObj[0]
                if filen=='':
                    return
            elif action == export_action:
                filters = "LUT(*.txt *.lut)"
                opts = QtWidgets.QFileDialog.DontUseNativeDialog
                try:
                    filename = self._filesave_dialog(filters, opts)
                    if len(filename[0][0])==0:
                        return
                    filename = filename[0][0] + '.txt'
                    with open(filename, "w") as f:
                        line = f"#Index\t #Color name\t R\t G\t B\t A\t\n"
                        f.write(line)
                        # Loop through each row of the color_index_rgb array
                        for i, row in enumerate(self.color_index_rgb):
                            # Extract the index (first element) and use it to get color and name
                            label_index = int(row[0])
                            color_name = self.color_name[i]
                            rgba = self.colorsCombinations[label_index]
                            color_name = "_".join(color_name.split('_')[1:])
                            if color_name == 'Combined' and label_index==9876:
                                continue
                            # Format each line: Index LabelName R G B A
                            # Convert RGBA values to integers within range [0-255] for FSL format
                            r, g, b, a = (int(c * 255)  if i<3 else 1 for i, c in enumerate(rgba))
                            line = f"{label_index}\t {color_name}\t {r}\t {g}\t {b}\t {a}\t\n"
                            f.write(line)
                except Exception as e:
                    print(e)


            # 3. Handle ALL dynamically created scheme actions with one check
            elif action.data():
                filen = action.data()

            else:
                return
            try: #TODO : Sep 11 2024
                possible_color_name, possible_color_index_rgb, _ = read_txt_color(filen, from_one=False, mode='None')
            except:
                return
            #if not (hasattr(self, 'readView1') and hasattr(self, 'readView2')):

            ind_avail = [9876]
            try:
                ind_avail += list(np.unique(self.readView1.npSeg))
            except:
                pass
            try:
                ind_avail += list(np.unique(self.readView2.npSeg))
            except:
                pass
            if 0 in ind_avail:
                ind_avail.remove(0)

            color_vec = possible_color_index_rgb[:, 0].astype('int')
            list_avail = list(set(color_vec) & set(ind_avail))
            mask_color = np.isin(color_vec, list_avail)
            if mask_color.sum()==0:
                mask_color[0] = True
            possible_color_index_rgb = possible_color_index_rgb[mask_color, :]
            possible_color_name = list(np.array(possible_color_name)[mask_color])

            uq = []
            set_not_in_new_list = set(uq) - (set(possible_color_index_rgb[:, 0].astype('int')))
            set_kept_new_list = set_not_in_new_list - (
                        set_not_in_new_list - set(self.color_index_rgb[:, 0].astype('int')))
            set_create_new_list = set_not_in_new_list - set_kept_new_list
            for element in list(set_kept_new_list):
                new_color_rgb = self.color_index_rgb[self.color_index_rgb[:, 0] == element, :]
                possible_color_index_rgb = np.vstack((possible_color_index_rgb, new_color_rgb))
                try:
                    new_colr_name = [l for l in self.color_name if l.split('_')[0] == str(element)][0]
                except:
                    r, l = [[r, l] for r, l in enumerate(self.color_name) if l.split('_')[0] == str(float(element))][0]
                    l2 = str(int(float(l.split('_fre')[0]))) + '_' + '_'.join(l.split('_')[1:])
                    self.color_name[r] = l2
                    new_colr_name = [l for l in self.color_name if l.split('_')[0] == str(element)][0]
                possible_color_name.append(new_colr_name)

            for element in set_create_new_list:
                new_colr_name = '{}_structure_unknown'.format(element)
                possible_color_name.append(new_colr_name)
                new_color_rgb = [element, np.random.rand(), np.random.rand(), np.random.rand(), 1]
                possible_color_index_rgb = np.vstack((possible_color_index_rgb, np.array(new_color_rgb)))
            if 9876 not in possible_color_index_rgb[:, 0]:
                new_colr_name = '9876_Combined'
                new_color_rgb = [9876, 1, 0, 0, 1]
                possible_color_name.append(new_colr_name)
                possible_color_index_rgb = np.vstack((possible_color_index_rgb, np.array(new_color_rgb)))




            # self.color_index_rgb, self.color_name, self.colorsCombinations = combinedIndex(self.colorsCombinations, possible_color_index_rgb, possible_color_name, np.unique(data), uq1)
            self.color_index_rgb, self.color_name, self.colorsCombinations = generate_color_scheme_info(
                self, possible_color_index_rgb, possible_color_name)


            try:
                # self.dw2_cb.currentTextChanged.disconnect(self.changeColorPen)
                self.tree_colors.itemChanged.disconnect(self.changeColorPen)
            except:
                pass

            set_new_color_scheme(self)
            try:
                # self.dw2_cb.currentTextChanged.connect(self.changeColorPen)
                self.tree_colors.itemChanged.connect(self.changeColorPen)
            except:
                pass
            widgets_num = [0, 1, 2, 3, 4, 5, 10, 11, 13, 23]
            for num in widgets_num:
                name = 'openGLWidget_' + str(num + 1)
                widget = getattr(self, name)
                if hasattr(widget, 'colorsCombinations'):
                    widget.colorsCombinations = self.colorsCombinations
                if hasattr(widget, 'color_name'):
                    widget.color_name = self.color_name
                if not widget.isVisible():
                    continue
                if hasattr(widget, 'makeObject'):
                    widget.makeObject()


                if num in [13]:
                    try:
                        widget.paint(self.readView1.npSeg,
                                           self.readView1.npImage, None)
                    except:
                        pass
                elif num in [23]:
                    try:
                        widget.paint(self.readView2.npSeg,
                          self.readView2.npImage, None)
                    except:
                        pass
                widget.update()


        return

    @QtCore.pyqtSlot("QWidget*")
    def _on_closeEditor(self, editor):
        p = editor.pos()
        index = self.tree_colors.indexAt(p)
        if index.column()==0:
            return
        _ind = self.tree_colors.model().sourceModel().item(index.row(), 0).text()
        _txt = self.tree_colors.model().sourceModel().item(index.row(), 1).text()
        new = '_'.join([_ind, _txt])

        if new not in self.color_name:
            cls = [r for r, col in enumerate(self.color_name) if col.split('_')[0] == _ind]
            if len(cls)>0:
                r = cls[0]
                self.color_name[r] = new

    def reset_view_pages(self, index=0):
        if index==0:
            self.t1_1.setValue(0)
            self.t1_2.setValue(100)
            self.t1_3.setValue(100)
            self.t1_4.setValue(0)
            self.t1_5.setValue(0)
            self.t1_7.setValue(0)

            self.toggle1_1.setChecked(False)
        elif index==1:
            self.t2_1.setValue(0)
            self.t2_2.setValue(100)
            self.t2_3.setValue(100)
            self.t2_4.setValue(0)
            self.t2_5.setValue(0)
            self.t2_7.setValue(0)

            self.toggle2_1.setChecked(False)





    def _double_clicked(self, item):
        if item.column()==1:
            _ind = self.tree_colors.model().sourceModel().item(item.row(), 0).text()
            if _ind == '9876':
                return
            #text = item.data(role=QtCore.Qt.DisplayRole)
            self.tree_colors.edit(item)

    def searchTreeChanged(self, text=None):
        """

        :param text:
        :return:
        """
        if text == 'image':
            text = self.line_text_image.text()
            tags_model = self.tags_model_image
            trees = self.tree_images
        elif text == 'color':
            text = self.line_text.text()
            tags_model = self.tags_model
            trees = self.tree_colors
        regExp = QtCore.QRegExp(text, QtCore.Qt.CaseInsensitive, QtCore.QRegExp.FixedString)

        tags_model.text = text.lower()
        tags_model.setFilterRegExp(regExp)

        if len(text) >= 1 and tags_model.rowCount() > 0:
            trees.expandAll()
        else:
            trees.collapseAll()



class SearchProxyModel(QtCore.QSortFilterProxyModel):
    """
    Class to search for available color lists
    """
    def __init__(self, parent=None):
        super(SearchProxyModel, self).__init__(parent)
        self.text = ''

    # Recursive search
    def _accept_index(self, idx):
        if idx.isValid():
            text = idx.data(role=QtCore.Qt.DisplayRole).lower()
            condition = text.find(self.text) >= 0

            if condition:
                return True
            for childnum in range(idx.model().rowCount(parent=idx)):
                if self._accept_index(idx.model().index(childnum, 0, parent=idx)):
                    return True
        return False

    def filterAcceptsRow(self, sourceRow, sourceParent):
        # Only first column in model for search
        idx = self.sourceModel().index(sourceRow, 1, sourceParent)
        return self._accept_index(idx)

    def lessThan(self, left, right):
        leftData = self.sourceModel().data(left)
        rightData = self.sourceModel().data(right)
        try:
            return float(leftData) < float(rightData)
        except ValueError:
            return leftData < rightData
