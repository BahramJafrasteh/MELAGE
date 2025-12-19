__AUTHOR__ = 'Bahram Jafrasteh'

from pathlib import Path
import sys
sys.path.append("../")
from PyQt5 import QtWidgets, QtCore, QtGui
from functools import partial
from qtwidgets import AnimatedToggle
from melage.utils.utils import generate_color_scheme_info
from PyQt5 import Qt
import numpy as np
import os
colorNames =("#FFCC08","darkRed","red", "darkOrange", "orange", "#8b8b00","yellow",
             "darkGreen","green","darkCyan","cyan",
             "darkBlue","blue","magenta","darkMagenta", 'red')


from melage.utils.utils import read_txt_color, set_new_color_scheme, addTreeRoot, update_color_scheme, addLastColor, update_image_sch
from melage.config import settings





class dockWidgets():
    """
    This class has been implemented for dock widgets in MELAGE
    """
    def __init__(self):
        pass

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




    def createDockWidget(self, Main):
        """
        Creating main attributes for the main widgets
        :param Main:
        :return:
        """
        ################################################### Segmentation Intensity ##############################################
        self.dockSegmentationIntensity = QtWidgets.QDockWidget(Main)
        self.dockSegmentationIntensity.setObjectName("dockSegmentationIntensity")
        self.dockSegmentationIntensity.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 12))

        # Create main content widget
        self.content_segInt = QtWidgets.QWidget()
        self.content_segInt.setObjectName("content_segInt")

        # Create layout for main content widget
        self.gridLayout_segIn = QtWidgets.QGridLayout(self.content_segInt)
        self.gridLayout_segIn.setObjectName("gridLayout_segIn")

        # Create label and scroll bar
        self.label_seg_intensity_title = QtWidgets.QLabel("Segmentation Intensity")
        self.label_intensity_value = QtWidgets.QLabel("100")
        self.label_intensity_value.setAlignment(QtCore.Qt.AlignCenter)
        #self.gridLayout_segIn.addWidget(self.label_intensity_value, 0, 0, 1, 1)

        self.scroll_intensity = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.scroll_intensity.setObjectName("scroll_intensity")
        self.scroll_intensity.setRange(0, 100)
        self.scroll_intensity.setValue(100)
        self.scroll_intensity.setSingleStep(1)
        #self.gridLayout_segIn.addWidget(self.scroll_intensity, 1, 0, 1, 1)
        self.scroll_intensity.valueChanged.connect(self.label_intensity_value.setNum)


        self.line_intensity = QtWidgets.QFrame()
        self.line_intensity.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_intensity.setFrameShadow(QtWidgets.QFrame.Sunken)

        # Create second label and scroll bar for image intensity
        self.label_image_intensity_title = QtWidgets.QLabel("Image Intensity")

        self.label_image_intensity_value = QtWidgets.QLabel("100")
        self.label_image_intensity_value.setAlignment(QtCore.Qt.AlignCenter)

        self.scroll_image_intensity = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.scroll_image_intensity.setObjectName("scroll_image_intensity")
        self.scroll_image_intensity.setRange(0, 100)
        self.scroll_image_intensity.setValue(100)
        self.scroll_image_intensity.setSingleStep(1)
        self.scroll_image_intensity.valueChanged.connect(self.label_image_intensity_value.setNum)

        self.line_image_intensity = QtWidgets.QFrame()
        self.line_image_intensity.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_image_intensity.setFrameShadow(QtWidgets.QFrame.Sunken)


        # Group label and scroll bar in a horizontal layout
        seg_group_layout = QtWidgets.QVBoxLayout()
        seg_group_layout.addWidget(self.label_seg_intensity_title)
        seg_group_layout.addWidget(self.label_intensity_value)
        seg_group_layout.addWidget(self.scroll_intensity)
        seg_group_layout.addWidget(self.line_intensity)

        # Group second scroll bar and label in the same vertical layout
        im_group_layout = QtWidgets.QVBoxLayout()
        im_group_layout.addWidget(self.label_image_intensity_title)
        im_group_layout.addWidget(self.label_image_intensity_value)
        im_group_layout.addWidget(self.scroll_image_intensity)
        im_group_layout.addWidget(self.line_image_intensity)

        # Add the horizontal layout to the main grid layout
        self.gridLayout_segIn.addLayout(seg_group_layout, 0, 0, 1, 1)
        self.gridLayout_segIn.addLayout(im_group_layout, 1, 0, 1, 1)

        # Set layout for main content widget
        self.dockSegmentationIntensity.setWidget(self.content_segInt)

        # Add dock widget to main window
        #Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockSegmentationIntensity)
        #self.dockSegmentationIntensity.setVisible(True)
        #################################################################################################

        self.dockImageEnh = QtWidgets.QDockWidget(Main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockImageEnh.sizePolicy().hasHeightForWidth())
        self.dockImageEnh.setSizePolicy(sizePolicy)
        self.dockImageEnh.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))
        self.dockImageEnh.setObjectName("dockImageEnh")
        self.content_imageEnh = QtWidgets.QWidget()
        self.content_imageEnh.setObjectName("content_imageEnh")


        self.Settings_widget = QtWidgets.QWidget()
        self.Settings_widget.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))

        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.ImageEnh_view1.sizePolicy().hasHeightForWidth())
        #self.ImageEnh_view1.setSizePolicy(sizePolicy)
        self.Settings_widget.setObjectName("setting")
        self.gridLayout_settings = QtWidgets.QGridLayout(self.Settings_widget)
        self.gridLayout_settings.setObjectName("gridLayout_settings")




        self.gridLayout_5 = QtWidgets.QGridLayout(self.content_imageEnh)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.main_toolbox = QtWidgets.QToolBox(self.content_imageEnh)
        self.main_toolbox.setObjectName("main_toolbox")
        self.main_toolbox.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))



        #self.dockImageConf = QtWidgets.QDockWidget(Main)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.dockImageConf.sizePolicy().hasHeightForWidth())
        #self.dockImageConf.setSizePolicy(sizePolicy)
        #self.dockImageConf.setMinimumSize(QtCore.QSize(200, 167))
        #self.dockImageConf.setObjectName("dockImageConf")

        #self.content_imageConf = QtWidgets.QWidget()
        #self.content_imageConf.setObjectName("content_imageConf")
        #self.gridLayout_imageConf = QtWidgets.QGridLayout(self.content_imageConf)
        #self.gridLayout_imageConf.setObjectName("gridLayout_5")
        #self.toolbox_imageConf = QtWidgets.QToolBox(self.content_imageConf)
        #self.toolbox_imageConf.setObjectName("toolbox_imageConf")
        #self.toolbox_imageConf.setMinimumSize(self.width() // 7, self.height()//2)
        #self.gridLayout_imageConf.addWidget(self.toolbox_imageConf, 0, 0, 1, 1)


        #########
        ################ Widget MRI COLORS ####################################

        self.page1_color = QtWidgets.QWidget()

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





        #self.gridLayout_color.addWidget(self.tree_colors, 1, 0, 1, 1)

        self.main_toolbox.addItem(self.page1_color, "")
        ###############












        self.ImageEnh_view1 = QtWidgets.QWidget()
        self.ImageEnh_view1.setGeometry(QtCore.QRect(0, 0, self.width()//8, self.height()//2))

        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.ImageEnh_view1.sizePolicy().hasHeightForWidth())
        #self.ImageEnh_view1.setSizePolicy(sizePolicy)
        self.ImageEnh_view1.setObjectName("page")
        self.gridLayout_view1 = QtWidgets.QGridLayout(self.ImageEnh_view1)
        self.gridLayout_view1.setObjectName("gridLayout_view1")

        self.line_5 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout_view1.addWidget(self.line_5, 0, 0, 1, 1)



        self.lb_ft1_1 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_1.sizePolicy().hasHeightForWidth())
        self.lb_ft1_1.setSizePolicy(sizePolicy)
        self.lb_ft1_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lb_ft1_1.setObjectName("lb_ft1_1")
        self.gridLayout_view1.addWidget(self.lb_ft1_1, 1, 0, 1, 1)


        self.lb_t1_1 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_1.sizePolicy().hasHeightForWidth())
        self.lb_t1_1.setSizePolicy(sizePolicy)
        self.lb_t1_1.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_1.setObjectName("lb_t1_1")
        self.gridLayout_view1.addWidget(self.lb_t1_1, 2, 0, 1, 1)


        self.hs_t1_1 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        self.hs_t1_1.setMinimum(-100)
        self.hs_t1_1.setMaximum(100)
        self.hs_t1_1.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_1.setObjectName("hs_t1_1")
        self.gridLayout_view1.addWidget(self.hs_t1_1, 3, 0, 1, 1)



        self.line_6 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout_view1.addWidget(self.line_6, 4, 0, 1, 1)


        self.lb_ft1_2 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_2.sizePolicy().hasHeightForWidth())
        self.lb_ft1_2.setSizePolicy(sizePolicy)
        self.lb_ft1_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lb_ft1_2.setObjectName("lb_ft1_2")
        self.gridLayout_view1.addWidget(self.lb_ft1_2, 5, 0, 1, 1)



        self.lb_t1_2 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_2.sizePolicy().hasHeightForWidth())
        self.lb_t1_2.setSizePolicy(sizePolicy)
        self.lb_t1_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_2.setObjectName("lb_t1_2")
        self.gridLayout_view1.addWidget(self.lb_t1_2, 6, 0, 1, 1)


        self.hs_t1_2 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        self.hs_t1_2.setMaximum(100)
        self.hs_t1_2.setMinimum(-100)
        self.hs_t1_2.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_2.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_2.setObjectName("hs_t1_2")
        self.gridLayout_view1.addWidget(self.hs_t1_2, 7, 0, 1, 1)



        self.line_7 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout_view1.addWidget(self.line_7, 8, 0, 1, 1)


        self.lb_ft1_3 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_3.sizePolicy().hasHeightForWidth())
        self.lb_ft1_3.setSizePolicy(sizePolicy)
        self.lb_ft1_3.setObjectName("lb_ft1_3")
        self.gridLayout_view1.addWidget(self.lb_ft1_3, 9, 0, 1, 1)



        self.lb_t1_3 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_3.sizePolicy().hasHeightForWidth())
        self.lb_t1_3.setSizePolicy(sizePolicy)
        self.lb_t1_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_3.setObjectName("lb_t1_3")
        self.gridLayout_view1.addWidget(self.lb_t1_3, 10, 0, 1, 1)



        self.hs_t1_3 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        self.hs_t1_3.setMaximum(100)
        self.hs_t1_3.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_3.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_3.setObjectName("hs_t1_3")
        self.gridLayout_view1.addWidget(self.hs_t1_3, 11, 0, 1, 1)

        self.line_8 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.gridLayout_view1.addWidget(self.line_8, 12, 0, 1, 1)





        self.lb_ft1_7 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_7.sizePolicy().hasHeightForWidth())
        self.lb_ft1_7.setSizePolicy(sizePolicy)
        self.lb_ft1_7.setObjectName("lb_ft1_7")
        self.gridLayout_view1.addWidget(self.lb_ft1_7, 13, 0, 1, 1)


        self.lb_t1_7 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_7.sizePolicy().hasHeightForWidth())
        self.lb_t1_7.setSizePolicy(sizePolicy)
        self.lb_t1_7.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_7.setObjectName("lb_t1_7")
        self.gridLayout_view1.addWidget(self.lb_t1_7, 14, 0, 1, 1)



        self.hs_t1_7 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hs_t1_7.sizePolicy().hasHeightForWidth())
        self.hs_t1_7.setSizePolicy(sizePolicy)
        self.hs_t1_7.setMinimum(0)
        self.hs_t1_7.setMaximum(100)
        self.hs_t1_7.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_7.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_7.setObjectName("hs_t1_4")
        self.gridLayout_view1.addWidget(self.hs_t1_7, 15, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_view1.addWidget(self.line_11, 16, 0, 1, 1)



        self.lb_ft1_4 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_4.sizePolicy().hasHeightForWidth())
        self.lb_ft1_4.setSizePolicy(sizePolicy)
        self.lb_ft1_4.setObjectName("lb_ft1_4")
        self.gridLayout_view1.addWidget(self.lb_ft1_4, 17, 0, 1, 1)


        self.lb_t1_4 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_4.sizePolicy().hasHeightForWidth())
        self.lb_t1_4.setSizePolicy(sizePolicy)
        self.lb_t1_4.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_4.setObjectName("lb_t1_4")
        self.gridLayout_view1.addWidget(self.lb_t1_4, 18, 0, 1, 1)



        self.hs_t1_4 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hs_t1_4.sizePolicy().hasHeightForWidth())
        self.hs_t1_4.setSizePolicy(sizePolicy)
        self.hs_t1_4.setMinimum(0)
        self.hs_t1_4.setMaximum(100)
        self.hs_t1_4.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_4.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_4.setObjectName("hs_t1_4")
        self.gridLayout_view1.addWidget(self.hs_t1_4, 19, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_view1.addWidget(self.line_11, 20, 0, 1, 1)


        self.lb_ft1_5 = QtWidgets.QLabel(self.ImageEnh_view1)
        self.lb_ft1_5.setObjectName("lb_ft1_5")
        self.gridLayout_view1.addWidget(self.lb_ft1_5, 21, 0, 1, 1)


        ################### WIDGET COMBOX ROTATION ###########################

        self.page1_rot_cor = QtWidgets.QComboBox(self.ImageEnh_view1)
        cbstyle = """
            QComboBox QAbstractItemView {border: 1px solid grey;
            background: #03211c; 
            selection-background-color: #03211c;} 
            QComboBox {background: #03211c;margin-right: 1px;}
            QComboBox::drop-down {
        subcontrol-origin: margin;}
            """
        self.page1_rot_cor.setStyleSheet(cbstyle)
        self.page1_rot_cor.setObjectName("dw2_cb")
        self.page1_rot_cor.addItem("")
        self.page1_rot_cor.addItem("")
        self.page1_rot_cor.addItem("")


        self.gridLayout_view1.addWidget(self.page1_rot_cor, 22, 0, 1, 1)


        self.lb_t1_5 = QtWidgets.QLabel(self.ImageEnh_view1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_5.sizePolicy().hasHeightForWidth())
        self.lb_t1_5.setSizePolicy(sizePolicy)
        self.lb_t1_5.setObjectName("lb_t1_5")
        self.lb_t1_5.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout_view1.addWidget(self.lb_t1_5, 23, 0, 1, 1)


        #self.hs_t1_5 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        self.hs_t1_5 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        #self.hs_t1_5.setPageStep(0.5)
        self.hs_t1_5.setMinimum(-50)
        self.hs_t1_5.setMaximum(50)
        #self.hs_t1_5.setTickInterval(1)
        self.hs_t1_5.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_5.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_5.setObjectName("hs_t1_5")
        self.gridLayout_view1.addWidget(self.hs_t1_5, 24, 0, 1, 1)



        #self.page1_rot_cor = QtWidgets.QCheckBox(self.ImageEnh_view1)
        #self.page1_rot_cor.setObjectName("page1_rot_cor")






        self.line_ = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_.setObjectName("line_")
        self.gridLayout_view1.addWidget(self.line_, 25, 0, 1, 1)




        self.page1_s2c = QtWidgets.QCheckBox(self.ImageEnh_view1)
        self.page1_s2c.setObjectName("page1_s2c")
        self.gridLayout_view1.addWidget(self.page1_s2c, 26, 0, 1, 1)





        self.line_10 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.gridLayout_view1.addWidget(self.line_10, 27, 0, 1, 1)

        self.lb_ft1_6 = QtWidgets.QLabel(self.ImageEnh_view1)
        self.lb_ft1_6.setObjectName("lb_ft1_6")
        self.gridLayout_view1.addWidget(self.lb_ft1_6, 28, 0, 1, 1)

        self.toggle1_1 = AnimatedToggle(
            checked_color="#FFB000",
            pulse_checked_color="#44FFB000"
        )

        self.toggle1_1.setObjectName('toggle1_1')

        self.gridLayout_view1.addWidget(self.toggle1_1, 29, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.ImageEnh_view1)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_10")
        self.gridLayout_view1.addWidget(self.line_11, 30, 0, 1, 1)


        #self.colorize = QtWidgets.QCheckBox(self.ImageEnh_view1)
        #self.colorize.setObjectName("Colorize")
        #self.gridLayout_view1.addWidget(self.colorize, 31, 0, 1, 1)



        # self.hs_t1_5 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        #self.hs_t1_8 = QtWidgets.QScrollBar(self.ImageEnh_view1)
        #self.hs_t1_8.setPageStep(0.5)
        #self.hs_t1_8.setMinimum(2)
        #self.hs_t1_8.setMaximum(50)
        #self.hs_t1_8.setValue(0)

        # self.hs_t1_5.setTickInterval(1)
        #self.hs_t1_8.setOrientation(QtCore.Qt.Horizontal)
        # self.hs_t1_5.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        #self.hs_t1_8.setObjectName("hs_t1_8")
        #self.gridLayout_view1.addWidget(self.hs_t1_8, 33, 0, 1, 1)



        self.main_toolbox.addItem(self.ImageEnh_view1, "")


        ########### PAGE 1 MRI ################

        self.page1_mri = QtWidgets.QWidget()
        #palet = QtGui.QPalette()
        #palet.setColor(QtGui.QPalette.Window, QtCore.Qt.blue)
        #self.page1_mri.setStyleSheet("background-color: black")
        #self.page1_mri.setAutoFillBackground(True);
        #self.page1_mri.setPalette(palet)
        self.page1_mri.setGeometry(QtCore.QRect(0, 0, self.width()//8, self.height()//2))
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.page1_mri.sizePolicy().hasHeightForWidth())
        #self.page1_mri.setSizePolicy(sizePolicy)
        self.page1_mri.setObjectName("page")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.page1_mri)
        self.gridLayout_8.setObjectName("gridLayout_view1")

        self.line_5 = QtWidgets.QFrame(self.page1_mri)
        self.line_5.setStyleSheet('background-color: rgb(50,50,50)')
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout_8.addWidget(self.line_5, 0, 0, 1, 1)

        self.lb_ft2_1 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft2_1.sizePolicy().hasHeightForWidth())
        self.lb_ft2_1.setSizePolicy(sizePolicy)
        self.lb_ft2_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lb_ft2_1.setObjectName("lb_ft2_1")
        self.gridLayout_8.addWidget(self.lb_ft2_1, 1, 0, 1, 1)

        self.lb_t2_1 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t2_1.sizePolicy().hasHeightForWidth())
        self.lb_t2_1.setSizePolicy(sizePolicy)
        self.lb_t2_1.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t2_1.setObjectName("lb_t2_1")
        self.gridLayout_8.addWidget(self.lb_t2_1, 2, 0, 1, 1)

        self.hs_t2_1 = QtWidgets.QScrollBar(self.page1_mri)
        self.hs_t2_1.setMinimum(-100)
        self.hs_t2_1.setMaximum(100)
        self.hs_t2_1.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t2_1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_1.setObjectName("hs_t2_1")
        self.gridLayout_8.addWidget(self.hs_t2_1, 3, 0, 1, 1)

        self.line_6 = QtWidgets.QFrame(self.page1_mri)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout_8.addWidget(self.line_6, 4, 0, 1, 1)

        self.lb_ft2_2 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft2_2.sizePolicy().hasHeightForWidth())
        self.lb_ft2_2.setSizePolicy(sizePolicy)
        self.lb_ft2_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lb_ft2_2.setObjectName("lb_ft2_2")
        self.gridLayout_8.addWidget(self.lb_ft2_2, 5, 0, 1, 1)

        self.lb_t2_2 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t2_2.sizePolicy().hasHeightForWidth())
        self.lb_t2_2.setSizePolicy(sizePolicy)
        self.lb_t2_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t2_2.setObjectName("lb_t2_2")
        self.gridLayout_8.addWidget(self.lb_t2_2, 6, 0, 1, 1)

        self.hs_t2_2 = QtWidgets.QScrollBar(self.page1_mri)
        self.hs_t2_2.setMaximum(100)
        self.hs_t2_2.setMinimum(-100)
        self.hs_t2_2.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t2_2.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_2.setObjectName("hs_t2_2")
        self.gridLayout_8.addWidget(self.hs_t2_2, 7, 0, 1, 1)

        self.line_7 = QtWidgets.QFrame(self.page1_mri)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout_8.addWidget(self.line_7, 8, 0, 1, 1)

        self.lb_ft2_3 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft2_3.sizePolicy().hasHeightForWidth())
        self.lb_ft2_3.setSizePolicy(sizePolicy)
        self.lb_ft2_3.setObjectName("lb_ft2_3")
        self.gridLayout_8.addWidget(self.lb_ft2_3, 9, 0, 1, 1)

        self.lb_t2_3 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t2_3.sizePolicy().hasHeightForWidth())
        self.lb_t2_3.setSizePolicy(sizePolicy)
        self.lb_t2_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t2_3.setObjectName("lb_t2_3")
        self.gridLayout_8.addWidget(self.lb_t2_3, 10, 0, 1, 1)

        self.hs_t2_3 = QtWidgets.QScrollBar(self.page1_mri)
        self.hs_t2_3.setMaximum(100)
        self.hs_t2_3.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t2_3.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_3.setObjectName("hs_t2_3")
        self.gridLayout_8.addWidget(self.hs_t2_3, 11, 0, 1, 1)

        self.line_8 = QtWidgets.QFrame(self.page1_mri)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.gridLayout_8.addWidget(self.line_8, 12, 0, 1, 1)







        self.lb_ft2_7 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft2_7.sizePolicy().hasHeightForWidth())
        self.lb_ft2_7.setSizePolicy(sizePolicy)
        self.lb_ft2_7.setObjectName("lb_ft2_7")
        self.gridLayout_8.addWidget(self.lb_ft2_7, 13, 0, 1, 1)


        self.lb_t2_7 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t2_7.sizePolicy().hasHeightForWidth())
        self.lb_t2_7.setSizePolicy(sizePolicy)
        self.lb_t2_7.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t2_7.setObjectName("lb_t2_7")
        self.gridLayout_8.addWidget(self.lb_t2_7, 14, 0, 1, 1)



        self.hs_t2_7 = QtWidgets.QScrollBar(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hs_t2_7.sizePolicy().hasHeightForWidth())
        self.hs_t2_7.setSizePolicy(sizePolicy)
        self.hs_t2_7.setMinimum(0)
        self.hs_t2_7.setMaximum(100)
        self.hs_t2_7.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t2_7.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_7.setObjectName("hs_t2_7")
        self.gridLayout_8.addWidget(self.hs_t2_7, 15, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.page1_mri)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_8.addWidget(self.line_11, 16, 0, 1, 1)







        self.lb_ft2_4 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft2_4.sizePolicy().hasHeightForWidth())
        self.lb_ft2_4.setSizePolicy(sizePolicy)
        self.lb_ft2_4.setObjectName("lb_ft2_4")
        self.gridLayout_8.addWidget(self.lb_ft2_4, 17, 0, 1, 1)

        self.lb_t2_4 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t2_4.sizePolicy().hasHeightForWidth())
        self.lb_t2_4.setSizePolicy(sizePolicy)
        self.lb_t2_4.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t2_4.setObjectName("lb_t2_4")
        self.gridLayout_8.addWidget(self.lb_t2_4, 18, 0, 1, 1)

        self.hs_t2_4 = QtWidgets.QScrollBar(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hs_t2_4.sizePolicy().hasHeightForWidth())
        self.hs_t2_4.setSizePolicy(sizePolicy)
        self.hs_t2_4.setMinimum(0)
        self.hs_t2_4.setMaximum(100)
        self.hs_t2_4.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t2_4.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_4.setObjectName("hs_t2_4")
        self.gridLayout_8.addWidget(self.hs_t2_4, 19, 0, 1, 1)

        self.line_11 = QtWidgets.QFrame(self.page1_mri)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_8.addWidget(self.line_11, 20, 0, 1, 1)

        self.lb_ft2_5 = QtWidgets.QLabel(self.page1_mri)
        self.lb_ft2_5.setObjectName("lb_ft2_5")
        self.gridLayout_8.addWidget(self.lb_ft2_5, 21, 0, 1, 1)


        self.page2_rot_cor = QtWidgets.QComboBox(self.page1_mri)
        cbstyle = """
            QComboBox QAbstractItemView {border: 1px solid grey;
            background: white; 
            selection-background-color: #03211c;} 
            QComboBox {background: #03211c;margin-right: 1px;}
            QComboBox::drop-down {
        subcontrol-origin: margin;}
            """
        self.page2_rot_cor.setStyleSheet(cbstyle)
        self.page2_rot_cor.setObjectName("page2_rot_cor")
        self.page2_rot_cor.addItem("")
        self.page2_rot_cor.addItem("")
        self.page2_rot_cor.addItem("")


        self.gridLayout_8.addWidget(self.page2_rot_cor, 22, 0, 1, 1)

        self.lb_t2_5 = QtWidgets.QLabel(self.page1_mri)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t2_5.sizePolicy().hasHeightForWidth())
        self.lb_t2_5.setSizePolicy(sizePolicy)
        self.lb_t2_5.setObjectName("lb_t2_5")
        self.lb_t2_5.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout_8.addWidget(self.lb_t2_5, 23, 0, 1, 1)

        #self.hs_t2_5 = QtWidgets.QScrollBar(self.page1_mri)
        self.hs_t2_5 = QtWidgets.QScrollBar(self.page1_mri)
        self.hs_t2_5.setMinimum(-25)
        self.hs_t2_5.setMaximum(25)
        #self.hs_t2_5.setTickInterval(1)
        self.hs_t2_5.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t2_5.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_5.setObjectName("hs_t2_5")
        self.gridLayout_8.addWidget(self.hs_t2_5, 24, 0, 1, 1)

        #self.page2_rot_cor = QtWidgets.QCheckBox(self.page1_mri)
        #self.page2_rot_cor.setObjectName("page2_rot_cor")



        line_ = QtWidgets.QFrame(self.page1_mri)
        line_.setFrameShape(QtWidgets.QFrame.HLine)
        line_.setFrameShadow(QtWidgets.QFrame.Sunken)
        line_.setObjectName("line_")
        self.gridLayout_8.addWidget(line_, 25, 0, 1, 1)


        self.page2_s2c = QtWidgets.QCheckBox(self.page1_mri)
        self.page2_s2c.setObjectName("page2_s2c")
        self.gridLayout_8.addWidget(self.page2_s2c, 26, 0, 1, 1)

        self.line_10 = QtWidgets.QFrame(self.page1_mri)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.gridLayout_8.addWidget(self.line_10, 27, 0, 1, 1)

        self.lb_ft2_6 = QtWidgets.QLabel(self.page1_mri)
        self.lb_ft2_6.setObjectName("lb_ft2_6")
        self.gridLayout_8.addWidget(self.lb_ft2_6, 28, 0, 1, 1)

        self.toggle2_1 = AnimatedToggle(
            checked_color="#FFB000",
            pulse_checked_color="#44FFB000"
        )

        self.toggle2_1.setObjectName('toggle2_1')

        self.gridLayout_8.addWidget(self.toggle2_1, 29, 0, 1, 1)

        self.line_11 = QtWidgets.QFrame(self.page1_mri)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_10")
        self.gridLayout_8.addWidget(self.line_11, 30, 0, 1, 1)




        self.main_toolbox.addItem(self.page1_mri, "")





        self.gridLayout_5.addWidget(self.main_toolbox, 0, 0, 1, 1)
        self.dockImageEnh.setWidget(self.content_imageEnh)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockImageEnh)
        self.dockImageEnh.setVisible(True)




        self.page1_mri.setVisible(True)
        self.ImageEnh_view1.setVisible(True)

        self.hs_t1_1.valueChanged.connect(self.lb_t1_1.setNum)
        self.hs_t1_2.valueChanged.connect(self.lb_t1_2.setNum)
        self.hs_t1_3.valueChanged.connect(self.lb_t1_3.setNum)
        self.hs_t1_4.valueChanged.connect(self.lb_t1_4.setNum)
        self.hs_t1_5.valueChanged.connect(self.lb_t1_5.setNum)
        self.hs_t1_7.valueChanged.connect(self.lb_t1_7.setNum)
        #self.hs_t1_8.valueChanged.connect(self.lb_t1_8.setNum)


        self.hs_t2_1.valueChanged.connect(self.lb_t2_1.setNum)
        self.hs_t2_2.valueChanged.connect(self.lb_t2_2.setNum)
        self.hs_t2_3.valueChanged.connect(self.lb_t2_3.setNum)
        self.hs_t2_4.valueChanged.connect(self.lb_t2_4.setNum)
        self.hs_t2_5.valueChanged.connect(self.lb_t2_5.setNum)
        #self.hs_t2_8.valueChanged.connect(self.lb_t2_8.setNum)
        self.hs_t2_7.valueChanged.connect(self.lb_t2_7.setNum)

        ################ NEW WIDGET ####################################



        #self.dw2_cb = QtWidgets.QComboBox(self.dockWidgetContents_2)
        cbstyle = """
        QComboBox QAbstractItemView {border: 1px solid grey;
        background: white; 
        selection-background-color: blue;} 
        QComboBox {background: #03211c;margin-right: 1px;}
        QComboBox::drop-down {
    subcontrol-origin: margin;}
        """
        #self.dw2_cb.setStyleSheet(cbstyle)
        #self.dw2_cb.setObjectName("dw2_cb")


        #size = self.dw2_cb.style().pixelMetric(QtWidgets.QStyle.PM_SmallIconSize)
        #pixmp = QtGui.QPixmap(size, size)
        #color_name, color_index_rgb, _ = read_txt_color(settings.RESOURCE_DIR+"/color/LUT_albert.txt", from_one=True)
        #set_new_color_scheme(self, color_name, color_index_rgb)

        self.color_name, self.color_index_rgb, _ = read_txt_color(settings.RESOURCE_DIR+"/color/Simple.txt", mode= '', from_one=True)
        #update_color_scheme(self, None, dialog=False, update_widget=False)
        from collections import defaultdict
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
        #####


        """
        
        self.colorsCombinations = defaultdict(list)
        for i in range(16):
            pixmp.fill(QtGui.QColor(colorNames[i]))
            if i == 14:
                self.colorsCombinations[i+1] = (1,1,1,1)
            else:
                self.colorsCombinations[i + 1] = QtGui.QColor(colorNames[i]).getRgbF()
            self.dw2_cb.setItemData(i, pixmp, QtCore.Qt.DecorationRole)
        """


        ################### WIDGET 3 #############################
        self.dockWidget_3 = QtWidgets.QDockWidget(Main)
        self.dockWidget_3.setObjectName("dockWidget_3")
        #self.dockWidget_3.setWindowState(QtCore.Qt.WindowMinimized)
        #self.dockWidget_3.setFloating(True)
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.dockWidgetContents_3)
        self.formLayout_3.setObjectName("formLayout_3")
        self.progressBarSaving = QtWidgets.QProgressBar(self.dockWidgetContents_3)
        self.progressBarSaving.setProperty("value", 24)

        #self.progressBarSaving.setWindowState(QtCore.Qt.WindowMinimized)
        self.progressBarSaving.setObjectName("progressBar")

        #self.progressBarSaving.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        #self.dockWidget_3.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.progressBarSaving)
        self.dockWidget_3.setWidget(self.dockWidgetContents_3)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockWidget_3)
        self.dockWidget_3.setVisible(False)



        ################### WIDGET 4 #############################
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


        ################### WIDGET 5 (track based) #############################
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

        ################### WIDGET Color Intensity #############################

        #self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.dw2_cb)
        self.label_assigned_rad_circle = QtWidgets.QLabel(self.Settings_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_assigned_rad_circle.sizePolicy().hasHeightForWidth())
        self.label_assigned_rad_circle.setSizePolicy(sizePolicy)
        self.label_assigned_rad_circle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_assigned_rad_circle.setObjectName("label_assigned_rad_circle")
        self.gridLayout_settings.addWidget(self.label_assigned_rad_circle, 0, 0, 1, 1)

        #self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_assigned_rad_circle)
        self.label_rad_circle = QtWidgets.QLabel(self.Settings_widget)
        self.label_rad_circle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rad_circle.setObjectName("label_rad_circle")
        self.gridLayout_settings.addWidget(self.label_rad_circle, 1, 0, 1, 1)
        #self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.SpanningRole, self.label_rad_circle)
        self.scrol_rad_circle = QtWidgets.QScrollBar(self.Settings_widget)
        self.scrol_rad_circle.setOrientation(QtCore.Qt.Horizontal)
        #self.scrol_rad_circle.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.scrol_rad_circle.setObjectName("scrol_rad_circle")
        self.scrol_rad_circle.setRange(50,1000)
        self.scrol_rad_circle.setSingleStep(1)
        #self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.SpanningRole, self.scrol_rad_circle)
        self.gridLayout_settings.addWidget(self.scrol_rad_circle, 2, 0, 1, 1)


        self.label_tol_rad_circle = QtWidgets.QLabel(self.Settings_widget)
        self.label_tol_rad_circle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tol_rad_circle.setObjectName("label_tol_rad_circle")
        self.gridLayout_settings.addWidget(self.label_tol_rad_circle, 3, 0, 1, 1)
        #self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.SpanningRole, self.label_tol_rad_circle)

        self.scrol_tol_rad_circle = QtWidgets.QScrollBar(self.Settings_widget)
        self.scrol_tol_rad_circle.setOrientation(QtCore.Qt.Horizontal)
        #self.scrol_rad_circle.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.scrol_tol_rad_circle.setObjectName("scrol_rad_circle")
        self.scrol_tol_rad_circle.setRange(0,10)
        self.scrol_tol_rad_circle.setSingleStep(1)
        #self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.SpanningRole, self.scrol_tol_rad_circle)
        self.gridLayout_settings.addWidget(self.scrol_tol_rad_circle, 4, 0, 1, 1)


        self.label_assigned_tol_rad_circle = QtWidgets.QLabel(self.Settings_widget)
        self.label_assigned_tol_rad_circle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_assigned_tol_rad_circle.setObjectName("label_assigned_tol_rad_circle")
        self.gridLayout_settings.addWidget(self.label_assigned_tol_rad_circle, 5, 0, 1, 1)


        self.dw2_l1 = QtWidgets.QFrame(self.Settings_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dw2_l1.sizePolicy().hasHeightForWidth())
        self.dw2_l1.setSizePolicy(sizePolicy)
        self.dw2_l1.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))
        self.dw2_l1.setFrameShape(QtWidgets.QFrame.HLine)
        self.dw2_l1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.dw2_l1.setObjectName("dw2_l1")
        #self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.dw2_l1)
        self.gridLayout_settings.addWidget(self.dw2_l1, 6, 0, 1, 1)
        self.dw2_l2 = QtWidgets.QFrame(self.Settings_widget)
        self.dw2_l2.setMinimumSize(QtCore.QSize(self.width()//8, self.height() // 2))
        self.dw2_l2.setFrameShape(QtWidgets.QFrame.HLine)
        self.dw2_l2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.dw2_l2.setObjectName("dw2_l2")
        self.gridLayout_settings.addWidget(self.dw2_l2, 7, 0, 1, 1)
        #self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.SpanningRole, self.dw2_l2)



        self.scrol_rad_circle.valueChanged.connect(self.label_rad_circle.setNum)
        self.scrol_tol_rad_circle.valueChanged.connect(self.label_tol_rad_circle.setNum)

        #self.dwIntensitylS1.valueChanged.connect(self.dwIntensitylb1.setNum)

        #self.formLayout_4.addWidget(self.line4_11, 29, 0, 1, 1)
        #self.formLayout_intensity.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.lineInten_11)
        #self.formLayout_intensity.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dwIntensitylS1)

        #self.formLayout_intensity.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.dwIntensitylb1)
        #self.formLayout_intensity.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.dwintensity_flb1)

        #self.dockWidget_colorintensity.setWidget(self.dockWidgetContents_intensity)
        #Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_colorintensity)
        #self.dockWidget_colorintensity.setVisible(True)




        ############################# WIDGET TABLE ######################################

        self.dock_widget_table = QtWidgets.QDockWidget(Main)
        self.dock_widget_table.setObjectName("dock_widget_table")
        self.dock_widget_table.setVisible(False)

        dock_widget_content_table = QtWidgets.QWidget()

        dock_widget_content_table.setObjectName("dock_widget_content_table")
        gridLayout_6 = QtWidgets.QGridLayout(dock_widget_content_table)
        gridLayout_6.setObjectName("gridLayout_6")
        #self.table_widget = QtWidgets.QTableWidget(dock_widget_content_table)
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
        gridLayout_6.addWidget(self.table_widget, 0, 1, 1, 1)
        splitter_2 = QtWidgets.QSplitter(dock_widget_content_table)
        splitter_2.setOrientation(QtCore.Qt.Horizontal)

        self.table_update = QtWidgets.QPushButton(splitter_2)
        self.table_update.setObjectName("table_update")
        self.table_link = QtWidgets.QPushButton(splitter_2)
        self.table_link.setObjectName("table_link")
        self.table_link.setCheckable(True)


        gridLayout_6.addWidget(splitter_2, 1, 1, 1, 1)
        self.dock_widget_table.setWidget(dock_widget_content_table)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock_widget_table)



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

        ############################# WIDGET TABLE Measure ######################################

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


        gridLayout_6.addWidget(splitter_2, 1, 1, 1, 1)

        self.dock_widget_measure.setWidget(dock_widget_content_table)
        self.main_toolbox.addItem(self.dock_widget_measure, "")
        #Main.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock_widget_measure)

        _translate = QtCore.QCoreApplication.translate
        self.dock_widget_measure.setWindowTitle(_translate("Main", "Table Measure"))


        __sortingEnabled = self.table_widget.isSortingEnabled()
        self.table_widget_measure.setSortingEnabled(True)
        self.table_widget_measure.setSortingEnabled(__sortingEnabled)

        ################################

        ################# WIDGET_IMAGE############
        #########
        ################ Widget MRI COLORS ####################################

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

        self.main_toolbox.addItem(self.page1_images, "")
        #self.toolbox_imageConf.addItem(self.page1_images, "")

        #self.main_toolbox.addItem(self.dockWidget_2, "")
        self.main_toolbox.addItem(self.Settings_widget, "")

        self.dockSegmentationIntensity.setWidget(self.content_segInt)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockSegmentationIntensity)
        self.dockSegmentationIntensity.setVisible(True)


        ##################################


        # Extra
        self.dockImageEnh.setWindowTitle(_translate("Main", "Image Enhancement"))
        #self.dockImageConf.setWindowTitle(_translate("Main", "Settings"))


        self.dockWidget_3.setWindowTitle(_translate("Main", "Progress Bar"))
        self.dockWidget_4.setWindowTitle(_translate("Main", "Thresholding net"))
        self.dockWidget_5.setWindowTitle(_translate("Main", "Tracking Distance"))
        self.dockSegmentationIntensity.setWindowTitle(_translate("Main", "Color intensity"))
        self.lb_t1_4.setText(_translate("Main", "0"))
        self.lb_t1_3.setText(_translate("Main", "0"))
        self.lb_ft1_1.setText(_translate("Main", "Brightness"))
        self.dw5_flb1.setText(_translate("Main", "Track Distance"))
        self.dw5_flb2.setText(_translate("Main", "Track Width"))
        self.lb_t1_2.setText(_translate("Main", "0"))
        self.lb_ft1_2.setText(_translate("Main", "Contrast"))
        self.lb_ft1_3.setText(_translate("Main", "BandPass R1"))
        self.lb_ft1_4.setText(_translate("Main", "Sobel"))
        self.lb_ft1_5.setText(_translate("Main", "Rotate"))

        self.lb_t1_1.setText(_translate("Main", "0"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.ImageEnh_view1), _translate("Main", "View 1"))

        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.dock_widget_measure), _translate("Main", "Tables"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page1_color), _translate("Main", "Color"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page1_images), _translate("Main", "Images"))

        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.Settings_widget), _translate("Main", "Settings"))
        self.lb_ft2_1.setText(_translate("Main", "Brightness"))
        self.lb_t2_1.setText(_translate("Main", "0"))
        self.lb_ft2_2.setText(_translate("Main", "Contrast"))
        self.lb_t2_2.setText(_translate("Main", "0"))
        self.lb_ft2_3.setText(_translate("Main", "BandPass R1"))
        self.lb_t2_3.setText(_translate("Main", "0"))
        self.lb_ft2_4.setText(_translate("Main", "Sobel"))
        self.lb_ft2_5.setText(_translate("Main", "Rotate"))
        self.lb_t2_4.setText(_translate("Main", "0"))
        self.page2_s2c.setText(_translate("Main", "Sagittal2Coronal"))
        self.page1_s2c.setText(_translate("Main", "Sagittal2Coronal"))
        #self.page1_rot_cor.setText(_translate("Main", "Coronal Rotation"))
        #self.page2_rot_cor.setText(_translate("Main", "Coronal Rotation"))
        #self.colorize.setText(_translate("Main", "Colorize"))
        #self.colorize_MRI.setText(_translate("Main", "Colorize"))
        self.main_toolbox.setItemText(self.main_toolbox.indexOf(self.page1_mri), _translate("Main", "View 2"))

        self.toggle1_1.setText(_translate("Main", "Toggle"))
        self.lb_ft1_6.setText(_translate("Main", "Hamming"))
        self.lb_ft2_6.setText(_translate("Main", "Hamming"))


        self.lb_ft1_7.setText(_translate("Main", "BandPass R2"))
        self.lb_t1_7.setText(_translate("Main", "0"))
        self.lb_ft2_7.setText(_translate("Main", "BandPass R2"))
        self.lb_t2_7.setText(_translate("Main", "0"))


        ################# WIDGET 2 ############################

        self.page1_rot_cor.setItemText(0, _translate("Main", "Coronal"))
        self.page1_rot_cor.setItemText(1, _translate("Main", "Sagittal"))
        self.page1_rot_cor.setItemText(2, _translate("Main", "Axial"))

        self.page2_rot_cor.setItemText(0, _translate("Main", "Coronal"))
        self.page2_rot_cor.setItemText(1, _translate("Main", "Sagittal"))
        self.page2_rot_cor.setItemText(2, _translate("Main", "Axial"))


        self.label_assigned_rad_circle.setText(_translate("Main", "Effect strength"))
        self.label_assigned_tol_rad_circle.setText(_translate("Main", "Tolerance AutoSeg"))
        self.label_rad_circle.setText(_translate("Main", "0"))


    def ShowContextMenu_table1(self, pos):
        """
        Context Menu of the segmentation table
        :param pos:
        :return:
        """
        index = self.table_widget_measure.indexAt(pos)
        if not index.isValid():
            return
        from PyQt5.QtWidgets import QMenu, QAction
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
            from melage.utils.utils import export_tables
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
            from melage.widgets.helpers.fileDialog_widget import QFileDialogPreview
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


        from PyQt5.QtWidgets import QMenu, QAction
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
                from PyQt5.QtWidgets import QFileDialog
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
            #if not (hasattr(self, 'readImECO') and hasattr(self, 'readImMRI')):

            ind_avail = [9876]
            try:
                ind_avail += list(np.unique(self.readImECO.npSeg))
            except:
                pass
            try:
                ind_avail += list(np.unique(self.readImMRI.npSeg))
            except:
                pass
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
                        widget.paint(self.readImECO.npSeg,
                                           self.readImECO.npImage, None)
                    except:
                        pass
                elif num in [23]:
                    try:
                        widget.paint(self.readImMRI.npSeg,
                          self.readImMRI.npImage, None)
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

    def reset_page1_eco(self):
        self.hs_t1_1.setValue(0)
        self.hs_t1_2.setValue(0)
        self.hs_t1_3.setValue(0)
        self.hs_t1_4.setValue(0)
        self.hs_t1_5.setValue(0)
        self.hs_t1_7.setValue(0)
        self.page1_s2c.setChecked(False)
        self.toggle1_1.setChecked(False)

    def reset_page1_mri(self):
        self.hs_t2_1.setValue(0)
        self.hs_t2_2.setValue(0)
        self.hs_t2_3.setValue(0)
        self.hs_t2_4.setValue(0)
        self.hs_t2_5.setValue(0)
        self.hs_t2_7.setValue(0)
        self.page2_s2c.setChecked(False)
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
