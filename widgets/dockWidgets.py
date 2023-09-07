__AUTHOR__ = 'Bahram Jafrasteh'


import sys
sys.path.append("../")
from PyQt5 import QtWidgets, QtCore, QtGui

from qtwidgets import AnimatedToggle
from utils.utils import generate_color_scheme_info
from PyQt5 import Qt
import numpy as np
import os


from utils.utils import read_txt_color, set_new_color_scheme, addTreeRoot, update_color_scheme, addLastColor, update_image_sch
from utils.source_folder import source_folder


class dockWidgets():
    """
    This class has been implemented for dock widgets in MELAGE
    """
    def __init__(self):
        pass

    def createDockWidget(self, Main):
        """
        Creating main attributes for the main widgets
        :param Main:
        :return:
        """
        self.dockWidget = QtWidgets.QDockWidget(Main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget.sizePolicy().hasHeightForWidth())
        self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidget.setMinimumSize(QtCore.QSize(200, 167))
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.toolBox = QtWidgets.QToolBox(self.dockWidgetContents)
        self.toolBox.setObjectName("toolBox")
        self.toolBox.setMinimumSize(self.width() // 7, self.height()//2)




        #########
        ################ Widget MRI COLORS ####################################

        self.page1_color = QtWidgets.QWidget()

        self.page1_color.setGeometry(QtCore.QRect(0, 0, 182, self.height()//2))
        self.page1_color.setObjectName("page")
        self.gridLayout_color = QtWidgets.QVBoxLayout(self.page1_color)
        self.gridLayout_color.setObjectName("gridLayout_7")





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



        self.gridLayout_color.addWidget(self.line_text)
        self.gridLayout_color.addWidget(self.tree_colors)


        # signals
        self.tree_colors.doubleClicked.connect(self._double_clicked)
        self.line_text.textChanged.connect(self.searchTreeChanged)
        self.tree_colors.itemDelegate().closeEditor.connect(self._on_closeEditor)
        self.tree_colors.customContextMenuRequested.connect(self.ShowContextMenu_tree)
        # init
        model = self.tree_colors.model().sourceModel()
        model.setColumnCount(2)
        model.setHorizontalHeaderLabels(['Index', 'Name'])
        self.tree_colors.sortByColumn(0, QtCore.Qt.AscendingOrder)





        #self.gridLayout_color.addWidget(self.tree_colors, 1, 0, 1, 1)

        self.toolBox.addItem(self.page1_color, "")
        ###############












        self.page1_eco = QtWidgets.QWidget()
        self.page1_eco.setGeometry(QtCore.QRect(0, 0, 182, self.height()//2))

        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.page1_eco.sizePolicy().hasHeightForWidth())
        #self.page1_eco.setSizePolicy(sizePolicy)
        self.page1_eco.setObjectName("page")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.page1_eco)
        self.gridLayout_7.setObjectName("gridLayout_7")

        self.line_5 = QtWidgets.QFrame(self.page1_eco)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout_7.addWidget(self.line_5, 0, 0, 1, 1)



        self.lb_ft1_1 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_1.sizePolicy().hasHeightForWidth())
        self.lb_ft1_1.setSizePolicy(sizePolicy)
        self.lb_ft1_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lb_ft1_1.setObjectName("lb_ft1_1")
        self.gridLayout_7.addWidget(self.lb_ft1_1, 1, 0, 1, 1)


        self.lb_t1_1 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_1.sizePolicy().hasHeightForWidth())
        self.lb_t1_1.setSizePolicy(sizePolicy)
        self.lb_t1_1.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_1.setObjectName("lb_t1_1")
        self.gridLayout_7.addWidget(self.lb_t1_1, 2, 0, 1, 1)


        self.hs_t1_1 = QtWidgets.QScrollBar(self.page1_eco)
        self.hs_t1_1.setMinimum(-100)
        self.hs_t1_1.setMaximum(100)
        self.hs_t1_1.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_1.setObjectName("hs_t1_1")
        self.gridLayout_7.addWidget(self.hs_t1_1, 3, 0, 1, 1)



        self.line_6 = QtWidgets.QFrame(self.page1_eco)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout_7.addWidget(self.line_6, 4, 0, 1, 1)


        self.lb_ft1_2 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_2.sizePolicy().hasHeightForWidth())
        self.lb_ft1_2.setSizePolicy(sizePolicy)
        self.lb_ft1_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lb_ft1_2.setObjectName("lb_ft1_2")
        self.gridLayout_7.addWidget(self.lb_ft1_2, 5, 0, 1, 1)



        self.lb_t1_2 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_2.sizePolicy().hasHeightForWidth())
        self.lb_t1_2.setSizePolicy(sizePolicy)
        self.lb_t1_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_2.setObjectName("lb_t1_2")
        self.gridLayout_7.addWidget(self.lb_t1_2, 6, 0, 1, 1)


        self.hs_t1_2 = QtWidgets.QScrollBar(self.page1_eco)
        self.hs_t1_2.setMaximum(100)
        self.hs_t1_2.setMinimum(-100)
        self.hs_t1_2.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_2.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_2.setObjectName("hs_t1_2")
        self.gridLayout_7.addWidget(self.hs_t1_2, 7, 0, 1, 1)



        self.line_7 = QtWidgets.QFrame(self.page1_eco)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout_7.addWidget(self.line_7, 8, 0, 1, 1)


        self.lb_ft1_3 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_3.sizePolicy().hasHeightForWidth())
        self.lb_ft1_3.setSizePolicy(sizePolicy)
        self.lb_ft1_3.setObjectName("lb_ft1_3")
        self.gridLayout_7.addWidget(self.lb_ft1_3, 9, 0, 1, 1)



        self.lb_t1_3 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_3.sizePolicy().hasHeightForWidth())
        self.lb_t1_3.setSizePolicy(sizePolicy)
        self.lb_t1_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_3.setObjectName("lb_t1_3")
        self.gridLayout_7.addWidget(self.lb_t1_3, 10, 0, 1, 1)



        self.hs_t1_3 = QtWidgets.QScrollBar(self.page1_eco)
        self.hs_t1_3.setMaximum(100)
        self.hs_t1_3.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_3.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_3.setObjectName("hs_t1_3")
        self.gridLayout_7.addWidget(self.hs_t1_3, 11, 0, 1, 1)

        self.line_8 = QtWidgets.QFrame(self.page1_eco)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.gridLayout_7.addWidget(self.line_8, 12, 0, 1, 1)





        self.lb_ft1_7 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_7.sizePolicy().hasHeightForWidth())
        self.lb_ft1_7.setSizePolicy(sizePolicy)
        self.lb_ft1_7.setObjectName("lb_ft1_7")
        self.gridLayout_7.addWidget(self.lb_ft1_7, 13, 0, 1, 1)


        self.lb_t1_7 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_7.sizePolicy().hasHeightForWidth())
        self.lb_t1_7.setSizePolicy(sizePolicy)
        self.lb_t1_7.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_7.setObjectName("lb_t1_7")
        self.gridLayout_7.addWidget(self.lb_t1_7, 14, 0, 1, 1)



        self.hs_t1_7 = QtWidgets.QScrollBar(self.page1_eco)
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
        self.gridLayout_7.addWidget(self.hs_t1_7, 15, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.page1_eco)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_7.addWidget(self.line_11, 16, 0, 1, 1)



        self.lb_ft1_4 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_ft1_4.sizePolicy().hasHeightForWidth())
        self.lb_ft1_4.setSizePolicy(sizePolicy)
        self.lb_ft1_4.setObjectName("lb_ft1_4")
        self.gridLayout_7.addWidget(self.lb_ft1_4, 17, 0, 1, 1)


        self.lb_t1_4 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_4.sizePolicy().hasHeightForWidth())
        self.lb_t1_4.setSizePolicy(sizePolicy)
        self.lb_t1_4.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_4.setObjectName("lb_t1_4")
        self.gridLayout_7.addWidget(self.lb_t1_4, 18, 0, 1, 1)



        self.hs_t1_4 = QtWidgets.QScrollBar(self.page1_eco)
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
        self.gridLayout_7.addWidget(self.hs_t1_4, 19, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.page1_eco)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_7.addWidget(self.line_11, 20, 0, 1, 1)


        self.lb_ft1_5 = QtWidgets.QLabel(self.page1_eco)
        self.lb_ft1_5.setObjectName("lb_ft1_5")
        self.gridLayout_7.addWidget(self.lb_ft1_5, 21, 0, 1, 1)


        ################### WIDGET COMBOX ROTATION ###########################

        self.page1_rot_cor = QtWidgets.QComboBox(self.page1_eco)
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


        self.gridLayout_7.addWidget(self.page1_rot_cor, 22, 0, 1, 1)


        self.lb_t1_5 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_5.sizePolicy().hasHeightForWidth())
        self.lb_t1_5.setSizePolicy(sizePolicy)
        self.lb_t1_5.setObjectName("lb_t1_5")
        self.lb_t1_5.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout_7.addWidget(self.lb_t1_5, 23, 0, 1, 1)


        #self.hs_t1_5 = QtWidgets.QScrollBar(self.page1_eco)
        self.hs_t1_5 = QtWidgets.QScrollBar(self.page1_eco)
        #self.hs_t1_5.setPageStep(0.5)
        self.hs_t1_5.setMinimum(-50)
        self.hs_t1_5.setMaximum(50)
        #self.hs_t1_5.setTickInterval(1)
        self.hs_t1_5.setOrientation(QtCore.Qt.Horizontal)
        #self.hs_t1_5.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_5.setObjectName("hs_t1_5")
        self.gridLayout_7.addWidget(self.hs_t1_5, 24, 0, 1, 1)



        #self.page1_rot_cor = QtWidgets.QCheckBox(self.page1_eco)
        #self.page1_rot_cor.setObjectName("page1_rot_cor")






        self.line_ = QtWidgets.QFrame(self.page1_eco)
        self.line_.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_.setObjectName("line_")
        self.gridLayout_7.addWidget(self.line_, 25, 0, 1, 1)




        self.page1_s2c = QtWidgets.QCheckBox(self.page1_eco)
        self.page1_s2c.setObjectName("page1_s2c")
        self.gridLayout_7.addWidget(self.page1_s2c, 26, 0, 1, 1)





        self.line_10 = QtWidgets.QFrame(self.page1_eco)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.gridLayout_7.addWidget(self.line_10, 27, 0, 1, 1)

        self.lb_ft1_6 = QtWidgets.QLabel(self.page1_eco)
        self.lb_ft1_6.setObjectName("lb_ft1_6")
        self.gridLayout_7.addWidget(self.lb_ft1_6, 28, 0, 1, 1)

        self.toggle1_1 = AnimatedToggle(
            checked_color="#FFB000",
            pulse_checked_color="#44FFB000"
        )

        self.toggle1_1.setObjectName('toggle1_1')

        self.gridLayout_7.addWidget(self.toggle1_1, 29, 0, 1, 1)


        self.line_11 = QtWidgets.QFrame(self.page1_eco)
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_10")
        self.gridLayout_7.addWidget(self.line_11, 30, 0, 1, 1)


        self.colorize = QtWidgets.QCheckBox(self.page1_eco)
        self.colorize.setObjectName("Colorize")
        self.gridLayout_7.addWidget(self.colorize, 31, 0, 1, 1)

        self.lb_t1_8 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_5.sizePolicy().hasHeightForWidth())
        self.lb_t1_8.setSizePolicy(sizePolicy)
        self.lb_t1_8.setObjectName("lb_t1_8")
        self.lb_t1_8.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t1_8.setText('2')
        self.gridLayout_7.addWidget(self.lb_t1_8, 32, 0, 1, 1)

        # self.hs_t1_5 = QtWidgets.QScrollBar(self.page1_eco)
        self.hs_t1_8 = QtWidgets.QScrollBar(self.page1_eco)
        #self.hs_t1_8.setPageStep(0.5)
        self.hs_t1_8.setMinimum(2)
        self.hs_t1_8.setMaximum(50)
        self.hs_t1_8.setValue(0)

        # self.hs_t1_5.setTickInterval(1)
        self.hs_t1_8.setOrientation(QtCore.Qt.Horizontal)
        # self.hs_t1_5.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t1_8.setObjectName("hs_t1_8")
        self.gridLayout_7.addWidget(self.hs_t1_8, 33, 0, 1, 1)



        self.toolBox.addItem(self.page1_eco, "")


        ########### PAGE 1 MRI ################

        self.page1_mri = QtWidgets.QWidget()
        #palet = QtGui.QPalette()
        #palet.setColor(QtGui.QPalette.Window, QtCore.Qt.blue)
        #self.page1_mri.setStyleSheet("background-color: black")
        #self.page1_mri.setAutoFillBackground(True);
        #self.page1_mri.setPalette(palet)
        self.page1_mri.setGeometry(QtCore.QRect(0, 0, 182, self.height()//2))
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.page1_mri.sizePolicy().hasHeightForWidth())
        #self.page1_mri.setSizePolicy(sizePolicy)
        self.page1_mri.setObjectName("page")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.page1_mri)
        self.gridLayout_8.setObjectName("gridLayout_7")

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

        self.colorize_MRI = QtWidgets.QCheckBox(self.page1_mri)
        self.colorize_MRI.setObjectName("Colorize")
        self.gridLayout_8.addWidget(self.colorize_MRI, 31, 0, 1, 1)

        self.lb_t2_8 = QtWidgets.QLabel(self.page1_eco)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lb_t1_5.sizePolicy().hasHeightForWidth())
        self.lb_t2_8.setSizePolicy(sizePolicy)
        self.lb_t2_8.setObjectName("lb_t2_8")
        self.lb_t2_8.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_t2_8.setText('2')
        self.gridLayout_8.addWidget(self.lb_t2_8, 32, 0, 1, 1)
        # self.hs_t1_5 = QtWidgets.QScrollBar(self.page1_eco)
        self.hs_t2_8 = QtWidgets.QScrollBar(self.page1_eco)
        #self.hs_t2_8.setPageStep(0.5)
        self.hs_t2_8.setMinimum(2)
        self.hs_t2_8.setMaximum(50)
        self.hs_t2_8.setValue(0)

        # self.hs_t1_5.setTickInterval(1)
        self.hs_t2_8.setOrientation(QtCore.Qt.Horizontal)
        # self.hs_t1_5.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.hs_t2_8.setObjectName("hs_t2_8")
        self.gridLayout_8.addWidget(self.hs_t2_8, 33, 0, 1, 1)

        self.toolBox.addItem(self.page1_mri, "")





        self.gridLayout_5.addWidget(self.toolBox, 0, 0, 1, 1)
        self.dockWidget.setWidget(self.dockWidgetContents)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)

        self.dockWidget.setVisible(True)
        self.page1_mri.setVisible(False)
        self.page1_eco.setVisible(False)

        self.hs_t1_1.valueChanged.connect(self.lb_t1_1.setNum)
        self.hs_t1_2.valueChanged.connect(self.lb_t1_2.setNum)
        self.hs_t1_3.valueChanged.connect(self.lb_t1_3.setNum)
        self.hs_t1_4.valueChanged.connect(self.lb_t1_4.setNum)
        self.hs_t1_5.valueChanged.connect(self.lb_t1_5.setNum)
        self.hs_t1_7.valueChanged.connect(self.lb_t1_7.setNum)
        self.hs_t1_8.valueChanged.connect(self.lb_t1_8.setNum)


        self.hs_t2_1.valueChanged.connect(self.lb_t2_1.setNum)
        self.hs_t2_2.valueChanged.connect(self.lb_t2_2.setNum)
        self.hs_t2_3.valueChanged.connect(self.lb_t2_3.setNum)
        self.hs_t2_4.valueChanged.connect(self.lb_t2_4.setNum)
        self.hs_t2_5.valueChanged.connect(self.lb_t2_5.setNum)
        self.hs_t2_8.valueChanged.connect(self.lb_t2_8.setNum)
        self.hs_t2_7.valueChanged.connect(self.lb_t2_7.setNum)

        ################ NEW WIDGET ####################################

        self.dockWidget_2 = QtWidgets.QDockWidget(Main)
        self.dockWidget_2.setObjectName("dockWidget_2")

        self.dockWidgetContents_2 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page1_eco.sizePolicy().hasHeightForWidth())
        self.dockWidget_2.setSizePolicy(sizePolicy)
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.dockWidgetContents_2)
        self.formLayout_2.setObjectName("formLayout_2")





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
        #color_name, color_index_rgb, _ = read_txt_color(source_folder+"/color/LUT_albert.txt", from_one=True)
        #set_new_color_scheme(self, color_name, color_index_rgb)
        self.color_name, self.color_index_rgb, _ = read_txt_color(source_folder+"/color/Simple.txt", mode= '', from_one=True)
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


        self.tree_colors.model().sourceModel().itemChanged.connect(self.changeColorPen)

        #####


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



        ################### WIDGET Color Intensity #############################
        self.dockWidget_colorintensity = QtWidgets.QDockWidget(Main)
        self.dockWidget_colorintensity.setObjectName("dockWidget_colorintensity")
        self.dockWidgetContents_intensity = QtWidgets.QWidget()
        self.dockWidgetContents_intensity.setObjectName("dockWidgetContents_intensity")
        self.formLayout_intensity = QtWidgets.QFormLayout(self.dockWidgetContents_intensity)
        self.formLayout_intensity.setObjectName("formLayout_intensity")


        self.dwintensity_flb1 = QtWidgets.QLabel(self.dockWidgetContents_intensity)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dwintensity_flb1.sizePolicy().hasHeightForWidth())
        self.dwintensity_flb1.setSizePolicy(sizePolicy)
        self.dwintensity_flb1.setAlignment(QtCore.Qt.AlignCenter)
        self.dwintensity_flb1.setObjectName("dwintensity_flb1")

        self.dwIntensitylb1 = QtWidgets.QLabel(self.dockWidgetContents_intensity)
        self.dwIntensitylb1.setAlignment(QtCore.Qt.AlignCenter)
        self.dwIntensitylb1.setObjectName("dwIntensitylb1")
        self.dwIntensitylb1.setText('100')

        self.dwIntensitylS1 = QtWidgets.QScrollBar(self.dockWidgetContents_intensity)
        self.dwIntensitylS1.setOrientation(QtCore.Qt.Horizontal)
        #self.dwIntensitylS1.setTickPosition(QtWidgets.QScrollBar.TicksBothSides)
        self.dwIntensitylS1.setObjectName("dwIntensitylS1")
        self.dwIntensitylS1.setRange(0,100)
        self.dwIntensitylS1.setValue(100)


        self.dwIntensitylS1.setSingleStep(1)


        self.lineInten_11 = QtWidgets.QFrame(self.dockWidgetContents_intensity)
        self.lineInten_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.lineInten_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lineInten_11.setObjectName("lineInten_11")
        #self.formLayout_4.addWidget(self.line4_11, 29, 0, 1, 1)
        self.formLayout_intensity.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.lineInten_11)
        self.formLayout_intensity.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dwIntensitylS1)

        self.formLayout_intensity.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.dwIntensitylb1)
        self.formLayout_intensity.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.dwintensity_flb1)

        self.dockWidget_colorintensity.setWidget(self.dockWidgetContents_intensity)
        Main.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget_colorintensity)
        self.dockWidget_colorintensity.setVisible(True)
        self.dwIntensitylS1.valueChanged.connect(self.dwIntensitylb1.setNum)



        ############################# WIDGET TABLE Measure ######################################

        self.dock_widget_measure = QtWidgets.QDockWidget(Main)
        self.dock_widget_measure.setObjectName("dock_widget_table")
        self.dock_widget_measure.setVisible(False)

        dock_widget_content_table = QtWidgets.QWidget()

        dock_widget_content_table.setObjectName("dock_widget_content_table")
        gridLayout_7 = QtWidgets.QGridLayout(dock_widget_content_table)
        gridLayout_7.setObjectName("gridLayout_7")
        # self.table_widget = QtWidgets.QTableWidget(dock_widget_content_table)
        self.table_widget_measure = QtWidgets.QTableWidget(dock_widget_content_table)

        self.table_widget_measure.setRowCount(5)
        self.table_widget_measure.setColumnCount(8)
        self.table_widget_measure.setEditTriggers(Qt.QAbstractItemView.NoEditTriggers)
        self.table_widget_measure.setHorizontalHeaderLabels(['Description','ImType', 'Measure1', 'Measure2', 'Slice', 'WindowName', 'CenterXY', 'FileName'])

        self.table_widget_measure.setObjectName("table_widget_measure")
        self.table_widget_measure.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table_widget_measure.customContextMenuRequested.connect(self.ShowContextMenu_table1)

        gridLayout_7.addWidget(self.table_widget_measure, 0, 1, 1, 1)
        splitter_2 = QtWidgets.QSplitter(dock_widget_content_table)
        splitter_2.setOrientation(QtCore.Qt.Horizontal)



        self.dock_widget_measure.setWidget(dock_widget_content_table)
        self.toolBox.addItem(self.dock_widget_measure, "")
        #Main.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock_widget_measure)

        _translate = QtCore.QCoreApplication.translate
        self.dock_widget_measure.setWindowTitle(_translate("Main", "Table Measure"))



        self.table_widget_measure.setSortingEnabled(True)


        ################################

        ################# WIDGET_IMAGE############
        #########
        ################ Widget MRI COLORS ####################################

        self.page1_images = QtWidgets.QWidget()

        self.page1_images.setGeometry(QtCore.QRect(0, 0, 182, self.height()//2))
        self.page1_images.setObjectName("page")
        self.gridLayout_images = QtWidgets.QVBoxLayout(self.page1_images)
        self.gridLayout_images.setObjectName("gridLayout_7")





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


        self.gridLayout_images.addWidget(self.line_text_image)
        self.gridLayout_images.addWidget(self.tree_images)


        # signals
        self.tree_images.doubleClicked.connect(self._double_clicked)
        self.line_text_image.textChanged.connect(self.searchTreeChanged)
        self.tree_images.itemDelegate().closeEditor.connect(self._on_closeEditor)
        self.tree_images.customContextMenuRequested.connect(self.ShowContextMenu_images)
        # init
        model = self.tree_images.model().sourceModel()
        model.setColumnCount(2)
        model.setHorizontalHeaderLabels(['Index', 'Name'])
        self.tree_images.sortByColumn(1, QtCore.Qt.AscendingOrder)
        self.imported_images = []


        self.tree_images.model().sourceModel().itemChanged.connect(self.changeImage)

        #self.gridLayout_color.addWidget(self.tree_colors, 1, 0, 1, 1)

        self.toolBox.addItem(self.page1_images, "")

        ##################################


        # Extra
        self.dockWidget.setWindowTitle(_translate("Main", "Image Enhancement"))

        self.dockWidget_2.setWindowTitle(_translate("Main", "Markers"))
        self.dockWidget_3.setWindowTitle(_translate("Main", "Progress Bar"))

        self.dockWidget_colorintensity.setWindowTitle(_translate("Main", "Segmentation intensity"))
        self.lb_t1_4.setText(_translate("Main", "0"))
        self.lb_t1_3.setText(_translate("Main", "0"))
        self.lb_ft1_1.setText(_translate("Main", "Brightness"))
        self.lb_t1_2.setText(_translate("Main", "0"))
        self.lb_ft1_2.setText(_translate("Main", "Contrast"))
        self.lb_ft1_3.setText(_translate("Main", "BandPass R1"))
        self.lb_ft1_4.setText(_translate("Main", "Sobel"))
        self.lb_ft1_5.setText(_translate("Main", "Rotate"))

        self.lb_t1_1.setText(_translate("Main", "0"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page1_eco), _translate("Main", "UltraSound"))

        self.toolBox.setItemText(self.toolBox.indexOf(self.dock_widget_measure), _translate("Main", "Tables"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page1_color), _translate("Main", "Color"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page1_images), _translate("Main", "Images"))
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
        self.colorize.setText(_translate("Main", "Colorize"))
        self.colorize_MRI.setText(_translate("Main", "Colorize"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page1_mri), _translate("Main", "MRI"))

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
            from utils.utils import export_tables
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
            from widgets.fileDialog_widget import QFileDialogPreview
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
        import_image_action = QtWidgets.QAction(self)
        import_image_action.setIconText('Images')
        menu_import.addAction(import_image_action)
        import_seg_action = QtWidgets.QAction(self)
        import_seg_action.setIconText('Segmentations')
        menu_import.addAction(import_seg_action)

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

        elif action==import_image_action:
            #if action==import_action:
            if self._basefileSave is None or self._basefileSave=='':
                return
            [fileObj, index] = dialog()
            filen = fileObj[0][0]
            if filen=='':
                return
            if index<2:
                color = [1,0,0]
            else:
                color = [1, 1, 0]
            update_image_sch(self, [fileObj, index], color =color)
        elif action==import_seg_action:
            if self._basefileSave is None or self._basefileSave=='':
                return
            [fileObj, index] = dialog()
            filen = fileObj[0][0]
            if filen=='':
                return
            index += 3
            if index<5:
                color = [0.5,1,0]
            else:
                color = [0.5, 0.5, 1]
            update_image_sch(self, [fileObj, index], color = color)

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
        menu_color = QtWidgets.QMenu(menu)
        menu_color.setObjectName('Color')
        menu_color.setWindowIconText('CCC')
        _translate = QtCore.QCoreApplication.translate
        menu_color.setTitle(_translate('Main', "Scheme"))
        action_albert = QtWidgets.QAction(self)
        action_albert.setIconText('Albert')
        menu_color.addAction(action_albert)

        action_tissue = QtWidgets.QAction(self)
        action_tissue.setIconText('Tissue')
        menu_color.addAction(action_tissue)

        action_tissue12 = QtWidgets.QAction(self)
        action_tissue12.setIconText('Tissue12')
        menu_color.addAction(action_tissue12)

        action_mcrib = QtWidgets.QAction(self)
        action_mcrib.setIconText('MCrib')
        menu_color.addAction(action_mcrib)

        action_pediatrics1 = QtWidgets.QAction(self)
        action_pediatrics1.setIconText('Pediatric1')
        menu_color.addAction(action_pediatrics1)

        action_pediatrics2 = QtWidgets.QAction(self)
        action_pediatrics2.setIconText('Pediatric2')
        menu_color.addAction(action_pediatrics2)

        action_simple = QtWidgets.QAction(self)
        action_simple.setIconText('Simple')
        menu_color.addAction(action_simple)


        menu.addMenu(menu_color)
        #menu_color.addAction(import_action)

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
            if action==import_action:
                from PyQt5.QtWidgets import QFileDialog
                filters = "TXT(*.txt)"
                opts = QFileDialog.DontUseNativeDialog
                fileObj = QFileDialog.getOpenFileName(self, "Open COLOR File", self.source_dir, filters, options=opts)
                filen = fileObj[0]
                if filen=='':
                    return
            elif action == action_simple:
                filen = source_folder+"/color/Simple.txt"
            elif action == action_pediatrics1:
                filen = source_folder+"/color/pediatric1.txt"
            elif action == action_pediatrics2:
                filen = source_folder+"/color/pediatric2.txt"
            elif action == action_albert:
                filen = source_folder + "/color/albert_LUT.txt"
            elif action == action_mcrib:
                filen = source_folder + "/color/mcrib_LUT.txt"
            elif action == action_tissue:
                filen = source_folder + "/color/Tissue.txt"
            elif action == action_tissue12:
                filen = source_folder + "/color/Tissue12.txt"
            else:
                return
            possible_color_name, possible_color_index_rgb, _ = read_txt_color(filen, from_one=False, mode='None')
            #if not (hasattr(self, 'readImECO') and hasattr(self, 'readImMRI')):
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
                possible_color_index_rgb, possible_color_name)
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
        regExp = QtCore.QRegExp(self.line_text.text(), QtCore.Qt.CaseInsensitive, QtCore.QRegExp.FixedString)

        self.tags_model.text = self.line_text.text().lower()
        self.tags_model.setFilterRegExp(regExp)

        if len(self.line_text.text()) >= 1 and self.tags_model.rowCount() > 0:
            self.tree_colors.expandAll()
        else:
            self.tree_colors.collapseAll()



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
