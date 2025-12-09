__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from melage.rendering.DisplayIm import GLWidget
from melage.rendering.glScientific import glScientific
class openglWidgets():
    """
    Maing OPENGL WIDGETS
    """
    def __init__(self):
        pass



    def create_mutual_view(self, colorsCombinations):
        self.mutulaViewTab = QtWidgets.QWidget()
        self.mutulaViewTab.setObjectName("tab1")
        self.gridLayout = QtWidgets.QGridLayout(self.mutulaViewTab)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setSpacing(0)

        #self.gridLayout.setColumnStretch(0, 1)
        #self.gridLayout.setColumnStretch(1, 1)
        #self.gridLayout.setColumnStretch(2, 1)
        #self.gridLayout.setSpacing(3)

        self.openGLWidget_1 = GLWidget(colorsCombinations, self.mutulaViewTab,imdata = None,
                                       currentWidnowName = 'coronal', type='eco',id=1
                                       )
        self.openGLWidget_1.setObjectName("openGLWidget_1")
        self.openGLWidget_1.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_1.setEnabled(True)
        self.openGLWidget_1.setVisible(False)

        from melage.dialogs.helpers import custom_qscrollbar
        self.horizontalSlider_1 = custom_qscrollbar(self.mutulaViewTab, id=1)
        self.horizontalSlider_1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_1.setObjectName("horizontalSlider_4")

        self.horizontalSlider_1.cut_limit.connect(
            lambda A: self._cutIM(A)
        )

        self.label_1 = QtWidgets.QLabel(self.mutulaViewTab)
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName("label_1")

        self.openGLWidget_4 = GLWidget(colorsCombinations, self.mutulaViewTab,imdata = None,
                                       currentWidnowName = 'coronal', type='mri',id=4
                                       )
        self.openGLWidget_4.setObjectName("openGLWidget_3")
        self.openGLWidget_4.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_4.setEnabled(True)
        self.openGLWidget_4.setVisible(False)

        self.horizontalSlider_4 = custom_qscrollbar(self.mutulaViewTab, id=4)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")

        self.label_4 = QtWidgets.QLabel(self.mutulaViewTab)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")

        ############################### MIDDDLE #########################################################

        self.openGLWidget_2 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'sagittal', type='eco',id=2)
        self.openGLWidget_2.setObjectName("openGLWidget_2")
        self.openGLWidget_2.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_2.setEnabled(True)
        self.openGLWidget_2.setVisible(False)

        self.horizontalSlider_2 = custom_qscrollbar(self.mutulaViewTab, id=2)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.horizontalSlider_2.sizePolicy().hasHeightForWidth())
        #self.horizontalSlider_2.setSizePolicy(sizePolicy)
        #self.horizontalSlider_2.setMinimumSize(QtCore.QSize(100, 20))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")


        self.label_2 = QtWidgets.QLabel(self.mutulaViewTab)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.openGLWidget_5 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'sagittal', type='mri',id=5)
        self.openGLWidget_5.setObjectName("openGLWidget_5")
        self.openGLWidget_5.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_5.setEnabled(True)
        self.openGLWidget_5.setVisible(False)


        self.horizontalSlider_5 = custom_qscrollbar(self.mutulaViewTab, id=5)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")


        self.label_5 = QtWidgets.QLabel(self.mutulaViewTab)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")

        ############################### END #########################################################

        self.openGLWidget_3 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'axial', type='eco', id=3)
        self.openGLWidget_3.setObjectName("openGLWidget_5")
        self.openGLWidget_3.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_3.setEnabled(True)
        self.openGLWidget_3.setVisible(False)

        self.horizontalSlider_3 = custom_qscrollbar(self.mutulaViewTab, id=3)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.horizontalSlider_3.sizePolicy().hasHeightForWidth())
        #self.horizontalSlider_3.setSizePolicy(sizePolicy)
        #self.horizontalSlider_3.setMinimumSize(QtCore.QSize(200, 20))
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")


        self.label_3 = QtWidgets.QLabel(self.mutulaViewTab)
        #self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.openGLWidget_6 = GLWidget(colorsCombinations, self.mutulaViewTab, imdata=None, currentWidnowName='axial',
                                       type='mri', id=6)
        self.openGLWidget_6.setObjectName("openGLWidget_6")
        self.openGLWidget_6.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_6.setEnabled(True)
        self.openGLWidget_6.setVisible(False)


        self.horizontalSlider_6 = custom_qscrollbar(self.mutulaViewTab, id=6)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")


        self.label_6 = QtWidgets.QLabel(self.mutulaViewTab)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        #self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)


    def create_horizontal_mutualview(self):
        self.clear_layout(self.gridLayout)

        # Reset stretch factors
        for i in range(10):
            self.gridLayout.setRowStretch(i, 0)
        for j in range(10):
            self.gridLayout.setColumnStretch(j, 0)

        # Reset layout margins and spacing
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)

        # Reset size policies
        for w in [
            self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3,
            self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6
        ]:
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Restore horizontal orientation for sliders
        for slider in [
            self.horizontalSlider_1, self.horizontalSlider_2, self.horizontalSlider_3,
            self.horizontalSlider_4, self.horizontalSlider_5, self.horizontalSlider_6
        ]:
            slider.setOrientation(QtCore.Qt.Horizontal)

        self.horizontalSlider_1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)

        self.gridLayout.addWidget(self.openGLWidget_1, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_1, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.label_1, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_4, 5, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_4, 7, 0, 1, 1)
        self.gridLayout.addWidget(self.label_4, 9, 0, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_2, 4, 1, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_2, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_5, 5, 1, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_5, 7, 1, 1, 1)
        self.gridLayout.addWidget(self.label_5, 9, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_3, 4, 2, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_3, 2, 2, 1, 1)
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_6, 5, 2, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_6, 7, 2, 1, 1)
        self.gridLayout.addWidget(self.label_6, 9, 2, 1, 1)

    def create_vertical_mutualview(self):
        self.clear_layout(self.gridLayout)
        for w in [
            self.openGLWidget_1, self.openGLWidget_2, self.openGLWidget_3,
            self.openGLWidget_4, self.openGLWidget_5, self.openGLWidget_6
        ]:
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Set stretch factors
        for i in range(3):
            self.gridLayout.setRowStretch(i, 1)
        for j in range(6):
            self.gridLayout.setColumnStretch(j, 3 if j in [2, 3] else 1)

        # Optional: minimize margins
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.horizontalSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Vertical)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Vertical)

        self.gridLayout.addWidget(self.label_1, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_1, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_1, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_4, 0, 3, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_4, 0, 4, 1, 1)
        self.gridLayout.addWidget(self.label_4, 0, 5, 1, 1)
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_2, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_2, 1, 2, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_5, 1, 3, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_5, 1, 4, 1, 1)
        self.gridLayout.addWidget(self.label_5, 1, 5, 1, 1)
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_3, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_3, 2, 2, 1, 1)
        self.gridLayout.addWidget(self.openGLWidget_6, 2, 3, 1, 1)
        self.gridLayout.addWidget(self.horizontalSlider_6, 2, 4, 1, 1)
        self.gridLayout.addWidget(self.label_6, 2, 5, 1, 1)

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

        self.reservedTab = QtWidgets.QWidget()
        self.reservedTab.setObjectName("tab_2")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.reservedTab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.openGLWidget_10 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'axial', type='3d', id=10)
        self.openGLWidget_10.setObjectName("openGLWidget_10")
        self.openGLWidget_10.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_10.setEnabled(True)
        self.openGLWidget_10.setVisible(False)
        self.gridLayout_3.addWidget(self.openGLWidget_10, 3, 2, 1, 1)
        self.openGLWidget_9 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'axial', type='eco', id=9)
        self.openGLWidget_9.setObjectName("openGLWidget_9")
        self.openGLWidget_9.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_9.setEnabled(True)
        self.openGLWidget_9.setVisible(False)
        self.gridLayout_3.addWidget(self.openGLWidget_9, 3, 0, 1, 1)
        self.horizontalSlider_9 = QtWidgets.QSlider(self.reservedTab)
        self.horizontalSlider_9.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_9.setObjectName("horizontalSlider_9")
        self.gridLayout_3.addWidget(self.horizontalSlider_9, 4, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.reservedTab)

        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 0, 2, 1, 1)
        self.horizontalSlider_10 = QtWidgets.QSlider(self.reservedTab)
        self.horizontalSlider_10.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_10.setObjectName("horizontalSlider_10")
        self.gridLayout_3.addWidget(self.horizontalSlider_10, 4, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.reservedTab)

        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 0, 1, 1)
        self.openGLWidget_7 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'coronal', type='eco', id=7)
        self.openGLWidget_7.setObjectName("openGLWidget_7")
        self.openGLWidget_7.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_7.setEnabled(True)
        self.openGLWidget_7.setVisible(False)
        self.gridLayout_3.addWidget(self.openGLWidget_7, 2, 0, 1, 1)
        self.horizontalSlider_7 = QtWidgets.QSlider(self.reservedTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_7.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_7.setSizePolicy(sizePolicy)
        self.horizontalSlider_7.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_7.setObjectName("horizontalSlider_7")
        self.gridLayout_3.addWidget(self.horizontalSlider_7, 1, 0, 1, 1)
        self.openGLWidget_8 = GLWidget(colorsCombinations,self.mutulaViewTab,imdata = None, currentWidnowName = 'sagittal', type='eco', id=8)
        self.openGLWidget_8.setObjectName("openGLWidget_8")
        self.openGLWidget_8.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_8.setEnabled(True)
        self.openGLWidget_8.setVisible(False)
        self.gridLayout_3.addWidget(self.openGLWidget_8, 2, 2, 1, 1)
        self.horizontalSlider_8 = QtWidgets.QSlider(self.reservedTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider_8.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_8.setSizePolicy(sizePolicy)
        self.horizontalSlider_8.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_8.setObjectName("horizontalSlider_8")
        self.gridLayout_3.addWidget(self.horizontalSlider_8, 1, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.reservedTab)

        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 5, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.reservedTab)

        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 5, 2, 1, 1)
        self.tabWidget.addTab(self.reservedTab, "")

        ################# LEFT SIDE ######################################
        self.segmentationTab = QtWidgets.QWidget()
        self.segmentationTab.setObjectName("segmentationTab")
        self.gridLayout_seg = QtWidgets.QGridLayout(self.segmentationTab)
        self.gridLayout_seg.setObjectName("gridLayout_seg")
        self.splitter_main = QtWidgets.QSplitter(self.segmentationTab)
        self.splitter_main.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_main.setObjectName("splitter_main")
        self.splitter_left = QtWidgets.QSplitter(self.splitter_main)
        self.splitter_left.setOrientation(QtCore.Qt.Vertical)
        self.splitter_left.setObjectName("splitter_left")



        width_3d, height_3d = self.width()//3, int(self.height()/1.2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)

        self.openGLWidget_11 = GLWidget(colorsCombinations,self.splitter_left, imdata=None, currentWidnowName='coronal', type='eco', id=11)


        self.openGLWidget_11.setSizePolicy(sizePolicy)
        self.openGLWidget_11.setMaximumSize(QtCore.QSize(self.width(), self.height()))
        self.openGLWidget_11.setObjectName("openGLWidget_11")

        self.splitter_slider = QtWidgets.QSplitter(self.splitter_left)
        self.splitter_slider.setOrientation(QtCore.Qt.Vertical)
        self.splitter_slider.setObjectName("splitter_slider")
        self.label_11 = QtWidgets.QLabel(self.splitter_slider)

        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setMinimumSize(QtCore.QSize(10, 10))
        self.label_11.setMaximumSize(QtCore.QSize(self.width()-self.width()//4, self.height()//44))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")

        self.horizontalSlider_11 = QtWidgets.QScrollBar(self.splitter_slider)
        self.horizontalSlider_11.setSizePolicy(sizePolicy)
        self.horizontalSlider_11.setMaximumSize(QtCore.QSize(self.width(), self.height()//44))
        self.horizontalSlider_11.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_11.setObjectName("horizontalSlider_11")

        self.splitterRadioButton = QtWidgets.QSplitter(self.splitter_left)
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


        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        splitter_right = QtWidgets.QSplitter(self.splitter_main)
        splitter_right.setOrientation(QtCore.Qt.Vertical)
        splitter_right.setObjectName("splitter_right")
        self.openGLWidget_14 = glScientific(colorsCombinations,splitter_right, id=0)
        self.openGLWidget_14.initiate_actions()
        self.openGLWidget_14.setObjectName("openGLWidget_14")
        self.openGLWidget_14.setFocusPolicy(Qt.StrongFocus)
        self.openGLWidget_14.setSizePolicy(sizePolicy)
        self.openGLWidget_14.setFixedSize(QtCore.QSize(width_3d, self.height() - self.height()//3))
        self.openGLWidget_11.setFocusPolicy(Qt.StrongFocus)



        self.widget = QtWidgets.QWidget(splitter_right)
        self.widget.setObjectName("widget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.label_points = QtWidgets.QLabel(self.widget)
        self.label_points.setAlignment(QtCore.Qt.AlignCenter)
        self.label_points.setObjectName("label_points")
        #self.label_points.setSizePolicy(sizePolicy)
        #self.label_points.setFixedSize(QtCore.QSize(width_3d, 100))
        txt = 'Sagittal:' + '0' + ', Coronal: ' + '0' + ', Axial: ' + '0'
        self.label_points.setText(txt)
        self.verticalLayout.addWidget(self.label_points)
        spacerItem = QtWidgets.QSpacerItem(14, 118, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_seg.addWidget(self.splitter_main, 0, 0, 1, 1)

        self.tabWidget.addTab(self.segmentationTab, "")


        ########################################## MRI TAB #########################################
        ################# LEFT SIDE ######################################
        self.MRISegTab = QtWidgets.QWidget()
        self.MRISegTab.setObjectName("segmentationTab")
        self.gridLayout_seg_2 = QtWidgets.QGridLayout(self.MRISegTab)
        self.gridLayout_seg_2.setObjectName("gridLayout_seg")
        splitter_main = QtWidgets.QSplitter(self.MRISegTab)
        splitter_main.setOrientation(QtCore.Qt.Horizontal)
        splitter_main.setObjectName("splitter_main")
        splitter_left = QtWidgets.QSplitter(splitter_main)
        splitter_left.setOrientation(QtCore.Qt.Vertical)
        splitter_left.setObjectName("splitter_left")

        width_3d, height_3d = self.width() // 3, int(self.height() // 1.2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)

        self.openGLWidget_12 = GLWidget(colorsCombinations, splitter_left, imdata=None, currentWidnowName='coronal',
                                        type='mri', id=12)

        self.openGLWidget_12.setSizePolicy(sizePolicy)
        self.openGLWidget_12.setMaximumSize(QtCore.QSize(self.width(), self.height()))
        self.openGLWidget_12.setObjectName("openGLWidget_11")
        self.openGLWidget_12.setFocusPolicy(Qt.StrongFocus)

        splitter_slider = QtWidgets.QSplitter(splitter_left)
        splitter_slider.setOrientation(QtCore.Qt.Vertical)
        splitter_slider.setObjectName("splitter_slider")
        self.label_12 = QtWidgets.QLabel(splitter_slider)

        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(10, 10))
        self.label_12.setMaximumSize(QtCore.QSize(self.width() - self.width() // 4, self.height() // 44))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_11")

        self.horizontalSlider_12 = QtWidgets.QScrollBar(splitter_slider)
        self.horizontalSlider_12.setSizePolicy(sizePolicy)
        self.horizontalSlider_12.setMaximumSize(QtCore.QSize(self.width(), self.height() // 44))
        self.horizontalSlider_12.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_12.setObjectName("horizontalSlider_11")

        splitterRadioButton = QtWidgets.QSplitter(splitter_left)
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
        splitter_right = QtWidgets.QSplitter(splitter_main)
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

        self.label_points_2 = QtWidgets.QLabel(self.widget_2)
        self.label_points_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_points_2.setObjectName("label_points")
        # self.label_points_2.setSizePolicy(sizePolicy)
        # self.label_points_2.setFixedSize(QtCore.QSize(width_3d, 100))
        txt = 'Sagittal:' + '0' + ', Coronal: ' + '0' + ', Axial: ' + '0'
        self.label_points_2.setText(txt)
        self.verticalLayout_2.addWidget(self.label_points_2)
        spacerItem = QtWidgets.QSpacerItem(14, 118, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.gridLayout_seg_2.addWidget(splitter_main, 0, 0, 1, 1)

        self.tabWidget.addTab(self.MRISegTab, "")





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





        #self.tabWidget.addTab(self.tab, "")
        #self.gridLayout_main.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider_1.valueChanged.connect(self.label_1.setNum)
        self.horizontalSlider_2.valueChanged.connect(self.label_2.setNum)
        self.horizontalSlider_3.valueChanged.connect(self.label_3.setNum)
        self.horizontalSlider_4.valueChanged.connect(self.label_4.setNum)
        self.horizontalSlider_5.valueChanged.connect(self.label_5.setNum)
        self.horizontalSlider_6.valueChanged.connect(self.label_6.setNum)
        self.horizontalSlider_7.valueChanged.connect(self.label_7.setNum)
        self.horizontalSlider_8.valueChanged.connect(self.label_8.setNum)
        self.horizontalSlider_9.valueChanged.connect(self.label_9.setNum)
        self.horizontalSlider_10.valueChanged.connect(self.label_10.setNum)
        self.horizontalSlider_11.valueChanged.connect(self.label_11.setNum)
        self.horizontalSlider_12.valueChanged.connect(self.label_12.setNum)
        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)


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
