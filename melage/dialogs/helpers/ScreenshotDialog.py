__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5.QtWidgets import QWidget, QDialog, QFileDialog
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
class screenshot(QtWidgets.QDialog):
    """
    Screen shot widget
    """
    numberN = pyqtSignal(object)
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle("ScreenShot")
        self.setupUi()


    def setupUi(self):
        Info = self.window()
        Info.setObjectName("Repeat")
        Info.resize(400, 190)
        widget = QtWidgets.QWidget(Info)
        widget.setGeometry(QtCore.QRect(10, 0, 391, 171))
        widget.setObjectName("widget")
        verticalLayout = QtWidgets.QVBoxLayout(widget)
        verticalLayout.setContentsMargins(0, 0, 0, 0)
        verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        verticalLayout.addItem(spacerItem)
        horizontalLayout = QtWidgets.QHBoxLayout()
        horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem1)

        self.screencombo_data = QtWidgets.QComboBox(widget)
        cbstyle = """
            QComboBox QAbstractItemView {border: 1px solid grey;
            background: #03211c; 
            selection-background-color: #03211c;} 
            QComboBox {background: #03211c;margin-right: 1px;}
            QComboBox::drop-down {
        subcontrol-origin: margin;}
            """
        self.screencombo_data.setStyleSheet(cbstyle)
        self.screencombo_data.setObjectName("dw2_cb")
        self.screencombo_data.addItem("")
        self.screencombo_data.addItem("")
        self.screencombo_data.addItem("")



        #self.label_screenshot = QtWidgets.QLabel(widget)
        #self.label_screenshot.setLayoutDirection(QtCore.Qt.LeftToRight)
        #self.label_screenshot.setAutoFillBackground(False)
        #self.label_screenshot.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_screenshot.setObjectName("label_4")
        #horizontalLayout.addWidget(self.screencombo_data)
        verticalLayout.addWidget(self.screencombo_data)



        #self.doubleSpinBox = QtWidgets.QDoubleSpinBox(widget)
        #self.doubleSpinBox.setObjectName("doubleSpinBox")
        #horizontalLayout.addWidget(self.doubleSpinBox)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem2)
        verticalLayout.addLayout(horizontalLayout)

        self.screencombo_plane = QtWidgets.QComboBox(widget)
        cbstyle = """
            QComboBox QAbstractItemView {border: 1px solid grey;
            background: #03211c; 
            selection-background-color: #03211c;} 
            QComboBox {background: #03211c;margin-right: 1px;}
            QComboBox::drop-down {
        subcontrol-origin: margin;}
            """
        self.screencombo_plane.setStyleSheet(cbstyle)
        self.screencombo_plane.setObjectName("dw2_cb")
        self.screencombo_plane.addItem("")
        self.screencombo_plane.addItem("")
        self.screencombo_plane.addItem("")
        self.screencombo_data.currentIndexChanged.connect(self.changeItems)

        verticalLayout.addWidget(self.screencombo_plane)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        verticalLayout.addItem(spacerItem3)
        horizontalLayout_2 = QtWidgets.QHBoxLayout()
        horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout_2.addItem(spacerItem4)
        self.OK = QtWidgets.QPushButton(widget)
        self.OK.setObjectName("pushButton")
        horizontalLayout_2.addWidget(self.OK)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout_2.addItem(spacerItem5)
        verticalLayout.addLayout(horizontalLayout_2)
        #self.doubleSpinBox.setMinimum(2)
        #self.doubleSpinBox.setValue(2)

        self.retranslateUi(Info)
        QtCore.QMetaObject.connectSlotsByName(Info)

        self.retranslateUi(Info)
        self.OK.clicked.connect(self.accept)
        QtCore.QMetaObject.connectSlotsByName(Info)
        self.screenErrorMsg = self.screen_error_msgbox



    def changeItems(self):
        if self.screencombo_data.currentText().lower() == 'all':
            self.screencombo_plane.setVisible(False)
        else:
            self.screencombo_plane.setVisible(True)


    def retranslateUi(self, Info):
        _translate = QtCore.QCoreApplication.translate
        Info.setWindowTitle(_translate("Info", "Take a screenshot"))
        _translate = QtCore.QCoreApplication.translate
        #self.label_screenshot.setText(_translate("Info", "Number of repetition"))
        self.OK.setText(_translate("Info", "OK"))
        self.screencombo_data.setItemText(0, _translate("Main", "UltraSound"))
        self.screencombo_data.setItemText(1, _translate("Main", "MRI"))
        self.screencombo_data.setItemText(2, _translate("Main", "All"))

        self.screencombo_plane.setItemText(0, _translate("Main", "Coronal"))
        self.screencombo_plane.setItemText(1, _translate("Main", "Sagittal"))
        self.screencombo_plane.setItemText(2, _translate("Main", "Axial"))


    def screen_error_msgbox(self, text= None):
        if text is None:
            text = 'There is an error. Screen is not captured. Please check the content.'
        MessageBox = QtWidgets.QMessageBox(self)
        MessageBox.setText(text)
        MessageBox.setWindowTitle('Warning')
        MessageBox.show()