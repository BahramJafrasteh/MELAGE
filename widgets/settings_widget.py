__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal
class settingsBN(QtWidgets.QMainWindow):
    """
    Setting MELAGE
    """
    newConfig = pyqtSignal(object)
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.setupUi()


    def setupUi(self):
        Info = self.window()
        Info.setObjectName("Info")
        Info.resize(728, 569)
        widget = QtWidgets.QWidget(Info)
        widget.setGeometry(QtCore.QRect(10, 10, 676, 501))
        widget.setObjectName("widget")
        verticalLayout = QtWidgets.QVBoxLayout(widget)
        verticalLayout.setContentsMargins(0, 0, 0, 0)
        verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 88, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        verticalLayout.addItem(spacerItem)
        horizontalLayout_2 = QtWidgets.QHBoxLayout()
        horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(158, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout_2.addItem(spacerItem1)
        self.label_4 = QtWidgets.QLabel(widget)
        self.label_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_4.setAutoFillBackground(False)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        horizontalLayout_2.addWidget(self.label_4)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(widget)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.doubleSpinBox.setValue(10)
        horizontalLayout_2.addWidget(self.doubleSpinBox)
        self.label_5 = QtWidgets.QLabel(widget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        horizontalLayout_2.addWidget(self.label_5)
        spacerItem2 = QtWidgets.QSpacerItem(198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout_2.addItem(spacerItem2)
        verticalLayout.addLayout(horizontalLayout_2)
        spacerItem3 = QtWidgets.QSpacerItem(17, 338, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        verticalLayout.addItem(spacerItem3)
        horizontalLayout = QtWidgets.QHBoxLayout()
        horizontalLayout.setObjectName("horizontalLayout")
        spacerItem4 = QtWidgets.QSpacerItem(290, 22, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem4)
        self.OK = QtWidgets.QPushButton(widget)
        self.OK.setObjectName("OK")
        horizontalLayout.addWidget(self.OK)
        spacerItem5 = QtWidgets.QSpacerItem(290, 22, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem5)
        verticalLayout.addLayout(horizontalLayout)

        self.retranslateUi(Info)
        self.OK.clicked.connect(self.closeWindow)
        QtCore.QMetaObject.connectSlotsByName(Info)

    def closeWindow(self):
        txt = self.doubleSpinBox.value()
        self.newConfig.emit([float(txt)])
        self.close()

    def retranslateUi(self, Info):
        _translate = QtCore.QCoreApplication.translate
        Info.setWindowTitle(_translate("Info", "Settings"))
        self.label_4.setText(_translate("Info", "Save every "))
        self.label_5.setText(_translate("Info", "minutes"))
        self.OK.setText(_translate("Info", "OK"))
