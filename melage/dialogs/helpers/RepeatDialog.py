__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal
class repeatN(QtWidgets.QDialog):
    """
    Repeating widgets
    """
    numberN = pyqtSignal(object)
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowTitle("Number of repetition")
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
        self.label_4 = QtWidgets.QLabel(widget)
        self.label_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_4.setAutoFillBackground(False)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        horizontalLayout.addWidget(self.label_4)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(widget)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        horizontalLayout.addWidget(self.doubleSpinBox)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem2)
        verticalLayout.addLayout(horizontalLayout)
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
        self.doubleSpinBox.setMinimum(-1000)
        self.doubleSpinBox.setValue(2)

        self.retranslateUi(Info)
        QtCore.QMetaObject.connectSlotsByName(Info)

        self.retranslateUi(Info)
        self.OK.clicked.connect(self.accept)
        QtCore.QMetaObject.connectSlotsByName(Info)



    def retranslateUi(self, Info):
        _translate = QtCore.QCoreApplication.translate
        Info.setWindowTitle(_translate("Info", "Number of repetition"))
        _translate = QtCore.QCoreApplication.translate
        self.label_4.setText(_translate("Info", "Number of repetition"))
        self.OK.setText(_translate("Info", "OK"))
