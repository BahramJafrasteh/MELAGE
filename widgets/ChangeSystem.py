__AUTHOR__ = 'Bahram Jafrasteh'


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal




class ChangeCoordSys(QtWidgets.QDialog):
    """
    This class has been implemented to change coordinate system of the current image
    """
    buttonpressed = pyqtSignal(object)
    closeSig = pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.inputimage = None


        self.setupUi()

    def setupUi(self):
        Form = self.window()
        Form.setObjectName("N4")
        #Form.resize(759, 140)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(0,0,0,0)
        self.grid_main.setObjectName("gridLayout")
        self.widget = QtWidgets.QWidget()
        #self.widget.setGeometry(QtCore.QRect(43, 20, 644, 89))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setObjectName("gridLayout")

        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        #self.gridLayout.addWidget(self.splitter, 0, 1, 1, 1)
        self._labelfrom = QtWidgets.QLabel(self.widget)
        self._labelfrom.setAlignment(QtCore.Qt.AlignCenter)
        self._labelfrom.setObjectName("From")
        self.gridLayout.addWidget(self._labelfrom, 0, 1, 1, 1)

        self.label_current = QtWidgets.QLabel(self.widget)
        self.label_current.setText('RAS')
        self.label_current.setStyleSheet('color: Green')
        self.gridLayout.addWidget(self.label_current, 0, 2, 1, 1)

        self._labelto = QtWidgets.QLabel(self.widget)
        self._labelto.setAlignment(QtCore.Qt.AlignCenter)
        self._labelto.setObjectName("To")
        self.gridLayout.addWidget(self._labelto, 0, 3, 1, 1)
        self._combobox_coords = QtWidgets.QComboBox(self.widget)
        self._combobox_coords.setObjectName("_combobox_coords")

        self.gridLayout.addWidget(self._combobox_coords, 0,4, 1, 2)


        #self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        #self.pushButton_2.setObjectName("pushButton")
        #self.gridLayout.addWidget(self.pushButton_2, 0, 6, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton, 1, 5, 1, 3)
        self.label_warning = QtWidgets.QLabel(self.widget)

        self.gridLayout.addWidget(self.label_warning, 1, 0, 1, 5)
        self.label_warning.setStyleSheet('color: Red')
        self.label_warning.setText('This options is for advanced users')

        #spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.gridLayout.addItem(spacerItem, 0, 1, 1, 4)
        self.comboBox_image = QtWidgets.QComboBox(self.widget)
        self.comboBox_image.setObjectName("comboBox_image")
        self.comboBox_image.addItem("")
        self.comboBox_image.addItem("")
        #self.comboBox_image.currentIndexChanged.connect()
        self.gridLayout.addWidget(self.comboBox_image, 0, 0, 1, 1)

        #self.label_timer = QtWidgets.QLabel(self.widget)
        #self.label_timer.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_timer.setObjectName("label")
        #self.gridLayout.addWidget(self.label_timer, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        (('L', 'R'), ('P', 'A'), ('I', 'S'))

        self.systems = ['RAS', 'RSA', 'RPI', 'RPS',
                   'RIP', 'RIA', 'RSP', 'RSA',
                    'LAS', 'LAI', 'LPI', 'LPS',
                   'LSA', 'LIA', 'LIP', 'LSP',
                   'PIL', 'PIR', 'PSL', 'PSR',
                   'PLI', 'PRI', 'PLS', 'PRS',
                   'AIL', 'AIR', 'ASL', 'ASR',
                   'ALI', 'ARI', 'ALS', 'ARS',
                   'IPL', 'IPR', 'IAL', 'IAR',
                   'ILP', 'IRP', 'ILA', 'IRA',
                   'SPL', 'SPR', 'SAL', 'SAR',
                   'SLP', 'SRP', 'SLA', 'SRA'
                   ]
        for i , ss in enumerate(self.systems):
            self._combobox_coords.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_coords.addItem(" "*9+ss+" "*9)
        self.grid_main.addWidget(self.widget)
        self.pushButton.clicked.connect(self.accepted_emit)




    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Change Image Coordinate System"))


        self._labelto.setText(_translate("Form", "To"))
        self._labelfrom.setText(_translate("Form", "From"))
        self.comboBox_image.setItemText(0, _translate("Form", "        Top Image        "))
        self.comboBox_image.setItemText(1, _translate("Form", "      Bottom Image      "))
        #self.label_timer.setText(_translate("Form", "00:00"))
        self.pushButton.setText(_translate("Form", "Apply"))
        self.pushButton.setVisible(True)
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(ChangeCoordSys, self).closeEvent(a0)

    def accepted_emit(self):


        index_image = self.comboBox_image.currentIndex()

        index_color = self._combobox_coords.currentIndex()
        CS = self.systems[index_color]
        self.buttonpressed.emit([index_image, CS])



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ChangeCoordSys()
    window.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    run()