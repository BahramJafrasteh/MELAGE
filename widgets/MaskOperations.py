__AUTHOR__ = 'Bahram Jafrasteh'


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal



class MaskOperations(QtWidgets.QDialog):
    """
    Image MASKING OPERATION
    """
    buttonpressed = pyqtSignal(object)
    closeSig = pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.inputimage = None

        self.MessageBox = QtWidgets.QMessageBox(self)
        self.setupUi()

    def setupUi(self):
        Form = self.window()
        Form.setObjectName("N4")
        Form.resize(759, 140)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(43, 20, 644, 89))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        #self.gridLayout.addWidget(self.splitter, 0, 1, 1, 1)

        self.mask_operation = QtWidgets.QComboBox(self.widget)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.mask_operation.setMaximumSize(50,50)
        sizePolicy.setHeightForWidth(self.mask_operation.sizePolicy().hasWidthForHeight())
        #self.mask_operation.setSizePolicy(sizePolicy)
        self.mask_operation.setObjectName("+")
        self.gridLayout.addWidget(self.mask_operation, 0, 2, 1, 1)
        self._combobox_colors = QtWidgets.QComboBox(self.widget)
        self._combobox_colors.setObjectName("_combobox_colors")

        self._combobox_colors2 = QtWidgets.QComboBox(self.widget)
        self._combobox_colors2.setObjectName("_combobox_colors")
        self.gridLayout.addWidget(self._combobox_colors2, 0, 0, 1, 2)
        self.gridLayout.addWidget(self._combobox_colors, 0,5, 1, 2)



        #self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        #self.pushButton_2.setObjectName("pushButton")
        #self.gridLayout.addWidget(self.pushButton_2, 0, 6, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton, 2, 7, 1, 1)

        #self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 5)
        #spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.gridLayout.addItem(spacerItem, 0, 1, 1, 4)
        self.comboBox_image = QtWidgets.QComboBox(self.widget)
        self.comboBox_image.setObjectName("comboBox_image")
        self.comboBox_image.addItem("")
        self.comboBox_image.addItem("")
        self.gridLayout.addWidget(self.comboBox_image, 1, 0, 1, 1)


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        for i, operation in enumerate(['+', '-']):
            self.mask_operation.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self.mask_operation.addItem(" {} ".format(operation))

        self.pushButton.clicked.connect(self.accepted_emit)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(MaskOperations, self).closeEvent(a0)

    def accept_emit(self):
        self.mask_operation.text()
        num = self.mask_operation.text()
        if num not in ['+', '-', '']:
            self.message()
            return False
        return True

    def setComboBoxColors(self, color_name):
        self._combobox_colors.clear()
        self._combobox_colors2.clear()
        for colr in color_name:
            #self._combobox_colors.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_colors.addItem("   {}   ".format(colr))
            self._combobox_colors2.addItem("   {}   ".format(colr))

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Mask Operations"))

        self.comboBox_image.setItemText(0, _translate("Form", "        Top Image        "))
        self.comboBox_image.setItemText(1, _translate("Form", "      Bottom Image      "))
        #self.label_timer.setText(_translate("Form", "00:00"))
        self.pushButton.setText(_translate("Form", "Apply"))
        self.pushButton.setVisible(True)

    def message(self):

        self.MessageBox.setText('Please fill with +,-')
        self.mask_operation.setText('+')
        self.MessageBox.setWindowTitle('Warning')
        self.MessageBox.show()

    def accepted_emit(self):

        #self._selectedColor = int(float(self._combobox_colors.currentText()))

        index_image = self.comboBox_image.currentIndex()

        index_color = self._combobox_colors.currentIndex()
        index_color2 = self._combobox_colors2.currentIndex()
        if index_color2==index_color:
            return
        operation = self.mask_operation.currentText().strip(' ')
        self.buttonpressed.emit([index_image, index_color2, index_color, operation])



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MaskOperations()
    window.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    run()