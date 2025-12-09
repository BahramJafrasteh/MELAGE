__AUTHOR__ = 'Bahram Jafrasteh'


from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal


class NewDialog(QtWidgets.QDialog):
    """
    This is a dialogue to pick and select a color
    """
    ColorIndName = pyqtSignal(object)
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.setupUi()
    def setupUi(self):
        Dialog = self.window()
        Dialog.setObjectName("Dialog")
        Dialog.resize(381, 119)

        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setValidator(QtGui.QIntValidator())
        self.lineEdit.setMaxLength(7)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 2, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout.addItem(spacerItem, 2, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 3, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.lineEdit2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit2.setObjectName("lineEdit2")

        self.gridLayout.addWidget(self.lineEdit2, 1, 2, 1, 2)
        #self.buttonBox.clicked.connect(self.accepted_emit)
        self.buttonBox.accepted.connect(self.accepted_emit)
        self.buttonBox.rejected.connect(Dialog.reject)
        self.dialog = Dialog
        self.retranslateUi(Dialog)
        self.MessageBox = QtWidgets.QMessageBox(Dialog)


        #Dialog.accepted.connect(self.accepted_emit)
        #QtCore.QMetaObject.connectSlotsByName(Dialog)
    def message(self):

        self.MessageBox.setText('Please fill with numeric (!=0) and text')
        self.MessageBox.setWindowTitle('Warning')
        self.MessageBox.show()


    def accepted_emit(self):
        self.lineEdit.text()
        num = self.lineEdit.text()
        if num=='' or float(num)==0:
            self.message()
            return False
        txt = self.lineEdit2.text()
        if txt=='':
            self.message()
            return False
        self.ColorIndName.emit([num, txt])
        self.accept()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Index (numeric)"))
        self.label_2.setText(_translate("Dialog", "Name"))



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = NewDialog()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()