__AUTHOR__ = 'Bahram Jafrasteh'



from PyQt5 import QtWidgets, QtCore

class enhanceIm(QtWidgets.QMainWindow):
    """
    Attributes related to image enhancement
    """
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.setupUi()
    def setupUi(self):
        Form = self.window()
        Form.setObjectName("Form")
        Form.resize(550, 544)
        self.gridLayoutWidget = QtWidgets.QWidget(Form)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 30, 481, 451))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")


        self.verticalSlider_1 = QtWidgets.QSlider(self.gridLayoutWidget)
        self.verticalSlider_1.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_1.setObjectName("verticalSlider_6")

        self.verticalSlider_2 = QtWidgets.QSlider(self.gridLayoutWidget)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider")
        self.gridLayout.addWidget(self.verticalSlider_2, 0, 1, 1, 1)
        self.verticalSlider_3 = QtWidgets.QSlider(self.gridLayoutWidget)
        self.verticalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_3.setObjectName("verticalSlider_5")
        self.gridLayout.addWidget(self.verticalSlider_3, 1, 0, 1, 1)
        self.verticalSlider_4 = QtWidgets.QSlider(self.gridLayoutWidget)
        self.verticalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_4.setObjectName("verticalSlider_2")
        self.gridLayout.addWidget(self.verticalSlider_4, 1, 1, 1, 1)
        self.verticalSlider_1.setRange(-100, 100)
        self.verticalSlider_2.setRange(0, 100)
        self.verticalSlider_3.setRange(0, 100)
        self.verticalSlider_4.setRange(0, 100)



        self.gridLayout.addWidget(self.verticalSlider_1, 0, 0, 1, 1)
        self.radioButton = QtWidgets.QRadioButton(Form)
        self.radioButton.setGeometry(QtCore.QRect(310, 490, 121, 51))
        self.radioButton.setObjectName("radioButton")
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(90, 220, 85, 31))
        self.textEdit.setOverwriteMode(True)
        self.textEdit.setAcceptRichText(True)
        self.textEdit.setText("Brightness")
        self.textEdit.setVisible(True)

        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(Form)
        self.textEdit_2.setGeometry(QtCore.QRect(100, 450, 71, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.setText("Contrast")
        self.textEdit_2.setVisible(True)
        self.textEdit_3 = QtWidgets.QTextEdit(Form)
        self.textEdit_3.setGeometry(QtCore.QRect(370, 450, 91, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_3.setText("Sobel Threshold")
        self.textEdit_4 = QtWidgets.QTextEdit(Form)
        self.textEdit_4.setGeometry(QtCore.QRect(370, 220, 81, 31))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_4.setVisible(True)
        self.textEdit_4.setText("Threshold")
        self.radioButton.clicked.connect(self.verticalSlider_4.setVisible)
        self.radioButton.setText('Sobel')
        self.verticalSlider_4.setVisible(False)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.radioButton.setText(_translate("Form", "RadioButton"))
        self.textEdit.setPlaceholderText(_translate("Form", "Brightness"))
        self.textEdit_2.setPlaceholderText(_translate("Form", "Contrast"))
        self.textEdit_3.setPlaceholderText(_translate("Form", "Sobel filter"))
        self.textEdit_4.setPlaceholderText(_translate("Form", "Reserved"))


class MainWindow0(enhanceIm):
    def __init__(self, *args, obj = None, **kwargs):
        super(enhanceIm, self).__init__(*args, **kwargs)
        QWidget.__init__(self)
        self.setupUi()
        #QtCore.QTimer.singleShot(5000, self.showChildWindow)






if __name__ == '__main__':
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import QWidget
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow0()
    window.show()
    sys.exit(app.exec_())
