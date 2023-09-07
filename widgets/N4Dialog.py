__AUTHOR__ = 'Bahram Jafrasteh'


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal


class Worker(QtCore.QObject):

    finished = pyqtSignal()
    def __init__(self, id: int):
        super().__init__()
        self.__id = id
        self.param_set = False
        self.inputimage = None
        self.out = None
    #@QtCore.pyqtSlot()
    def run(self):
        if self.inputimage is None or not self.param_set:
            return
        from utils.utils import N4_bias_correction
        #self.progressBar.setVisible(True)
        #self.progressBar.setEnabled(True)
        #self.setEnabled(True)
        #self.progressBar.setValue(10)


        out = N4_bias_correction(self.inputimage, use_otsu=self._outsu,
                                 shrinkFactor=self._shrinkfactor, numberFittingLevels=self._filtering_level,
                                 max_iter=self._max_iter)

        ##self.pushButton_2.setVisible(True)
        ##self.progressBar.setValue(100)
        ##self.progressBar.setVisible(False)
        self.inputimage = None
        self.out = out
        self.param_set = False
        self.finished.emit()
        #self.close()

    def set_params(self, inpim, params, affine):
        self.param_set = True
        [self._max_iter, self._shrinkfactor, self._filtering_level, self._outsu] = params
        from utils.utils import make_image_using_affine
        self.inputimage = make_image_using_affine(inpim, affine)


class N4Dialog(QtWidgets.QDialog):
    """
    N4 bias filed correction dialogue
    """
    buttonpressed = pyqtSignal(object)
    buttonpressed2 = pyqtSignal(object)
    closeSig = pyqtSignal()
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.inputimage = None


        self.setupUi()

    def setupUi(self):
        Form = self.window()
        Form.setObjectName("N4")
        Form.resize(759, 140)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(0,0,0,0)
        self.grid_main.setObjectName("gridLayout")
        self.widget = QtWidgets.QWidget()
        #self.widget.setGeometry(QtCore.QRect(43, 20, 644, 89))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(10, 10, 10,10)
        self.gridLayout.setObjectName("gridLayout")
        self.checkBox = QtWidgets.QCheckBox(self.widget)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 0, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_filterlevel = QtWidgets.QLabel(self.splitter)
        self.label_filterlevel.setAlignment(QtCore.Qt.AlignCenter)
        self.label_filterlevel.setObjectName("label_filterlevel")
        self.gridLayout.addWidget(self.splitter, 0, 1, 1, 1)
        self.comboBox_filterlevel = QtWidgets.QComboBox(self.widget)
        self.comboBox_filterlevel.setObjectName("comboBox_filterlevel")

        self.gridLayout.addWidget(self.comboBox_filterlevel, 0, 2, 1, 1)
        self.label_shrink = QtWidgets.QLabel(self.widget)
        self.label_shrink.setAlignment(QtCore.Qt.AlignCenter)
        self.label_shrink.setObjectName("label_shrink")
        self.gridLayout.addWidget(self.label_shrink, 0, 3, 1, 1)
        self.combobox_shrink = QtWidgets.QComboBox(self.widget)
        self.combobox_shrink.setObjectName("combobox_shrink")

        self.gridLayout.addWidget(self.combobox_shrink, 0, 4, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 5, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setAutoFillBackground(False)
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setDragEnabled(False)
        self.lineEdit.setReadOnly(False)
        self.lineEdit.setPlaceholderText("")
        self.lineEdit.setCursorMoveStyle(QtCore.Qt.LogicalMoveStyle)
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 6, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton_2, 1, 6, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton, 2, 6, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.widget)
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 6)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 2, 1, 4)
        self.comboBox_image = QtWidgets.QComboBox(self.widget)
        self.comboBox_image.setObjectName("comboBox_image")
        self.comboBox_image.addItem("")
        self.comboBox_image.addItem("")
        self.gridLayout.addWidget(self.comboBox_image, 2, 0, 1, 2)

        #self.label_timer = QtWidgets.QLabel(self.widget)
        #self.label_timer.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_timer.setObjectName("label")
        #self.gridLayout.addWidget(self.label_timer, 1, 0, 1, 1)
        self.grid_main.addWidget(self.widget)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)



        for i in range(12):
            self.comboBox_filterlevel.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self.comboBox_filterlevel.addItem("  {}  ".format(i+1))

        for i in range(4):
            self.combobox_shrink.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self.combobox_shrink.addItem("    {}    ".format(i+1))

        self.pushButton.clicked.connect(self.accepted_emit)


        self.pushButton_2.clicked.connect(self.back_to_orig)
        self.lineEdit.setValidator(QtGui.QIntValidator())
        self.lineEdit.setMaxLength(2)
        self.progressBar.setVisible(False)
        self.comboBox_filterlevel.setCurrentIndex(4)
        self.lineEdit.setText('5')
        self.pushButton_2.setVisible(False)
        self.checkBox.setCheckState(Qt.Checked)




    def timerTimeout(self):
        self.time_spent += 1
        if self.time_spent == 0:
            self.time_spent = 50
        self.label_timer.setText(str(self.time_spent))



    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(N4Dialog, self).closeEvent(a0)
    def back_to_orig(self):
        botton = self.comboBox_image.currentIndex()
        self.buttonpressed2.emit(botton)
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "N4 Bias Field Correction"))
        self.checkBox.setText(_translate("Form", "Otsu"))

        for i in range(12):
            if i <4:
                self.combobox_shrink.setItemText(i, _translate("Form", "{}".format(i + 1)))
            self.comboBox_filterlevel.setItemText(i, _translate("Form", "{}".format(i + 1)))

        #self.comboBox.setCurrentIndex(1)
        self.comboBox_filterlevel.setCurrentIndex(2)
        self.checkBox.setCheckState(Qt.Checked)
        self.label_filterlevel.setText(_translate("Form", "Fitting Level"))
        self.label_5.setText(_translate("Form", "Max Iterations"))
        self.pushButton_2.setText(_translate("Form", "Original"))
        self.label_shrink.setText(_translate("Form", "ShrinkFactor"))
        self.comboBox_image.setItemText(0, _translate("Form", "        Top Image        "))
        self.comboBox_image.setItemText(1, _translate("Form", "      Bottom Image      "))
        #self.label_timer.setText(_translate("Form", "00:00"))
        self.pushButton.setText(_translate("Form", "Apply"))
        self.pushButton.setVisible(True)
    def message(self):

        self.MessageBox.setText('Please fill with numeric (!=0) and text')
        self.MessageBox.setWindowTitle('Warning')
        self.MessageBox.show()

    def accepted_emit(self):
        self.lineEdit.text()
        num = self.lineEdit.text()
        if num == '' or float(num) == 0:
            self.message()
            return False
        self._outsu = False
        if self.checkBox.checkState()==Qt.Checked:
            self._outsu = True
        self._max_iter = int(float(self.lineEdit.text()))
        self._shrinkfactor = int(float(self.combobox_shrink.currentText()))
        self._filtering_level = int(float(self.comboBox_filterlevel.currentText()))
        botton = self.comboBox_image.currentIndex()
        self.params = [self._max_iter, self._shrinkfactor, self._filtering_level, self._outsu]



        self.buttonpressed.emit(botton)



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = N4Dialog()
    window.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    run()