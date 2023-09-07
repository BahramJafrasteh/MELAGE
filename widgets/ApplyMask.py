__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal


class Worker(QtCore.QObject):
    """

    """
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


        out = N4_bias_correction(self.inputimage, use_otsu=self._outsu,
                                 shrinkFactor=self._selectedColor, numberFittingLevels=self._filtering_level,
                                 max_iter=self._max_iter)


        self.inputimage = None
        self.out = out
        self.param_set = False
        self.finished.emit()
        #self.close()

    def set_params(self, inpim, params):
        self.param_set = True
        [self._max_iter, self._selectedColor, self._filtering_level, self._outsu] = params
        self.inputimage = inpim


class Masking(QtWidgets.QDialog):
    """
    Apply segmentation mask to the image
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
        Form.resize(759, 140)
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

        self.mask_color = QtWidgets.QLabel(self.widget)
        self.mask_color.setAlignment(QtCore.Qt.AlignCenter)
        self.mask_color.setObjectName("mask_color")
        self.gridLayout.addWidget(self.mask_color, 0, 2, 1, 1)

        self.mask_keep_remove = QtWidgets.QComboBox(self.widget)
        self.mask_keep_remove.setObjectName("comboBox_image")
        self.mask_keep_remove.addItem("Keep")
        self.mask_keep_remove.addItem("Remove")
        self.gridLayout.addWidget(self.mask_keep_remove, 0, 1, 1, 1)

        self._combobox_colors = QtWidgets.QComboBox(self.widget)
        self._combobox_colors.setObjectName("_combobox_colors")

        self.gridLayout.addWidget(self._combobox_colors, 0,3, 1, 3)



        #self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        #self.pushButton_2.setObjectName("pushButton")
        #self.gridLayout.addWidget(self.pushButton_2, 0, 6, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton, 1, 5, 1, 3)
        self.progressBar = QtWidgets.QProgressBar(self.widget)
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 5)
        #spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.gridLayout.addItem(spacerItem, 0, 1, 1, 4)
        self.comboBox_image = QtWidgets.QComboBox(self.widget)
        self.comboBox_image.setObjectName("comboBox_image")
        self.comboBox_image.addItem("")
        self.comboBox_image.addItem("")
        self.gridLayout.addWidget(self.comboBox_image, 0, 0, 1, 1)

        #self.label_timer = QtWidgets.QLabel(self.widget)
        #self.label_timer.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_timer.setObjectName("label")
        #self.gridLayout.addWidget(self.label_timer, 1, 0, 1, 1)
        self.grid_main.addWidget(self.widget)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)





        for i in range(4):
            self._combobox_colors.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_colors.addItem("    fdfdffdff{}    ".format(i+1))

        self.pushButton.clicked.connect(self.accepted_emit)




        self.progressBar.setVisible(True)



    def setComboBoxColors(self, color_name):
        self._combobox_colors.clear()
        for colr in color_name:
            #self._combobox_colors.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_colors.addItem("   {}   ".format(colr))



    def timerTimeout(self):
        self.time_spent += 1
        if self.time_spent == 0:
            self.time_spent = 50
        self.label_timer.setText(str(self.time_spent))




    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Apply Mask"))









        self.mask_color.setText(_translate("Form", "Mask Color"))
        self.mask_keep_remove.setItemText(0, _translate("Form", "        Keep        "))
        self.mask_keep_remove.setItemText(1, _translate("Form", "        Remove        "))
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

        index_image = self.comboBox_image.currentIndex()

        index_keep_remove = self.mask_keep_remove.currentText().strip(' ')
        keep = False
        if index_keep_remove.lower()=='keep':
            keep = True
        index_color = self._combobox_colors.currentIndex()
        self.buttonpressed.emit([index_image, index_color, keep])

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(Masking, self).closeEvent(a0)

def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Masking()
    window.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    run()