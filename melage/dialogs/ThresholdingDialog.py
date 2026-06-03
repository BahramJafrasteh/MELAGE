__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
import random
import matplotlib
from melage.dialogs import FigureCanvas
matplotlib.use('Agg')


class ThresholdingImage(QDialog):
    """
    Image thresholding
    """
    closeSig = pyqtSignal()
    applySig = pyqtSignal()
    repltSig = pyqtSignal()
    histeqSig = pyqtSignal(int)

    def __init__(self, parent=None):
        super(ThresholdingImage, self).__init__(parent)

        Form = self.window()
        Form.setObjectName("N4")
        Form.resize(600, 600)
        #self.tabWidget = QtWidgets.QTabWidget(Form)
        #self.tabWidget.setGeometry(QtCore.QRect(50, 50, 500, 500))
        #self.tabWidget.setObjectName("tabWidget")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                           QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())

        # self.tabWidget.setObjectName("tabWidget")
        self.widget = QtWidgets.QWidget(self)
        #self.widget.setGeometry(QtCore.QRect(20, 20, 400, 400))
        self.widget.setObjectName("widget")
        self.gridLayout_main = QtWidgets.QVBoxLayout(self)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setParent(self)

        self.gridLayout.addWidget(self.canvas, 5, 5, 2, 2)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.gridLayout2 = QtWidgets.QGridLayout()

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.applySig)
        # self.gridLayout.addWidget(self.button, 7, 0, 1, 1)
        self._number_class = QtWidgets.QComboBox(self.widget)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self._number_class.setMaximumSize(50,50)
        sizePolicy.setHeightForWidth(self._number_class.sizePolicy().hasWidthForHeight())
        self._number_class.setSizePolicy(sizePolicy)
        self._number_class.setObjectName("+")
        self.gridLayout2.addWidget(self._number_class, 0, 2, 1, 2)
        self._label_numberClasses = QtWidgets.QLabel(self.widget)
        self._label_numberClasses.setObjectName("_label_numberClasses")

        self._histeq = QtWidgets.QCheckBox(self.widget)
        self._histeq.setObjectName("check box")
        self._histeq.setChecked(False)
        self._histeq.clicked.connect(self.histEQ)

        self.gridLayout2.addWidget(self.button, 0, 4, 1, 1)
        self.gridLayout2.addWidget(self._histeq, 0,3, 1, 1)
        self.gridLayout2.addWidget(self._label_numberClasses, 0,1, 1, 1)

        # self.toolbar2 = NavigationToolbar(self.canvas2, self)
        # self.canvas2.setParent(self)
        for i in range(9):
            self._number_class.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._number_class.addItem(" {} ".format(i+1))
        _translate = QtCore.QCoreApplication.translate
        self._label_numberClasses.setText(_translate("Form", "No. classes"))
        self._histeq.setText(_translate("Form", "Hist EQ."))
        self.button.setText(_translate("Form", "Apply"))
        self.comboBox_image = QtWidgets.QComboBox(self.widget)
        self.comboBox_image.setObjectName("comboBox_image")
        self.comboBox_image.addItem("")
        self.comboBox_image.addItem("")
        self.gridLayout2.addWidget(self.comboBox_image,0, 0, 1, 1)
        Form.setWindowTitle(_translate("Form", "Image Thresholding"))
        self.comboBox_image.setItemText(0, _translate("Form", "        View 1        "))
        self.comboBox_image.setItemText(1, _translate("Form", "      View 2      "))
        #self._number_class.currentIndexChanged.connect(self.plot)

        self.gridLayout_main.addLayout(self.gridLayout, 0)
        self.gridLayout_main.addLayout(self.gridLayout2, 1)

        self.comboBox_image.currentIndexChanged.connect(self.repltSig)
        self.button.clicked.connect(self.applySig)
        self._number_class.currentIndexChanged.connect(self.repltSig)

        #self.plot([])

    def histEQ(self):
        if not hasattr(self, '_a'):
            return
        if self._histeq.isChecked():
            a = self._a
            from melage.utils.utils import histogram_equalization
            self.im_rec = histogram_equalization(a)
            self.plot(self.im_rec)
            self.histeqSig.emit(True)
        else:
            self.histeqSig.emit(0)



    def new_thresholds(self, a):

        self._a = a
        numc = self._number_class.currentIndex()+2
        from melage.utils.utils import Threshold_MultiOtsu
        thresholds = Threshold_MultiOtsu(a, numc)
        return thresholds


    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        if hasattr(self, '_a'):
            delattr(self,'_a')
        self._number_class.setCurrentIndex(0)
        self._histeq.setChecked(False)
        super(ThresholdingImage, self).closeEvent(a0)



    def emptyPlot(self):


        figure, canvas = self.figure, self.canvas

        ''' plot some random stuff '''
        # random data

        # instead of ax.hold(False)
        figure.clear()
        #import numpy as np
        #data = np.random.rand(40,40,40)*255
        # create an axis
        ax = figure.add_subplot(111)

        # ax.axis('off')
        #ax.grid('on')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.set_title('Image histogram', fontsize=6 * 1.7, weight='bold')
        ax.set_xlabel('Range', fontsize=6 * 1.5, weight='bold')

        ax.set_ylabel('Frequency', fontsize=6 * 1.5, weight='bold')
        #ax.set_xlim([0, a.max() + 1])
        figure.tight_layout(pad=2)

        # ax.legend(loc="upper right", prop={'size': 8 * 2})
        # refresh canvas
        canvas.draw()



    def plot(self, data):
        if data is None:
            return  # data = [random.random() for i in range(10)]

        figure, canvas = self.figure, self.canvas


        figure.clear()

        # create an axis
        ax = figure.add_subplot(111)

        a = data.flatten()
        a = a[a > 0]

        ax.hist(a, 25)
        self._currentThresholds = self.new_thresholds(data)
        yl = ax.get_ylim()[1]
        for el in self._currentThresholds:
            ax.plot([el,el], [0, yl], )
        # ax.axis('off')
        #ax.grid('on')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.set_title('Image histogram', fontsize=6 * 1.7, weight='bold')
        ax.set_xlabel('Range', fontsize=6 * 1.5, weight='bold')

        ax.set_ylabel('Frequency', fontsize=6 * 1.5, weight='bold')
        #ax.set_xlim([0, a.max() + 1])
        figure.tight_layout(pad=2)


        canvas.draw()


def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ThresholdingImage()
    window.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    run()