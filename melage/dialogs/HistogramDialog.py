__AUTHOR__ = 'Bahram Jafrasteh'

import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvasQT
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
import random
import matplotlib
matplotlib.use('Agg')

from matplotlib.figure import Figure

class FigureCanvas(FigureCanvasQTAgg):
    """
    Class implemented to show figures
    """
    def __init__(self, parent=None, width=5, height=4, dpi=1000):
        super(FigureCanvas, self).__init__(parent)
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self._canvas = FigureCanvasQT(parent)

    def mousePressEvent(self, a0) -> None:
        x, y = a0.x(), a0.y()
        inaxes = self._canvas.inaxes((x, y))
        if inaxes is not None:
            try:
                trans = inaxes.transData.inverted()
                xdata, ydata = trans.transform((x, y))
            except ValueError:
                pass
            else:
                print(xdata, ydata)
        super(FigureCanvas, self).mousePressEvent(a0)


class HistImage(QDialog):
    """
    Histogram images
    """
    closeSig = pyqtSignal()
    def __init__(self, parent=None):
        super(HistImage, self).__init__(parent)

        Form = self.window()
        Form.setObjectName("N4")
        Form.resize(600, 600)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(0,0,0,0)
        self.grid_main.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.setGeometry(QtCore.QRect(50, 50, 500, 500))
        self.tabWidget.setObjectName("tabWidget")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(100, 100))
        #self.tabWidget.setObjectName("tabWidget")
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(20, 20, 400, 400))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        # a figure instance to plot on
        self.figure = plt.figure()


        self.canvas = FigureCanvas(self.figure)

        self.canvas.setParent(self)

        self.gridLayout.addWidget(self.canvas, 0, 0, 2, 2)

        self.figure2 = plt.figure()

        self.widget2 = QtWidgets.QWidget()
        self.widget2.setGeometry(QtCore.QRect(20, 20, 400, 400))
        self.widget2.setObjectName("widget")
        
        self.gridLayout2 = QtWidgets.QGridLayout(self.widget2)
        self.canvas2 = FigureCanvas(self.figure2)

        self.gridLayout2.addWidget(self.canvas2, 0, 0, 2, 2)
        self.gridLayout2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout2.setObjectName("gridLayout")
        self.tabWidget.addTab(self.widget2,"")
        self.tabWidget.addTab(self.widget, "")
        _translate = QtCore.QCoreApplication.translate
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget), _translate("Dialog", "View 2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget2), _translate("Dialog", "View 1"))

        self.grid_main.addWidget(self.tabWidget)

        Form.setWindowTitle(_translate("Form", "Image Histogram"))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(HistImage, self).closeEvent(a0)

    def UpdateName(self, a, b):
        _translate = QtCore.QCoreApplication.translate
        if a is not None:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget), _translate("Dialog", a))
        if b is not None:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget2), _translate("Dialog", b))
    def plot(self, data, id):
        if data is None:
            return #data = [random.random() for i in range(10)]
        if id == 1:
            figure, canvas = self.figure, self.canvas
        elif id == 2:
            figure, canvas = self.figure2, self.canvas2
        else:
            return


        figure.clear()

        ax = figure.add_subplot(111)

        a = data.mean(axis=2).flatten()
        a = a[a>0]

        ax.hist(a, 25)
        ax.grid('on')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.set_title('Image histogram', fontsize=6 * 1.7, weight='bold')
        ax.set_xlabel('Range', fontsize=6 * 1.5, weight='bold')

        ax.set_ylabel('Frequency', fontsize=6 * 1.5, weight='bold')
        ax.set_xlim([0,a.max()+1])
        figure.tight_layout(pad=2)

        canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = HistImage()
    main.show()

    sys.exit(app.exec_())