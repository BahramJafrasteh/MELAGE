from PyQt5.QtWidgets import QScrollBar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenu, QAction, QDialog
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets
class custom_qscrollbar(QScrollBar):
    """
    HORIZONTAL SLIDER
    """
    cut_limit = pyqtSignal(object)
    def __init__(self, parent=None, id = 0):
        self.id = id
        QScrollBar.__init__(self, parent)
        Dialog = self.window()
        Dialog.setObjectName("Image info")
        Dialog.resize(40, 200)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.ShowContextMenu)
        self._txtA = "A"
        self._txtB = "B"
        self._first = 0
        self._second = 0



    def ShowContextMenu(self, pos):
        menu = QMenu('Menu')
        first_action = QAction(self._txtA)
        second_action = QAction(self._txtB)

        cut_action = QAction("Cut")
        menu.addAction(first_action)
        menu.addAction(second_action)
        menu.addAction(cut_action)
        action = menu.exec_(self.mapToGlobal(pos))
        if action==first_action:
            #mouse_p = int((pos.x() / self.width()) * (self.maximum() - self.minimum()) + self.minimum())
            #self.setSliderPosition(mouse_p)
            self._first = self.value()
            self._txtA = "A:{}".format(self._first)
            #self.setSliderPosition(105)
        elif action==second_action:
            self._second = self.value()
            self._txtB = "B:{}".format(self._second)
        elif action == cut_action:
            if self._first == self._second:
                return
            cut_limit = [self._first, self._second]
            self.cut_limit.emit(cut_limit)

def run():
    import sys
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import pyqtSignal
    app = QtWidgets.QApplication(sys.argv)
    window = custom_qscrollbar()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()