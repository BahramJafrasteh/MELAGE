# This Python file uses the following encoding: utf-8
import sys
sys.path.append("../")
import os
from PyQt5 import QtWidgets, QtCore, QtGui

from widgets.mainwindow_widget import Ui_Main



from utils.source_folder import source_folder, desktop

class MainWindow(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self, *args, app = None, **kwargs):



        super(MainWindow, self).__init__(*args, **kwargs)
        QtWidgets.QWidget.__init__(self)
        self.app = app
        self.setupUi(self)
        if os.path.isdir(desktop):
            self.source_dir = desktop
        #self.settings.setPath(self.settings.IniFormat, self.settings.UserScope, '.')

        self.setWindowIcon(QtGui.QIcon(source_folder+'/main.ico'))


        if not self.settings.value("geometry") == None:
            self.restoreGeometry(self.settings.value("geometry"))
        #QtCore.QTimer.singleShot(5000, self.showChildWindow)

        self._openUSEnabled = True
        ######################### Load connect ################################
        self.actionLoad.triggered.connect(self.loadChanges)
        self.setEnabled(True)


    def saveChanges(self):
        from utils.utils import getAttributeWidget
        try:
            import cPickle as pickle
        except ModuleNotFoundError:
            import pickle
        from collections import defaultdict
        from PyQt5.QtCore import QSettings
        from time import gmtime, strftime
        self.settings.sync()
        dic = defaultdict(list)

        self.settings = QSettings(self._basefileSave+ '.ini', self.settings.IniFormat)

        self.progressBarSaving.setValue(10)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.progressBarSaving.setValue(20)


        name = 'openGLWidget_'
        widgets_num = [0, 1, 2, 3, 4, 5, 10, 11,13,23]
        for i in widgets_num:
            nameWidget = name+str(i+1)
            if hasattr(self, name+str(i+1)):
                widget =getattr(self, name+str(i+1))
                dic = getAttributeWidget(widget, nameWidget, dic)
        self.progressBarSaving.setValue(40)
        names = ['readImECO', 'readImMRI']
        for name in names:
            if hasattr(self, name):
                readD = getattr(self, name)
                #dic[name] = {}
                dic = getAttributeWidget(readD, name, dic)
        table_items = []
        rows = self.table_widget_measure.rowCount()
        cols = self.table_widget_measure.columnCount()
        for row in range(rows):
            txts = []
            for col in range(cols):
                if hasattr(self.table_widget_measure.item(row, col),'text'):
                    txt = self.table_widget_measure.item(row, col).text()
                    txts.append(txt)
            table_items.append(txts)
        dic['measurements'] = table_items


        self.progressBarSaving.setValue(60)
        dic = getAttributeWidget(self, 'main', dic)
        self.progressBarSaving.setValue(80)

        dic['versionInfo'] = defaultdict(list)
        dic['versionInfo']['__version__'] = 'Neobrain_1.0.6'
        dic['versionInfo']['dataTime'] = strftime("%Y_%m_%d_%H%M%S", gmtime())


 #       with open(self._basefileSave+'.bn', 'w+') as f:
#            f.write('0\n')

        self.progressBarSaving.setValue(90)


        with open(self._basefileSave+'.bn', 'wb') as output:
            pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)

        with open(self._basefileSave+'.bn', 'rb') as file:
            # read all file data
            file_data = file.read()
        self.progressBarSaving.setValue(100)

    def createPopupMenu(self): # overriding create popup menu
        menu = super().createPopupMenu()
        for action in menu.actions():
            if action.text()=='Progress Bar':
                menu.removeAction(action)
        return menu

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.Close: # on closing the window
            if self._basefileSave == '':
                event.accept()
            else:
                MessageBox = QtWidgets.QMessageBox(self)
                MessageBox.setDefaultButton(QtWidgets.QMessageBox.Yes)

                MessageBox.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel)
                reply = MessageBox.question(self, 'Close','Do you want to save the changes before closing?',QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel)
                if reply == QtWidgets.QMessageBox.Yes:
                    print('saving the changes ...')
                    self.saveChanges()
                    event.accept()
                elif reply == QtWidgets.QMessageBox.No:
                    event.accept()
                elif reply == QtWidgets.QMessageBox.Cancel:
                    event.ignore()

    #def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:

            #self.saveChanges()






def main():
    app = QtWidgets.QApplication(sys.argv)
    root = os.path.dirname(os.path.abspath(__file__))
    QtCore.QDir.addSearchPath('resource', os.path.join(root, 'resource'))
    QtCore.QDir.addSearchPath('theme', os.path.join(root, 'resource/theme'))
    QtCore.QDir.addSearchPath('rc', os.path.join(root, 'resource/theme/rc'))
    QtCore.QDir.addSearchPath('color', os.path.join(root, 'resource/color'))
    file = QtCore.QFile("theme:style.qss")
    file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
    stream = QtCore.QTextStream(file)
    app.setStyleSheet(stream.readAll())
    window = MainWindow(app=app)
    window.show()
    #sys.excepthook = excepthook
    ret = app.exec_()
    #print("Exit")
    #sys.exit(ret)
    #sys.exc_info()







if __name__ == "__main__":
    main()

