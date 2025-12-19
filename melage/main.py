# This Python file uses the following encoding: utf-8
import sys

import os
sys.path.append('.')
sys.path.append("../")
#print(sys.path)
from PyQt5 import QtWidgets, QtCore, QtGui
#from PyQt5.QtWidgets import QWidget
from melage.mainwindow_widget import Ui_Main
from melage.dialogs import activation_dialog
from melage.config import settings, __VERSION__

## verifyClass
class verifyClass(QtCore.QObject):
    sig = QtCore.pyqtSignal(int)
    def __init__(self):
        super().__init__()

    def verify_func(self):
        ## your verification code
        self.sig.emit(1) ## mean the verification is okay, do it if verification success



class MainWindow(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self, *args, app = None, **kwargs):



        super(MainWindow, self).__init__(*args, **kwargs)
        QtWidgets.QWidget.__init__(self)
        self.app = app
        self.setupUi(self)
        #if os.path.isdir(SOURCE_DIR):
            #self.source_dir = SOURCE_DIR
        #self.settings.setPath(self.settings.IniFormat, self.settings.UserScope, '.')

        self.setWindowIcon(QtGui.QIcon(settings.RESOURCE_DIR+'/main.ico'))
        self._key_picke = b'PPQ0ByoMsieWGv6bMEyJ9rSYXQDoa5D4ldAkwaNNpw0='

        if not self.settings.value("geometry") == None:
            self.restoreGeometry(self.settings.value("geometry"))
        #QtCore.QTimer.singleShot(5000, self.showChildWindow)

        self._openUSEnabled = True
        ######################### Load connect ################################
        self.actionLoad.triggered.connect(self.loadChanges)
        key_needed = False
        if key_needed:


            self.setEnabled(False)
            self.activation_dialog = activation_dialog(self,settings.RESOURCE_DIR)
            self.activation_dialog.correct_key.connect(self.EnableAll)
            verifyInstance = verifyClass()
            verifyInstance.sig.connect(self.verify_func)
            thread = QtCore.QThread()
            self._v_thread = (thread, verifyInstance)
            verifyInstance.moveToThread(thread)
            thread.started.connect(verifyInstance.verify_func)
            thread.start()
        else:
            self.setEnabled(True)




    def verify_func(self, i):
        if i == 1:
            if self.activation_dialog.verify():
                self.EnableAll(self.activation_dialog.options)
            else:
                self.activation_dialog.exec_()


            ## enable your buttons

    def EnableAll(self, options):
        if options=='FULL':
            #self.menuExport.setEnabled(False)
            #self.menuPreprocess.setEnabled(False)
            #self.actionCircles.setEnabled(False)

            '''
            
            self.actionPoints.setEnabled(False)
            self.menuSeg.setEnabled(False)
            
            self.actionsaveModified.setVisible(False)
            self.actionconvert.setVisible(False)
            self.menuCalc.setEnabled(False)
            '''
            self._openUSEnabled = True
            self.setEnabled(True)
            self.menuSeg.setEnabled(False)
            self.menuCalc.setEnabled(False)

            self.menuRegistration.setEnabled(False)

    def saveChanges(self):
        from melage.utils.utils import getAttributeWidget
        try:
            import cPickle as pickle
        except ModuleNotFoundError:
            import pickle
        from collections import defaultdict
        from PyQt5.QtCore import QSettings
        from time import gmtime, strftime
        self.settings.sync()
        dic = defaultdict(list)


        """
        
        if filePath == None:
            basefile = ''
            if hasattr(self, 'readImECO'):
                basefile =self.readImECO.basefile
            if basefile == '':
                if hasattr(self, 'readImMRI'):
                    basefile = self.readImMRI.basefile
        else:
            basefile = filePath[0]

        if basefile[-3:] == '.bn':
            basefile = basefile[:-3]

        self._basefileSave = basefile
        """
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
        dic['versionInfo']['__version__'] = __VERSION__
        dic['versionInfo']['dataTime'] = strftime("%Y_%m_%d_%H%M%S", gmtime())
        dic["settings"] = {}
        dic["settings"] = vars(settings).copy()

 #       with open(self._basefileSave+'.bn', 'w+') as f:
#            f.write('0\n')

        self.progressBarSaving.setValue(90)


        #fp = gzip.open('primes.data', 'wb')
        #from cryptography.fernet import Fernet
        with open(self._basefileSave+'.bn', 'wb') as output:
            pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)

        #f = Fernet(self._key_picke)
        #with open(self._basefileSave+'.bn', 'rb') as file:
            # read all file data
        #    file_data = file.read()
        #encrypted_data = f.encrypt(file_data)
        #with open(self._basefileSave+'.bn', 'wb') as file:
        #    file.write(encrypted_data)
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
    """
    
    # Set the default surface format for the application
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(2, 0)  # Set the desired OpenGL version
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)  # Set the desired profile
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    """
    #root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    QtCore.QDir.addSearchPath('resource', os.path.join(root, "assets", 'resource'))
    QtCore.QDir.addSearchPath('theme', os.path.join(root, "assets",'resource', 'theme'))
    QtCore.QDir.addSearchPath('rc', os.path.join(root, "assets",'resource','theme', 'rc'))
    QtCore.QDir.addSearchPath('color', os.path.join(root, "assets",'resource', 'color'))
    file = QtCore.QFile("theme:style.qss")
    file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
    stream = QtCore.QTextStream(file)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))


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

