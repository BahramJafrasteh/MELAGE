# This Python file uses the following encoding: utf-8
# --- 1. Python Standard Library ---
import sys
import os
import pickle
from collections import defaultdict
from time import gmtime, strftime
import argparse
# Ensure project root (parent of melage/) is on sys.path.
# When running this file directly (e.g. from PyCharm), Python inserts the
# melage/ directory as sys.path[0], which causes melage.py to shadow the
# melage package.  Remove it and add the true project root instead.
_here = os.path.dirname(os.path.abspath(__file__))        # .../melage/
_project_root = os.path.dirname(_here)                    # .../MELAGE/MELAGE/
while _here in sys.path:
    sys.path.remove(_here)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- 2. GUI (PyQt5) ---
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSettings, QEvent

# --- 3. Melage Project Imports ---
from melage.config import settings, __VERSION__
from melage.mainwindow_widget import Ui_Main
from melage.utils.utils import getAttributeWidget
from melage.core.headless import run_headless_mode

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

        self.setEnabled(True)




    def saveChanges(self):

        self.settings.sync()
        dic = defaultdict(list)


        """
        
        if filePath == None:
            basefile = ''
            if hasattr(self, 'readView1'):
                basefile =self.readView1.basefile
            if basefile == '':
                if hasattr(self, 'readView2'):
                    basefile = self.readView2.basefile
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
        """
        
        names = ['readView1', 'readView2']
        for name in names:
            if hasattr(self, name):
                readD = getattr(self, name)
                #dic[name] = {}
                dic = getAttributeWidget(readD, name, dic)
        """
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

        with open(self._basefileSave+'.bn', 'wb') as output:
            pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)

        self.progressBarSaving.setValue(100)

    def createPopupMenu(self): # overriding create popup menu
        menu = super().createPopupMenu()
        for action in menu.actions():
            if action.text()=='Progress Bar':
                menu.removeAction(action)
        return menu

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:

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

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        handle = self.windowHandle()
        if handle:
            try:
                handle.screenChanged.disconnect(self._apply_toolbar_icon_size)
            except TypeError:
                pass
            handle.screenChanged.connect(self._apply_toolbar_icon_size)

    def _apply_toolbar_icon_size(self, *_):
        size = QtCore.QSize(36, 36)
        if hasattr(self, 'toolBar'):
            self.toolBar.setIconSize(size)
        if hasattr(self, 'toolBar2'):
            self.toolBar2.setIconSize(size)

    #def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:

            #self.saveChanges()






def main():
    # 1. Setup Argument Parser

    from melage.utils.headless_utils import list_available_tools
    tools_map = list_available_tools()
    tool_ids = list(tools_map.keys())
    # Create a helpful description for the --tool argument
    tool_help_text = "Choose a tool to run:\n"
    for tid, tname in tools_map.items():
        tool_help_text += f"  - {tid:<12} ({tname})\n"
    # Add arguments
    parser = argparse.ArgumentParser(description="MELAGE: Neuroimaging Tool")
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--tool', type=str, choices=tool_ids, help=tool_help_text)
    parser.add_argument('--input', type=str,default='', help='Input image path')
    parser.add_argument('--output', type=str, default='',help='Output image path')

    # Parse known args to avoid conflict with Qt args
    args, unknown = parser.parse_known_args()
    # 2. Check for Headless Mode
    if args.headless:
        run_headless_mode(args)
    else:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
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

