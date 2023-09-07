__AUTHOR__ = 'Bahram Jafrasteh'


from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMenu, QAction, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage, qRgb
import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from utils.utils import help_dialogue_open_image

class QFileSaveDialogPreview(QFileDialog):
    """
    Customizing save dialogue
    """
    def __init__(self, *args, check_state_csv, **kwargs):
        QFileDialog.__init__(self, *args, **kwargs)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        #self.setViewMode(QFileDialog.Detail)
        self.setWindowTitle('Saving image...')
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)



        self.checkBox_csv = QtWidgets.QCheckBox(self)
        self.checkBox_csv.setObjectName("checkBox")
        self.checkBox_csv.setText('Save table')
        self.checkBox_csv.setChecked(check_state_csv)

        box = QHBoxLayout()
        box.addWidget(self.checkBox_csv)

        box.addStretch()

        self.layout().addLayout(box, 4, 2, 1, 1)

        _translate = QtCore.QCoreApplication.translate


        #self.layout().addLayout(box, 4, 2, 1, 1)

        self.label = QtWidgets.QLabel(self)
        self.label.setText(_translate("Dialog", "RAS"))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet('color: Red')



        self.combbox_asto = QtWidgets.QComboBox(self)
        self.combbox_asto.setObjectName("label")

        for i , ss in enumerate(['as', 'to']):
            self.combbox_asto.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self.combbox_asto.addItem(" "*9+ss+" "*9)

        self._combobox_coords = QtWidgets.QComboBox(self)
        self._combobox_coords.setObjectName("_combobox_coords")



        self.systems = ['RAS', 'RSA', 'RPI', 'RPS',
                   'RIP', 'RIA', 'RSP', 'RSA',
                    'LAS', 'LAI', 'LPI', 'LPS',
                   'LSA', 'LIA', 'LIP', 'LSP',
                   'PIL', 'PIR', 'PSL', 'PSR',
                   'PLI', 'PRI', 'PLS', 'PRS',
                   'AIL', 'AIR', 'ASL', 'ASR',
                   'ALI', 'ARI', 'ALS', 'ARS',
                   'IPL', 'IPR', 'IAL', 'IAR',
                   'ILP', 'IRP', 'ILA', 'IRA',
                   'SPL', 'SPR', 'SAL', 'SAR',
                   'SLP', 'SRP', 'SLA', 'SRA'
                   ]
        for i , ss in enumerate(self.systems):
            self._combobox_coords.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_coords.addItem(" "*9+ss+" "*9)
        self._combobox_coords.setEnabled(False)
        self.combbox_asto.setEnabled(False)

        #box = QHBoxLayout()
        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText('Advanced')

        box = QHBoxLayout()
        box.addWidget(self.checkBox)
        box.addWidget(self.label)
        box.addWidget(self.combbox_asto)



        #self.layout().addLayout(box, 4, 0, 1, 1)

        box.addWidget(self._combobox_coords)
        #box.addStretch()
        self.layout().addLayout(box, 4, 1, 1, 1)
        self.checkBox.stateChanged.connect(self.activate_combobox)
        self._fileSelected = ''
        self.fileSelected.connect(self.onFileSelected)


    def setCS(self, cs):
        try:
            index = [i for i, el in enumerate(self.systems) if el == cs][0]
            self._combobox_coords.setCurrentIndex(0)
            self.label.setText(cs)
        except:
            pass

    def activate_combobox(self, value):
        self._combobox_coords.setEnabled(value)
        self.combbox_asto.setEnabled(value)

    def getInfo(self):
        return self._combobox_coords.currentText().strip(' '), self.combbox_asto.currentText().strip(' '), self.checkBox_csv.isChecked()

    def getFileSelected(self):
        return [self._fileSelected, '']

    def onFileSelected(self, file):
        self._fileSelected = file



class QFileDialogPreview(QFileDialog):
    """
    Open dialogue preview
    """
    def __init__(self, *args, **kwargs):
        if 'index' in kwargs:
            index = kwargs['index']
            kwargs.pop('index')
        else:
            index = 0
        if 'last_state' in kwargs:
            last_state = kwargs['last_state']
            kwargs.pop('last_state')
        else:
            last_state = False


        QFileDialog.__init__(self, *args, **kwargs)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.ExistingFile)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        #self.setViewMode(QFileDialog.Detail)
        self.setWindowTitle('Load image...')

        box = QVBoxLayout()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

        #self.setFixedSize(self.width() + 500, self.height())

        self.mpPreview = QLabel("Preview", self)
        #self.mpPreview.setFixedSize(500, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.mpPreview.setMinimumSize(400, 400)
        sizePolicy.setHeightForWidth(self.mpPreview.sizePolicy().hasHeightForWidth())
        self.mpPreview.setSizePolicy(sizePolicy)
        self.mpPreview.setAlignment(Qt.AlignCenter)
        self.mpPreview.setObjectName("labelPreview")

        self.checkBox_preview = QtWidgets.QCheckBox(self)
        self.checkBox_preview.setObjectName("checkBox")
        self.checkBox_preview.setText('Preview')
        self.checkBox_preview.setChecked(last_state)

        box.addWidget(self.mpPreview)
        box.addWidget(self.checkBox_preview)
        box2 = QHBoxLayout()
        self._combobox_type = QtWidgets.QComboBox(self)
        self._combobox_type.setObjectName("_combobox_coords")



        self.systems = ['Neonatal', 'Fetal', 'MRI']
        for i , ss in enumerate(self.systems):
            self._combobox_type.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
            self._combobox_type.addItem(" "*9+ss+" "*9)
        self._combobox_type.setCurrentIndex(index)
        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText('Advanced')
        self.checkBox.stateChanged.connect(self.activate_combobox)
        box2.addWidget(self.checkBox)
        box2.addWidget(self._combobox_type)


        #box.addStretch()

        self.layout().addLayout(box, 1, 3, 1, 1)
        self.layout().addLayout(box2, 4, 1, 1, 1)


        self.currentChanged.connect(self.onChange)
        self.fileSelected.connect(self.onFileSelected)
        self.filesSelected.connect(self.onFilesSelected)

        self._fileSelected = None
        self._filesSelected = None



    def onChange(self, path):
        #pixmap = QPixmap(path)
        if not self.checkBox_preview.isChecked():
            return
        fileName, file_extension = os.path.splitext(path)
        if file_extension ==  '.gz':
            fileName, file_extension = os.path.splitext(fileName)
        if not os.path.isfile(path):
            return

        try:
            s = help_dialogue_open_image(path)
            height, width,_ = s.shape

            bytesPerLine = 3 * width
            #qImg = QImage(normal.data, width, height, bytesPerLine, QImage.Format_RGB888)
            #gray_color_table = [qRgb(i, i, i) for i in range(256)]
            qImg = QImage(s.data, s.shape[1]//3, s.shape[0]//3, bytesPerLine, QImage.Format_RGB888)
            #qImg.setColorTable(gray_color_table)
            #from PyQt5.QtCore import QByteArray
            #qb = QByteArray(np.ndarray.tobytes(s))
            #qImg = QImage.fromData(qb)
            pixmap01 = QPixmap.fromImage(qImg)
            pixmap = QPixmap(pixmap01)
            if(pixmap.isNull()):
                self.mpPreview.setText("Preview")
            else:
                self.mpPreview.setPixmap(pixmap.scaled(self.mpPreview.width(), self.mpPreview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                #self.mpPreview.setPixmap(pixmap)
        except:
            pass
    def activate_combobox(self, value):
        self._combobox_type.setEnabled(value)
    def onFileSelected(self, file):
        self._fileSelected = file

    def onFilesSelected(self, files):
        self._filesSelected = files

    def getFileSelected(self):
        return [self._fileSelected, '']

    def getFilesSelected(self):
        return self._filesSelected



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QFileDialogPreview()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()