__AUTHOR__ = 'Bahram Jafrasteh'

from utils.utils import make_affine
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog
import numpy as np
class TransformationDialog(QtWidgets.QDialog):
    """
    Image to image transformation widget
    """
    closeSig = pyqtSignal()


    def __init__(self, parent=None, source_dir = 'None'):
        self.source_dir = source_dir
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Child Window!")
        self.setupUi()
        self.reg_weights_path = None
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(TransformationDialog, self).closeEvent(a0)

    def setupUi(self):
        self.filters = "NifTi (*.nii *.nii.gz)"
        Form = self.window()
        Form.setObjectName("Registration form")
        self.MessageBox = QtWidgets.QMessageBox(Form)
        self.setObjectName("widget")
        self.setMinimumSize(750, 120)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_target = QtWidgets.QLineEdit(self)
        self.lineEdit_target.setObjectName("lineEdit_target")

        self.label_moving = QtWidgets.QLabel(self)
        self.label_moving.setAlignment(QtCore.Qt.AlignCenter)
        self.label_moving.setObjectName("label_moving")

        self.button_weights = QtWidgets.QPushButton(self)
        self.button_weights.setObjectName("button_weights")

        self.lineEdit_moving = QtWidgets.QLineEdit(self)
        self.lineEdit_moving.setObjectName("lineEdit")

        self.label_fixed = QtWidgets.QLabel(self)
        self.label_fixed.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fixed.setObjectName("label_fixed")

        self.button_fixed = QtWidgets.QPushButton(self)
        self.button_fixed.setObjectName("button_fixed")

        self.lineEdit_fixed = QtWidgets.QLineEdit(self)
        self.lineEdit_fixed.setObjectName("lineEdit_fixed")
        self.lineEdit_fixed.setReadOnly(True)
        #self.gridLayout.addWidget(self.button_fixed, 2, 5, 1, 1)
        #self.gridLayout.addWidget(self.label_fixed, 1, 0, 1, 1)
        #self.gridLayout.addWidget(self.lineEdit_fixed, 1, 0, 1, 1)

        self.button_moving = QtWidgets.QPushButton(self)
        self.button_moving.setObjectName("button_moving")

        self.label_out = QtWidgets.QLabel(self)
        self.label_out.setAlignment(QtCore.Qt.AlignCenter)
        self.label_out.setObjectName("label_out")

        self.lineEdit_out = QtWidgets.QLineEdit(self)
        self.lineEdit_out.setObjectName("lineEdit")

        self.lineEdit_out.setReadOnly(True)
        self.lineEdit_moving.setReadOnly(True)
        self.lineEdit_target.setReadOnly(True)

        self.button_out = QtWidgets.QPushButton(self)
        self.button_out.setObjectName("button_out")

        self.label_weights = QtWidgets.QLabel(self)
        self.label_weights.setAlignment(QtCore.Qt.AlignCenter)
        self.label_weights.setObjectName("label_weights")


        self.splitter = QtWidgets.QSplitter(self)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_nch = QtWidgets.QLabel(self.splitter)
        self.label_nch.setAlignment(QtCore.Qt.AlignCenter)
        self.label_nch.setObjectName("label_ref")
        #self.gridLayout.addWidget(self.label_nch, 4, 1, 2, 2)
        self.noch = QtWidgets.QSpinBox(self.splitter)
        self.noch.setAlignment(QtCore.Qt.AlignCenter)
        self.noch.setObjectName("label_ref")
        self.noch.setValue(3)
        self.noch.setMaximum(99)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)

        self.OK = QtWidgets.QPushButton(self)
        self.OK.setObjectName("checkBox_5")

        self.gridLayout.addWidget(self.button_fixed, 1, 5, 1, 1)
        self.gridLayout.addWidget(self.lineEdit_fixed, 1, 1, 1, 4)
        self.gridLayout.addWidget(self.label_fixed, 1, 0, 1, 1)

        self.gridLayout.addWidget(self.button_moving, 2, 5, 1, 1)
        self.gridLayout.addWidget(self.lineEdit_moving, 2, 1, 1, 4)
        self.gridLayout.addWidget(self.label_moving, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.splitter, 2,6,1,1)

        self.gridLayout.addWidget(self.lineEdit_target, 3, 1, 1, 4)
        self.gridLayout.addWidget(self.label_weights, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.button_weights, 3, 5, 1, 1)

        self.gridLayout.addWidget(self.button_out, 4, 5, 1, 1)
        self.gridLayout.addWidget(self.lineEdit_out, 4, 1, 1, 4)
        self.gridLayout.addWidget(self.label_out, 4, 0, 1, 1)

        self.gridLayout.addWidget(self.OK, 5, 5, 2, 2)

        self.button_moving.clicked.connect(self.browse_ref)
        self.button_weights.clicked.connect(self.browse_weights)
        self.button_out.clicked.connect(self.save_out)
        self.button_fixed.clicked.connect(self.browse_fixed)
        self.OK.clicked.connect(self.ApplyTransform)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def save_out(self, value):
        from PyQt5.QtWidgets import QFileDialog
        #from utils.utils import getCurrentCoordSystem

        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        dialg = QFileDialog(self, "Open File", self.source_dir, filter="NifTi (*.nii *.nii.gz)",options=opts)
        dialg.setFileMode(QFileDialog.AnyFile)
        dialg.setAcceptMode(QFileDialog.AcceptSave)
        if dialg.exec_() == QFileDialog.Accepted:
            self.file_out = dialg.selectedFiles()[0] + '.nii.gz'
            self.lineEdit_out.setText(self.file_out)
        return

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Transformation"))

        self.label_moving.setText(_translate("Form", "Moving"))


        self.button_weights.setText(_translate("Form", "Weights"))
        self.button_moving.setText(_translate("Form", "Moving"))
        self.button_out.setText(_translate("Form", "Out"))
        self.label_weights.setText(_translate("Form", "Weights"))
        self.label_out.setText(_translate("Form", "Out"))

        self.label_fixed.setText(_translate("Form", "Fixed"))
        self.button_fixed.setText(_translate("Form", "Fixed"))
        self.label_nch.setText(_translate("Form", "NoCh"))

        self.OK.setText(_translate("Form", "APPLY"))


    def accepted_emit(self):
        self.lineEdit.text()
        num = self.lineEdit.text()
        if num=='' or float(num)==0:
            self.message()
            return False
        txt = self.lineEdit2.text()
        if txt=='':
            self.message()
            return False
        self.ColorIndName.emit([num, txt])
        self.accept()

    def browse_fixed(self):
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        dialg = QFileDialog( self, "Open File", self.source_dir, self.filters, options=opts)
        dialg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialg.exec_() == QFileDialog.Accepted:
            fileObj = dialg.selectedFiles()
            if len(fileObj)>0:
                self.file_fixed = fileObj[0]
                self.lineEdit_fixed.setText(self.file_fixed)

    def browse_ref(self):
        fileObj = ['', '']
        #if fileObj is None or type(fileObj) is bool:
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        dialg = QFileDialog( self, "Open File", self.source_dir, self.filters, options=opts)
        dialg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialg.exec_() == QFileDialog.Accepted:
            fileObj = dialg.selectedFiles()
            if len(fileObj)>0:
                self.file_moving = fileObj[0]
                self.lineEdit_moving.setText(self.file_moving)

    def browse_weights(self):
        fileObj = ['', '']
        #if fileObj is None or type(fileObj) is bool:
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        dialg = QFileDialog( self, "Open File", self.source_dir, "TFM (*.tfm)", options=opts)
        dialg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialg.exec_() == QFileDialog.Accepted:
            fileObj = dialg.selectedFiles()
            if len(fileObj)>0:
                self.file_weights = fileObj[0]
                self.lineEdit_target.setText(self.file_weights)
    def ApplyTransform(self):

        from utils.registration import apply_reg
        if not (hasattr(self, 'file_weights') and hasattr(self, 'file_moving')):
            return
        if self.file_out is None:
            return
        import SimpleITK as sitk
        from utils.utils import read_file_with_cs, make_image

        fixed = sitk.ReadImage(self.file_fixed)
        outTx = sitk.ReadTransform(self.file_weights)
        f_reader = sitk.ImageFileReader()
        f_reader.SetFileName(self.file_moving)
        f_reader.ReadImageInformation()
        size = list(f_reader.GetSize())
        if len(size) not in [3,4]:
            self.MessageBox.setText('The number of dimensions should be 3 or 4 while it is {}'.format(len(size)))
            self.MessageBox.setWindowTitle('Warning')
            self.MessageBox.show()
            return
        total_labels = []
        if len(size)==4:
            if size[-1]!=self.noch.value():
                self.MessageBox.setText('The number of channels is not as expected {}!={}'.format(size[-1], self.noch.value()))
                self.MessageBox.setWindowTitle('Warning')
                self.MessageBox.show()
                return
            size_total = size[-1]
        else:
            size_total = 1

        for l in range(size_total):
            f_reader.SetExtractIndex([0, 0, 0, l])
            f_reader.SetExtractSize((size[0], size[1], size[2], 0))
            moving = f_reader.Execute()
            predicted = apply_reg(outTx, fixed, moving)
            predicted = sitk.GetArrayFromImage(predicted).transpose()
            total_labels.append(predicted)

        fixed = read_file_with_cs(self.file_fixed)
        total_labels = np.stack(total_labels, -1)
        if len(total_labels)==1:
            total_labels = total_labels[...,0]
        total_labels = make_image(total_labels, fixed)
        total_labels.to_filename(self.file_out)



def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = TransformationDialog()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()