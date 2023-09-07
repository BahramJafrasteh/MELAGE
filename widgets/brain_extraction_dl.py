__AUTHOR__ = 'Bahram Jafrasteh'

"""
    Main BrainExtractor class
"""
import os
import numpy as np
import nibabel as nib
from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial


from PyQt5.QtCore import pyqtSignal
import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt
from utils.utils import convert_to_ras
def to_str(val):
    return '{:.2f}'.format(val)

def remove_zero(f_data, value=0):
    """
    Remove non segmented areas from image
    :param f_data:
    :param value:
    :return:
    """

    xs, ys, zs = np.where(f_data > value) #find zero values
    tol = 4

    min_max = []
    for x in [xs, ys, zs]:
        minx = min(x)-tol if min(x)-tol>1 else min(x)
        maxx = max(x) + tol if max(x) + tol < f_data.shape[0]-1 else max(x)
        min_max.append([minx, maxx])
    f_data = f_data[min_max[0][0]:min_max[0][1] + 1, min_max[1][0]:min_max[1][1] + 1, min_max[2][0]:min_max[2][1] + 1]

    return f_data, min_max


def centralize_image(img, maxas=128, border=None):
    """
    Put image in the center
    :param img:
    :param maxas:
    :param border:
    :return:
    """

    n = img.shape
    if type(maxas) != list:
        maxas= [maxas, maxas, maxas]
    pads = np.array([maxas[i] - a for i, a in enumerate(n)])
    pads_r = pads // 2
    pads_l = pads - pads_r
    npads_l = pads_l * -1
    npads_r = pads_r * -1
    if border is None:
        border = img[0,0,0]
    new_img = np.ones((maxas[0], maxas[1], maxas[2]))*border

    pads_r[pads_r < 0] = 0
    pads_l[pads_l < 0] = 0
    npads_l[npads_l < 0] = 0
    npads_r[npads_r < 0] = 0
    # print(pads_l, pads_r)
    new_img[pads_r[0]:maxas[0] - pads_l[0], pads_r[1]:maxas[1] - pads_l[1], pads_r[2]:maxas[2] - pads_l[2]] = img[
                                                                                                  npads_r[0]:n[0] -npads_l[0],
                                                                                                  npads_r[1]:n[1] -npads_l[1],
                                                                                                  npads_r[2]:n[2] -npads_l[2]]
    return new_img, [pads, pads_l, pads_r, npads_l, npads_r, n]

class BE_DL(QDialog):
    """
    This class has been implemented to use deep learning algorithms for brain extraction
    """
    closeSig = pyqtSignal()
    betcomp = pyqtSignal()
    datachange = pyqtSignal()

    """

    """
    def __init__(self, parent=None
                 ):
        super(BE_DL, self).__init__(parent)
        #self.load_filepath = 'widgets/Hybrid_latest.pth'
        self._curent_weight_dir = os.path.join(os.getcwd())
        """
        Initialization of Brain Extractor

        Computes image range/thresholds and
        estimates the brain radius
        """
    def set_pars(self, threshold=-0.5, remove_extra_bone=True):
        #print("Initializing...")

        # get image resolution

        self.threshold = threshold
        self.borderp = 0.0

        # store brain extraction parameters
        self.setupUi()
    def setData(self, img, res):
        # store the image
        self.img = img
        self.initial_mask = None
        self.shape = img.shape  # 3D shape




    def activate_advanced(self, value):
        self.widget.setEnabled(value)

    def setupUi(self):
        Dialog = self.window()
        Dialog.setObjectName("N4")
        Dialog.resize(500, 220)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(0, 0, 0, 0)
        self.grid_main.setObjectName("gridLayout")

        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setObjectName("gridLayout")
        self.grid_main.addLayout(self.hbox, 0, 0)

        self.checkBox = QtWidgets.QCheckBox()
        self.hbox.addWidget(self.checkBox, 0)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.stateChanged.connect(self.activate_advanced)
        self.comboBox_image = QtWidgets.QComboBox()
        self.comboBox_image.setGeometry(QtCore.QRect(20, 10, 321, 25))
        self.comboBox_image.setObjectName("comboBox")
        for i in range(2):
            self.comboBox_image.addItem("")
        self.comboBox_image.currentIndexChanged.connect(self.datachange)

        self.comboBox_models = QtWidgets.QComboBox()
        self.comboBox_models.setGeometry(QtCore.QRect(20, 10, 321, 25))
        self.comboBox_models.setObjectName("comboBox")
        for i in range(2):
            self.comboBox_models.addItem("")
        self.comboBox_models.currentIndexChanged.connect(partial(self.parChanged, True))


        self.hbox.addWidget(self.comboBox_image, 1)
        self.hbox.addWidget(self.comboBox_models, 2)


        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setMaximum(100)



        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.setContentsMargins(0, 0, 0, 0)
        self.hbox2.setObjectName("gridLayout")
        self.hbox2.addWidget(self.progressBar, 1)



        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter_3 = QtWidgets.QSplitter(self.widget)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")


        self.gridLayout.addWidget(self.splitter_3, 0, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.checkbox_thresholding = QtWidgets.QCheckBox(self.splitter)
        #self.checkbox_thresholding.setAlignment(QtCore.Qt.AlignCenter)
        self.checkbox_thresholding.setObjectName("checkbox_thresholding")
        self.checkbox_thresholding.setChecked(False)
        #self.checkbox_bone = QtWidgets.QCheckBox(self.splitter)
        #self.checkbox_bone.setObjectName("histogram_bone")
        #self.checkbox_bone.setChecked(False)
        self.label_type = QtWidgets.QLabel(self.splitter)
        self.label_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_type.setObjectName("label")
        self.comboBox_image_type = QtWidgets.QComboBox(self.splitter)
        self.comboBox_image_type.setGeometry(QtCore.QRect(20, 10, 321, 25))
        self.comboBox_image_type.setObjectName("comboBox")
        self.comboBox_image_type.setCurrentIndex(0)
        self.comboBox_image_type.currentIndexChanged.connect(partial(self.parChanged, False))
        for i in range(2):
            self.comboBox_image_type.addItem("")


        #self.histogram_threshold_min = QtWidgets.QDoubleSpinBox(self.splitter)
        #self.histogram_threshold_min.setObjectName("histogram_threshold_min")
        #self.histogram_threshold_min.setValue(6)#(self.ht_min)*100)
        #self.histogram_threshold_min.setMaximum(10)
        #self.histogram_threshold_min.setMinimum(0)

        #self.histogram_threshold_max = QtWidgets.QDoubleSpinBox(self.splitter)
        #self.histogram_threshold_max.setObjectName("histogram_threshold_max")
        #self.histogram_threshold_max.setValue((self.ht_max) * 100)
        #self.histogram_threshold_max.setMaximum(100)
        #self.histogram_threshold_max.setMinimum(0)

        #self.histogram_threshold_min.setEnabled(True)
        #self.histogram_threshold_max.setEnabled(False)

        self.gridLayout.addWidget(self.splitter, 1, 0, 1, 1)



        self.splitter_2 = QtWidgets.QSplitter(self.widget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")


        self.label = QtWidgets.QLabel(self.splitter_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.fractional_threshold = QtWidgets.QDoubleSpinBox(self.splitter_2)
        self.fractional_threshold.setObjectName("fractional_threshold")

        self.fractional_threshold.setMaximum(10)
        self.fractional_threshold.setMinimum(-10)
        self.fractional_threshold.setSingleStep(0.1)
        self.fractional_threshold.setValue(self.threshold)



        self.gridLayout.addWidget(self.splitter_2, 2, 0, 1, 1)


        self.splitter_3 = QtWidgets.QSplitter(self.widget)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")

        self.label_fl = QtWidgets.QLabel(self.splitter_3)
        self.label_fl.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fl.setObjectName("label_5")
        self.load_filepath = os.path.join('NONE.pth')
        self.bt_load_weight = QtWidgets.QPushButton(self.splitter_3)
        self.bt_load_weight.setObjectName("pushButton")
        self.bt_load_weight.pressed.connect(self.load_weight_dialog)
        self.bt_load_weight.setDefault(False)
        self.pushButton = QtWidgets.QPushButton()

        self.pushButton.setObjectName("pushButton")
        self.pushButton.pressed.connect(self.accepted_emit)
        self.hbox2.addWidget(self.pushButton, 0)
        self.pushButton.setDefault(True)
        self.gridLayout.addWidget(self.splitter_3, 3, 0, 1, 1)

        self.widget.setEnabled(False)


        self.grid_main.addWidget(self.widget)
        self.grid_main.addLayout(self.hbox2, 20, 0)

        self.label_pr = QtWidgets.QLabel()
        self.label_pr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_pr.setObjectName("label_2")
        self.label_pr.setText('fdfdf')
        self.label_pr.setVisible(False)

        self.grid_main.addWidget(self.label_pr)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def parChanged(self, model_changed=True):
        """
        if model parameter changed the user needs to run the algorithm again
        :param model_changed:
        :return:
        """
        self.initial_mask = None
        if model_changed:
            ci = self.comboBox_models.currentIndex()

            if ci == 0:
                self.load_filepath = os.path.join(self._curent_weight_dir,'synthstrip.1.pt')
                self.fractional_threshold.setValue(1)
            else:
                return
            self.label_fl.setText(self.load_filepath)
    def load_weight_dialog(self):
        opts = QtWidgets.QFileDialog.DontUseNativeDialog
        pwd = os.path.abspath(__file__)
        source_dir = os.path.dirname(os.path.dirname(pwd))
        fileObj = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", source_dir, "pth (*.pth *.pt)", options=opts)
        if fileObj[0] == '':
            return
        self.load_filepath = fileObj[0]
        self.label_fl.setText(self.load_filepath)
        self.initial_mask = None
    def clear(self):
        # store the image
        self.img = None
        self.shape = None  # 3D shape
        self.mask = None
        self.initial_mask = None

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "DEEPL BET"))
        self.checkBox.setText(_translate("Dialog", "Advanced"))
        self.comboBox_image.setItemText(0, _translate("Dialog", "Top Image"))
        self.comboBox_image.setItemText(1, _translate("Dialog", "Bottom Image"))
        self.comboBox_image_type.setItemText(0, _translate("Dialog", "Ultrasound"))
        self.comboBox_image_type.setItemText(1, _translate("Dialog", "MRI"))
        self.comboBox_models.setItemText(0, _translate("Dialog", "SynthStrip"))
        self.comboBox_models.setItemText(1, _translate("Dialog", "Custom"))

        self.pushButton.setText(_translate("Dialog", "Apply"))
        self.checkbox_thresholding.setText(_translate("Dialog", "CUDA"))
        #self.checkbox_bone.setText(_translate("Dialog", "Remove extra bone"))
        self.bt_load_weight.setText(_translate("Dialog", "Load Network Weights"))
        self.label.setText(_translate("Dialog", "Fractional Threshold"))
        self.label_type.setText(_translate("Dialog", "Image type"))


        self.label_fl.setText(_translate("Dialog", self.load_filepath))


    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(BE_DL, self).closeEvent(a0)

    def accepted_emit(self):
        if not hasattr(self,'initial_mask'):
            return
        try:
            self.label_pr.setVisible(True)
            self.label_pr.setText('Initialization...')


            self.progressBar.setValue(5)
            self._progress = 5
            if self.initial_mask is None:
                self.initial_mask = self.initialization()
            self.label_pr.setText('prediction...')
            self.mask = self.compute_mask(self.initial_mask)
            self.progressBar.setValue(98)
            self.label_pr.setVisible(False)
            self._progress = 100
            self.progressBar.setValue(self._progress)
            self._progress =0
            self.betcomp.emit()
        except Exception as e:
            print(e)
            self.screen_error_msgbox(e.args[0])
    def screen_error_msgbox(self, text= None):
        if text is None:
            text = 'There is an error. Screen is not captured. Please check the content.'
        MessageBox = QtWidgets.QMessageBox(self)
        MessageBox.setText(str(text))
        MessageBox.setWindowTitle('Warning')
        MessageBox.show()
        self.progressBar.setValue(0)

    def initialization(self):
        def rescaleint8(x):
            """
            y = a+(b-a)*(x-min(x))/(max(x)-min(x))
            Parameters
            ----------
            x

            Returns
            -------

            """
            oldMin, oldMax = int(x.min()), x.max()
            NewMin, NewMax = 0, 1000
            OldRange = (oldMax - oldMin)
            NewRange = (NewMax - NewMin)
            y = NewMin + (NewRange) * ((x - oldMin) / (OldRange))
            return y
        def signdf(df):
            if df<=0:
                return -1
            else:
                return 1
        pv = self._progress
        self.progressBar.setValue(pv+1)
        from widgets.Synthstrip import StripModel
        # find the center of mass of image
        from nibabel.processing import resample_to_output, resample_from_to
        import torch
        #imA = self.img
        imA = self.img.__class__(self.img.dataobj[:], self.img.affine, self.img.header)

        if self.checkbox_thresholding.isChecked():
            if torch.cuda.is_available() :
                # device = torch.device("cuda")
                device = torch.device("cuda:0")
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        ci = self.comboBox_models.currentIndex()


        if ci == 0:
            model = StripModel()
            #self.fractional_threshold.setValue(1)
        else:
            return


        self.borderp = 0
        if ci==0:
            transform, source = convert_to_ras(imA.affine, target='LIA')
            imAa = imA.as_reoriented(transform)
            shape_initial = imAa.shape
            image_used, pad_zero = remove_zero(imAa.get_fdata(), 0)
            imAzero = nib.Nifti1Image(image_used, imAa.affine, imAa.header)
            affine_used, header_used = imAa.affine, imAa.header.copy()
            NewSpacing = 1
            imAa = resample_to_output(imAzero, [NewSpacing, NewSpacing, NewSpacing])

            transform, source = convert_to_ras(imAa.affine, target='LIA')
            imAa = imAa.as_reoriented(transform)
            target_shape = np.clip(np.ceil(np.array(imAa.shape[:3]) / 64).astype(int) * 64, 192, 320)

        #transform, source = convert_to_ras(imAa.affine, target='RAS')
        #imAa = imAa.as_reoriented(transform)
        pixdim = imAa.header['pixdim'][1:4]
        affine = imAa.affine


        imA = imAa.get_fdata()
        imA -= imA.min()
        if np.percentile(imA, 99)!=0:
            imA = (imA / np.percentile(imA, 99)).clip(0, 1)
        else:
            imA = (imA / imA.max()).clip(0, 1)
        if ci==0:
            imA, info_back = centralize_image(imA, list(target_shape))
        del imAa
        [pads, pads_l, pads_r, npads_l, npads_r, shape_img] = info_back



        imA = torch.from_numpy(imA).to(torch.float).unsqueeze(0).unsqueeze(0)

        print('loading model weight...')


        imA = imA.to(device)
        model.to(device)
        self.progressBar.setValue(40)
        self.label_pr.setText('loading model...')
        if ci==0:
            state_dict = torch.load(self.load_filepath, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])

        self.progressBar.setValue(50)
        self.label_pr.setText('computing mask...')
        im_mask = model.forward(imA).detach().cpu().squeeze().numpy()
        maxas = [i for i in im_mask.shape]
        im_mask = im_mask[pads_r[0]:maxas[0] - pads_l[0], pads_r[1]:maxas[1] - pads_l[1], pads_r[2]:maxas[2] - pads_l[2]]
        if ci==0:
            mask = nib.Nifti1Image(im_mask, affine, header_used)
            mask = resample_from_to(mask, imAzero)
            im_mask = np.zeros(shape_initial)
            im_mask[pad_zero[0][0]:pad_zero[0][1] + 1, pad_zero[1][0]:pad_zero[1][1] + 1,
            pad_zero[2][0]:pad_zero[2][1] + 1] = mask.get_fdata()
        self.progressBar.setValue(80)
        header = self.img.header.copy()

        header['pixdim'][1:4] = pixdim
        mask = nib.Nifti1Image(im_mask, affine, header)
        if ci==0:
            _, source = convert_to_ras(self.img.affine, target='LIA')
            transform, _ = convert_to_ras(mask.affine, target=source)
            mask = mask.as_reoriented(transform)
        return mask


    def compute_mask(self, mask):
        """
        Convert surface mesh to volume
        """
        threshold = self.fractional_threshold.value()

        im_mask = mask.get_fdata().copy()

        ind = im_mask >= threshold
        im_mask[ind] = 0
        im_mask[~ind] = 1
        labels = im_mask

        return labels.astype('int')


def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    file = 't1_withoutmask.nii.gz'
    import nibabel as nib
    m = nib.load(file)
    res = m.header["pixdim"][1]
    nibf = m.get_fdata()
    from utils.utils import standardize
    #nibf = standardize(nibf)
    window = BE_DL()
    window.setData(m, res)
    window.set_pars()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()

