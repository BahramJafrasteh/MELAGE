__AUTHOR__ = 'Bahram Jafrasteh'

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal

class iminfo_dialog(QtWidgets.QDialog):
    """
    This calss has been implemented to show image info
    """

    closeSig = pyqtSignal()
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Activation")

        self.setupUi()
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.closeSig.emit()
        super(iminfo_dialog, self).closeEvent(a0)

    def setupUi(self):
        Dialog = self.window()
        Dialog.setObjectName("Image info")
        Dialog.resize(803, 340)
        self.grid_main = QtWidgets.QVBoxLayout(self)
        self.grid_main.setContentsMargins(10,10,10,10)
        self.grid_main.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.setGeometry(QtCore.QRect(20, 30, 761, 251))
        self.tabWidget.setObjectName("tabWidget")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(100, 100))
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_mri = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_mri.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_mri.setObjectName("gridLayout_6")
        self.label_mri = QtWidgets.QTextEdit(self.tab)
        #a=[['Specific Character Set', 'ISO 2022 IR 100'], ['Image Type', ['ORIGINAL', 'SECONDARY', 'M', 'ND', 'NORM', 'FM', 'FIL']], ['Instance Creation Date', '20221122'], ['Instance Creation Time', '132720'], ['SOP Class UID', '1.2.840.10008.5.1.4.1.1.4'], ['SOP Instance UID', '1.2.840.113704.7.1.0.24314716597252255.1669120040.34722'], ['Study Date', '20210918'], ['Series Date', '20210918'], ['Acquisition Date', '20210918'], ['Content Date', '20210918'], ['Study Time', '125148.593000'], ['Series Time', '130306.453000'], ['Acquisition Time', '125906.770000'], ['Content Time', '130306.453000'], ['Accession Number', '124599971'], ['Modality', 'MR'], ['Modalities in Study', ['MR', 'SR']], ['Manufacturer', 'SIEMENS'], ['Institution Name', 'Hospital Puerta del Mar'], ['Institution Address', 'Av. Ana deb Viya 21,Cadiz,District,ES,11009'], ["Referring Physician's Name", 'RUIZ^ESTEFANIA'], ['Station Name', 'MRC37557'], ['Study Description', 'RM sin Contraste I.V. de Cráneo'], ['Procedure Code Sequence', 1], ['Series Description', 't1_mprage_tra_p2_iso'], ['Institutional Department Name', 'Department'], ['Physician(s) of Record', 'RUIZ^ESTEFANIA'], ["Performing Physician's Name", ''], ["Manufacturer's Model Name", 'SymphonyTim'], ['Referenced Image Sequence', 3], ['Source Image Sequence', 1], ["Patient's Name", 'BEARDO NAVAS^ADRIANA'], ['Patient ID', 'AN1337834619'], ['Issuer of Patient ID', 'NUHSA'], ["Patient's Birth Date", '20110128'], ["Patient's Sex", 'F'], ["Patient's Age", '010Y'], ["Patient's Size", '0.0'], ["Patient's Weight", '35.0'], ['Pregnancy Status', 4], ['Scanning Sequence', ['GR', 'IR']], ['Sequence Variant', ['SP', 'MP', 'OSP']], ['Scan Options', ['IR', 'PFP']], ['MR Acquisition Type', '3D'], ['Sequence Name', '*tfl3d1'], ['Angio Flag', 'N'], ['Slice Thickness', '1.0'], ['Repetition Time', '1910.0'], ['Echo Time', '3.53'], ['Inversion Time', '1100.0'], ['Number of Averages', '1.0'], ['Imaging Frequency', '63.686144'], ['Imaged Nucleus', '1H'], ['Echo Number(s)', '1'], ['Magnetic Field Strength', '1.5'], ['Number of Phase Encoding Steps', '223'], ['Echo Train Length', '1'], ['Percent Sampling', '100.0'], ['Percent Phase Field of View', '75.0'], ['Pixel Bandwidth', '130.0'], ['Device Serial Number', '37557'], ['Software Versions', 'syngo MR B19'], ['Protocol Name', 't1_mprage_tra_p2_iso'], ['Radionuclide Total Dose', None], ['Transmit Coil Name', 'Body'], ['Acquisition Matrix', [0, 256, 192, 0]], ['In-plane Phase Encoding Direction', 'ROW'], ['Flip Angle', '15.0'], ['Variable Flip Angle Flag', 'N'], ['SAR', '0.02698522422428'], ['dB/dt', '0.0'], ['Patient Position', 'HFS'], ['Private Creator', 'SIEMENS MR HEADER'], ['Study Instance UID', '1.2.840.113564.99.1.71094327813832.73.202191614221430.18408.2'], ['Series Instance UID', '1.2.840.113704.7.32.0.2.33.37557.2021091813030271133005070.0.0.0'], ['Study ID', '124599971'], ['Series Number', '5'], ['Acquisition Number', '1'], ['Instance Number', '1'], ['Image Position (Patient)', [-100.65369859619, -134.58704029078, -54.95581203964]], ['Image Orientation (Patient)', [0.99978196469588, 0.00571799450922, 0.02008301789151, -0.0087249648952, 0.98818840483709, 0.15299526637481]], ['Frame of Reference UID', '1.3.12.2.1107.5.2.33.37557.1.20210918125148781.0.0.0'], ['Position Reference Indicator', ''], ['Slice Location', '-31.777824626422'], ['Number of Study Related Instances', '513'], ['Samples per Pixel', 1], ['Photometric Interpretation', 'MONOCHROME2'], ['Rows', 256], ['Columns', 192], ['Pixel Spacing', [1, 1]], ['Bits Allocated', 16], ['Bits Stored', 12], ['High Bit', 11], ['Pixel Representation', 0], ['Smallest Image Pixel Value', 0], ['Largest Image Pixel Value', 328], ['Window Center', '170.0'], ['Window Width', '404.0'], ['Rescale Intercept', '0.0'], ['Rescale Slope', '1.0'], ['Window Center & Width Explanation', 'Selección del usuario'], ['VOI LUT Function', 'LINEAR'], ['Private Creator', 'SIEMENS CSA HEADER'], ['Private Creator', 'SIEMENS MEDCOM HEADER2'], ['Requesting Physician', 'RUIZ^ESTEFANIA'], ['Requesting Service', 'Servicio Andaluz de Salud'], ['Requested Procedure Description', 'RM sin Contraste I.V. de Cráneo'], ['Requested Procedure Code Sequence', 1], ['Performed Procedure Step Start Date', '20210918'], ['Performed Procedure Step Start Time', '125148.718000'], ['Performed Procedure Step ID', '124599971'], ['Performed Procedure Step Description', 'RM sin Contraste I.V. de Cráneo'], ['Request Attributes Sequence', 1], ['Private Creator', 'SIEMENS MR HEADER'], ['Icon Image Sequence', 1], ['Private Creator', 'ELSCINT1'], ['[Unknown]', 'N'], ['[Presentation Relative Center]', [0, 0]], ['[Presentation Relative Part]', [1, 1]], ['[Presentation Horizontal Invert]', 'N'], ['[Unknown]', 'N'], ['Private Creator', 'ELSCINT1'], ['[number of images in series]', 144], ['[Tamar Software Version]', '12.1.5.5'], ['[Tamar Study Status]', 'UNREAD'], ['Private tag data', 'NOT ASSIGNED'], ['[Unknown]', '1'], ['Private tag data', 'AXIAL'], ['[Tamar Site Id]', 10009], ['Private tag data', 'Y'], ['Private tag data', 'S'], ['Private tag data', '20210918131109.000000'], ['Private tag data', 'N'], ['[Unknown]', ['2D', 'PALETTE']], ['[Tamar Translate Flags]', 3], ['Private tag data', '00021563706'], ['Private tag data', '02009'], ['Private tag data', '10009'], ['Private tag data', '9022727127041'], ['Private Creator', 'ELSCINT1'], ['[Tamar Exe Software Version]', '12.1.5.5'], ['[Tamar Study Has Sticky Note]', 'N'], ['Private tag data', 'RMCRA-SCIV'], ['Private tag data', 'Pediatría'], ['Private tag data', 'SC'], ['Private tag data', 'ZUAZO'], ['Private tag data', 'MIREN AMAYA'], ['Private tag data', '7044233'], ['Private tag data', 'SG10009MR01'], ['[Unknown]', [0, 0, 0, 0, 1, 0, 0]], ['Private tag data', 'T1'], ['[Unknown]', 'N']]#, ['[Unknown]', <Sequence, length 8>], ['Private tag data', 'N'], ['[Unknown]', 'N'], ['[Unknown]', 'N'], ['Private tag data', 0], ['Private tag data', 'N'], ['Private tag data', 'N']]
        #txt = ''
        #for el in a:
            #txt += '<p style="color:#0088ff">{}'.format(el[0])  + "  "+'<span style="color:#ff0026;"></span>{}</p><br>'.format(el[1])
            #txt = '<p style="font-size:14px; color:#538b01; font-weight:bold; font-style:italic;">Enter the competition by <span style="font-size:14px; color:#ff00; font-weight:bold; font-style:italic;"</span>summer</p>'
        #    txt += '<p style="font-size:14px; color:#538b01; font-weight:bold; font-style:italic;"> {} <span style="color:#FF0000">{}</span> </p>'.format(el[0], el[1])
        #self.label_mri.setHtml(txt)
        self.label_mri.setReadOnly(True)
        self.gridLayout_mri.addWidget(self.label_mri,1,1,1,1)


        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        self.gridLayout_eco = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_eco.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_eco.setObjectName("gridLayout_6")
        self.label_eco = QtWidgets.QTextEdit(self.tab)

        self.label_eco.setReadOnly(True)
        self.gridLayout_eco.addWidget(self.label_eco,0,0,0,0)

        self.tabWidget.addTab(self.tab_2, "")
        self.hbox = QtWidgets.QHBoxLayout()
        self.buttonBox = QtWidgets.QPushButton()
        self.buttonBox.clicked.connect(self._next_find)

        self.buttonBox.setGeometry(QtCore.QRect(610, 290, 166, 25))
        #self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        #self.gridLayout_6.addWidget(self.buttonBox, 4, 4, 1, 1)
        #self.gridLayout_4.addWidget(self.buttonBox, 4, 4, 1, 1)


        #self.grid_main.addLayout(self.gridLayout_6)
        #self.grid_main.addWidget(self.tab_2)

        #
        self.label_search = QtWidgets.QLineEdit()
        # spacer = QtWidgets.QSpacerItem(1120, 4, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.label_search.textChanged.connect(self.search_for_text)
        self.hbox.addWidget(self.label_search)
        self.hbox.addWidget(self.buttonBox)
        # self.hbox.addItem(spacer)

        self.grid_main.addWidget(self.tabWidget)
        self.grid_main.addLayout(self.hbox, 0)
        #self.gridLayout_mri.addWidget(self.label_search)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def set_tag_value(self, a, ind):
        txt = ''
        for el in a:
            # txt += '<p style="color:#0088ff">{}'.format(el[0])  + "  "+'<span style="color:#ff0026;"></span>{}</p><br>'.format(el[1])
            # txt = '<p style="font-size:14px; color:#538b01; font-weight:bold; font-style:italic;">Enter the competition by <span style="font-size:14px; color:#ff00; font-weight:bold; font-style:italic;"</span>summer</p>'
            txt += '<p style="font-size:14px; color:#538b01; font-weight:bold; font-style:italic;"> {} <span style="color:#FF0000">{}</span> </p>'.format(
                str(el[0]), str(el[1]))
        if ind ==0:
            self.label_mri.setHtml(txt)
        else:
            self.label_eco.setHtml(txt)

    def _next_find(self):
        if not hasattr(self, '_next_el'):
            return
        self._next_el += 1
        ind = self.tabWidget.currentIndex()
        if ind == 0:
            label = self.label_mri
        else:
            label = self.label_eco
        txt2 = self.label_search.text()
        if txt2 != '':
            try:
                txt = label.toPlainText().split('\n')
                founds = [i for i, el in enumerate(txt) if txt2.lower() in el.lower()]
                if self._next_el>len(founds):
                    self._next_el = 0
                cursor = QtGui.QTextCursor(label.document().findBlockByNumber(founds[self._next_el]))
                # format = QtGui.QTextCharFormat()
                # format.setBackground(QtGui.QBrush(QtGui.QColor("blue")))
                # cursor.setCharFormat(format)
                cursor.movePosition(QtGui.QTextCursor.EndOfLine, 1)

                label.setTextCursor(cursor)

            except:
                pass


    def search_for_text(self):
        ind =  self.tabWidget.currentIndex()
        if ind == 0:
            label = self.label_mri
        else:
            label = self.label_eco
        txt2 = self.label_search.text()
        if txt2!='':
            try:
                txt = label.toPlainText().split('\n')
                found= [i for i, el in enumerate(txt) if txt2.lower() in el.lower()]
                self._next_el = 0
                cursor = QtGui.QTextCursor(label.document().findBlockByNumber(found[0]))
                #format = QtGui.QTextCharFormat()
                #format.setBackground(QtGui.QBrush(QtGui.QColor("blue")))
                #cursor.setCharFormat(format)
                cursor.movePosition(QtGui.QTextCursor.EndOfLine, 1)

                label.setTextCursor(cursor)

            except:
                pass
        else:
            cursor = QtGui.QTextCursor(label.document())
            cursor.movePosition(QtGui.QTextCursor.Start, 0)

            label.setTextCursor(cursor)



    def UpdateName(self, a, b):
        _translate = QtCore.QCoreApplication.translate
        if a is not None:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", a))
        if b is not None:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", b))

    def updata_name_iminfo(self, name, index):
        _translate = QtCore.QCoreApplication.translate
        if index==0:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "MRI".format(name)))
        else:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "{}".format(name)))
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Image Information"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "MRI"))

        self.buttonBox.setText(_translate("Dialog", "OK"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "US"))


def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = iminfo0()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()