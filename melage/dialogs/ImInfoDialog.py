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
        self.grid_main.setContentsMargins(10, 10, 10, 10)
        self.grid_main.setObjectName("gridLayout")

        # Tab widget
        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.setObjectName("tabWidget")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                           QtWidgets.QSizePolicy.MinimumExpanding)
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(100, 100))

        # MRI Tab






        # Echo Tab
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab_2")

        self.meta_table_eco = QtWidgets.QTableWidget(self.tab)
        self.meta_table_eco.setColumnCount(2)
        self.meta_table_eco.setHorizontalHeaderLabels(["Tag", "Value"])
        self.meta_table_eco.horizontalHeader().setStretchLastSection(True)
        self.meta_table_eco.verticalHeader().setVisible(False)
        self.meta_table_eco.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.meta_table_eco.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.meta_table_eco.setShowGrid(True)



        self.gridLayout_eco = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_eco.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_eco.setObjectName("gridLayout_eco")
        self.gridLayout_eco.addWidget(self.meta_table_eco, 1, 0, 1, 1)


        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab")
        self.meta_table_mri = QtWidgets.QTableWidget(self.tab_2)
        self.meta_table_mri.setColumnCount(2)
        self.meta_table_mri.setHorizontalHeaderLabels(["Tag", "Value"])
        self.meta_table_mri.horizontalHeader().setStretchLastSection(True)
        self.meta_table_mri.verticalHeader().setVisible(False)
        self.meta_table_mri.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.meta_table_mri.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.meta_table_mri.setShowGrid(True)

        self.gridLayout_mri = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_mri.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_mri.setObjectName("gridLayout_mri")
        self.gridLayout_mri.addWidget(self.meta_table_mri, 1, 0, 1, 1)

        self.tabWidget.addTab(self.tab, "File 2")
        self.tabWidget.addTab(self.tab_2, "File 1")

        for table in [self.meta_table_mri, self.meta_table_eco]:
            table.setAlternatingRowColors(False)
            table.setStyleSheet("QTableWidget { background-color: white; }")
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)  # or SelectRows
            table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)  # or ExtendedSelection
        # Search and Next button layout
        self.hbox = QtWidgets.QHBoxLayout()

        self.label_search = QtWidgets.QLineEdit()
        self.label_search.setPlaceholderText("Search metadata...")
        self.label_search.textChanged.connect(self.search_for_text)
        self.hbox.addWidget(self.label_search)

        self.buttonBox = QtWidgets.QPushButton("Next")
        self.buttonBox.clicked.connect(self._next_find)
        self.hbox.addWidget(self.buttonBox)

        self.grid_main.addWidget(self.tabWidget)
        self.grid_main.addLayout(self.hbox)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)



    def set_tag_value(self, aa, ind):
        metadata = aa.im_metadata
        table = self.meta_table_mri if ind == 1 else self.meta_table_eco


        allowed = ['filename','affine', 'pixdim', 'dim']
        metadata['FileName'] = self.tabWidget.tabText(ind)
        table.setRowCount(len(allowed))
        row = 0
        for key in sorted(metadata.keys()):
            value = metadata[key]
            value_str = str(value)
            if key.lower() not in allowed:
                continue
            if value_str=='pixdim':
                value_str = 'Spacing'
            key_item = QtWidgets.QTableWidgetItem(key)
            val_item = QtWidgets.QTableWidgetItem(value_str)

            # Tooltips for clarity
            key_item.setToolTip(key)
            val_item.setToolTip(value_str)

            # Optional styling
            key_item.setForeground(QtGui.QColor("#2e3d49"))
            key_item.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
            val_item.setForeground(QtGui.QColor("#FF0000"))
            val_item.setFont(QtGui.QFont("Arial", 10))

            table.setItem(row, 0, key_item)
            table.setItem(row, 1, val_item)
            row += 1

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

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
        search_text = self.label_search.text().strip().lower()

        # Select the current table based on the active tab
        if self.tabWidget.currentIndex() == 0:
            table = self.meta_table_mri
        else:
            table = self.meta_table_eco

        # Reset all rows' background color
        for row in range(table.rowCount()):
            for col in range(2):
                item = table.item(row, col)
                if item:
                    item.setBackground(QtGui.QBrush(QtCore.Qt.white))

        if not search_text:
            return

        # Highlight matching rows
        for row in range(table.rowCount()):
            tag_item = table.item(row, 0)
            val_item = table.item(row, 1)

            tag_text = tag_item.text().lower() if tag_item else ""
            val_text = val_item.text().lower() if val_item else ""

            if search_text in tag_text or search_text in val_text:
                for col in range(2):
                    item = table.item(row, col)
                    if item:
                        item.setBackground(QtGui.QBrush(QtGui.QColor("#ffff99")))  # Light yellow
                table.scrollToItem(tag_item or val_item, QtWidgets.QAbstractItemView.PositionAtCenter)



    def UpdateName(self, a, b):
        _translate = QtCore.QCoreApplication.translate
        if a is not None:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", a))
        if b is not None:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", b))

    def updata_name_iminfo(self, name, index):
        _translate = QtCore.QCoreApplication.translate
        if index==0:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "{}".format(name)))
        else:
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "{}".format(name)))
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Image Information"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "View 2"))

        self.buttonBox.setText(_translate("Dialog", "OK"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "View 1"))
    def setmri(self, lst):
        try:
            #_eco = ['affine_eco_l', 'dt_eco_l', 'fn_eco_l', 'frmt_eco_l', 'dim_eco_l', 'spc_eco_l', 'voxel_offset_eco_l']
            #_mri = ['affine_mri_l', 'dt_mri_l', 'fn_mri_l', 'frmt_mri_l', 'dim_mri_l', 'spc_mri_l', 'voxel_offset_mri_l']
            header, affine, filename , format = lst
            self.affine_mri_l.setText(''.join(str(affine).split('\n')))

            self.fn_mri_l.setText(filename.split('.')[0])
            self.frmt_mri_l.setText(format)

            self.dt_mri_l.setText(header.get_data_dtype().name)
            spacing = ', '.join([str(el) for el in header['pixdim'][1:4]])
            self.spc_mri_l.setText(spacing)
            dims = ', '.join([str(el) for el in header['dim'][1:4]])
            self.dim_mri_l.setText(dims)
            vos = str(header['vox_offset'])
            self.voxel_offset_mri_l.setText(vos)
        except:
            pass
    def seteco(self, lst):
        try:
            #_eco = ['affine_eco_l', 'dt_eco_l', 'fn_eco_l', 'frmt_eco_l', 'dim_eco_l', 'spc_eco_l', 'voxel_offset_eco_l']
            #_mri = ['affine_mri_l', 'dt_mri_l', 'fn_mri_l', 'frmt_mri_l', 'dim_mri_l', 'spc_mri_l', 'voxel_offset_mri_l']
            header, affine, filename , format = lst
            self.affine_eco_l.setText(''.join(str(affine).split('\n')))

            self.fn_eco_l.setText(filename.split('.')[0])
            self.frmt_eco_l.setText(format)

            self.dt_eco_l.setText(header.get_data_dtype().name)
            spacing = ', '.join([str(el) for el in header['pixdim'][1:4]])
            self.spc_eco_l.setText(spacing)
            dims = ', '.join([str(el) for el in header['dim'][1:4]])
            self.dim_eco_l.setText(dims)
            vos = str(header['vox_offset'])
            self.voxel_offset_eco_l.setText(vos)
        except:
            pass









def run():
    import numpy as np
    array = np.array
    int32 = np.int32
    int16 = np.int16
    uint8 = np.uint8
    float16 = np.float16
    float32 = np.float32
    nan = np.nan
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = iminfo_dialog()
    im_metadata= {'sizeof_hdr': array(348, dtype=int32),
 'data_type': array(b'', dtype='|S10'),
 'db_name': array(b'', dtype='|S18'),
 'extents': array(0, dtype=int32),
 'session_error': array(0, dtype=int16),
 'regular': array(b'', dtype='|S1'),
 'dim_info': array(0, dtype=uint8),
 'dim': array([  3, 197, 233, 189,   1,   1,   1,   1], dtype=int16),
 'intent_p1': array(0., dtype=float32),
 'intent_p2': array(0., dtype=float32),
 'intent_p3': array(0., dtype=float32),
 'intent_code': array(0, dtype=int16),
 'datatype': array(4, dtype=int16),
 'bitpix': array(16, dtype=int16),
 'slice_start': array(0, dtype=int16),
 'pixdim': array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32),
 'vox_offset': array(0., dtype=float32),
 'scl_slope': array(nan, dtype=float32),
 'scl_inter': array(nan, dtype=float32),
 'slice_end': array(0, dtype=int16),
 'slice_code': array(0, dtype=uint8),
 'xyzt_units': array(0, dtype=uint8),
 'cal_max': array(0., dtype=float32),
 'cal_min': array(0., dtype=float32),
 'slice_duration': array(0., dtype=float32),
 'toffset': array(0., dtype=float32),
 'glmax': array(0, dtype=int32),
 'glmin': array(0, dtype=int32),
 'descrip': array(b'', dtype='|S80'),
 'aux_file': array(b'', dtype='|S24'),
 'qform_code': array(0, dtype=int16),
 'sform_code': array(2, dtype=int16),
 'quatern_b': array(0., dtype=float32),
 'quatern_c': array(0., dtype=float32),
 'quatern_d': array(0., dtype=float32),
 'qoffset_x': array(-98., dtype=float32),
 'qoffset_y': array(-134., dtype=float32),
 'qoffset_z': array(-72., dtype=float32),
 'srow_x': array([  1.,   0.,  -0., -98.], dtype=float32),
 'srow_y': array([   0.,    1.,   -0., -134.], dtype=float32),
 'srow_z': array([  0.,   0.,   1., -72.], dtype=float32),
 'intent_name': array(b'', dtype='|S16'),
 'magic': array(b'n+1', dtype='|S4'),
 'Affine': array([[   1.,    0.,   -0.,  -98.],
        [   0.,    1.,   -0., -134.],
        [   0.,    0.,    1.,  -72.],
        [   0.,    0.,    0.,    1.]])}
    window.set_tag_value(im_metadata, 0)
    window.updata_name_iminfo('Eco', 0)
    window.updata_name_iminfo('MRI', 1)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()