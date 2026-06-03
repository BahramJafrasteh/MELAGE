import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class RobustCheckTreeDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal/Slot Checkbox Demo")
        self.setGeometry(100, 100, 400, 300)

        # Use the standard QTreeView, no subclassing needed for this
        self.tree_view = QtWidgets.QTreeView()
        self.model = QtGui.QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.model.setHorizontalHeaderLabels(['Status', 'ID'])
        self.setCentralWidget(self.tree_view)

        self.populate_data()

        # 1. Connect the view's 'clicked' signal to our logic function
        self.tree_view.clicked.connect(self.on_row_clicked)

    def populate_data(self):
        """Adds items with flags set to disable the default toggle behavior."""
        root = self.model.invisibleRootItem()
        for i in range(5):
            item_status = QtGui.QStandardItem(f"{i + 1}")
            item_status.setCheckable(True)

            # 2. Disable the automatic toggle to give our slot full control
            flags = item_status.flags() & QtCore.Qt.ItemIsUserCheckable
            item_status.setFlags(flags)

            root.appendRow([item_status, QtGui.QStandardItem(f"ID-{100 + i}")])

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_row_clicked(self, index):
        """
        This slot is triggered by a click and safely toggles the item's state.
        """
        # 3. Get the item in the first column, which has the checkbox
        item = self.model.itemFromIndex(index.sibling(index.row(), 0))

        if item and item.isCheckable():
            # 4. Manually toggle the state
            new_state = QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
            item.setCheckState(new_state)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = RobustCheckTreeDemo()
    window.show()
    sys.exit(app.exec_())