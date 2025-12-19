from PyQt5 import QtWidgets, QtCore, QtGui


class VSCodeSidebar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Activity Bar (Icons)
        self.activity_bar = QtWidgets.QWidget()
        self.activity_bar.setMinimumWidth(50)
        #self.activity_bar.setStyleSheet("background-color: #333333;")

        self.bar_layout = QtWidgets.QVBoxLayout(self.activity_bar)
        self.bar_layout.setContentsMargins(5, 10, 5, 10)
        self.bar_layout.setSpacing(10)
        self.bar_layout.addStretch()

        # 2. Right Panel (Header + Content)
        self.right_panel = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        # Header Label
        self.header_label = QtWidgets.QLabel("")

        self.header_label.setStyleSheet("""
            QLabel {
                background-color: #0b0b40; 
                color: #f6f6f6; 
                font-weight: bold; 
                padding: 8px 15px; 
                font-size: 11px;
                border-bottom: 1px solid #3E3E42;
                text-transform: uppercase;
            }
        """)
        self.right_layout.addWidget(self.header_label)

        # Stack Content
        self.stack = QtWidgets.QStackedWidget()
        #self.stack.setStyleSheet("background-color: #252526;")
        self.right_layout.addWidget(self.stack)

        self.main_layout.addWidget(self.activity_bar)
        self.main_layout.addWidget(self.right_panel)

        self.buttons = []
        self.page_titles = {}

    def add_tab(self, widget, icon_path="", tooltip=""):
        index = self.stack.addWidget(widget)
        self.page_titles[index] = tooltip

        btn = QtWidgets.QToolButton()
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        btn.setAutoExclusive(False)  # Must be False to allow manual collapsing
        btn.setFixedSize(40, 40)

        if icon_path and QtCore.QFile.exists(icon_path):
            btn.setIcon(QtGui.QIcon(icon_path))
            btn.setIconSize(QtCore.QSize(45, 45))
        else:
            text = tooltip[:2].upper() if tooltip else "??"
            btn.setText(text)
            btn.setStyleSheet("font-weight: bold; color: #cccccc;")

        btn.setStyleSheet(btn.styleSheet() + """
            QToolButton { border: none; border-radius: 5px; padding: 5px; }
            QToolButton:hover { background-color: #3e3e42; }
            QToolButton:checked { background-color: #505050; border-left: 3px solid #007acc; }
        """)

        # Connect Click
        btn.clicked.connect(lambda checked, idx=index: self.toggle_page(idx))

        self.bar_layout.insertWidget(len(self.buttons), btn)
        self.buttons.append(btn)

        if len(self.buttons) == 1:
            self.toggle_page(0)

    def toggle_page(self, index):
        """
        Handles switching pages AND collapsing logic.
        """
        # Check current state
        is_current_page = (self.stack.currentIndex() == index)
        is_visible = self.right_panel.isVisible()

        # --- LOGIC: COLLAPSE ---
        if is_current_page and is_visible:
            self.right_panel.setVisible(False)
            self.buttons[index].setChecked(False)

            # CRITICAL FIX: Force the widget to snap to 50px width
            self.setFixedWidth(50)
            return

        # --- LOGIC: EXPAND ---
        # 1. Unlock the width constraints so it can grow
        self.setMinimumWidth(50)  # Reset to default min
        self.setMaximumWidth(16777215)  # Reset to QWIDGETSIZE_MAX (Unlimited)

        # 2. Show panel
        self.right_panel.setVisible(True)
        self.stack.setCurrentIndex(index)

        # 3. Update Header
        title_text = self.page_titles.get(index, "PANEL")
        self.header_label.setText(title_text.upper())

        # 4. Update Buttons
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)

        # Optional: Restore a reasonable width if it was just 50px
        if self.width() <= 55:
            self.resize(300, self.height())  # Pop out to a default width


class CollapsibleBox(QtWidgets.QWidget):
    # Custom signal to notify when this box is clicked
    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title="", content_widget=None, parent=None):
        super().__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(text=title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                text-align: left;
                background-color: #333333;
                color: #cccccc;
                font-weight: bold;
                padding: 4px;
            }
            QToolButton:hover { background-color: #3e3e42; }
            QToolButton:checked { /* Expanded state */ }
        """)

        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)

        # Connect internal click to internal handler
        self.toggle_button.clicked.connect(self.on_pressed)

        self.content_area = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        if content_widget:
            self.content_layout.addWidget(content_widget)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)

        # Initialize state (Hidden by default)
        self.content_area.setVisible(False)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        self.content_area.setVisible(checked)

        # Emit signal so the parent knows we changed state
        self.toggled.emit(checked)

    def collapse(self):
        """Force close"""
        if self.toggle_button.isChecked():
            self.toggle_button.setChecked(False)
            self.on_pressed()

    def expand(self):
        """Force open"""
        if not self.toggle_button.isChecked():
            self.toggle_button.setChecked(True)
            self.on_pressed()