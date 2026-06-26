from PyQt5 import QtWidgets, QtCore, QtGui

# ---------------------------------------------------------------------------
# VS Code-style sidebar
# ---------------------------------------------------------------------------
# Layout mirrors VS Code exactly:
#   ┌──────┬──────────────────────────────────────┐
#   │      │ PANEL TITLE                          │
#   │ icon │──────────────────────────────────────│
#   │ icon │                                      │
#   │ icon │   panel content (scrollable)         │
#   │  …   │                                      │
#   └──────┴──────────────────────────────────────┘
#
# Activity bar: 48 px wide, dark (#333333), 48×48 buttons, 24 px icons.
# Active button: 2 px left accent (#007ACC), slightly lighter background.
# Hover:         subtle white overlay.
# Right panel:   fills all remaining width; expands/shrinks freely.
# ---------------------------------------------------------------------------

_ACTIVITY_BG  = "#333333"
_PANEL_BG     = "#252526"
_ACCENT       = "#007ACC"
_TEXT_ACTIVE  = "#CCCCCC"
_TEXT_MUTED   = "#858585"
_BORDER       = "#3C3C3C"


class VSCodeSidebar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ── 1. Activity Bar ──────────────────────────────────────────────
        self.activity_bar = QtWidgets.QWidget()
        self.activity_bar.setFixedWidth(48)
        self.activity_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.activity_bar.setStyleSheet(
            f"background-color: {_ACTIVITY_BG};"
            f"border-right: 1px solid {_BORDER};"
        )

        self.bar_layout = QtWidgets.QVBoxLayout(self.activity_bar)
        self.bar_layout.setContentsMargins(0, 4, 0, 4)
        self.bar_layout.setSpacing(2)
        self.bar_layout.addStretch()

        # ── 2. Right Panel ───────────────────────────────────────────────
        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        # Panel header — VS Code style: uppercase, muted, thin bottom border
        self.header_label = QtWidgets.QLabel("")
        self.header_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.header_label.setStyleSheet(f"""
            QLabel {{
                background-color: {_PANEL_BG};
                color: {_TEXT_ACTIVE};
                font-weight: bold;
                font-size: 11px;
                padding: 7px 12px;
                border-bottom: 1px solid {_BORDER};
                letter-spacing: 1px;
            }}
        """)
        self.right_layout.addWidget(self.header_label)

        # Stacked content — stretch=1 fills all remaining vertical space
        self.stack = QtWidgets.QStackedWidget()
        self.stack.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.right_layout.addWidget(self.stack, 1)

        # Activity bar (stretch=0) | Right panel (stretch=1)
        self.main_layout.addWidget(self.activity_bar, 0)
        self.main_layout.addWidget(self.right_panel, 1)

        self.buttons: list[QtWidgets.QToolButton] = []
        self.page_titles: dict[int, str] = {}

    # ── Button factory ───────────────────────────────────────────────────

    _BTN_STYLE = f"""
        QToolButton {{
            border: none;
            border-left: 2px solid transparent;
            background: transparent;
            color: {_TEXT_MUTED};
            font-weight: bold;
            font-size: 9px;
            padding: 0px;
        }}
        QToolButton:hover {{
            background-color: rgba(255,255,255, 0.08);
            color: {_TEXT_ACTIVE};
        }}
        QToolButton:checked {{
            border-left: 2px solid {_ACCENT};
            background-color: rgba(255,255,255, 0.05);
            color: #ffffff;
        }}
    """

    def add_tab(self, widget, icon=None, tooltip=""):
        """
        Add a panel to the sidebar.

        Parameters
        ----------
        widget  : QWidget to show in the content area
        icon    : file-path str, QIcon, or None (falls back to two-letter initials)
        tooltip : panel title / hover text
        """
        index = self.stack.addWidget(widget)
        self.page_titles[index] = tooltip

        btn = QtWidgets.QToolButton()
        btn.setCheckable(True)
        btn.setAutoExclusive(False)
        btn.setToolTip(tooltip)
        btn.setFixedSize(48, 48)          # full activity-bar width, square
        btn.setStyleSheet(self._BTN_STYLE)

        q_icon: QtGui.QIcon | None = None
        if isinstance(icon, QtGui.QIcon):
            q_icon = icon
        elif isinstance(icon, str) and icon and QtCore.QFile.exists(icon):
            q_icon = QtGui.QIcon(icon)

        if q_icon is not None:
            btn.setIcon(q_icon)
            btn.setIconSize(QtCore.QSize(26, 26))   # 26 px icon in 48 px button
        else:
            initials = tooltip[:2].upper() if tooltip else "??"
            btn.setText(initials)

        btn.clicked.connect(lambda _checked, idx=index: self.toggle_page(idx))
        self.bar_layout.insertWidget(len(self.buttons), btn)
        self.buttons.append(btn)

        if len(self.buttons) == 1:
            self.toggle_page(0)

    # ── Toggle logic ─────────────────────────────────────────────────────

    def switch_to_tab(self, title: str) -> bool:
        """Expand and switch to the panel whose tooltip/title matches *title* (case-insensitive).
        Returns True if found, False otherwise."""
        target = title.lower()
        for index, t in self.page_titles.items():
            if t.lower() == target:
                # Only switch if this isn't already the active, visible tab
                if not (self.right_panel.isVisible() and self.stack.currentIndex() == index):
                    self.toggle_page(index)
                return True
        return False

    def toggle_page(self, index: int):
        """Switch to panel *index*, or collapse if it is already active & visible."""
        is_active  = (self.stack.currentIndex() == index)
        is_visible = self.right_panel.isVisible()

        # ── COLLAPSE ────────────────────────────────────────────────────
        if is_active and is_visible:
            self.right_panel.setVisible(False)
            self.buttons[index].setChecked(False)
            self.setFixedWidth(48)          # shrink to icon strip only
            return

        # ── EXPAND ──────────────────────────────────────────────────────
        # Release any prior fixed-width constraint so the dock can be resized.
        self.setMinimumWidth(48)
        self.setMaximumWidth(16_777_215)   # Qt "no limit"

        self.right_panel.setVisible(True)
        self.stack.setCurrentIndex(index)
        self.header_label.setText(self.page_titles.get(index, "PANEL").upper())

        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)

        # On first open from collapsed state: pop out to a comfortable default.
        # User can freely drag the dock wider or narrower afterward.
        if self.width() <= 50:
            screen_w  = QtWidgets.QApplication.primaryScreen().availableGeometry().width()
            default_w = min(420, max(280, int(screen_w * 0.28)))
            self.resize(default_w, self.height())


# ---------------------------------------------------------------------------
# CollapsibleBox — unchanged, kept for other uses in the codebase
# ---------------------------------------------------------------------------

class CollapsibleBox(QtWidgets.QWidget):
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
            QToolButton:checked { }
        """)
        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
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

        self.content_area.setVisible(False)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        self.content_area.setVisible(checked)
        self.toggled.emit(checked)

    def collapse(self):
        if self.toggle_button.isChecked():
            self.toggle_button.setChecked(False)
            self.on_pressed()

    def expand(self):
        if not self.toggle_button.isChecked():
            self.toggle_button.setChecked(True)
            self.on_pressed()
