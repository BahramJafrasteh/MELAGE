from PyQt5.QtCore import Qt, QPropertyAnimation, pyqtProperty, QEasingCurve, QRectF, QPointF
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen


class AnimatedToggle(QCheckBox):
    """
    A robust Toggle Switch inheriting from QCheckBox.
    Replaces qtwidgets.AnimatedToggle with a native PyQt implementation.
    """

    def __init__(self, parent=None, width=50, height=28):
        super().__init__(parent)

        # 1. UI Setup
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)

        # Remove default text/indicator of QCheckBox
        self.setContentsMargins(0, 0, 0, 0)

        # 2. Colors (Customizable)
        self._track_color_active = QColor("#007BFF")  # Blue (Bootstrap primary)
        self._track_color_inactive = QColor("#BDC3C7")  # Gray
        self._track_color_disabled = QColor("#E0E0E0")

        self._thumb_color = QColor("white")
        self._text_color = QColor("black")

        # 3. Animation Logic
        self._handle_position = 0.0  # 0.0 (Left/Off) to 1.0 (Right/On)

        self._animation = QPropertyAnimation(self, b"handle_position", self)
        self._animation.setDuration(200)  # Speed in ms
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Connect state changes to animation
        self.stateChanged.connect(self._setup_animation)

        # Sync initial state
        if self.isChecked():
            self._handle_position = 1.0

    # --- Property for Animation ---
    @pyqtProperty(float)
    def handle_position(self):
        return self._handle_position

    @handle_position.setter
    def handle_position(self, pos):
        self._handle_position = pos
        self.update()  # Trigger repaint

    def _setup_animation(self, value):
        """Starts animation when checked state changes."""
        self._animation.stop()
        if value == Qt.Checked:
            self._animation.setEndValue(1.0)
        else:
            self._animation.setEndValue(0.0)
        self._animation.start()

    def hitButton(self, pos):
        """Make the entire widget clickable."""
        return self.contentsRect().contains(pos)

    def paintEvent(self, e):
        """Custom painting of Track and Thumb."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # 1. Geometry Calculation
        rect = self.rect()
        track_height = rect.height()
        thumb_radius = track_height / 2.0 - 2  # 2px padding

        # 2. Determine Colors
        if not self.isEnabled():
            track_color = self._track_color_disabled
            opacity = 0.6
        elif self.isChecked() or self._handle_position > 0.5:
            track_color = self._track_color_active
            opacity = 1.0
        else:
            track_color = self._track_color_inactive
            opacity = 1.0

        p.setOpacity(opacity)
        p.setPen(Qt.NoPen)

        # 3. Draw Track (Rounded Rectangle)
        p.setBrush(QBrush(track_color))
        p.drawRoundedRect(0, 0, rect.width(), rect.height(), track_height / 2, track_height / 2)

        # 4. Draw Thumb (Circle)
        # Calculate current X position based on animation property (0.0 to 1.0)
        # Min X = radius + padding
        # Max X = width - radius - padding
        padding = 2
        thumb_radius = (rect.height() - 2 * padding) / 2

        min_x = padding + thumb_radius
        max_x = rect.width() - padding - thumb_radius

        curr_x = min_x + (max_x - min_x) * self._handle_position
        curr_y = rect.height() / 2

        p.setBrush(QBrush(self._thumb_color))
        p.drawEllipse(QPointF(curr_x, curr_y), thumb_radius, thumb_radius)

        p.end()