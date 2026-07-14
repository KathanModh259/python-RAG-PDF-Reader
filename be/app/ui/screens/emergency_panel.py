from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.domain.citizen_protection.emergency import EMERGENCY_HELPLINES


class EmergencyPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("emergencyPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMaximumWidth(200)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("EMERGENCY")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        style = "color: #ffffff; background-color: #c0392b; padding: 6px; border-radius: 4px;"
        title.setStyleSheet(style)
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(4)

        for name, number, description in EMERGENCY_HELPLINES:
            btn = QPushButton(f"{name}\n{number}")
            btn.setMinimumHeight(60)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(description)
            if "Police" in name or "112" in number:
                btn.setStyleSheet(
                    "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
                    "border: none; border-radius: 4px; padding: 8px; text-align: center; }"
                    "QPushButton:hover { background-color: #c0392b; }"
                )
            elif "Women" in name or "181" in number:
                btn.setStyleSheet(
                    "QPushButton { background-color: #e91e63; color: white; font-weight: bold; "
                    "border: none; border-radius: 4px; padding: 8px; text-align: center; }"
                    "QPushButton:hover { background-color: #c2185b; }"
                )
            elif "Legal Aid" in name or "15100" in number:
                btn.setStyleSheet(
                    "QPushButton { background-color: #27ae60; color: white; font-weight: bold; "
                    "border: none; border-radius: 4px; padding: 8px; text-align: center; }"
                    "QPushButton:hover { background-color: #219a52; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background-color: #34495e; color: white; font-weight: bold; "
                    "border: none; border-radius: 4px; padding: 8px; text-align: center; }"
                    "QPushButton:hover { background-color: #2c3e50; }"
                )
            btn.clicked.connect(lambda checked, n=number: self._copy_helpline(n))
            scroll_layout.addWidget(btn)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self._style()

    def _copy_helpline(self, number: str):
        from PyQt6.QtCore import QMimeData
        from PyQt6.QtWidgets import QApplication

        mime = QMimeData()
        mime.setText(number)
        QApplication.clipboard().setMimeData(mime)
        main_win = self.window()
        if main_win and hasattr(main_win, "statusBar"):
            main_win.statusBar().showMessage(f"Copied {number} to clipboard", 3000)

    def _style(self):
        self.setStyleSheet("""
            EmergencyPanel { background-color: #1a1a2e; border: 1px solid #c0392b; border-radius: 6px; }
            QScrollArea { background: transparent; }
            QWidget { background: transparent; }
            QPushButton { font-size: 11px; }
        """)
