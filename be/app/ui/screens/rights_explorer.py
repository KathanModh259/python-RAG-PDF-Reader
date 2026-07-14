from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.domain.rights.data import RIGHTS_DB

CATEGORY_LABELS: dict[str, str] = {
    "legal_aid": "Legal Aid",
    "criminal": "Criminal / Police",
    "governance": "Governance / RTI",
    "health": "Health",
    "employment": "Employment",
    "family": "Family / Marriage",
    "consumer": "Consumer",
    "property": "Property / Rent",
    "fundamental": "Fundamental Rights",
    "education": "Education",
}


class RightsExplorerPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rightsExplorer")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Know Your Rights")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        title.setStyleSheet(
            "color: #ffffff; background-color: #2c3e50; padding: 6px; border-radius: 4px;"
        )
        layout.addWidget(title)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search rights...")
        self.search_input.textChanged.connect(self._filter_rights)
        layout.addWidget(self.search_input)

        self.category_buttons = QHBoxLayout()
        self.category_buttons.setSpacing(3)
        self._all_btn = QPushButton("All")
        self._all_btn.setCheckable(True)
        self._all_btn.setChecked(True)
        self._all_btn.clicked.connect(lambda: self._filter_by_category(None))
        self.category_buttons.addWidget(self._all_btn)

        self._cat_buttons: dict[str, QPushButton] = {}
        for cat_key in sorted(CATEGORY_LABELS.keys()):
            btn = QPushButton(CATEGORY_LABELS[cat_key][:6])
            btn.setToolTip(CATEGORY_LABELS[cat_key])
            btn.setCheckable(True)
            btn.setMaximumWidth(70)
            btn.clicked.connect(lambda checked, c=cat_key: self._filter_by_category(c))
            self.category_buttons.addWidget(btn)
            self._cat_buttons[cat_key] = btn
        layout.addLayout(self.category_buttons)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.rights_list = QListWidget()
        self.rights_list.setAlternatingRowColors(True)
        self.rights_list.currentRowChanged.connect(self._show_right_detail)
        scroll.setWidget(self.rights_list)

        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(200)

        layout.addWidget(scroll)
        layout.addWidget(self.detail_view)

        self._all_rights = list(RIGHTS_DB)
        self._populate_list(self._all_rights)
        self._style()

    def _populate_list(self, rights: list[dict]):
        self.rights_list.clear()
        for r in rights:
            item = QListWidgetItem(r["right"])
            item.setData(Qt.ItemDataRole.UserRole, r)
            self.rights_list.addItem(item)

    def _filter_rights(self, text: str):
        text_lower = text.lower()
        if not text_lower:
            self._filter_by_category(self._current_category)
            return
        filtered = [
            r
            for r in self._all_rights
            if text_lower in r["right"].lower()
            or text_lower in r["plain"].lower()
            or text_lower in r.get("category", "").lower()
        ]
        self._populate_list(filtered)

    _current_category: str | None = None

    def _filter_by_category(self, category: str | None):
        self._current_category = category
        for cat, btn in self._cat_buttons.items():
            btn.setChecked(cat == category)
        self._all_btn.setChecked(category is None)

        if category is None:
            self._populate_list(self._all_rights)
        else:
            filtered = [r for r in self._all_rights if r.get("category") == category]
            self._populate_list(filtered)

        search_text = self.search_input.text().strip()
        if search_text:
            self._filter_rights(search_text)

    def _show_right_detail(self, row: int):
        if row < 0:
            self.detail_view.clear()
            return
        item = self.rights_list.item(row)
        if not item:
            return
        r = item.data(Qt.ItemDataRole.UserRole)
        if not r:
            return
        html = f"""
        <h3>{r['right']}</h3>
        <p><b>Provision:</b> {r['provision']}</p>
        <p>{r['plain']}</p>
        <p><i>Category: {CATEGORY_LABELS.get(r.get('category', ''), r.get('category', ''))}</i></p>
        """
        self.detail_view.setHtml(html)

    def _style(self):
        self.setStyleSheet("""
            RightsExplorerPanel { background-color: #f5f6fa; border: 1px solid #dcdde1; border-radius: 6px; }
            QListWidget { background: white; border: 1px solid #dcdde1; border-radius: 4px; }
            QListWidget::item:selected { background-color: #3498db; color: white; }
            QTextEdit { background: white; border: 1px solid #dcdde1; border-radius: 4px; font-size: 12px; }
            QLineEdit { border: 1px solid #bdc3c7; border-radius: 4px; padding: 4px 8px; }
            QPushButton { border: 1px solid #bdc3c7; border-radius: 4px; padding: 4px 6px; font-size: 10px; }
            QPushButton:checked { background-color: #3498db; color: white; border-color: #2980b9; }
        """)
