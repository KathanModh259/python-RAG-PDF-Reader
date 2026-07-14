from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.domain.cases.tracker import CaseStatus, CaseType, Case, tracker as case_tracker


class CaseDialog(QDialog):
    def __init__(self, case: Optional[Case] = None, parent=None):
        super().__init__(parent)
        self.case = case
        self._result: Optional[Case] = None
        self._init_ui()

    def _init_ui(self):
        is_edit = self.case is not None
        self.setWindowTitle("Edit Case" if is_edit else "Add New Case")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g., Property dispute with Sharma")
        if self.case:
            self.title_input.setText(self.case.title)
        form.addRow("Title *", self.title_input)

        self.type_combo = QComboBox()
        for ct in CaseType:
            self.type_combo.addItem(ct.display(), ct.value)
        if self.case:
            self.type_combo.setCurrentIndex(self.type_combo.findData(self.case.case_type.value))
        form.addRow("Case Type", self.type_combo)

        self.status_combo = QComboBox()
        for cs in CaseStatus:
            self.status_combo.addItem(cs.display(), cs.value)
        if self.case:
            self.status_combo.setCurrentIndex(self.status_combo.findData(self.case.status.value))
        form.addRow("Status", self.status_combo)

        self.court_input = QLineEdit()
        self.court_input.setPlaceholderText("e.g., District Court, Delhi")
        if self.case:
            self.court_input.setText(self.case.court)
        form.addRow("Court", self.court_input)

        self.case_no_input = QLineEdit()
        self.case_no_input.setPlaceholderText("e.g., CS/2024/1234")
        if self.case:
            self.case_no_input.setText(self.case.case_number)
        form.addRow("Case Number", self.case_no_input)

        self.opposite_input = QLineEdit()
        self.opposite_input.setPlaceholderText("Name of the other party")
        if self.case:
            self.opposite_input.setText(self.case.opposite_party)
        form.addRow("Opposite Party", self.opposite_input)

        self.hearing_input = QLineEdit()
        self.hearing_input.setPlaceholderText("DD/MM/YYYY")
        if self.case:
            self.hearing_input.setText(self.case.next_hearing_date)
        form.addRow("Next Hearing", self.hearing_input)

        self.lawyer_input = QLineEdit()
        self.lawyer_input.setPlaceholderText("Advocate name (or 'DLSA' for free legal aid)")
        if self.case:
            self.lawyer_input.setText(self.case.assigned_lawyer)
        form.addRow("Assigned Lawyer", self.lawyer_input)

        self.description_input = QTextEdit()
        self.description_input.setMaximumHeight(80)
        self.description_input.setPlaceholderText("Brief description of the case...")
        if self.case:
            self.description_input.setText(self.case.description)
        form.addRow("Description", self.description_input)

        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(80)
        self.notes_input.setPlaceholderText("Any important notes...")
        if self.case:
            self.notes_input.setText(self.case.notes)
        form.addRow("Notes", self.notes_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        if not self.title_input.text().strip():
            self.title_input.setStyleSheet("border: 1px solid red;")
            return
        if self.case:
            self.case.title = self.title_input.text().strip()
            self.case.case_type = CaseType(self.type_combo.currentData())
            self.case.status = CaseStatus(self.status_combo.currentData())
            self.case.court = self.court_input.text().strip()
            self.case.case_number = self.case_no_input.text().strip()
            self.case.opposite_party = self.opposite_input.text().strip()
            self.case.next_hearing_date = self.hearing_input.text().strip()
            self.case.assigned_lawyer = self.lawyer_input.text().strip()
            self.case.description = self.description_input.toPlainText().strip()
            self.case.notes = self.notes_input.toPlainText().strip()
            case_tracker.update_case(self.case)
            self._result = self.case
        else:
            self._result = case_tracker.add_case(
                title=self.title_input.text().strip(),
                case_type=CaseType(self.type_combo.currentData()),
                status=CaseStatus(self.status_combo.currentData()),
                court=self.court_input.text().strip(),
                case_number=self.case_no_input.text().strip(),
                opposite_party=self.opposite_input.text().strip(),
                next_hearing_date=self.hearing_input.text().strip(),
                assigned_lawyer=self.lawyer_input.text().strip(),
                description=self.description_input.toPlainText().strip(),
                notes=self.notes_input.toPlainText().strip(),
            )
        self.accept()

    def get_result(self) -> Optional[Case]:
        return self._result


class CaseTrackerView(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("caseTrackerView")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._init_ui()
        self._refresh()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()

        title = QLabel("My Cases")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        title.setFont(title_font)
        title.setStyleSheet(
            "color: #ffffff; background-color: #2c3e50; padding: 6px; border-radius: 4px;"
        )
        toolbar.addWidget(title)

        toolbar.addStretch()

        add_btn = QPushButton("+ Add Case")
        add_btn.clicked.connect(self._add_case)
        toolbar.addWidget(add_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        toolbar.addWidget(self.refresh_btn)

        layout.addLayout(toolbar)

        filter_row = QHBoxLayout()
        self.status_filter = QComboBox()
        self.status_filter.addItem("All Statuses", "")
        for cs in CaseStatus:
            self.status_filter.addItem(cs.display(), cs.value)
        self.status_filter.currentIndexChanged.connect(self._refresh)
        filter_row.addWidget(QLabel("Filter:"))
        filter_row.addWidget(self.status_filter)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search cases...")
        self.search_input.returnPressed.connect(self._search)
        filter_row.addWidget(self.search_input)
        layout.addLayout(filter_row)

        self.case_list = QListWidget()
        self.case_list.setAlternatingRowColors(True)
        self.case_list.currentRowChanged.connect(self._show_detail)
        layout.addWidget(self.case_list)

        self.detail_group = QGroupBox("Case Details")
        detail_layout = QVBoxLayout(self.detail_group)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(180)
        detail_layout.addWidget(self.detail_text)

        detail_buttons = QHBoxLayout()
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self._edit_case)
        self.edit_btn.setEnabled(False)
        detail_buttons.addWidget(self.edit_btn)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet("color: #e74c3c;")
        self.delete_btn.clicked.connect(self._delete_case)
        self.delete_btn.setEnabled(False)
        detail_buttons.addWidget(self.delete_btn)
        detail_buttons.addStretch()
        detail_layout.addLayout(detail_buttons)

        layout.addWidget(self.detail_group)

        self._style()

    def _refresh(self):
        self.case_list.clear()
        status_val = self.status_filter.currentData()
        status = CaseStatus(status_val) if status_val else None
        cases = case_tracker.list_cases(status=status, limit=200)
        for c in cases:
            display = f"{c.title} [{c.status.display()}]"
            if c.next_hearing_date:
                display += f" - Hearing: {c.next_hearing_date}"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, c.case_id)
            item.setData(Qt.ItemDataRole.ToolTipRole, c.description[:100] if c.description else "")
            self.case_list.addItem(item)

    def _search(self):
        query = self.search_input.text().strip()
        if not query:
            self._refresh()
            return
        self.case_list.clear()
        results = case_tracker.search_cases(query)
        for c in results:
            display = f"{c.title} [{c.status.display()}]"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, c.case_id)
            self.case_list.addItem(item)

    def _show_detail(self, row: int):
        has_selection = row >= 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        if row < 0:
            self.detail_text.clear()
            return
        item = self.case_list.item(row)
        if not item:
            return
        case_id = item.data(Qt.ItemDataRole.UserRole)
        case = case_tracker.get_case(case_id)
        if not case:
            return
        html = f"""
        <h3>{case.title}</h3>
        <table>
        <tr><td><b>Type:</b></td><td>{case.case_type.display()}</td></tr>
        <tr><td><b>Status:</b></td><td>{case.status.display()}</td></tr>
        <tr><td><b>Court:</b></td><td>{case.court or '-'}</td></tr>
        <tr><td><b>Case No:</b></td><td>{case.case_number or '-'}</td></tr>
        <tr><td><b>Opposite Party:</b></td><td>{case.opposite_party or '-'}</td></tr>
        <tr><td><b>Next Hearing:</b></td><td>{case.next_hearing_date or '-'}</td></tr>
        <tr><td><b>Lawyer:</b></td><td>{case.assigned_lawyer or '-'}</td></tr>
        </table>
        <p><b>Description:</b> {case.description or '-'}</p>
        <p><b>Notes:</b> {case.notes or '-'}</p>
        <p><i>Created: {case.created_at}</i></p>
        """
        self.detail_text.setHtml(html)

    def _add_case(self):
        dialog = CaseDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh()

    def _edit_case(self):
        item = self.case_list.currentItem()
        if not item:
            return
        case_id = item.data(Qt.ItemDataRole.UserRole)
        case = case_tracker.get_case(case_id)
        if not case:
            return
        dialog = CaseDialog(case=case, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh()

    def _delete_case(self):
        item = self.case_list.currentItem()
        if not item:
            return
        case_id = item.data(Qt.ItemDataRole.UserRole)
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Delete Case",
            "Are you sure you want to delete this case?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            case_tracker.delete_case(case_id)
            self._refresh()

    def _style(self):
        self.setStyleSheet("""
            CaseTrackerView { background-color: #f5f6fa; border: 1px solid #dcdde1; border-radius: 6px; }
            QListWidget { background: white; border: 1px solid #dcdde1; border-radius: 4px; }
            QListWidget::item:selected { background-color: #3498db; color: white; }
            QTextEdit, QLineEdit { background: white; border: 1px solid #dcdde1; border-radius: 4px; }
            QGroupBox { font-weight: bold; border: 1px solid #dcdde1; border-radius: 4px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { border: 1px solid #bdc3c7; border-radius: 4px; padding: 4px 10px; }
            QPushButton:hover { background-color: #ecf0f1; }
        """)
