from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.domain.templates.models import TEMPLATES, TemplateType
from app.domain.templates.renderer import renderer


class TemplateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draft Legal Document")
        self.setMinimumSize(800, 650)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("Draft a Legal Document")
        header_font = header.font()
        header_font.setBold(True)
        header_font.setPointSize(13)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        self.tabs = QTabWidget()

        self._form_tab = QWidget()
        self._init_form_tab()
        self.tabs.addTab(self._form_tab, "Fill Details")

        self._preview_tab = QWidget()
        self._init_preview_tab()
        self.tabs.addTab(self._preview_tab, "Preview")

        self._export_tab = QWidget()
        self._init_export_tab()
        self.tabs.addTab(self._export_tab, "Export")

        layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _init_form_tab(self):
        tab_layout = QVBoxLayout(self._form_tab)

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Document Type:"))
        self.type_combo = QComboBox()
        for tt in TemplateType:
            self.type_combo.addItem(tt.display_name(), tt.value)
        self.type_combo.currentIndexChanged.connect(self._on_type_change)
        select_row.addWidget(self.type_combo, 1)

        info_btn = QPushButton("Show Info")
        info_btn.clicked.connect(self._show_template_info)
        select_row.addWidget(info_btn)

        tab_layout.addLayout(select_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        self.form_layout.setSpacing(4)
        self.form_fields: dict[str, QTextEdit | QLineEdit] = {}
        scroll.setWidget(self.form_widget)
        tab_layout.addWidget(scroll, 1)

        generate_btn = QPushButton("Generate Document")
        generate_btn.clicked.connect(self._generate)
        tab_layout.addWidget(generate_btn)

        self._populate_form()

    def _init_preview_tab(self):
        tab_layout = QVBoxLayout(self._preview_tab)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        tab_layout.addWidget(self.preview_text)

    def _init_export_tab(self):
        tab_layout = QVBoxLayout(self._export_tab)

        tab_layout.addWidget(QLabel("Export Options"))

        save_md_btn = QPushButton("Save as Markdown (.md)")
        save_md_btn.clicked.connect(lambda: self._export("md"))
        tab_layout.addWidget(save_md_btn)

        save_txt_btn = QPushButton("Save as Plain Text (.txt)")
        save_txt_btn.clicked.connect(lambda: self._export("txt"))
        tab_layout.addWidget(save_txt_btn)

        save_docx_btn = QPushButton("Save as Word Document (.docx)")
        save_docx_btn.clicked.connect(lambda: self._export("docx"))
        tab_layout.addWidget(save_docx_btn)

        tab_layout.addStretch()

        self.export_info = QLabel("")
        tab_layout.addWidget(self.export_info)

    def _populate_form(self):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child and child.widget():
                child.widget().deleteLater()
        self.form_fields.clear()

        template_type = TemplateType(self.type_combo.currentData())
        template = TEMPLATES[template_type]
        fields_summary = renderer.get_fields_summary(template_type)

        info_label = QLabel(f"<b>{template.title}</b><br>{template.template_type.description()}")
        info_label.setWordWrap(True)
        self.form_layout.addRow(info_label)

        for field_info in fields_summary:
            label = field_info["label"]
            required = field_info["required"]
            placeholder = field_info["placeholder"]
            desc = field_info["description"]
            label_text = f"{label} {'*' if required else ''}"
            if desc != label:
                label_text += f"\n<span style='color: gray; font-size: 10px;'>{desc}</span>"

            if label_text in ["Incident Description", "Your Response", "Complaint Details",
                              "Information Requested", "Violence Details", "Case Details"]:
                widget = QTextEdit()
                widget.setMaximumHeight(100)
                widget.setPlaceholderText(placeholder)
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(placeholder)
            self.form_layout.addRow(label_text, widget)
            self.form_fields[field_info["key"]] = widget

    def _on_type_change(self):
        self._populate_form()
        self.preview_text.clear()

    def _show_template_info(self):
        template_type = TemplateType(self.type_combo.currentData())
        info = renderer.get_template_info(template_type)
        acts = "\n".join(f"- {a}" for a in info["applicable_acts"])
        msg = (
            f"<h3>{info['title']}</h3>"
            f"<p>{info['description']}</p>"
            f"<p><b>Applicable Law:</b><br>{acts}</p>"
            f"<p><b>Fee Info:</b> {info['fee_info']}</p>"
            f"<hr><p>{info['instructions'].replace(chr(10), '<br>')}</p>"
        )
        QMessageBox.information(self, f"About {info['title']}", msg)

    def _collect_data(self) -> dict[str, str]:
        data = {}
        for key, widget in self.form_fields.items():
            if isinstance(widget, QTextEdit):
                data[key] = widget.toPlainText().strip()
            else:
                data[key] = widget.text().strip()
        return data

    def _generate(self):
        template_type = TemplateType(self.type_combo.currentData())
        data = self._collect_data()
        template = TEMPLATES[template_type]
        missing = template.validate_fields(data)
        if missing:
            QMessageBox.warning(
                self,
                "Missing Fields",
                f"Please fill in: {', '.join(missing)}",
            )
            return
        md = renderer.render_markdown(template_type, data)
        self.preview_text.setHtml(
            f"<pre style='white-space: pre-wrap; font-family: inherit;'>{md}</pre>"
        )
        self.tabs.setCurrentIndex(1)
        self._current_data = data

    def _export(self, fmt: str):
        if not hasattr(self, "_current_data"):
            QMessageBox.warning(self, "No Document", "Please generate the document first.")
            return
        template_type = TemplateType(self.type_combo.currentData())
        data = self._current_data

        from PyQt6.QtWidgets import QFileDialog

        template = TEMPLATES[template_type]
        safe_title = template.title.lower().replace(" ", "_").replace("/", "_")
        default_name = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        if fmt == "md":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Markdown", default_name + ".md",
                "Markdown Files (*.md);;All Files (*)",
            )
            if path:
                content = renderer.render_markdown(template_type, data)
                Path(path).write_text(content, encoding="utf-8")
                self.export_info.setText(f"Saved to: {path}")
        elif fmt == "txt":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Text", default_name + ".txt",
                "Text Files (*.txt);;All Files (*)",
            )
            if path:
                content = renderer.render_text(template_type, data)
                Path(path).write_text(content, encoding="utf-8")
                self.export_info.setText(f"Saved to: {path}")
        elif fmt == "docx":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Word Document", default_name + ".docx",
                "Word Documents (*.docx);;All Files (*)",
            )
            if path:
                try:
                    renderer.render_docx(template_type, data, Path(path))
                    self.export_info.setText(f"Saved to: {path}")
                except RuntimeError as e:
                    QMessageBox.warning(self, "Export Error", str(e))
