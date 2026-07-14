import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.localization.detector import language_detector
from app.core.second_opinion.evaluator import second_opinion
from app.di.container import ApplicationContainer
from app.domain.citizen_protection.emergency import (
    EMERGENCY_HELPLINES,
    EMERGENCY_TRIGGER_WORDS,
    detect_urgency,
    get_helplines_text,
)
from app.domain.memory.session import get_memory
from app.infra.logging import logger
from app.ui.screens.case_tracker_view import CaseTrackerView
from app.ui.screens.emergency_panel import EmergencyPanel
from app.ui.screens.rights_explorer import RightsExplorerPanel
from app.ui.screens.template_dialog import TemplateDialog


class QueryWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, container: ApplicationContainer, question: str, mode: str = "citizen"):
        super().__init__()
        self.container = container
        self.question = question
        self.mode = mode

    def run(self):
        try:
            orchestrator = self.container.citizen_orchestrator()
            result = orchestrator.query(self.question, mode=self.mode)
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Query failed")
            self.finished.emit({"answer": f"Error: {str(e)}", "sources": [], "confidence": 0.0})


class SecondOpinionWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def run(self):
        try:
            result = second_opinion.evaluate(self.question)
            self.finished.emit(result.to_dict())
        except Exception as e:
            self.finished.emit({"error": str(e)})


class MainWindow(QMainWindow):
    def __init__(self, container: ApplicationContainer):
        super().__init__()
        self.container = container
        self._init_ui()
        self._init_state()

    def _init_ui(self):
        self.setWindowTitle("Nyaya Mitra - Your Legal Guardian")
        self.setMinimumSize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.emergency_panel = EmergencyPanel()
        self.emergency_panel.setVisible(True)
        main_layout.addWidget(self.emergency_panel)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        title_label = QLabel("\u1e8c Nyaya Mitra")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        toolbar.addWidget(title_label)
        toolbar.addStretch()

        self.second_opinion_btn = QPushButton("Second Opinion")
        self.second_opinion_btn.clicked.connect(self._on_second_opinion)
        toolbar.addWidget(self.second_opinion_btn)

        self.draft_btn = QPushButton("Draft Document")
        self.draft_btn.clicked.connect(self._open_template_dialog)
        toolbar.addWidget(self.draft_btn)

        self.explain_toggle = QPushButton("Explain Like I'm 10: OFF")
        self.explain_toggle.setCheckable(True)
        self.explain_toggle.clicked.connect(self._toggle_explain)
        self.explain_toggle.setStyleSheet(
            "QPushButton:checked { background-color: #f39c12; color: white; }"
        )
        toolbar.addWidget(self.explain_toggle)

        left_layout.addLayout(toolbar)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        left_layout.addWidget(self.chat_display, 1)

        input_row = QHBoxLayout()
        self.question_input = QTextEdit()
        self.question_input.setMaximumHeight(80)
        self.question_input.setPlaceholderText(
            "Describe your situation or upload a legal document...\n"
            "Press Ctrl+Enter to send"
        )
        input_row.addWidget(self.question_input, 1)

        ask_btn = QPushButton("Ask")
        ask_btn.setMinimumWidth(80)
        ask_btn.clicked.connect(self._on_ask)
        input_row.addWidget(ask_btn)

        left_layout.addLayout(input_row)

        right_tabs = QTabWidget()
        self.rights_explorer = RightsExplorerPanel()
        right_tabs.addTab(self.rights_explorer, "Know Your Rights")

        self.case_tracker = CaseTrackerView()
        right_tabs.addTab(self.case_tracker, "My Cases")

        sources_tab = QWidget()
        sources_layout = QVBoxLayout(sources_tab)
        self.sources_display = QTextEdit()
        self.sources_display.setReadOnly(True)
        sources_layout.addWidget(self.sources_display)
        right_tabs.addTab(sources_tab, "Sources")

        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_tabs)
        content_splitter.setSizes([700, 400])

        main_layout.addWidget(content_splitter, 1)

        self._build_menu()

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        import_action = QAction("Import Document...", self)
        import_action.triggered.connect(self._import_document)
        file_menu.addAction(import_action)
        file_menu.addSeparator()
        export_action = QAction("Export Chat...", self)
        export_action.triggered.connect(self._export_chat)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menubar.addMenu("Tools")
        draft_action = QAction("Draft Legal Document...", self)
        draft_action.triggered.connect(self._open_template_dialog)
        tools_menu.addAction(draft_action)
        second_opinion_action = QAction("Get Second Opinion...", self)
        second_opinion_action.triggered.connect(self._on_second_opinion)
        tools_menu.addAction(second_opinion_action)
        tools_menu.addSeparator()
        rights_action = QAction("Know Your Rights", self)
        rights_action.triggered.connect(lambda: self._show_tab(0))
        tools_menu.addAction(rights_action)
        cases_action = QAction("My Cases", self)
        cases_action.triggered.connect(lambda: self._show_tab(1))
        tools_menu.addAction(cases_action)

        help_menu = menubar.addMenu("Help")
        helplines_action = QAction("Emergency Helplines", self)
        helplines_action.triggered.connect(self._show_helplines)
        help_menu.addAction(helplines_action)
        about_action = QAction("About Nyaya Mitra", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_state(self):
        self._session = get_memory()
        self._explain_mode = False
        self._show_welcome()
        self.statusBar().showMessage("Ready | Nyaya Mitra is protecting your rights")

    def _show_welcome(self):
        welcome = (
            '<h1 style="color: #2c3e50;">\u1e8c Nyaya Mitra</h1>'
            '<p><b>Your 24/7 Legal Guardian &mdash; Offline, Private, Free</b></p>'
            '<hr>'
            '<h3>How Nyaya Mitra Helps You</h3>'
            '<ul>'
            '<li><b>Upload a document</b> &ndash; paste or upload any legal notice, summons, or agreement</li>'
            '<li><b>Describe your situation</b> &ndash; tell us what happened in your own words</li>'
            '<li><b>Get a clear, actionable response</b> &ndash; understand your rights, risks, and next steps</li>'
            '</ul>'
            '<h3>Key Features</h3>'
            '<ul>'
            '<li><b>Draft Documents</b> &ndash; Generate FIR, RTI, Legal Notices, and more</li>'
            '<li><b>Know Your Rights</b> &ndash; Browse 60+ citizen rights with plain explanations</li>'
            '<li><b>Case Tracker</b> &ndash; Keep track of your legal cases and deadlines</li>'
            '<li><b>Second Opinion</b> &ndash; Get an independent evaluation of your situation</li>'
            '<li><b>Emergency Help</b> &ndash; One-click access to helpline numbers (left panel)</li>'
            '</ul>'
            '<p><i>Disclaimer: Nyaya Mitra provides legal information and education, not legal advice. For serious matters, contact the District Legal Services Authority (DLSA) for free legal aid.</i></p>'
            '<hr>'
        )
        self.chat_display.setHtml(welcome)

    def _on_ask(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            return

        self._session.add_turn("user", question)
        self.chat_display.append(
            f'<div style="background-color: #f0f0f0; padding: 8px 12px; border-radius: 6px; margin: 4px 0;">'
            f'<b style="color: #2c3e50;">You:</b> {question}</div>'
        )
        self.question_input.clear()

        lang = language_detector.detect(question)
        if detect_urgency(question):
            helplines = get_helplines_text()
            urgency_msg = (
                f"<div style='background-color: #ffeaa7; padding: 10px; border-left: 4px solid #e74c3c;'>"
                f"<b>IMPORTANT:</b> Your situation appears urgent. If you are in immediate danger, "
                f"please contact emergency services immediately.<br><br>"
                f"<pre>{helplines}</pre>"
                f"</div>"
            )
            self.chat_display.append(urgency_msg)

        self.statusBar().showMessage("Analyzing your situation...")

        self.worker = QueryWorker(self.container, question)
        self.worker.finished.connect(self._on_answer)
        self.worker.start()

    def _on_answer(self, result: dict):
        answer = result.get("answer", "No answer could be generated.")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)

        if self._explain_mode:
            answer = answer + "\n\n---\n\n*This explanation has been simplified. For full legal details, toggle 'Explain Like I'm 10' OFF and ask again.*"

        self.chat_display.append(
            f'<div style="background-color: #e8f4fd; padding: 12px; border-radius: 6px; margin: 8px 0; border-left: 4px solid #3498db;">'
            f'<b style="color: #2c3e50;">Nyaya Mitra:</b><br>{answer}</div>'
        )
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

        self._session.add_turn("assistant", answer)

        sources_html = "<h3>Sources & Citations</h3>"
        for s in sources:
            heading = s.get("heading", "")
            source = s.get("source", "")
            score = s.get("score", 0)
            text = s.get("text", "")[:200]
            sources_html += f"<p><b>{heading}</b> ({source}) [score: {score:.3f}]<br>{text}...</p>"
        if not sources:
            sources_html += "<p>No sources cited. Response based on legal knowledge base.</p>"
        sources_html += f"<p><i>Confidence: {confidence:.1%}</i></p>"
        self.sources_display.setHtml(sources_html)

        self.statusBar().showMessage("Ready", 3000)

    def _on_second_opinion(self):
        text = self.question_input.toPlainText().strip()
        if not text:
            text = "Please provide your situation or document text for a second opinion."
            self.question_input.setText(text)
            return

        self.statusBar().showMessage("Getting second opinion...")
        self.chat_display.append(
            "<div style='background-color: #f0f0f0; padding: 8px; border-left: 4px solid #8e44ad;'>"
            "<b>Second Opinion requested...</b></div>"
        )

        self.opinion_worker = SecondOpinionWorker(text)
        self.opinion_worker.finished.connect(self._on_second_opinion_result)
        self.opinion_worker.start()

    def _on_second_opinion_result(self, result: dict):
        if "error" in result:
            self.chat_display.append(
                f"<p style='color: red;'>Second Opinion error: {result['error']}</p>"
            )
            self.statusBar().showMessage("Second opinion failed", 3000)
            return

        html = "<div style='background-color: #f3e8ff; padding: 12px; border-left: 4px solid #8e44ad; margin: 8px 0;'>"
        html += "<h3>Second Opinion Report</h3>"

        if result.get("scam_flags"):
            html += "<p style='color: #e74c3c;'><b>Warning:</b> Potential scam indicators detected</p>"
            for flag in result["scam_flags"]:
                html += f"<p>- {flag['description']}</p>"

        if result.get("fee_check"):
            fc = result["fee_check"]
            html += f"<p><b>Fee Check:</b> {fc.get('message', '')}</p>"

        if result.get("relevant_rights"):
            html += "<p><b>Your Rights:</b></p>"
            for right in result["relevant_rights"][:3]:
                html += f"<p><b>{right['right']}</b>: {right['plain'][:100]}...</p>"

        if result.get("alternative_perspective"):
            html += f"<p><b>Alternative Perspective:</b><br>{result['alternative_perspective']}</p>"

        if result.get("recommendations"):
            html += "<p><b>Recommendations:</b></p>"
            for rec in result["recommendations"]:
                html += f"<p>-> {rec}</p>"

        if result.get("strengths"):
            html += "<p><b>Strengths in your approach:</b></p>"
            for s in result["strengths"]:
                html += f"<p>+ {s}</p>"

        if result.get("weaknesses"):
            html += "<p><b>Watch out for:</b></p>"
            for w in result["weaknesses"]:
                html += f"<p>! {w}</p>"

        html += "</div>"
        self.chat_display.append(html)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        self.statusBar().showMessage("Second opinion ready", 3000)

    def _toggle_explain(self):
        self._explain_mode = self.explain_toggle.isChecked()
        if self._explain_mode:
            self.explain_toggle.setText("Explain Like I'm 10: ON")
            self.explain_toggle.setStyleSheet(
                "QPushButton:checked { background-color: #f39c12; color: white; font-weight: bold; }"
            )
        else:
            self.explain_toggle.setText("Explain Like I'm 10: OFF")
            self.explain_toggle.setStyleSheet("")

    def _open_template_dialog(self):
        dialog = TemplateDialog(self)
        dialog.exec()

    def _import_document(self):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Legal Document",
            "",
            "All Supported Files (*.pdf *.docx *.doc *.txt *.rtf *.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self.chat_display.append(
                f'<p><i>Imported document: {Path(path).name}</i></p>'
            )
            try:
                from app.core.ingestion.loader import DocumentLoader
                from app.infra.config import settings

                loader = DocumentLoader(ocr_engine=settings.ocr_engine)
                text = loader.load(path)
                preview = text[:500]
                self._session.document_text = text
                self.question_input.setText(
                    f"I have uploaded '{Path(path).name}'. "
                    f"Here is the document content:\n\n{preview}\n\n---\n\n"
                    f"Can you analyze this document and tell me what it means?"
                )
                self.statusBar().showMessage(f"Loaded: {Path(path).name}", 3000)
            except Exception as e:
                self.chat_display.append(
                    f'<p style="color: red;">Error loading document: {e}</p>'
                )

    def _export_chat(self):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat",
            f"nyaya_mitra_chat_{datetime.now().strftime('%Y%m%d')}.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            content = self.chat_display.toPlainText()
            Path(path).write_text(content, encoding="utf-8")
            self.statusBar().showMessage(f"Chat exported to {path}", 3000)

    def _show_tab(self, index: int):
        parent = self.centralWidget()
        right_tabs = parent.findChild(QTabWidget)
        if right_tabs:
            right_tabs.setCurrentIndex(index)

    def _show_helplines(self):
        helplines = get_helplines_text()
        self.chat_display.append(
            f"<div style='background-color: #e74c3c; padding: 12px; border-radius: 6px; color: white; margin: 8px 0;'>"
            f"<b>EMERGENCY HELPLINES</b><br><pre style='color: white; font-size: 12px;'>{helplines}</pre></div>"
        )

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About Nyaya Mitra",
            "<h3>Nyaya Mitra v0.1.0</h3>"
            "<p>Offline-first Legal AI assistant for ordinary Indian citizens.</p>"
            "<p>Protects citizens from exploitation by unethical lawyers and legal middlemen.</p>"
            "<hr>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>100% offline - no internet required</li>"
            "<li>Document analysis (PDF, DOCX, images)</li>"
            "<li>Draft legal documents (FIR, RTI, notices)</li>"
            "<li>60+ citizen rights with explanations</li>"
            "<li>Case tracker with hearing reminders</li>"
            "<li>Scam detection and fee verification</li>"
            "<li>200+ legal terms explained in plain language</li>"
            "<li>AES-256 encryption for all stored data</li>"
            "</ul>"
            "<p><i>Built with open-source software for the public good.</i></p>"
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._on_ask()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        logger.info("Shutting down Nyaya Mitra")
        event.accept()


def run_ui(container: ApplicationContainer) -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f6fa; }
        QTextEdit { font-size: 13px; padding: 6px; border: 1px solid #dcdde1; border-radius: 4px; background: white; }
        QTextEdit:focus { border-color: #3498db; }
        QPushButton {
            padding: 6px 14px;
            border-radius: 4px;
            background-color: #3498db;
            color: white;
            border: none;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #2980b9; }
        QPushButton:pressed { background-color: #2471a3; }
        QPushButton:checked { background-color: #f39c12; }
        QTabWidget::pane { border: 1px solid #dcdde1; border-radius: 4px; background: white; }
        QTabBar::tab {
            padding: 10px 20px;
            margin-right: 2px;
            background: #ecf0f1;
            border: 1px solid #dcdde1;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { background-color: white; border-bottom: 2px solid #3498db; font-weight: bold; }
        QTabBar::tab:hover:!selected { background-color: #d5dbdb; }
        QSplitter::handle { background-color: #dcdde1; width: 2px; }
        QMenuBar { background-color: #2c3e50; color: white; padding: 2px; }
        QMenuBar::item:selected { background-color: #3498db; }
        QMenu { background-color: white; border: 1px solid #dcdde1; }
        QMenu::item:selected { background-color: #3498db; color: white; }
        QStatusBar { background-color: #2c3e50; color: #ecf0f1; }
        QScrollBar:vertical { background: #ecf0f1; width: 8px; border-radius: 4px; }
        QScrollBar::handle:vertical { background: #bdc3c7; border-radius: 4px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: #95a5a6; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """)
    window = MainWindow(container)
    window.show()
    sys.exit(app.exec())
