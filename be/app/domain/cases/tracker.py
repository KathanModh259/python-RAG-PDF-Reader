import json
import sqlite3
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class CaseStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    STAYED = "stayed"
    CLOSED = "closed"
    APPEALED = "appealed"

    def display(self) -> str:
        labels = {
            "not_started": "Not Started",
            "in_progress": "In Progress",
            "stayed": "Stayed",
            "closed": "Closed",
            "appealed": "Appealed",
        }
        return labels[self.value]


class CaseType(str, Enum):
    CIVIL = "civil"
    CRIMINAL = "criminal"
    CONSUMER = "consumer"
    FAMILY = "family"
    LABOUR = "labour"
    TAX = "tax"
    PROPERTY = "property"
    RTI = "rti"
    OTHER = "other"

    def display(self) -> str:
        return self.value.capitalize()


class Case:
    def __init__(
        self,
        case_id: str,
        title: str,
        case_type: CaseType,
        status: CaseStatus,
        description: str = "",
        court: str = "",
        case_number: str = "",
        party_names: str = "",
        opposite_party: str = "",
        filing_date: str = "",
        next_hearing_date: str = "",
        last_hearing_date: str = "",
        assigned_lawyer: str = "",
        notes: str = "",
        documents: Optional[list[str]] = None,
        important_dates: Optional[dict[str, str]] = None,
        created_at: str = "",
        updated_at: str = "",
    ):
        self.case_id = case_id
        self.title = title
        self.case_type = case_type
        self.status = status
        self.description = description
        self.court = court
        self.case_number = case_number
        self.party_names = party_names
        self.opposite_party = opposite_party
        self.filing_date = filing_date
        self.next_hearing_date = next_hearing_date
        self.last_hearing_date = last_hearing_date
        self.assigned_lawyer = assigned_lawyer
        self.notes = notes
        self.documents = documents or []
        self.important_dates = important_dates or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "case_type": self.case_type.value,
            "status": self.status.value,
            "description": self.description,
            "court": self.court,
            "case_number": self.case_number,
            "party_names": self.party_names,
            "opposite_party": self.opposite_party,
            "filing_date": self.filing_date,
            "next_hearing_date": self.next_hearing_date,
            "last_hearing_date": self.last_hearing_date,
            "assigned_lawyer": self.assigned_lawyer,
            "notes": self.notes,
            "documents": json.dumps(self.documents),
            "important_dates": json.dumps(self.important_dates),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "Case":
        return cls(
            case_id=row["case_id"],
            title=row["title"],
            case_type=CaseType(row["case_type"]),
            status=CaseStatus(row["status"]),
            description=row.get("description", ""),
            court=row.get("court", ""),
            case_number=row.get("case_number", ""),
            party_names=row.get("party_names", ""),
            opposite_party=row.get("opposite_party", ""),
            filing_date=row.get("filing_date", ""),
            next_hearing_date=row.get("next_hearing_date", ""),
            last_hearing_date=row.get("last_hearing_date", ""),
            assigned_lawyer=row.get("assigned_lawyer", ""),
            notes=row.get("notes", ""),
            documents=json.loads(row.get("documents", "[]")),
            important_dates=json.loads(row.get("important_dates", "{}")),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )


class CaseTracker:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".nyaya_mitra" / "cases.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    case_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'not_started',
                    description TEXT DEFAULT '',
                    court TEXT DEFAULT '',
                    case_number TEXT DEFAULT '',
                    party_names TEXT DEFAULT '',
                    opposite_party TEXT DEFAULT '',
                    filing_date TEXT DEFAULT '',
                    next_hearing_date TEXT DEFAULT '',
                    last_hearing_date TEXT DEFAULT '',
                    assigned_lawyer TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    documents TEXT DEFAULT '[]',
                    important_dates TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cases_next_hearing ON cases(next_hearing_date)
            """)

    def add_case(
        self,
        title: str,
        case_type: CaseType,
        status: CaseStatus = CaseStatus.NOT_STARTED,
        description: str = "",
        court: str = "",
        case_number: str = "",
        party_names: str = "",
        opposite_party: str = "",
        filing_date: str = "",
        next_hearing_date: str = "",
        last_hearing_date: str = "",
        assigned_lawyer: str = "",
        notes: str = "",
        documents: Optional[list[str]] = None,
        important_dates: Optional[dict[str, str]] = None,
    ) -> Case:
        case_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()
        case = Case(
            case_id=case_id,
            title=title,
            case_type=case_type,
            status=status,
            description=description,
            court=court,
            case_number=case_number,
            party_names=party_names,
            opposite_party=opposite_party,
            filing_date=filing_date,
            next_hearing_date=next_hearing_date,
            last_hearing_date=last_hearing_date,
            assigned_lawyer=assigned_lawyer,
            notes=notes,
            documents=documents,
            important_dates=important_dates,
            created_at=now,
            updated_at=now,
        )
        self._upsert(case)
        return case

    def update_case(self, case: Case) -> None:
        case.updated_at = datetime.now().isoformat()
        self._upsert(case)

    def delete_case(self, case_id: str) -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
            return cursor.rowcount > 0

    def get_case(self, case_id: str) -> Optional[Case]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            return Case.from_dict(dict(row)) if row else None

    def list_cases(
        self,
        status: Optional[CaseStatus] = None,
        case_type: Optional[CaseType] = None,
        sort_by: str = "updated_at",
        sort_desc: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Case]:
        query = "SELECT * FROM cases WHERE 1=1"
        params: list[str] = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if case_type:
            query += " AND case_type = ?"
            params.append(case_type.value)
        order = "DESC" if sort_desc else "ASC"
        query += f" ORDER BY {sort_by} {order} LIMIT ? OFFSET ?"
        params.extend([str(limit), str(offset)])

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [Case.from_dict(dict(r)) for r in rows]

    def search_cases(self, query: str) -> list[Case]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT * FROM cases WHERE
                    title LIKE ? OR description LIKE ? OR
                    court LIKE ? OR case_number LIKE ? OR
                    party_names LIKE ? OR opposite_party LIKE ? OR
                    notes LIKE ?
                ORDER BY updated_at DESC LIMIT 50""",
                (pattern,) * 7,
            ).fetchall()
            return [Case.from_dict(dict(r)) for r in rows]

    def get_upcoming_hearings(self, days: int = 30) -> list[Case]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM cases WHERE
                    next_hearing_date != '' AND
                    next_hearing_date >= date('now') AND
                    next_hearing_date <= date('now', '+' || ? || ' days')
                ORDER BY next_hearing_date ASC""",
                (str(days),),
            ).fetchall()
            return [Case.from_dict(dict(r)) for r in rows]

    def add_document(self, case_id: str, document_path: str) -> Optional[Case]:
        case = self.get_case(case_id)
        if not case:
            return None
        if document_path not in case.documents:
            case.documents.append(document_path)
            self.update_case(case)
        return case

    def add_important_date(self, case_id: str, label: str, date: str) -> Optional[Case]:
        case = self.get_case(case_id)
        if not case:
            return None
        case.important_dates[label] = date
        self.update_case(case)
        return case

    def get_statistics(self) -> dict:
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            by_status = dict(
                conn.execute(
                    "SELECT status, COUNT(*) FROM cases GROUP BY status"
                ).fetchall()
            )
            by_type = dict(
                conn.execute(
                    "SELECT case_type, COUNT(*) FROM cases GROUP BY case_type"
                ).fetchall()
            )
            upcoming = conn.execute(
                """SELECT COUNT(*) FROM cases WHERE
                    next_hearing_date != '' AND
                    next_hearing_date >= date('now')"""
            ).fetchone()[0]
        return {
            "total_cases": total,
            "by_status": by_status,
            "by_type": by_type,
            "upcoming_hearings": upcoming,
        }

    def _upsert(self, case: Case) -> None:
        data = case.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        updates = ", ".join(f"{k}=excluded.{k}" for k in data if k != "case_id")
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                f"""INSERT INTO cases ({columns}) VALUES ({placeholders})
                    ON CONFLICT(case_id) DO UPDATE SET {updates}""",
                list(data.values()),
            )


tracker = CaseTracker()
