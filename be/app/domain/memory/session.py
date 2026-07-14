import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.infra.logging import logger


class ConversationTurn:
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ConversationMemory:
    def __init__(self, session_id: Optional[str] = None, persist_path: Optional[Path] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.turns: list[ConversationTurn] = []
        self.persist_path = persist_path
        self.context: dict = {}
        self.document_text: Optional[str] = None
        self.user_name: Optional[str] = None

    def add_turn(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        if self.persist_path:
            self._persist()

    def get_last_user_message(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def get_recent_context(self, n: int = 5) -> str:
        recent = self.turns[-n:] if len(self.turns) > n else self.turns
        lines = []
        for turn in recent:
            label = "You" if turn.role == "user" else "Nyaya Mitra"
            lines.append(f"{label}: {turn.content[:300]}")
        return "\n\n".join(lines)

    def set_document(self, text: str) -> None:
        self.document_text = text

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "context": self.context,
        }

    def _persist(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, session_id: str, persist_dir: Path) -> Optional["ConversationMemory"]:
        path = persist_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mem = cls(session_id=session_id, persist_path=path)
            for t in data.get("turns", []):
                turn = ConversationTurn(
                    role=t["role"],
                    content=t["content"],
                    timestamp=t.get("timestamp"),
                    metadata=t.get("metadata"),
                )
                mem.turns.append(turn)
            return mem
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None


_sessions: dict[str, ConversationMemory] = {}
_persist_dir: Optional[Path] = None


def init_memory_persistence(dir_path: Path) -> None:
    global _persist_dir
    _persist_dir = dir_path
    _persist_dir.mkdir(parents=True, exist_ok=True)


def get_memory(session_id: Optional[str] = None) -> ConversationMemory:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    if session_id and _persist_dir:
        loaded = ConversationMemory.load(session_id, _persist_dir)
        if loaded:
            _sessions[session_id] = loaded
            return loaded
    mem = ConversationMemory(
        session_id=session_id,
        persist_path=_persist_dir / f"{session_id or 'default'}.json" if _persist_dir else None,
    )
    _sessions[mem.session_id] = mem
    return mem
