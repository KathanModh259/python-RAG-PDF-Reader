from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Clause:
    number: str
    text: str
    sub_clauses: list["Clause"] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"Clause {self.number}"


@dataclass
class LegalSection:
    number: str
    heading: Optional[str] = None
    text: str = ""
    clauses: list[Clause] = field(default_factory=list)
    explanation: Optional[str] = None
    illustrations: list[str] = field(default_factory=list)
    amendments: list[str] = field(default_factory=list)

    @property
    def full_citation(self) -> str:
        base = f"Section {self.number}"
        if self.heading:
            base = f"{base} - {self.heading}"
        return base
