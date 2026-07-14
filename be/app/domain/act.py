from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Act:
    title: str
    short_title: Optional[str] = None
    act_number: Optional[str] = None
    year: Optional[int] = None
    jurisdiction: str = "India"
    enacted_date: Optional[datetime] = None
    commencement_date: Optional[datetime] = None
    source_url: Optional[str] = None
    sections: list["Section"] = field(default_factory=list)
    chapters: list["Chapter"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        parts = [self.short_title or self.title]
        if self.act_number:
            parts.insert(0, f"Act No. {self.act_number}")
        if self.year:
            parts.append(f"({self.year})")
        return " ".join(parts)


@dataclass
class Chapter:
    number: int
    title: str
    sections: list["Section"] = field(default_factory=list)


@dataclass
class Section:
    number: str
    title: Optional[str] = None
    text: str = ""
    parent_act: Optional[str] = None
    parent_chapter: Optional[int] = None
    margin_note: Optional[str] = None
    amendment_history: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        base = f"Section {self.number}"
        if self.parent_act:
            base = f"{base} of {self.parent_act}"
        return base
