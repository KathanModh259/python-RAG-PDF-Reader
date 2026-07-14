from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Judgment:
    case_name: str
    citation: str
    court: str
    bench: Optional[str] = None
    date_of_judgment: Optional[datetime] = None
    judges: list[str] = field(default_factory=list)
    parties: dict[str, str] = field(default_factory=dict)
    headnote: str = ""
    held: str = ""
    full_text: str = ""
    overrules: list[str] = field(default_factory=list)
    overruled_by: list[str] = field(default_factory=list)
    cited_cases: list[str] = field(default_factory=list)
    statutes_referred: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
