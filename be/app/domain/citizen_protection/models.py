from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def display(self) -> str:
        icons = {
            "low": "GREEN",
            "medium": "YELLOW",
            "high": "RED",
            "critical": "BLACK",
        }
        return f"{icons[self.value]}"


class Urgency(str, Enum):
    TODAY = "do_today"
    THIS_WEEK = "do_this_week"
    BEFORE_DEADLINE = "before_deadline"
    NO_URGENCY = "no_urgency"


@dataclass
class ActionStep:
    step_number: int
    description: str
    urgency: Urgency
    is_free_or_low_cost: bool = True
    helpline_number: Optional[str] = None
    reference_info: Optional[str] = None


@dataclass
class LegalCitation:
    provision: str
    plain_explanation: str
    knowledge_base_link: Optional[str] = None


@dataclass
class RedFlag:
    description: str
    why_it_matters: str
    what_to_do: str


@dataclass
class EightPartResponse:
    what_document_is: str
    what_happened_simple: str
    risk_level: RiskLevel
    risk_explanation: str
    deadline: Optional[str] = None
    action_plan: list[ActionStep] = field(default_factory=list)
    red_flags: list[RedFlag] = field(default_factory=list)
    prevention_tips: list[str] = field(default_factory=list)
    legal_basis: list[LegalCitation] = field(default_factory=list)
    final_conclusion: str = ""
    emergency_numbers: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "what_document_is": self.what_document_is,
            "what_happened_simple": self.what_happened_simple,
            "risk_level": self.risk_level.value,
            "risk_explanation": self.risk_explanation,
            "deadline": self.deadline,
            "action_plan": [
                {
                    "step": a.step_number,
                    "action": a.description,
                    "urgency": a.urgency.value,
                    "free_or_low_cost": a.is_free_or_low_cost,
                    "helpline": a.helpline_number,
                }
                for a in self.action_plan
            ],
            "red_flags": [
                {"flag": r.description, "why": r.why_it_matters, "what_to_do": r.what_to_do}
                for r in self.red_flags
            ],
            "prevention_tips": self.prevention_tips,
            "legal_basis": [
                {"section": c.provision, "explanation": c.plain_explanation}
                for c in self.legal_basis
            ],
            "final_conclusion": self.final_conclusion,
            "emergency_numbers": [
                {"name": n, "number": p} for n, p in self.emergency_numbers
            ],
        }
