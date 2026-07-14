import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.core.anti_exploitation.scam_detector import (
    FeeSanityChecker,
    RightsReminder,
    ScamDetector,
)
from app.core.localization.detector import language_detector
from app.domain.glossary.data import explain_text
from app.domain.rights.data import find_relevant_rights


@dataclass
class OpinionResult:
    query: str
    detected_language: str = "en"
    scam_flags: list[dict] = field(default_factory=list)
    fee_check: Optional[dict] = None
    relevant_rights: list[dict] = field(default_factory=list)
    relevant_glossary_terms: list[dict] = field(default_factory=list)
    alternative_perspective: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "detected_language": self.detected_language,
            "scam_flags": self.scam_flags,
            "fee_check": self.fee_check,
            "relevant_rights": self.relevant_rights,
            "relevant_glossary_terms": self.relevant_glossary_terms,
            "alternative_perspective": self.alternative_perspective,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }


class SecondOpinionEngine:
    def __init__(self):
        self.scam_detector = ScamDetector()
        self.fee_checker = FeeSanityChecker()
        self.rights_reminder = RightsReminder()

    def evaluate(self, query: str, context: Optional[str] = None) -> OpinionResult:
        text = context or query
        lang = language_detector.detect(query)

        result = OpinionResult(
            query=query,
            detected_language=lang,
            scam_flags=self._check_scams(text),
            fee_check=self._check_fees(text),
            relevant_rights=self._get_rights(text),
            relevant_glossary_terms=self._get_glossary(text),
            alternative_perspective=self._generate_alternative_perspective(text),
            strengths=self._identify_strengths(text),
            weaknesses=self._identify_weaknesses(text),
            recommendations=self._generate_recommendations(text),
        )

        if result.scam_flags:
            result.weaknesses.append(
                "Document contains potential scam indicators. Review with caution."
            )

        return result

    def _check_scams(self, text: str) -> list[dict]:
        return self.scam_detector.analyze(text)

    def _check_fees(self, text: str) -> Optional[dict]:
        matter_type = self.fee_checker.detect_matter_type(text)
        if not matter_type:
            return None
        quoted = self._extract_quoted_fee(text)
        if quoted is None:
            return {
                "matter_type": matter_type,
                "message": f"Matter appears to be '{matter_type}'. No specific fee quoted for verification.",
            }
        return self.fee_checker.check_fee(matter_type, quoted)

    def _get_rights(self, text: str) -> list[dict]:
        return find_relevant_rights(text)

    def _get_glossary(self, text: str) -> list[dict]:
        return explain_text(text)

    def _generate_alternative_perspective(self, text: str) -> str:
        return self._build_alternative_perspective(text)

    def _identify_strengths(self, text: str) -> list[str]:
        strengths = []
        text_lower = text.lower()
        if any(w in text_lower for w in ["fir", "complaint", "report"]):
            strengths.append("Formal documentation creates an official record")
        if any(w in text_lower for w in ["legal aid", "free", "dlsa", "nalsa"]):
            strengths.append("Awareness of free legal aid resources")
        if any(w in text_lower for w in ["police", "fir", "zero fir"]):
            strengths.append("Engaging with law enforcement through proper channels")
        if any(w in text_lower for w in ["rti", "information", "document"]):
            strengths.append("Using RTI as an evidence-gathering tool")
        if not strengths:
            strengths.append("Seeking legal information proactively")
        return strengths

    def _identify_weaknesses(self, text: str) -> list[str]:
        weaknesses = []
        text_lower = text.lower()
        if "lawyer" in text_lower and "legal aid" not in text_lower:
            weaknesses.append("May need lawyer -- explore free legal aid via DLSA first")
        if "sign" in text_lower and "lawyer" not in text_lower:
            weaknesses.append("Avoid signing documents without independent legal review")
        if "pay" in text_lower and "money" in text_lower:
            weaknesses.append("Verify payment demands before transferring money")
        if "urgent" in text_lower and "deadline" in text_lower:
            weaknesses.append("Urgency tactics are often used to pressure you into bad decisions")
        return weaknesses

    def _generate_recommendations(self, text: str) -> list[str]:
        recs = []
        text_lower = text.lower()
        if any(w in text_lower for w in ["notice", "summons", "legal notice"]):
            recs.append("DO NOT ignore this notice -- respond within the given time")
            recs.append("Get a free copy of the notice evaluated by DLSA if unsure")
        if any(w in text_lower for w in ["cheque", "bounce", "ni act"]):
            recs.append("Section 138 NI Act notice must be responded within 15 days")
            recs.append("Preserve all bank records and cheque return memos as evidence")
        if any(w in text_lower for w in ["arrest", "police", "custody"]):
            recs.append("You have the right to remain silent under Article 20(3)")
            recs.append("Ask for legal aid immediately -- call NALSA 15100 or DLSA")
        if any(w in text_lower for w in ["property", "land", "sale", "registry"]):
            recs.append("Always verify property title through encumbrance certificate")
            recs.append("Register all property transactions (unregistered deeds have limited value)")
        if any(w in text_lower for w in ["divorce", "maintenance", "alimony"]):
            recs.append("Try mediation first -- it saves time, money, and relationships")
            recs.append("Women: you are entitled to free legal aid and maintenance during proceedings")
        if not recs:
            recs.append("Document all communications and evidence in writing")
            recs.append("Keep copies of every document related to your case")
        return recs

    def _build_alternative_perspective(self, text: str) -> str:
        text_lower = text.lower()
        if "notice" in text_lower and ("arrears" in text_lower or "due" in text_lower):
            return (
                "Alternative perspective: The sender may be using a legal notice "
                "as a intimidation tactic without genuine legal basis. Verify if "
                "the sender has actually filed a case or obtained a court order. "
                "Many legal notices are sent as 'scare letters' to force payment "
                "from people who don't know their rights."
            )
        if "arrest" in text_lower or "police" in text_lower:
            return (
                "Alternative perspective: Police threats of arrest without court "
                "order may be illegal for non-cognizable offenses. If no FIR has "
                "been registered, there is no formal case. You have the right to "
                "be informed of grounds of arrest and to consult a lawyer."
            )
        if "property" in text_lower or "eviction" in text_lower:
            return (
                "Alternative perspective: Landlords and builders often threaten "
                "eviction without following due process. Forced eviction is illegal "
                "under the Transfer of Property Act and state rent control laws. "
                "Only a court can order eviction after following the proper procedure."
            )
        if "consumer" in text_lower or "product" in text_lower or "refund" in text_lower:
            return (
                "Alternative perspective: Consumer disputes can often be resolved "
                "through the National Consumer Helpline (1915) without filing a "
                "formal complaint. Mediation through consumer forums is faster "
                "and less stressful than litigation. The Consumer Protection Act "
                "provides for strict liability of sellers."
            )
        return (
            "Alternative perspective: Legal situations often have multiple valid "
            "interpretations. What seems like a straightforward case may have "
            "nuances that benefit your side. Always get complete information "
            "before making decisions. Consider consulting DLSA for free legal "
            "advice before spending money on private lawyers."
        )

    def _extract_quoted_fee(self, text: str) -> Optional[int]:
        import re
        patterns = [
            r"(?:fee|cost|charge|payment|amount|demand).{0,20}?(?:rs\.?|rupees)\s*([0-9,]+)",
            r"(?:rs\.?|rupees)\s*([0-9,]+).{0,20}?(?:fee|cost|charge|payment|amount|demand)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).replace(",", ""))
                except ValueError:
                    pass
        return None


second_opinion = SecondOpinionEngine()
