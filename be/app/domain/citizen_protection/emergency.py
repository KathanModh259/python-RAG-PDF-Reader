EMERGENCY_HELPLINES: list[tuple[str, str, str]] = [
    ("Women Helpline", "181", "National helpline for women in distress"),
    ("NALSA Legal Aid", "15100", "Free legal aid helpline"),
    ("Police Emergency", "112", "National police emergency number"),
    ("Cyber Crime", "1930", "National cyber crime reporting helpline"),
    ("Child Helpline", "1098", "National helpline for children in distress"),
    ("National Human Rights Commission", "15446", "Human rights complaints"),
    ("Anti-Corruption Helpline", "1800118800", "Central Vigilance Commission"),
    ("Senior Citizen Helpline", "14567", "Help for senior citizens"),
    ("Disaster Management", "1070", "National disaster response"),
]

URGENT_KEYWORDS: list[str] = [
    "domestic violence", "gharelu hinsa", "mar peet", "dowry", "dahej",
    "threat to life", "jaan ka khatra", "kidnap", "apaharan",
    "child abuse", "sexual assault", "rape", "balatkar",
    "police custody", "arrest", "girftari", "warrant",
    "suicide", "aatmahatya",
]

EMERGENCY_TRIGGER_WORDS: set[str] = {
    "kill", "murder", "death threat", "acid", "rape", "beaten",
    "torture", "hostage", "gun", "weapon", "bleeding", "hospital",
}


def detect_urgency(text: str) -> bool:
    lower = text.lower()
    for word in EMERGENCY_TRIGGER_WORDS:
        if word in lower:
            return True
    for phrase in URGENT_KEYWORDS:
        if phrase in lower:
            return True
    return False


def get_helplines_text() -> str:
    lines = ["EMERGENCY HELPLINES"]
    lines.append("=" * 40)
    for name, number, desc in EMERGENCY_HELPLINES:
        lines.append(f"{name}: {number}")
    lines.append("")
    return "\n".join(lines)
