import re

GLOSSARY: dict[str, str] = {
    "acquittal": "The court says you are not guilty. You are free to go, no punishment.",
    "adjournment": "The court case is postponed to another date. Happens a lot in India.",
    "affidavit": "A written statement where you swear under oath that everything is true. Like a promise on paper.",
    "appeal": "If you lose a case, you can ask a higher court to look at it again. That is called an appeal.",
    "arrest": "Police takes you into custody because they believe you committed a crime.",
    "bail": "Temporary freedom while your case is going on. You pay some money or give a guarantee that you will come to court when needed.",
    "bailable offense": "A less serious crime where the police MUST give you bail. They cannot keep you in jail.",
    "bench warrant": "If you do not come to court when told, the judge can order the police to arrest you and bring you.",
    "caveat": "A warning you file in court saying: if anyone files a case about this matter, please hear me too before deciding.",
    "charge sheet": "The final report police files in court after investigating a crime. It says who they think committed the crime and what evidence they have.",
    "cognizable offense": "A serious crime where police can arrest you without a warrant. Like murder, robbery, rape.",
    "complaint": "A written statement to the police or court saying someone did something wrong to you.",
    "consent": "Saying yes willingly and with full understanding. Without pressure or fear.",
    "contempt of court": "Disrespecting the court or disobeying its orders. Can lead to fine or jail.",
    "conviction": "The court finds you guilty. You are punished according to law.",
    "cross examination": "When the other side's lawyer asks you questions in court to check if your story is true.",
    "damages": "Money the court orders someone to pay you because they caused you loss or harm.",
    "decree": "The final order of a civil court saying who won and what must happen.",
    "defamation": "Saying or writing something false about someone that damages their reputation.",
    "deposition": "Your sworn statement recorded outside court, usually used as evidence later.",
    "disclaimer": "A statement saying you are not responsible for something. Like the one at the end of every Nyaya Mitra response.",
    "dismissal": "The court throws out the case without fully hearing it. Can be for many reasons.",
    "domicile": "The place you call your permanent home for legal purposes.",
    "due diligence": "Carefully checking all facts and documents before signing or agreeing to something.",
    "encumbrance": "A legal claim on your property. Like if you took a loan against your house, the bank has an encumbrance.",
    "evidence": "Anything that proves or disproves a fact in court. Documents, photos, witness statements, etc.",
    "ex parte": "When the court hears one side because the other side did not come. The absent person loses by default.",
    "FIR": "First Information Report. The first written record of a crime made at a police station. The starting point of a criminal case.",
    "garnishee order": "Court orders your bank or employer to give your money to someone you owe.",
    "habeas corpus": "A court order saying: 'produce the person in court.' Used when someone is illegally detained. Your fundamental right under Article 226.",
    "injunction": "A court order telling someone to STOP doing something or to DO something. Like a STOP sign from the judge.",
    "interim order": "A temporary order until the final decision. Like temporary maintenance, temporary custody, etc.",
    "ipso facto": "Latin for 'by the fact itself.' Means something is true automatically because of the situation.",
    "judgment": "The final decision of the court after hearing both sides.",
    "jurisdiction": "Which court has the power to hear your case. Depends on where you live, how much money is involved, and what kind of case.",
    "laches": "If you wait too long to file a case, the court may refuse to hear it. 'Delay defeats equity.'",
    "legal aid": "Free legal help for people who cannot afford a lawyer. Available through DLSA and NALSA.",
    "liable": "Legally responsible. If you are liable, you have to pay or face consequences.",
    "litigation": "The process of taking a case through court. Going to court, filing papers, hearings, etc.",
    "locus standi": "The right to file a case. You must show that you are personally affected by the issue.",
    "magistrate": "A lower court judge who handles smaller criminal cases and grants bail.",
    "maintenance": "Money paid by one spouse to the other after separation or divorce for daily expenses.",
    "mandamus": "A court order commanding a government official to do their duty. Your right if a government office is not doing its job.",
    "mediation": "A process where a neutral person helps both sides reach a settlement without going to court. Like a friendly negotiator.",
    "misdemeanor": "A less serious crime (rarely used in India, we use 'bailable offense' instead).",
    "moot": "No longer relevant. If the issue is already resolved, the case becomes moot.",
    "mutation": "Changing the name of the owner in government land records after a property transfer.",
    "non-bailable offense": "A serious crime where bail is not a right. The court decides based on the case. Murder, rape, etc.",
    "notice": "A formal written warning or information. Like a legal notice, eviction notice, termination notice.",
    "oath": "Swearing to tell the truth. Usually done before giving evidence in court.",
    "order": "A direction from the court that must be followed. Not the final judgment, but an instruction during the case.",
    "parole": "Temporary release of a convicted person from jail for a specific reason (medical, family emergency).",
    "petition": "A formal written request to a court asking for something specific.",
    "plaint": "The document that starts a civil case. It describes what happened and what relief you want.",
    "plaintiff": "The person who files a civil case (the one complaining).",
    "plea": "Your answer to a criminal charge: guilty or not guilty.",
    "power of attorney": "A legal document giving someone the authority to act on your behalf. Like a permission slip.",
    "prima facie": "Latin for 'at first look.' If there is prima facie evidence, it means there is enough evidence to proceed with the case.",
    "probate": "A court process to prove that a will is valid and legal.",
    "pro bono": "Free legal services. Lawyers helping people without charging fees.",
    "prosecution": "The government lawyer who argues that you committed a crime. Their job is to prove you guilty.",
    "quo warranto": "A court order asking someone: 'by what authority are you holding this public position?'",
    "recess": "A short break in court proceedings. Like lunch recess.",
    "recovery": "Getting back something that belongs to you, especially money. A recovery suit is to get your money back.",
    "remedy": "What the court can give you to fix your problem. Money, an order, or a declaration.",
    "res judicata": "If a court has already decided a matter, you cannot file the same case again. The matter is closed.",
    "respondent": "The person who has to answer a petition. The one being complained about.",
    "restitution": "Returning something to its rightful owner. Or paying back what was taken.",
    "sanction": "Permission. For example, to prosecute a government official, you need government sanction.",
    "search warrant": "A court order allowing police to search your home or property for evidence.",
    "sentence": "The punishment given by the court after a conviction. Fine, jail time, or both.",
    "sequestration": "Taking away property until a court order is obeyed.",
    "service": "Formally delivering legal documents to the other party. Like a summons or notice.",
    "settlement": "Agreement between both sides before the final judgment. Both parties agree on a solution.",
    "show cause": "A court order asking you to explain why something should not be done against you.",
    "standing": "Same as locus standi. Your right to file a case because you are personally affected.",
    "status quo": "Keep things as they are right now. A court often orders this until the final decision.",
    "stay order": "A court order that STOPS something from happening until further orders. Like stopping an eviction.",
    "sub judice": "A matter that is currently being decided by a court. Under judicial consideration.",
    "subpoena": "A court order requiring you to appear in court or produce documents.",
    "suit": "Another word for a civil case. A lawsuit.",
    "summons": "A document telling you that a case has been filed against you and you must appear in court on a specific date.",
    "tenancy": "The right to live in or use someone else's property after paying rent. Your rights as a tenant.",
    "tort": "A civil wrong that causes harm or loss to someone. Not a crime, but you can sue for compensation.",
    "trespass": "Entering someone's property without permission.",
    "trust": "A legal arrangement where one person holds property for the benefit of another.",
    "ultra vires": "Latin for 'beyond powers.' When someone acts beyond their legal authority. The action is invalid.",
    "unilateral": "Done by one side only. Without agreement of the other party.",
    "vacate": "To leave a property. A vacate order means you must move out. Also, to cancel an earlier court order.",
    "verdict": "The decision of the judge or jury in a case. Guilty or not guilty, liable or not liable.",
    "void": "Having no legal effect. Like it never happened. A void contract cannot be enforced.",
    "voidable": "Valid until one party decides to cancel it. Like a contract signed under pressure.",
    "waiver": "Voluntarily giving up a known right. If you waive something, you cannot claim it later.",
    "warrant": "A court order authorizing police to arrest someone or search a place.",
    "will": "A legal document saying who gets your property after you die.",
    "witness": "Someone who saw or knows something relevant to the case and tells the court about it.",
    "writ": "A formal written order from a court. In India, High Courts can issue writs under Article 226.",
    "zero FIR": "An FIR you can file at ANY police station, even if the crime happened elsewhere. They must take it and transfer it.",
    "DLSA": "District Legal Services Authority. Government office in every district that provides FREE lawyers to those who qualify.",
    "NALSA": "National Legal Services Authority. The central body that runs free legal aid across India. Helpline: 15100.",
    "Lok Adalat": "A people's court. Free, informal settlement process where cases are resolved amicably. No court fees.",
    "Nyaya Bandhu": "Pro bono legal mentorship program connecting young lawyers with those who need free legal help.",
    "RTI": "Right to Information. A law that lets you ask any government office for information. Costs only Rs. 10.",
    "e-FIR": "FIR filed online. Available in many states for certain crimes. No need to go to the police station physically.",
    "IPC": "Indian Penal Code, 1860. The main criminal code of India. Being replaced by BNS (Bharatiya Nyaya Sanhita).",
    "CrPC": "Code of Criminal Procedure, 1973. The law that says how criminal cases are processed. Being replaced by BNSS.",
    "BNS": "Bharatiya Nyaya Sanhita, 2023. The new criminal code replacing IPC.",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023. The new procedural law replacing CrPC.",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023. The new evidence law replacing the Indian Evidence Act, 1872.",
    "POSH Act": "Sexual Harassment of Women at Workplace Act, 2013. Protects women from sexual harassment at work.",
    "DV Act": "Protection of Women from Domestic Violence Act, 2005. Protects women from domestic violence.",
    "CP Act": "Consumer Protection Act, 2019. Protects consumers from unfair practices by sellers and service providers.",
    "NI Act": "Negotiable Instruments Act, 1881. The law about cheques, promissory notes, and bills of exchange.",
}


def lookup_term(word: str) -> str | None:
    word_clean = word.strip().rstrip(".,;:!?").lower()
    for key, val in GLOSSARY.items():
        if key.lower() == word_clean:
            return f"{key}: {val}"
        if key.lower().startswith(word_clean) and len(word_clean) >= 3:
            return f"{key}: {val}"
        if word_clean.startswith(key.lower()) and len(word_clean) <= len(key) + 2:
            return f"{key}: {val}"
    return None


def explain_text(text: str) -> list[str]:
    found = set()
    explanations = []
    words = set(re.findall(r"[A-Za-z]{4,}(?:\s+[A-Za-z]{2,})?", text))
    for word in words:
        explanation = lookup_term(word)
        if explanation and word.lower() not in found:
            explanations.append(explanation)
            found.add(word.lower())
    return explanations[:10]
