RIGHTS_DB: list[dict] = [
    {"right": "Right to Free Legal Aid", "provision": "Article 39A, Constitution of India; Legal Services Authorities Act 1987", "plain": "If you cannot afford a lawyer, you can get one for FREE. Call NALSA helpline 15100 or visit your District Legal Services Authority (DLSA).", "category": "legal_aid"},
    {"right": "Right to Free Copy of FIR", "provision": "Section 207 BNSS (formerly Section 207 CrPC); Satya Prakash Singh vs State of UP", "plain": "If an FIR is filed against you or by you, you have the right to get a free copy within 24 hours. No one can deny this.", "category": "criminal"},
    {"right": "Right to Bail in Bailable Offenses", "provision": "Section 478 BNSS (formerly Section 436 CrPC)", "plain": "For less serious offenses, the police MUST release you on bail immediately. They cannot keep you in custody. The bail amount cannot be excessive.", "category": "criminal"},
    {"right": "Right to be Produced Before Magistrate within 24 Hours", "provision": "Article 22(2), Constitution of India; Section 57 BNSS", "plain": "If arrested, police must produce you before a magistrate within 24 hours (excluding travel time). If they don't, the detention is illegal.", "category": "criminal"},
    {"right": "Right Against Self-Incrimination", "provision": "Article 20(3), Constitution of India", "plain": "No one can force you to be a witness against yourself. You have the right to remain silent and not answer questions that might incriminate you.", "category": "criminal"},
    {"right": "Right to Information (RTI)", "provision": "Right to Information Act 2005", "plain": "You can get any government document or information by filing an RTI application. Cost: Rs. 10 only. Use this to collect evidence against exploitation.", "category": "governance"},
    {"right": "Right to Emergency Medical Care", "provision": "Parmanand Katara vs Union of India; Section 166B BNSS", "plain": "Hospitals cannot refuse treatment to an injured or sick person, even without police papers or identification. Medical help comes first.", "category": "health"},
    {"right": "Right to Zero FIR", "provision": "Supreme Court guidelines; Lalita Kumari vs Government of UP", "plain": "You can file an FIR at ANY police station, even if the crime happened elsewhere. They MUST register it and transfer it. They cannot refuse.", "category": "criminal"},
    {"right": "Right to Free Legal Services in Custody", "provision": "Legal Services Authorities Act 1987; Sheela Barse vs State of Maharashtra", "plain": "If you are in police custody or jail, the magistrate must inform you of your right to free legal aid. If they don't, the proceedings can be challenged.", "category": "criminal"},
    {"right": "Protection from Arrest Without Warrant (for non-cognizable offenses)", "provision": "Section 35 BNSS (formerly Section 41 CrPC)", "plain": "For less serious offenses, police cannot arrest you without a warrant from a magistrate. They can only issue a notice asking you to appear.", "category": "criminal"},
    {"right": "Right to Equal Pay for Equal Work", "provision": "Article 39(d), Constitution of India; Equal Remuneration Act 1976", "plain": "Men and women must be paid the same salary for the same work. If your employer pays you less because of gender, that is illegal.", "category": "employment"},
    {"right": "Right to Minimum Wage", "provision": "Minimum Wages Act 1948", "plain": "Your employer MUST pay you at least the minimum wage set by the government. If they pay less, you can complain to the Labour Department.", "category": "employment"},
    {"right": "Right to Safe Working Conditions", "provision": "Factories Act 1948; Occupational Safety and Health Code 2020", "plain": "Your workplace must be safe. Proper ventilation, fire exits, clean drinking water, and toilets are legal requirements.", "category": "employment"},
    {"right": "Protection from Sexual Harassment at Workplace", "provision": "POSH Act 2013 (Sexual Harassment of Women at Workplace Act)", "plain": "Every workplace must have an Internal Complaints Committee (ICC) to handle sexual harassment complaints. You can file a complaint without fear.", "category": "employment"},
    {"right": "Right to Register FIR for Free", "provision": "Section 173 BNSS; Police Standing Orders", "plain": "Filing an FIR is FREE. Police cannot charge you money to register a complaint. If they ask for money, complain to the SP/DCP.", "category": "criminal"},
    {"right": "Right to Medical Examination After Arrest", "provision": "Section 53 BNSS; D.K. Basu vs State of West Bengal", "plain": "If arrested, you have the right to be examined by a doctor. If you are injured, get the injuries recorded in the medical report.", "category": "criminal"},
    {"right": "Right to Inform Family About Arrest", "provision": "D.K. Basu vs State of West Bengal (Supreme Court)", "plain": "When arrested, you have the right to tell your family or a friend. Police MUST inform them within 24 hours.", "category": "criminal"},
    {"right": "Right Against Handcuffing (General Rule)", "provision": "Prem Shankar Shukla vs Delhi Administration (Supreme Court)", "plain": "Police cannot handcuff you as a routine. Handcuffing is only allowed in exceptional circumstances. If handcuffed without reason, it is illegal.", "category": "criminal"},
    {"right": "Right to Default Bail", "provision": "Section 187 BNSS (formerly Section 167 CrPC)", "plain": "If police do not file their charge sheet within the allowed time (60 or 90 days), you have the RIGHT to be released on bail. This is called default bail.", "category": "criminal"},
    {"right": "Right to Compensation for Illegal Detention", "provision": "Article 21; Rudul Sah vs State of Bihar (Supreme Court)", "plain": "If police keep you illegally (without following the law), you can claim compensation from the court. The government has to pay you for the injustice.", "category": "criminal"},
    {"right": "Right to Protection from Domestic Violence", "provision": "Protection of Women from Domestic Violence Act 2005", "plain": "If you are a woman facing physical, emotional, sexual, or economic abuse at home, you can file a complaint. You can get protection orders, monetary relief, and a place to stay.", "category": "family"},
    {"right": "Right to Maintenance (Financial Support)", "provision": "Section 125 BNSS; Section 24 Hindu Marriage Act; Section 20 DV Act", "plain": "If your spouse does not support you financially after separation, you can claim maintenance. The court decides how much they must pay monthly.", "category": "family"},
    {"right": "Right to Streedhan (Woman's Property)", "provision": "Hindu law; Supreme Court judgments", "plain": "Gifts given to a woman before, during, or after marriage (streedhan) are HER property. Husband or in-laws cannot keep or sell it without her consent.", "category": "family"},
    {"right": "Right to Live in Shared Household", "provision": "Section 17, Domestic Violence Act 2005", "plain": "A woman has the right to live in her marital home (shared household), even if she is facing domestic violence. She cannot be thrown out.", "category": "family"},
    {"right": "Right Against Dowry Demand", "provision": "Dowry Prohibition Act 1961; Section 304B BNS (IPC 498A)", "plain": "Demanding dowry (dahej) is illegal. If your husband or in-laws demand more money or gifts after marriage, you can file a complaint.", "category": "family"},
    {"right": "Right to Custody and Visitation of Children", "provision": "Guardians and Wards Act 1890; Hindu Minority and Guardianship Act", "plain": "Both parents have rights over their children. In case of separation, the court decides what is in the 'best interest of the child.' Mothers get equal consideration.", "category": "family"},
    {"right": "Right to Consumer Protection", "provision": "Consumer Protection Act 2019", "plain": "If you buy a defective product or get poor service, you can file a complaint in the Consumer Forum. Filing costs very little. No lawyer needed for small claims.", "category": "consumer"},
    {"right": "Right to Refund and Return", "provision": "Consumer Protection Act 2019; e-commerce rules", "plain": "If you buy something online, you have the right to return it within the return period. The seller cannot refuse a refund for genuine defects.", "category": "consumer"},
    {"right": "Right Against Unfair Trade Practices", "provision": "Consumer Protection Act 2019", "plain": "False advertising, misleading prices, selling fake products -- all are illegal. You can complain to the Consumer Forum or file on the National Consumer Helpline (1915).", "category": "consumer"},
    {"right": "Right to Product Safety", "provision": "Consumer Protection Act 2019; BIS standards", "plain": "Products must be safe to use. If a product harms you (e.g., defective appliance causes fire), you can claim compensation.", "category": "consumer"},
    {"right": "Right to Rent Receipt", "provision": "Income Tax Act (for HRA); Rent Control Acts", "plain": "Your landlord must give you a rent receipt when you pay. This is your proof. Without it, you cannot claim HRA and cannot prove you are a tenant.", "category": "property"},
    {"right": "Right Against Unlawful Eviction", "provision": "Rent Control Acts (state-specific); Transfer of Property Act", "plain": "A landlord cannot throw you out without following the legal process. They must give you notice, go to court, and get an eviction order. Forced eviction is illegal.", "category": "property"},
    {"right": "Right to Get Your Security Deposit Back", "provision": "Transfer of Property Act 1882; Contract Act", "plain": "When you vacate a rental property, the landlord must return your security deposit minus actual damages. They cannot deduct money without proof.", "category": "property"},
    {"right": "Right to Register Property in Your Name", "provision": "Registration Act 1908", "plain": "When you buy property, the sale deed must be registered at the Sub-Registrar's office. Registration is proof of ownership. Unregistered agreements have limited legal value.", "category": "property"},
    {"right": "Right to Mutation of Property Records", "provision": "State land revenue codes", "plain": "After buying property, get your name entered in the government land records (mutation). This is your proof that the government recognizes you as the owner.", "category": "property"},
    {"right": "Right to Equality (Before Law)", "provision": "Article 14, Constitution of India", "plain": "Every person is equal before the law. The government cannot discriminate between people in similar situations. Rich or poor, the same law applies.", "category": "fundamental"},
    {"right": "Right Against Discrimination", "provision": "Article 15, Constitution of India", "plain": "The state cannot discriminate against you based on religion, race, caste, sex, or place of birth. Everyone gets equal opportunity.", "category": "fundamental"},
    {"right": "Right to Freedom of Speech and Expression", "provision": "Article 19(1)(a), Constitution of India", "plain": "You have the right to say what you think, write what you believe, and express yourself, within reasonable limits. No one can silence you without legal cause.", "category": "fundamental"},
    {"right": "Right to Assemble Peacefully", "provision": "Article 19(1)(b), Constitution of India", "plain": "You have the right to gather peacefully with others for a lawful purpose. You can protest, march, or hold meetings without arms.", "category": "fundamental"},
    {"right": "Right to Form Associations", "provision": "Article 19(1)(c), Constitution of India", "plain": "You can form or join unions, associations, societies, or cooperatives. No one can stop you from organizing with others.", "category": "fundamental"},
    {"right": "Right to Move Freely Throughout India", "provision": "Article 19(1)(d), Constitution of India", "plain": "You can travel anywhere in India and live in any part of the country. No state can stop you from entering or staying.", "category": "fundamental"},
    {"right": "Right to Practice Any Profession", "provision": "Article 19(1)(g), Constitution of India", "plain": "You can choose any profession, business, or occupation. The government cannot force you into a particular job or stop you from working.", "category": "fundamental"},
    {"right": "Right to Life and Personal Liberty", "provision": "Article 21, Constitution of India", "plain": "Your life and freedom cannot be taken away except by following the law. This is the most important fundamental right. It includes the right to live with dignity.", "category": "fundamental"},
    {"right": "Right to Education", "provision": "Article 21A, Constitution of India; RTE Act 2009", "plain": "Every child aged 6-14 has the right to free and compulsory education. No child can be denied admission or thrown out of school.", "category": "fundamental"},
    {"right": "Right to Constitutional Remedies", "provision": "Article 32, Constitution of India", "plain": "If your fundamental rights are violated, you can directly go to the Supreme Court or High Court for justice. You do not need to go through lower courts first.", "category": "fundamental"},
    {"right": "Protection Against Double Jeopardy", "provision": "Article 20(2), Constitution of India", "plain": "You cannot be tried and punished twice for the same crime. If you were already acquitted, you cannot be charged again for the same offense.", "category": "criminal"},
    {"right": "Protection Against Ex-Post-Facto Laws", "provision": "Article 20(1), Constitution of India", "plain": "You cannot be punished for something that was NOT a crime when you did it. Laws cannot be applied backwards to punish you.", "category": "criminal"},
    {"right": "Right to File Complaint Under POSH Act (even after leaving job)", "provision": "POSH Act 2013; Section 4", "plain": "If you faced sexual harassment at work, you can file a complaint even after leaving the job. The employer must still investigate.", "category": "employment"},
    {"right": "Right Against Unlawful Strike / Lockout", "provision": "Industrial Disputes Act 1947", "plain": "Workers have the right to strike, and employers have the right to lockout, but both must follow the law. Illegal strikes or lockouts can be challenged.", "category": "employment"},
    {"right": "Right to Provident Fund", "provision": "Employees' Provident Fund Act 1952", "plain": "If you work in a company with 20+ employees, both you and your employer must contribute to PF. This is YOUR retirement money. Check your passbook regularly.", "category": "employment"},
    {"right": "Right to Insurance (ESI)", "provision": "Employees State Insurance Act 1948", "plain": "If you earn less than Rs. 21,000/month, you and your employer contribute to ESI. This gives you free medical treatment, maternity benefits, and disability support.", "category": "employment"},
    {"right": "Right Against Bonded Labour", "provision": "Bonded Labour System (Abolition) Act 1976; Article 23", "plain": "Forcing someone to work to pay off a debt is illegal. If you or someone you know is forced to work without freedom, report immediately. Helpline: 15100.", "category": "employment"},
    {"right": "Right to Report Cyber Crime Anonymously", "provision": "IT Act 2000; Cyber Crime Portal", "plain": "You can report cyber crimes (online fraud, harassment, hacking) anonymously through the national cyber crime portal. Your identity is protected.", "category": "cyber"},
    {"right": "Right to Remove Intimate Images (Right to Be Forgotten)", "provision": "IT Act 2000; Supreme Court guidelines", "plain": "If someone shares your private photos or videos without consent, you can file a complaint. You also have the right to request removal of such content.", "category": "cyber"},
    {"right": "Right Against Online Harassment", "provision": "IT Act 2000; Section 78 BNS (IPC 354D - stalking)", "plain": "Repeated online harassment, threats, or stalking is a crime. Save screenshots and file a complaint on cybercrime.gov.in or call 1930.", "category": "cyber"},
    {"right": "Right to File Zero FIR for Cyber Crime", "provision": "IT Act 2000; Supreme Court guidelines", "plain": "You can file a cyber crime complaint at ANY police station. If the local police refuse, go to the cyber crime cell or file online at cybercrime.gov.in.", "category": "cyber"},
    {"right": "Right Against Fake News and Misinformation", "provision": "IT Act 2000; Section 505 BNS (IPC)", "plain": "Spreading fake news that causes fear or violence is a crime. If someone spreads false information about you, you can take legal action.", "category": "cyber"},
    {"right": "Right to File FIR for Online Fraud", "provision": "IT Act 2000; Section 318 BNS (IPC 420)", "plain": "If someone cheated you online (fake product, UPI fraud, phishing), file an FIR. Also report at cybercrime.gov.in or call 1930. Quick action can freeze the scammer's account.", "category": "cyber"},
]


CATEGORIES: dict[str, str] = {
    "criminal": "Police, Arrest, and Criminal Cases",
    "fundamental": "Your Fundamental Rights (Constitution)",
    "family": "Marriage, Divorce, and Family Matters",
    "consumer": "Shopping, Services, and Consumer Rights",
    "employment": "Job, Workplace, and Labour Rights",
    "property": "Rent, Land, and Property Rights",
    "cyber": "Internet, Social Media, and Cyber Crime",
    "legal_aid": "Free Legal Help and Aid",
    "governance": "Government, RTI, and Public Services",
    "health": "Health, Medical Care, and Emergencies",
}


def find_relevant_rights(text: str, max_results: int = 8) -> list[dict]:
    lower = text.lower()
    scored = []
    for right in RIGHTS_DB:
        score = 0
        right_words = set(right["plain"].lower().split())
        query_words = set(lower.split())
        common = right_words & query_words
        score += len(common) * 2

        category_keywords = CATEGORIES.get(right["category"], "").lower()
        cat_words = set(category_keywords.split())
        common_cat = cat_words & query_words
        score += len(common_cat)

        for trigger in ["arrest", "bail", "fir", "police", "lawyer", "court", "cheque", "rent", "landlord", "property", "consumer", "product", "job", "salary", "work", "marriage", "divorce", "dowry", "domestic", "cyber", "online", "fraud", "rti", "information", "education", "school"]:
            if trigger in lower and trigger in right["right"].lower():
                score += 5

        if score > 0:
            scored.append((score, right))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max_results]]


def get_rights_by_category() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for right in RIGHTS_DB:
        cat = right.get("category", "other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(right)
    return grouped
