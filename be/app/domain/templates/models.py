from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TemplateType(str, Enum):
    FIR = "fir"
    RTI = "rti"
    LEGAL_NOTICE_REPLY = "legal_notice_reply"
    CONSUMER_COMPLAINT = "consumer_complaint"
    DV_COMPLAINT = "domestic_violence_complaint"
    LEGAL_AID_APPLICATION = "legal_aid_application"
    MONEY_RECOVERY_NOTICE = "money_recovery_notice"

    def display_name(self) -> str:
        names = {
            "fir": "FIR (First Information Report)",
            "rti": "RTI Application",
            "legal_notice_reply": "Reply to Legal Notice",
            "consumer_complaint": "Consumer Complaint",
            "domestic_violence_complaint": "Domestic Violence Complaint",
            "legal_aid_application": "Free Legal Aid Application",
            "money_recovery_notice": "Legal Notice for Money Recovery",
        }
        return names[self.value]

    def applicable_acts(self) -> list[str]:
        acts = {
            "fir": [
                "Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023, Section 173",
                "Code of Criminal Procedure 1973, Section 154 (before BNSS adoption)",
            ],
            "rti": ["Right to Information Act 2005"],
            "legal_notice_reply": ["Indian Contract Act 1872", "Specific Relief Act 1963"],
            "consumer_complaint": ["Consumer Protection Act 2019"],
            "domestic_violence_complaint": [
                "Protection of Women from Domestic Violence Act 2005",
            ],
            "legal_aid_application": [
                "Legal Services Authorities Act 1987",
                "Article 39A, Constitution of India",
            ],
            "money_recovery_notice": [
                "Indian Contract Act 1872",
                "Negotiable Instruments Act 1881",
            ],
        }
        return acts[self.value]

    def description(self) -> str:
        descs = {
            "fir": "Report a crime to the police. Use this to register a First Information Report for any criminal offense.",
            "rti": "Request information from any government department. Cost: Rs. 10. Get documents, file records, decisions.",
            "legal_notice_reply": "Respond to a legal notice sent to you. Protect your rights when someone threatens legal action.",
            "consumer_complaint": "File a complaint against a seller or service provider for defective goods or poor service.",
            "domestic_violence_complaint": "Seek protection from domestic violence. Get protection orders, monetary relief, and shelter.",
            "legal_aid_application": "Apply for free legal aid if you cannot afford a lawyer. Available through DLSA.",
            "money_recovery_notice": "Send a legal notice demanding payment of money owed to you. First step before filing a suit.",
        }
        return descs[self.value]


@dataclass
class TemplateField:
    key: str
    label: str
    description: str
    required: bool = True
    placeholder: str = ""
    field_type: str = "text"

    def __post_init__(self):
        if not self.placeholder:
            self.placeholder = f"Enter {self.label.lower()}"


@dataclass
class DocumentTemplate:
    template_type: TemplateType
    title: str
    fields: list[TemplateField]
    body_markdown: str
    instructions: str = ""
    fee_info: str = ""

    def validate_fields(self, data: dict[str, str]) -> list[str]:
        missing = []
        for field_def in self.fields:
            if field_def.required and (field_def.key not in data or not data[field_def.key].strip()):
                missing.append(field_def.label)
        return missing


FIR_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.FIR,
    title="First Information Report (FIR)",
    fields=[
        TemplateField("police_station", "Police Station", "Name of police station where FIR is being filed"),
        TemplateField("district", "District", "District where the police station is located"),
        TemplateField("date", "Date of Incident", "Date when the incident occurred", placeholder="DD/MM/YYYY"),
        TemplateField("time", "Time of Incident", "Time when the incident occurred", placeholder="HH:MM AM/PM"),
        TemplateField("place", "Place of Incident", "Full address/description of where the incident happened"),
        TemplateField("complainant_name", "Complainant Name", "Your full name"),
        TemplateField("complainant_father", "Father/Husband Name", "Your father's or husband's name"),
        TemplateField("complainant_address", "Complainant Address", "Your complete residential address"),
        TemplateField("complainant_phone", "Complainant Phone", "Your mobile/phone number", placeholder="10-digit mobile number"),
        TemplateField("accused_name", "Accused Name(s)", "Full name(s) of person(s) you are reporting", required=False),
        TemplateField("accused_description", "Accused Description", "Description, address, or identifying details of accused", required=False),
        TemplateField("incident_details", "Incident Description", "Full description of what happened -- write in chronological order"),
        TemplateField("witnesses", "Witness Name(s)", "Names and contact of any witnesses, if any", required=False),
        TemplateField("property_loss", "Property Loss (if any)", "Details of any property stolen or damaged", required=False),
        TemplateField("injuries", "Injuries (if any)", "Details of any injuries sustained", required=False),
    ],
    body_markdown="""# First Information Report (FIR)

**Under Section 173, Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023**

---

## Information

| Field | Details |
|-------|---------|
| Police Station | **{police_station}** |
| District | **{district}** |
| Date of Incident | **{date}** |
| Time of Incident | **{time}** |
| Place of Incident | **{place}** |

## Complainant Details

| Field | Details |
|-------|---------|
| Name | **{complainant_name}** |
| Father/Husband Name | **{complainant_father}** |
| Address | **{complainant_address}** |
| Phone | **{complainant_phone}** |

## Details of Incident

I, {complainant_name}, son/daughter/wife of {complainant_father}, resident of {complainant_address}, do hereby state as follows:

On {date} at approximately {time}, at {place}, the following incident occurred:

{incident_details}

{f"{'The accused person(s) is/are: ' + accused_name + '.'}' if accused_name else ''}
{f"{'Description of accused: ' + accused_description + '.'}' if accused_description else ''}
{f"{'The following persons witnessed the incident: ' + witnesses + '.'}' if witnesses else ''}
{f"{'Property loss/damage: ' + property_loss + '.'}' if property_loss else ''}
{f"{'Injuries sustained: ' + injuries + '.'}' if injuries else ''}

## Request

I request that:
1. This information be registered as a First Information Report
2. Necessary investigation be conducted
3. Legal action be taken against the accused as per law
4. A free copy of this FIR be provided to me as per Section 207 BNSS

## Declaration

I declare that the information given above is true to the best of my knowledge and belief.

**Signature:** ___________________

**Date:** ___________________

**Place:** ___________________
""",
    instructions="""## How to File this FIR

1. **Go to the police station** where the incident happened or the nearest police station
2. **Take a printed copy** of this filled form with you
3. **The police MUST register your FIR** -- if they refuse, ask to speak to the SHO (Station House Officer)
4. **If still refused**, contact:
   - Deputy Commissioner of Police (DCP) / Superintendent of Police (SP)
   - Women Helpline: 181 (for women-related crimes)
   - Cyber Crime: 1930 (for cyber crimes)
   - Or file a Zero FIR at any police station
5. **Get a free copy** of the FIR -- you are entitled to it by law
6. **Check the FIR** before signing -- make sure all facts are correctly recorded

## Important Rights
- Filing an FIR is **FREE** -- no one can charge you money
- You are entitled to a **free copy** within 24 hours
- If police refuse, you can write directly to the **Magistrate** under Section 175 BNSS
- For women: you can file a complaint at **any police station** in India""",
    fee_info="Filing an FIR is completely free of cost. No stamp paper or fee required.",
)

RTI_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.RTI,
    title="RTI Application (Right to Information Act, 2005)",
    fields=[
        TemplateField("public_authority", "Public Authority", "Name of the government department/office you are writing to"),
        TemplateField("authority_address", "Authority Address", "Full address of the department/office"),
        TemplateField("applicant_name", "Applicant Name", "Your full name"),
        TemplateField("applicant_address", "Applicant Address", "Your complete address for correspondence"),
        TemplateField("applicant_phone", "Applicant Phone", "Your phone number", required=False),
        TemplateField("applicant_email", "Applicant Email", "Your email address", required=False),
        TemplateField("information_requested", "Information Requested", "Clearly describe what information/documents you want. Be specific."),
        TemplateField("purpose", "Purpose of Request", "Why you need this information (optional but helpful)", required=False),
        TemplateField("fee_mode", "Fee Payment Mode", "How are you paying the Rs. 10 fee", placeholder="Cash / IPO / DD / Online"),
    ],
    body_markdown="""# RTI Application

**Under the Right to Information Act, 2005**

---

**To,**
The Public Information Officer (PIO),
{public_authority}
{authority_address}

**Date:** ___________________

**From:**
{applicant_name}
{applicant_address}
{f'Phone: {applicant_phone}' if applicant_phone else ''}
{f'Email: {applicant_email}' if applicant_email else ''}

**Subject: Request for information under RTI Act, 2005**

Sir/Madam,

In exercise of the right conferred by Section 6 of the Right to Information Act, 2005, I hereby request the following information:

---

**Information Requested:**

{information_requested}

{f'**Purpose/Background:** {purpose}' if purpose else ''}

---

**Details:**

1. Please provide the above information in {place here: written/printed/photocopy/digital} format
2. I am enclosing the application fee of Rs. 10/- vide {fee_mode}
3. If the requested information falls under any exemption under Section 8 or 9 of the RTI Act, please inform me with reasons
4. If partial information can be provided, please provide the disclosable portion
5. If this application needs to be transferred to another PIO under Section 6(3), please do so

**Declaration:**

I certify that the information sought is not covered under any exempted category under Section 8 of the RTI Act.

**Yours faithfully,**

Signature: ___________________
Name: {applicant_name}
Date: ___________________

---

**FEE PAYMENT DETAILS:**
Mode: {fee_mode}
Amount: Rs. 10/-
Date: ___________________
""",
    instructions="""## How to File this RTI

1. **Address it to the PIO** (Public Information Officer) of the concerned department
2. **Fee**: Rs. 10 -- pay by:
   - Cash (get receipt)
   - Indian Postal Order (IPO) payable to the PIO
   - Demand Draft / Banker's Cheque
   - Online payment (many departments accept it)
3. **Send** by:
   - Registered Post / Speed Post
   - Hand-deliver to the department's RTI cell
   - Online through RTI portals (rtionline.gov.in for central departments)
4. **Response time**: 30 days (48 hours for life/liberty matters)
5. **If no response**: File first appeal with First Appellate Authority (FAA) within 30 days
6. **Appeal fee**: Rs. 25 (for central RTI, no fee for state RTI)

## Important
- You do **NOT** need to give a reason for seeking information
- The PIO cannot refuse without giving a valid exemption under Section 8
- Extra pages beyond 100: Rs. 2 per page (free for BPL applicants)
- Inspection of records is **free**""",
    fee_info="Application fee: Rs. 10. Additional Rs. 2 per page for more than 100 pages. Free for Below Poverty Line (BPL) applicants.",
)

LEGAL_NOTICE_REPLY_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.LEGAL_NOTICE_REPLY,
    title="Reply to Legal Notice",
    fields=[
        TemplateField("sender_name", "Notice Sender Name", "Name of the person/lawyer who sent you the notice"),
        TemplateField("sender_address", "Sender Address", "Address of the person/lawyer who sent the notice"),
        TemplateField("notice_date", "Notice Date", "Date on the notice you received", placeholder="DD/MM/YYYY"),
        TemplateField("notice_ref", "Notice Reference", "Reference number on the notice, if any", required=False),
        TemplateField("respondent_name", "Your Name", "Your full name"),
        TemplateField("respondent_address", "Your Address", "Your complete address"),
        TemplateField("response_body", "Your Response", "Your point-by-point reply to the allegations in the notice"),
        TemplateField("legal_grounds", "Legal Grounds", "Legal basis for your defense (cite specific laws if possible)", required=False),
    ],
    body_markdown="""# Reply to Legal Notice

---

**To,**
{sender_name}
{sender_address}

**Date:** ___________________

**From:**
{respondent_name}
{respondent_address}

**Reference:** Your Notice dated {notice_date}
{f'Notice Ref: {notice_ref}' if notice_ref else ''}

**Subject: Reply to legal notice dated {notice_date}**

Sir/Madam,

I acknowledge receipt of your notice dated {notice_date} {f'reference {notice_ref}' if notice_ref else ''}. I hereby respond as follows:

---

## Preliminary Submissions

1. I deny each and every allegation made in the said notice unless specifically admitted herein.

2. The notice is vague, baseless, and appears to be a coercive tactic aimed at pressuring me rather than a genuine legal grievance.

## Response to Allegations

{response_body}

{f'## Legal Grounds\n\n{legal_grounds}' if legal_grounds else ''}

## Demand / Prayer

In light of the above:
1. The notice dated {notice_date} is baseless and rejected in its entirety
2. No amount is due or payable as claimed
3. The sender is put to strict proof of each allegation
4. Any legal proceedings initiated on the basis of this notice will be defended vigorously at the sender's risk as to costs

## Caution

This reply is without prejudice to my rights and contentions. I reserve the right to take appropriate legal action including filing a counter-claim for damages/harassment caused by this baseless notice.

**Yours faithfully,**

Signature: ___________________
Name: {respondent_name}

**Copy to:** ___________________
""",
    instructions="""## How to Respond to a Legal Notice

1. **Do NOT panic** -- a legal notice is not a court order. It is a demand letter.
2. **Do NOT ignore it** -- if you ignore it, the other side may get an ex-parte order against you.
3. **Read carefully** -- understand exactly what is being claimed/alleged.
4. **Respond within the time given** -- usually 15-30 days.
5. **Keep it factual** -- stick to facts, do not get emotional.
6. **Send by registered post** -- keep proof of sending.

## Important
- If you cannot afford a lawyer, visit your **District Legal Services Authority (DLSA)** for free legal aid
- Many legal notices are **bluffing tactics** -- unethical lawyers send them to intimidate you
- A reply does NOT mean you admit anything -- it is your defense
- If the notice involves a court case already filed, you MUST respond through a lawyer""",
    fee_info="You can draft and send this reply yourself. No lawyer required for this stage. If you need legal aid, DLSA provides free lawyers. For postal charges: approximately Rs. 50 for registered post.",
)

CONSUMER_COMPLAINT_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.CONSUMER_COMPLAINT,
    title="Consumer Complaint (Consumer Protection Act, 2019)",
    fields=[
        TemplateField("consumer_forum", "Consumer Forum", "Which forum: District / State / National Commission based on claim value"),
        TemplateField("forum_address", "Forum Address", "Address of the Consumer Forum"),
        TemplateField("complainant_name", "Complainant Name", "Your full name"),
        TemplateField("complainant_address", "Complainant Address", "Your complete address"),
        TemplateField("complainant_phone", "Complainant Phone", "Your phone number", required=False),
        TemplateField("opposite_party", "Opposite Party Name", "Name of the seller/service provider you are complaining against"),
        TemplateField("opposite_party_address", "Opposite Party Address", "Address of the seller/service provider"),
        TemplateField("product_service", "Product/Service Details", "Description of the product purchased or service availed"),
        TemplateField("purchase_date", "Date of Purchase", "When did you buy the product or service?", placeholder="DD/MM/YYYY"),
        TemplateField("amount_paid", "Amount Paid", "How much did you pay?", placeholder="Rs."),
        TemplateField("complaint_details", "Complaint Details", "Describe the defect/deficiency in detail"),
        TemplateField("relief_sought", "Relief Sought", "What do you want? Refund / Replacement / Compensation / Repair", placeholder="Full refund of Rs. X + compensation of Rs. Y"),
        TemplateField("documents", "Documents Attached", "List of documents you are attaching (bill, warranty card, photos, etc.)"),
    ],
    body_markdown="""# Consumer Complaint

**Under the Consumer Protection Act, 2019**

---

**Before the Hon'ble {consumer_forum} Consumer Disputes Redressal Commission**
{forum_address}

**Complaint No.:** ___________________

## 1. Details of the Complainant

| Field | Details |
|-------|---------|
| Name | **{complainant_name}** |
| Address | **{complainant_address}** |
| Phone | **{complainant_phone}** |

## 2. Details of the Opposite Party

| Field | Details |
|-------|---------|
| Name | **{opposite_party}** |
| Address | **{opposite_party_address}** |

## 3. Facts of the Case

The complainant states as follows:

1. The complainant purchased/availed **{product_service}** from the opposite party on **{purchase_date}** for a total consideration of **Rs. {amount_paid}**.
2. The complainant paid the full amount as consideration.
3. The said product/service suffered from the following defect/deficiency:

**{complaint_details}**

4. The complainant brought this to the notice of the opposite party, but they failed to resolve the issue.
5. Hence, this complaint.

## 4. Relief Sought

The complainant prays for the following relief:

**{relief_sought}**

Along with:
- Interest on the amount @ 12% per annum from the date of payment
- Compensation for mental agony and harassment
- Cost of litigation

## 5. List of Documents Attached

{documents}

## Verification

I, {complainant_name}, do hereby verify that the contents of this complaint are true and correct to the best of my knowledge and belief. No part of this complaint is false or misleading.

**Place:** ___________________
**Date:** ___________________

**Signature:** ___________________
{complainant_name}
Complainant
""",
    instructions="""## How to File a Consumer Complaint

1. **Determine the right forum** based on claim value:
   - **District Commission**: Up to Rs. 1 Crore
   - **State Commission**: Rs. 1 Crore to Rs. 10 Crore
   - **National Commission**: Above Rs. 10 Crore
2. **Fee** is very nominal (e.g., Rs. 50 to Rs. 500 for District Forum)
3. **No lawyer required** for claims up to Rs. 50 lakhs
4. **Time limit**: File within 2 years of the cause of action
5. **Documents needed**: Bill/receipt, warranty card, photos, correspondence with seller
6. **Also try**: National Consumer Helpline (1915) for pre-litigation mediation

## Important Points
- The complaint can be filed **online** at edaakhil.nic.in
- You can file it **personally** or by **registered post**
- Cases are usually resolved within 3-6 months for simple matters
- If you win, the opposite party pays your costs
- **Free legal aid** is available through DLSA for consumer cases""",
    fee_info="District Forum: Rs. 50 (up to Rs. 1 lakh claim) to Rs. 500 (up to Rs. 10 lakh claim). Free for BPL applicants. No court fee stamp paper needed.",
)

DV_COMPLAINT_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.DV_COMPLAINT,
    title="Complaint under Domestic Violence Act, 2005",
    fields=[
        TemplateField("court", "Court / Magistrate", "Name of the court where filing the complaint", placeholder="Chief Judicial Magistrate / Metropolitan Magistrate"),
        TemplateField("court_place", "Court Place", "City/town where the court is located"),
        TemplateField("applicant_name", "Your Name", "Your full name"),
        TemplateField("applicant_age", "Your Age", "Your age"),
        TemplateField("applicant_address", "Your Address", "Your complete address"),
        TemplateField("applicant_phone", "Your Phone", "Your phone number", required=False),
        TemplateField("respondent_name", "Respondent Name", "Name of the person who committed domestic violence (husband/partner/relative)"),
        TemplateField("respondent_address", "Respondent Address", "Address of the respondent"),
        TemplateField("relationship", "Relationship", "Relationship with the respondent(s)", placeholder="Husband / Father-in-law / Partner"),
        TemplateField("marriage_date", "Date of Marriage", "Date of marriage (if applicable)", placeholder="DD/MM/YYYY", required=False),
        TemplateField("violence_details", "Violence Details", "Describe the domestic violence in detail -- physical, emotional, sexual, economic"),
        TemplateField("relief_sought", "Relief Sought", "What protection/relief do you need", placeholder="Protection order, right to stay in shared household, monetary relief, custody"),
        TemplateField("documents", "Documents Attached", "List of documents attached", required=False),
    ],
    body_markdown="""# Complaint under the Protection of Women from Domestic Violence Act, 2005

---

**Before the Hon'ble Court of {court}, {court_place}**

**Complaint No.:** ___________________

## 1. Details of the Applicant (Victim)

| Field | Details |
|-------|---------|
| Name | **{applicant_name}** |
| Age | **{applicant_age}** |
| Address | **{applicant_address}** |
| Phone | **{applicant_phone}** |

## 2. Details of the Respondent(s)

| Field | Details |
|-------|---------|
| Name(s) | **{respondent_name}** |
| Address | **{respondent_address}** |
| Relationship | **{relationship}** |

{f'| Date of Marriage | **{marriage_date}** |' if marriage_date else ''}

## 3. Facts of the Case

I, {applicant_name}, daughter/wife of ___________________, resident of {applicant_address}, do hereby state as follows:

{f'1. I was married to {respondent_name} on {marriage_date}.' if marriage_date else f'1. I am in a domestic relationship with {respondent_name}.'}

2. I have been living at {applicant_address} / the shared household.

3. The respondent(s) has/have subjected me to domestic violence as follows:

**{violence_details}**

4. Due to the above acts of domestic violence, my life, safety, and well-being are at risk.

5. I have not filed any other proceedings in relation to this matter except: ___________________

6. Hence, this complaint under Section 12 of the Protection of Women from Domestic Violence Act, 2005.

## 4. Relief Sought

It is therefore prayed that the Hon'ble Court may be pleased to pass the following orders:

**{relief_sought}**

Along with:
- **Protection Order** under Section 18 restraining respondent from committing violence
- **Residence Order** under Section 19 allowing me to stay in the shared household
- **Monetary Relief** under Section 20 for losses suffered
- **Custody Order** under Section 21 for any children
- **Compensation** under Section 22 for mental torture and emotional distress

## 5. List of Documents Attached

{f'documents' if documents else '(To be attached: medical reports, photographs, messages, call records, bank statements, etc.)'}

## Verification

I, {applicant_name}, do hereby verify that the contents of this complaint are true and correct to the best of my knowledge and belief.

**Place:** ___________________
**Date:** ___________________

**Signature:** ___________________
{applicant_name}
Applicant
""",
    instructions="""## How to Use This Complaint

1. **Where to file**: File this complaint before the **Magistrate** (Judicial Magistrate / Metropolitan Magistrate) in the area where:
   - You live / lived with the respondent, OR
   - The incident occurred, OR
   - The respondent lives
2. **Who can file**: Any woman who is or has been in a domestic relationship with the respondent
3. **No lawyer needed**: You can file this complaint yourself
4. **Police help**: You can also call the **Women Helpline (181)** for assistance
5. **Time**: The court usually gives interim protection orders within 3 days

## Rights Under DV Act
- You CANNOT be thrown out of your shared household
- You have the right to **free legal aid** through DLSA
- You can get **protection orders**, **monetary relief**, **custody of children**
- Police must assist you in serving the notice
- The court can appoint a **Protection Officer** to help you

## Emergency
If you are in immediate danger, call **112** (Police) or **181** (Women Helpline) or go to the nearest police station.""",
    fee_info="No court fee is required for filing a complaint under the Domestic Violence Act. It is designed to be accessible to all women regardless of financial status.",
)

LEGAL_AID_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.LEGAL_AID_APPLICATION,
    title="Application for Free Legal Aid (DLSA)",
    fields=[
        TemplateField("dlsa_name", "DLSA Name", "Name of the District Legal Services Authority", placeholder="DLSA [Your District Name]"),
        TemplateField("dlsa_address", "DLSA Address", "Address of the DLSA office"),
        TemplateField("applicant_name", "Applicant Name", "Your full name"),
        TemplateField("applicant_age", "Applicant Age", "Your age"),
        TemplateField("applicant_father", "Father/Husband Name", "Your father's or husband's name"),
        TemplateField("applicant_address", "Applicant Address", "Your complete residential address"),
        TemplateField("applicant_phone", "Applicant Phone", "Your phone number", required=False),
        TemplateField("annual_income", "Annual Income", "Your annual family income from all sources", placeholder="Rs."),
        TemplateField("category", "Eligibility Category", "Why you qualify for free legal aid", placeholder="SC/ST/Woman/Child/Victim of trafficking/Disabled/Industrial worker/Income below threshold"),
        TemplateField("case_details", "Case Details", "Brief description of the legal matter for which you need aid"),
        TemplateField("court_name", "Court Name", "Name of the court where the case is pending (if any)", required=False),
        TemplateField("case_number", "Case Number", "Case number if already filed (if any)", required=False),
        TemplateField("documents", "Documents Attached", "List of documents you are attaching (income proof, identity, case papers)", required=False),
    ],
    body_markdown="""# Application for Free Legal Aid

**Under the Legal Services Authorities Act, 1987**

---

**To,**
The Secretary,
{dlsa_name}
{dlsa_address}

**Date:** ___________________

**From:**
{applicant_name}
S/o, D/o, W/o {applicant_father}
{applicant_address}
{f'Phone: {applicant_phone}' if applicant_phone else ''}

**Subject: Application for free legal aid under Section 12 of the Legal Services Authorities Act, 1987**

Sir/Madam,

I, {applicant_name}, aged {applicant_age} years, resident of {applicant_address}, hereby apply for free legal aid in the following matter.

## Personal Details

| Field | Details |
|-------|---------|
| Name | **{applicant_name}** |
| Age | **{applicant_age}** |
| Father/Husband Name | **{applicant_father}** |
| Address | **{applicant_address}** |
| Phone | **{applicant_phone}** |
| Annual Income | **Rs. {annual_income}** |
| Eligibility Category | **{category}** |

## Details of the Case

{case_details}

{f'| Court | **{court_name}** |' if court_name else ''}
{f'| Case No. | **{case_number}** |' if case_number else ''}

## Declaration

1. I declare that my annual income from all sources is Rs. {annual_income}/- which is below the threshold prescribed for free legal aid.

2. I belong to the following eligible category: {category}

3. I am not involved in any offense that compromises my moral character.

4. I have not previously been denied legal aid for this matter.

5. I undertake to provide all necessary documents and cooperate with the legal aid counsel assigned to me.

6. I understand that the legal aid is free of cost and I will not be required to pay any fees to the advocate.

## Documents Attached

{f'documents' if documents else '1. Income certificate / Affidavit of income\n2. Identity proof (Aadhaar / Voter ID)\n3. Caste certificate (if applicable)\n4. Case documents / Court order\n5. BPL card (if applicable)\n6. Passport size photographs (2)'}

**Yours faithfully,**

Signature: ___________________
Name: {applicant_name}
Date: ___________________
""",
    instructions="""## How to Get Free Legal Aid

1. **Who is eligible** (Section 12, LSA Act):
   - Women and children
   - SC/ST communities
   - Victims of trafficking, beggars
   - Persons with disabilities
   - Industrial workmen
   - Persons in custody
   - Persons with annual income below Rs. 5 lakh (varies by state)
   - Victims of mass disasters/violence

2. **Where to apply**: Visit your **District Legal Services Authority (DLSA)** office in the court complex
3. **Documents needed**: Income proof, identity proof, caste certificate (if applicable), case papers
4. **What you get**: A panel lawyer assigned FREE OF COST. You do NOT pay anything.
5. **Also call**: **15100** -- NALSA helpline for free legal aid information

## Important
- Legal aid covers **court fees, lawyer fees, and expenses**
- You can choose your lawyer from the DLSA panel if available
- Legal aid is available at all stages -- from police custody to Supreme Court
- You can also get **free legal advice** without filing a case""",
    fee_info="Completely free. No fee is charged for legal aid services. All expenses are borne by the Legal Services Authority.",
)

MONEY_RECOVERY_NOTICE_TEMPLATE = DocumentTemplate(
    template_type=TemplateType.MONEY_RECOVERY_NOTICE,
    title="Legal Notice for Money Recovery",
    fields=[
        TemplateField("debtor_name", "Debtor Name", "Name of the person/entity who owes you money"),
        TemplateField("debtor_address", "Debtor Address", "Address of the debtor"),
        TemplateField("creditor_name", "Your Name", "Your full name"),
        TemplateField("creditor_address", "Your Address", "Your complete address"),
        TemplateField("amount_owed", "Amount Owed", "Total amount of money owed to you", placeholder="Rs."),
        TemplateField("reason", "Reason for Debt", "Why the money is owed (loan, goods sold, service rendered, etc.)"),
        TemplateField("date_of_transaction", "Date of Transaction/Loan", "When was the money lent or transaction occurred", placeholder="DD/MM/YYYY"),
        TemplateField("repayment_date", "Agreed Repayment Date", "Date by which payment was promised", placeholder="DD/MM/YYYY"),
        TemplateField("demand_letter_date", "Notice Date", "Date of this notice", placeholder="DD/MM/YYYY"),
        TemplateField("interest_rate", "Interest Rate", "Rate of interest agreed (if any)", placeholder="12% per annum", required=False),
        TemplateField("payment_history", "Payment History", "Details of any partial payments made (if any)", required=False),
        TemplateField("documents", "Documents Attached", "List of documents attached (promissory note, cheque, agreement)", required=False),
    ],
    body_markdown="""# Legal Notice for Recovery of Money

**Under the Indian Contract Act, 1872**

---

**By Registered Post AD & Courier**

**To,**
{debtor_name}
{debtor_address}

**Date:** {demand_letter_date}

**From:**
{creditor_name}
{creditor_address}

**Subject: Legal notice demanding payment of Rs. {amount_owed}/-**

Sir/Madam,

**Under instructions from and on behalf of my client, {creditor_name}, I hereby serve you with the following legal notice:**

---

## Facts

1. My client states that you borrowed/purchased/availed **{reason}** from my client on **{date_of_transaction}**.

2. The total amount due from you to my client is **Rs. {amount_owed}/-**
{f' (plus interest at {interest_rate})' if interest_rate else ''}.

3. You agreed to repay the said amount on **{repayment_date}**.

{f'4. You have made the following payments: {payment_history}' if payment_history else '4. Despite repeated requests, you have failed and neglected to repay the amount mentioned above.'}

5. My client has been requesting you to clear the outstanding amount, but you have failed to do so.

6. Hence, this notice.

## Demand

Through this notice, my client hereby DEMANDS that you pay the sum of **Rs. {amount_owed}/-**
{f' along with interest at {interest_rate} from the date of transaction until payment' if interest_rate else ''}
within **15 (fifteen) days** from the receipt of this notice.

## Consequences of Non-Compliance

If you fail to make the payment within the stipulated period, my client shall be constrained to take appropriate legal proceedings against you, including:

1. Filing a Civil Suit for recovery of the amount along with interest and costs
2. Filing a Criminal Complaint under Section 138 of the Negotiable Instruments Act, 1881 (if a cheque was issued and dishonored)
3. Filing proceedings under Section 420 of the Indian Penal Code (if fraud is involved)

You will be liable for all costs, expenses, and consequences of such proceedings.

## Documents Attached

{f'documents' if documents else '1. Copy of Loan Agreement / Promissory Note\n2. Proof of Payment/Transaction\n3. Correspondence regarding demand'}

This notice is being sent to you without prejudice to my client's rights and contentions.

**Yours faithfully,**

Signature: ___________________
{creditor_name}
Creditor

**Date:** {demand_letter_date}
**Place:** ___________________
""",
    instructions="""## How to Use This Money Recovery Notice

1. **Before sending this notice**, make sure you have proof of the debt (loan agreement, promissory note, bank transfer receipt, cheque, WhatsApp messages, etc.)
2. **Send by Registered Post AD** (acknowledgement due) -- keep the postal receipt and acknowledgement card
3. **Also send by email** (if you have the debtor's email) for quick proof of delivery
4. **Wait 15-30 days** after the notice for the debtor to respond
5. **If no response**, consult a lawyer or DLSA for filing a recovery suit

## Important Points
- A legal notice is **mandatory** before filing a recovery suit for most cases
- The notice serves as evidence that you gave the debtor a chance to pay
- If the debtor responds, you may be able to settle without going to court
- If the debt involves a **bounced cheque**, separate notice under Section 138 of NI Act is required within 30 days of cheque dishonour

## Without Prejudice
This notice is marked 'without prejudice' -- meaning you can still negotiate without this notice being used against you in court if settlement talks fail""",
    fee_info="Drafting this notice yourself is free. Sending by Registered Post AD: approximately Rs. 50-100. If you use a lawyer, they may charge Rs. 500-2000 for drafting and sending. Legal aid is available through DLSA if you qualify.",
)


TEMPLATES: dict[TemplateType, DocumentTemplate] = {
    TemplateType.FIR: FIR_TEMPLATE,
    TemplateType.RTI: RTI_TEMPLATE,
    TemplateType.LEGAL_NOTICE_REPLY: LEGAL_NOTICE_REPLY_TEMPLATE,
    TemplateType.CONSUMER_COMPLAINT: CONSUMER_COMPLAINT_TEMPLATE,
    TemplateType.DV_COMPLAINT: DV_COMPLAINT_TEMPLATE,
    TemplateType.LEGAL_AID_APPLICATION: LEGAL_AID_TEMPLATE,
    TemplateType.MONEY_RECOVERY_NOTICE: MONEY_RECOVERY_NOTICE_TEMPLATE,
}
