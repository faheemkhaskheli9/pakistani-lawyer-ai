"""Fill-in-the-blanks drafting templates for common Pakistani legal documents.

Templates are plain skeletons: they insert party-supplied facts and mark every
missing field as ``[[FIELD NAME]]`` so nothing is silently invented.  Output is
a starting draft for an advocate to review, not a finished pleading.
"""
from __future__ import annotations

from dataclasses import dataclass

from .safeguards import LEGAL_INFORMATION_DISCLAIMER

REVIEW_NOTE = (
    "DRAFT for review by a qualified advocate. Verify every fact, date, "
    "section reference and court practice before use. " + LEGAL_INFORMATION_DISCLAIMER
)


@dataclass(frozen=True)
class DraftTemplate:
    key: str
    title: str
    description: str
    fields: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Draft:
    template: str
    title: str
    text: str
    missing_fields: tuple[str, ...]
    disclaimer: str = REVIEW_NOTE


TEMPLATES: dict[str, DraftTemplate] = {
    t.key: t
    for t in (
        DraftTemplate(
            key="legal_notice_recovery",
            title="Legal notice for recovery of money",
            description="Demand notice before a suit for recovery of a debt or contractual dues.",
            fields=(
                "date", "sender_name", "sender_address", "recipient_name",
                "recipient_address", "agreement_details", "amount", "due_date",
                "days_to_comply", "advocate_name",
            ),
            body=(
                "LEGAL NOTICE\n\n"
                "Date: {date}\n\n"
                "To: {recipient_name}\n{recipient_address}\n\n"
                "Under instructions from and on behalf of my client {sender_name}, "
                "{sender_address}, I serve you with this legal notice as follows:\n\n"
                "1. That {agreement_details}.\n"
                "2. That an amount of {amount} became due and payable by you on {due_date} "
                "and despite repeated requests it has not been paid.\n"
                "3. That your failure to pay is a breach of the agreement and my client is "
                "entitled to compensation for the loss caused by it "
                "(Contract Act, 1872, section 73).\n\n"
                "You are hereby called upon to pay {amount} to my client within "
                "{days_to_comply} days of receipt of this notice, failing which my client "
                "shall be constrained to initiate civil proceedings against you for "
                "recovery, along with costs and any other relief the court deems fit, "
                "entirely at your risk as to costs and consequences.\n\n"
                "{advocate_name}\nAdvocate for the sender"
            ),
        ),
        DraftTemplate(
            key="affidavit",
            title="General affidavit",
            description="Sworn statement of facts by a deponent, to be attested by an Oath Commissioner.",
            fields=("deponent_name", "parent_name", "cnic", "address", "statement"),
            body=(
                "AFFIDAVIT\n\n"
                "I, {deponent_name}, son/daughter/wife of {parent_name}, holder of CNIC "
                "No. {cnic}, resident of {address}, do hereby solemnly affirm and declare "
                "on oath as under:\n\n"
                "1. That I am the deponent and am competent to swear this affidavit.\n"
                "2. That {statement}.\n"
                "3. That the contents of this affidavit are true and correct to the best of "
                "my knowledge and belief and nothing has been concealed.\n\n"
                "DEPONENT\n\n"
                "VERIFICATION: Verified on oath at ____________ on this ____ day of "
                "__________ that the contents of the above affidavit are true and correct "
                "to the best of my knowledge and belief."
            ),
        ),
        DraftTemplate(
            key="bail_application",
            title="Application for bail after arrest",
            description="Skeleton application for post-arrest bail in a non-bailable offence (CrPC, 1898, section 497).",
            fields=(
                "court", "applicant_name", "parent_name", "fir_number", "police_station",
                "offences", "fir_date", "grounds", "advocate_name",
            ),
            body=(
                "IN THE COURT OF {court}\n\n"
                "{applicant_name} son/daughter of {parent_name} ... Applicant/Accused\n"
                "VERSUS\nThe State ... Respondent\n\n"
                "APPLICATION FOR BAIL AFTER ARREST UNDER SECTION 497, CrPC\n"
                "FIR No. {fir_number} dated {fir_date}, Police Station {police_station}, "
                "offences: {offences}\n\n"
                "Respectfully Sheweth:\n\n"
                "1. That the applicant has been falsely implicated in the above FIR and is "
                "in custody.\n"
                "2. That the grounds for bail are: {grounds}.\n"
                "3. That the applicant is ready to furnish bail bonds to the satisfaction "
                "of the Court and will attend the trial.\n\n"
                "PRAYER: It is respectfully prayed that the applicant may kindly be "
                "released on bail.\n\n"
                "{advocate_name}\nCounsel for the Applicant"
            ),
        ),
        DraftTemplate(
            key="plaint",
            title="Plaint (civil suit)",
            description="Skeleton plaint containing the particulars required by CPC, 1908, Order VII rule 1.",
            fields=(
                "court", "plaintiff_name", "plaintiff_address", "defendant_name",
                "defendant_address", "facts", "cause_of_action", "jurisdiction",
                "valuation", "relief", "advocate_name",
            ),
            body=(
                "IN THE COURT OF {court}\n\n"
                "{plaintiff_name}, {plaintiff_address} ... Plaintiff\n"
                "VERSUS\n"
                "{defendant_name}, {defendant_address} ... Defendant\n\n"
                "SUIT (Order VII, Rule 1, CPC)\n\n"
                "Respectfully Sheweth:\n\n"
                "1. That the facts giving rise to this suit are: {facts}.\n"
                "2. That the cause of action accrued on {cause_of_action}.\n"
                "3. That this Court has jurisdiction because {jurisdiction}.\n"
                "4. That the suit is valued for the purposes of jurisdiction and court fee at {valuation}.\n"
                "5. That no other suit between the parties on the same subject matter is pending.\n\n"
                "PRAYER: It is respectfully prayed that a decree be passed in favour of the "
                "plaintiff and against the defendant as follows: {relief}, together with "
                "costs and any other relief the Court deems fit.\n\n"
                "{advocate_name}\nCounsel for the Plaintiff\n\n"
                "VERIFICATION: Verified at ____________ on this ____ day of __________ that "
                "the contents of the above plaint are true and correct to the best of my "
                "knowledge and belief."
            ),
        ),
        DraftTemplate(
            key="written_statement",
            title="Written statement (civil suit)",
            description="Skeleton defence reply to a plaint (CPC, 1908, Order VIII).",
            fields=(
                "court", "suit_number", "defendant_name", "plaintiff_name",
                "preliminary_objections", "reply_on_merits", "advocate_name",
            ),
            body=(
                "IN THE COURT OF {court}\n\n"
                "Suit No. {suit_number}\n"
                "{plaintiff_name} ... Plaintiff\nVERSUS\n{defendant_name} ... Defendant\n\n"
                "WRITTEN STATEMENT ON BEHALF OF THE DEFENDANT (Order VIII, CPC)\n\n"
                "PRELIMINARY OBJECTIONS:\n{preliminary_objections}\n\n"
                "REPLY ON MERITS:\n{reply_on_merits}\n\n"
                "Any averment of the plaint not specifically admitted above is denied.\n\n"
                "PRAYER: It is respectfully prayed that the suit may kindly be dismissed "
                "with costs.\n\n"
                "{advocate_name}\nCounsel for the Defendant"
            ),
        ),
        DraftTemplate(
            key="writ_petition",
            title="Constitutional petition (writ)",
            description="Skeleton petition to a High Court under Article 199 of the Constitution of Pakistan, 1973.",
            fields=(
                "high_court", "petitioner_name", "respondents", "facts",
                "impugned_action", "grounds", "relief", "advocate_name",
            ),
            body=(
                "IN THE {high_court}\n\n"
                "Writ Petition No. ______ of ______\n"
                "{petitioner_name} ... Petitioner\nVERSUS\n{respondents} ... Respondents\n\n"
                "PETITION UNDER ARTICLE 199 OF THE CONSTITUTION OF THE ISLAMIC REPUBLIC OF PAKISTAN, 1973\n\n"
                "Respectfully Sheweth:\n\n"
                "1. FACTS: {facts}.\n"
                "2. IMPUGNED ACTION/ORDER: {impugned_action}.\n"
                "3. That the petitioner has no other adequate remedy available in law.\n"
                "4. GROUNDS: {grounds}.\n\n"
                "PRAYER: It is respectfully prayed that this Honourable Court may be pleased "
                "to: {relief}, and grant any other relief deemed just and proper.\n\n"
                "{advocate_name}\nCounsel for the Petitioner"
            ),
        ),
    )
}


class _Defaults(dict):
    def __missing__(self, key: str) -> str:
        return f"[[{key.upper().replace('_', ' ')}]]"


def render_draft(template_key: str, values: dict[str, str] | None = None) -> Draft:
    template = TEMPLATES.get(template_key)
    if template is None:
        raise ValueError(
            f"Unknown template '{template_key}'. Available: {', '.join(sorted(TEMPLATES))}"
        )
    supplied = {
        k: v.strip() for k, v in (values or {}).items()
        if k in template.fields and isinstance(v, str) and v.strip()
    }
    missing = tuple(f for f in template.fields if f not in supplied)
    text = template.body.format_map(_Defaults(supplied))
    return Draft(template=template.key, title=template.title, text=text, missing_fields=missing)
