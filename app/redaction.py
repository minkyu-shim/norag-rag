import re
from dataclasses import dataclass, field
from app.config import REDACTION_ENABLED, REDACTION_NAMES_FILE

# V4 Part C — Personal data redaction.
#
# Personal data is replaced by a placeholder ([EMAIL_1], [PHONE_1], [PERSON_1]) in two places:
#   1. before a prompt leaves the machine (app/llm.py, the call to the LLM provider),
#   2. before anything is written to a trace (app/tracing.py, in clean()).
#
# HOW WE DETECT: rules only (regular expressions + a cue list + an optional names file).
#   No library, no LLM: an LLM detector would send the personal data to a provider to find it,
#   which is exactly what we want to avoid. The price: rules miss things (see KNOWN LIMITS).
#
# What is, and is not, personal data here:
#   - E-mail addresses of a PERSON (jane.doe@x.com): yes.  Role addresses (privacy@, support@...): no,
#     they identify a function, and the documents need them to answer ("who do I write to?").
#   - Phone numbers: yes, but not dates, versions, error codes (KST-2044) or IP addresses.
#   - IP addresses: not personal data here (they identify machines in our corpora): kept.
#   - Names: only when a cue says it is a person ("my name is Jane Doe", "Dr. Smith", "Dear Jane"),
#     or when the name is in data/redaction_names.txt (one name per line, filled by the user).
#
# KNOWN LIMITS (also in REPORT.md): a bare name without a cue ("Ask Jane about it") is NOT detected;
# phone numbers written with words are not detected; a name in a language with other capitalisation
# rules may be missed. Rules favour few false alarms over finding everything.
#
# The mapping placeholder -> original lives only in memory, for ONE LLM call: the answer comes back
# with the placeholders put back (restore()), so the user reads their own data again, on their machine.

ROLE_LOCAL_PARTS = {
    "privacy", "support", "billing", "info", "contact", "sales", "admin", "administrator", "noreply",
    "no-reply", "security", "legal", "hello", "help", "team", "ops", "oncall", "on-call", "abuse",
    "press", "hr", "jobs", "careers", "office", "postmaster", "webmaster", "dpo", "compliance", "finance",
    "accounts", "service", "status", "feedback", "mail", "engineering", "devops", "it",
}

EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?![\w-])")

# A phone candidate: optional +, then digits separated by space . - ( ), 9 to 15 digits in total.
PHONE = re.compile(r"(?<![\w.\-/])(\+?\(?\d[\d\s().\-]{7,22}\d)(?![\w\-/])")
DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")
IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

CAP = r"[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ'’\-]+"
NAME = rf"{CAP}(?:[ \t]+(?:(?:de|van|von|di|da|le|la|du|del|ben|bin)[ \t]+)?{CAP}){{0,2}}"
NAME_CUES = [
    # "my name is Jane Doe", "I am Jane Doe", "I'm Jane Doe", "this is Jane Doe"
    re.compile(rf"(?i:\b(?:my name is|my name's|i am|i'm|this is|je m'appelle|je suis))[ \t]+(?P<name>{NAME})"),
    # titles: "Dr. Smith", "Mr Smith", "Mme Dupont"
    re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mme|Mlle|M)\.?[ \t]+(?P<name>{NAME})"),
    # letters: "Dear Jane", "Regards, Jane Doe", "Thanks, Jane"
    re.compile(rf"(?i:\b(?:dear|regards,|thanks,|cheers,|sincerely,|cordialement,|bonjour))[ \t]+(?P<name>{NAME})"),
]
# Words that follow "I am" / "this is" but are not names.
NOT_NAMES = {"Kestrel", "Analytics", "Platform", "Data", "Business", "Enterprise", "Pro", "Free", "Team", "Starter"}


@dataclass
class Redaction:
    text: str
    mapping: dict = field(default_factory=dict)     # placeholder -> original
    counts: dict = field(default_factory=dict)      # kind -> number of replaced items (not of occurrences)
    kept: dict = field(default_factory=dict)        # what looked personal but was kept, and why (for debugging)


def load_known_names():
    try:
        lines = REDACTION_NAMES_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return sorted({l.strip() for l in lines if l.strip() and not l.startswith("#")}, key=len, reverse=True)


def looks_like_phone(candidate):
    digits = re.sub(r"\D", "", candidate)
    if not 9 <= len(digits) <= 15:
        return False
    if DATE.match(candidate.strip()) or IPV4.match(candidate.strip()):
        return False
    # Bare digit runs (an id, a timestamp) are not phone numbers: need a +, a leading 0, a bracket or separators.
    c = candidate.strip()
    return c.startswith(("+", "0", "(")) or bool(re.search(r"[\s.\-]", c))


def redact(text, enabled=None):
    """Returns a Redaction: the text with placeholders, the mapping to put the originals back, the counts."""
    result = Redaction(text=text)
    if not text or not (REDACTION_ENABLED if enabled is None else enabled):
        return result

    state = {}      # kind -> {original: placeholder}, so the same value always gets the same placeholder

    def placeholder(kind, original):
        table = state.setdefault(kind, {})
        if original not in table:
            table[original] = f"[{kind}_{len(table) + 1}]"
            result.mapping[table[original]] = original
            result.counts[kind] = result.counts.get(kind, 0) + 1
        return table[original]

    def email(m):
        if m.group(1).lower() in ROLE_LOCAL_PARTS:
            result.kept[m.group(0)] = "role address"
            return m.group(0)
        return placeholder("EMAIL", m.group(0))

    def phone(m):
        if looks_like_phone(m.group(1)):
            return m.group(0).replace(m.group(1), placeholder("PHONE", m.group(1)))
        return m.group(0)

    def person(m):
        name = m.group("name")
        if name.split()[0] in NOT_NAMES:
            return m.group(0)
        return m.group(0).replace(name, placeholder("PERSON", name))

    out = EMAIL.sub(email, text)
    out = PHONE.sub(phone, out)
    for pattern in NAME_CUES:
        out = pattern.sub(person, out)
    for name in load_known_names():
        out = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", lambda m: placeholder("PERSON", m.group(0)), out)
    result.text = out
    return result


def restore(text, mapping, json_safe=False):
    """Puts the originals back in `text` (the LLM answer). json_safe: the text is a JSON document,
    so the originals are escaped like JSON strings (a name with a quote must not break the JSON)."""
    if not text or not mapping:
        return text
    for ph, original in mapping.items():
        if json_safe:
            original = original.replace("\\", "\\\\").replace('"', '\\"')
        text = text.replace(ph, original)
    return text
