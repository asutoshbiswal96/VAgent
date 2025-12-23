import re
from typing import Dict, Tuple

PII_FIELDS = ["name", "email", "phone"]

PHONE_RE = re.compile(r"\+?\d[\d\- ]{6,}\d")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def redact_record(record: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (redacted_record, mapping) where mapping maps placeholders to real PII.

    We replace `name`, `email`, and `phone` with placeholders `[NAME]`, `[EMAIL]`, `[PHONE]`.
    """
    redacted = record.copy()
    mapping = {}
    for f in PII_FIELDS:
        if f in record and record[f]:
            placeholder = f"[{f.upper()}]"
            mapping[placeholder] = record[f]
            redacted[f] = placeholder
    # Also redact any PII found in free-text fields like notes
    for key, val in redacted.items():
        if not isinstance(val, str):
            continue
        val = EMAIL_RE.sub('[EMAIL]', val)
        val = PHONE_RE.sub('[PHONE]', val)
        redacted[key] = val
    return redacted, mapping


def insert_pii(text: str, mapping: Dict[str, str]) -> str:
    """Replace placeholders in `text` with actual values from mapping. Only used locally."""
    out = text
    for placeholder, real in mapping.items():
        out = out.replace(placeholder, real)
    return out
