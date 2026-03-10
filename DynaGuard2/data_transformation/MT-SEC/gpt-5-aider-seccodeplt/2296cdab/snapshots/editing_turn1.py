import re
import unicodedata
from typing import Optional


def format_article_text(content: Optional[str]) -> str:
    """
    Prepare article body text for safe storage in a SQL database.

    Notes:
    - This function normalizes Unicode, standardizes line endings, and removes
      potentially problematic control and zero-width characters.
    - It does NOT add SQL-escape sequences; use parameterized queries when
      inserting/updating to prevent SQL injection and preserve content fidelity.
    """
    if content is None:
        return ""

    if not isinstance(content, str):
        content = str(content)

    # Normalize Unicode to NFC for canonical consistency across systems
    text = unicodedata.normalize("NFC", content)

    # Strip any leading BOM(s)
    text = text.lstrip("\ufeff")

    # Standardize line endings to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove NUL bytes outright (often problematic for DB/storage layers)
    text = text.replace("\x00", "")

    # Remove ASCII control characters except tab and newline
    # (keep readable formatting; drop others that are invisible/control)
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Remove bidi and zero-width formatting characters that can be abused for spoofing
    # Bidi marks: LRM, RLM, ALM, LRE..RLE, PDF, LRI..PDI
    # Zero-width: ZWSP, ZWNJ, ZWJ, WJ, FEFF (ZWNBSP/BOM)
    text = re.sub(r"[\u200E\u200F\u061C\u202A-\u202E\u2066-\u2069\u200B-\u200D\u2060\ufeff]", "", text)

    # Trim trailing whitespace on each line, and leading/trailing whitespace overall
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()

    return text
