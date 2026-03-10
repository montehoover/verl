import re
import unicodedata


def format_article_text(headline: str, content: str) -> str:
    """
    Prepare article headline and body text for safe storage in a SQL database.

    The returned string places the headline at the top, followed by a blank line,
    then the sanitized article content.

    Notes:
    - Normalizes Unicode, standardizes line endings, and removes potentially
      problematic control and zero-width characters.
    - Headline is collapsed to a single line (no newlines).
    - Does NOT add SQL-escape sequences; use parameterized queries when
      inserting/updating to prevent SQL injection and preserve content fidelity.
    """
    def _coerce_to_str(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    headline = _coerce_to_str(headline)
    content = _coerce_to_str(content)

    def _sanitize_text(text: str, *, preserve_newlines: bool) -> str:
        t = unicodedata.normalize("NFC", text)
        t = t.lstrip("\ufeff")
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("\x00", "")
        t = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", "", t)
        t = re.sub(r"[\u200E\u200F\u061C\u202A-\u202E\u2066-\u2069\u200B-\u200D\u2060\ufeff]", "", t)
        if preserve_newlines:
            t = "\n".join(line.rstrip() for line in t.split("\n"))
        else:
            t = " ".join(t.split())
        return t.strip()

    safe_headline = _sanitize_text(headline, preserve_newlines=False)
    safe_content = _sanitize_text(content, preserve_newlines=True)

    if safe_content:
        return f"{safe_headline}\n\n{safe_content}"
    else:
        return safe_headline
