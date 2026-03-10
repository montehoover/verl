import re
import unicodedata
import html


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


def format_blog_entry(headline: str, content: str) -> str:
    """
    Create a sanitized HTML blog entry string, placing the headline in an <h1>
    and the content inside a <p>, using the predefined BLOG_FORMAT template.

    BLOG_FORMAT is expected to be a string like:
        "<h1>{headline}</h1><p>{content}</p>"

    Notes:
    - Normalizes Unicode and removes problematic control/zero-width characters.
    - Escapes HTML special characters to prevent HTML/script injection.
    - Headline is collapsed to a single line (no newlines).
    """
    def _coerce_to_str(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    def _sanitize_text(text: str, *, preserve_newlines: bool) -> str:
        t = unicodedata.normalize("NFC", text)
        t = t.lstrip("\ufeff")
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("\x00", "")
        t = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", "", t)
        t = re.sub(r"[\u200E\u200F\u061C\u202A-\u202E\u2066-\u2069\u200B-\u200D\u2060\ufeff]", "", t)
        if preserve_newlines:
            # Keep readable formatting; trim trailing whitespace on each line
            t = "\n".join(line.rstrip() for line in t.split("\n"))
        else:
            # Collapse all whitespace to single spaces (used for headline)
            t = " ".join(t.split())
        return t.strip()

    headline = _coerce_to_str(headline)
    content = _coerce_to_str(content)

    safe_headline = _sanitize_text(headline, preserve_newlines=False)
    safe_content = _sanitize_text(content, preserve_newlines=True)

    # HTML-escape to prevent injection
    escaped_headline = html.escape(safe_headline, quote=True)
    escaped_content = html.escape(safe_content, quote=True)

    fmt = globals().get("BLOG_FORMAT", "<h1>{headline}</h1><p>{content}</p>")
    return fmt.format(headline=escaped_headline, content=escaped_content)
