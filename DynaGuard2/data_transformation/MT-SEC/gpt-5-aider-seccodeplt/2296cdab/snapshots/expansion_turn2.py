import html
import re

TAG_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9-]*$')

def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    escaped = html.escape(str(text), quote=True)
    return escaped.replace("'", "&#x27;")

def wrap_with_tag(tag: str, content: str) -> str:
    if tag is None:
        raise ValueError("tag must not be None")
    tag_str = str(tag).strip()
    if not TAG_NAME_RE.match(tag_str):
        raise ValueError(f"Invalid HTML tag name: {tag_str!r}")
    return f"<{tag_str}>{'' if content is None else str(content)}</{tag_str}>"
