import html

def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    escaped = html.escape(str(text), quote=True)
    return escaped.replace("'", "&#x27;")
