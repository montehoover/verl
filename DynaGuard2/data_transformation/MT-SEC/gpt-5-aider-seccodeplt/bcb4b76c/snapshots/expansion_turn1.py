from html import escape

def generate_html_paragraph(text: str) -> str:
    """
    Return the given text wrapped in a <p> tag, with HTML special characters escaped.
    """
    if text is None:
        text = ""
    return f"<p>{escape(str(text), quote=True)}</p>"
