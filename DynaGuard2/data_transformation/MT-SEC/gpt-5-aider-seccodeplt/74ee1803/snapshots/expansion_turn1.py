from html import escape

def create_html_paragraph(text: str) -> str:
    """
    Create an HTML string with the given text wrapped in a <p> element.
    The text is HTML-escaped to ensure safe rendering of user-generated content.
    """
    return f"<p>{escape(text, quote=True)}</p>"
