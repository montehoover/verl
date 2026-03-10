from html import escape

def sanitize_html(text: str) -> str:
    """
    Return a version of the input string with special HTML characters escaped.
    """
    return escape(text, quote=True)

def create_html_paragraph(text: str) -> str:
    """
    Create an HTML string with the given text wrapped in a <p> element.
    The text is HTML-escaped to ensure safe rendering of user-generated content.
    """
    return f"<p>{sanitize_html(text)}</p>"
