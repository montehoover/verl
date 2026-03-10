from html import escape

def text_to_paragraph(text: str) -> str:
    """
    Convert a text string into an HTML paragraph by wrapping it in <p> tags.
    Special characters are escaped to ensure safe HTML output.
    """
    return f"<p>{escape(text, quote=True)}</p>"
