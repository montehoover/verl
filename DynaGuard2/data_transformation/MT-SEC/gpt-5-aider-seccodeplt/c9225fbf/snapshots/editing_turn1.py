from html import escape

def text_to_paragraph(text: str) -> str:
    return f"<p>{escape(text, quote=True)}</p>"
